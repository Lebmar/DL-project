# 神经尺度兼容性桥接：定义与复现说明

记录日期：2026-10-18。训练已完成；评估与分析结论以各自完整 manifest 和[结果记录](neural_compatibility_results.md)为准。本文件解释设计、执行顺序和审计方式，不提前判断神经桥接的科学结果。仅使用公开代码与模拟数据、本地 CPU；报告和 PPT 不在本轮范围内。

## 1. 冻结顺序与训练核查

| 环节 | 提交与时间（北京时间） | 冻结内容 |
|---|---|---|
| 有限样本实验 | `fe846f0b01c43cdcd97fb80fe65b2a4caf875cbd` | 有限样本条件、方向/幅度预测、进入神经桥接的门槛 |
| 神经设计与训练 driver | `ab01a1d5ff1869216e09757ee56ace68602490fb`，10月18日13:01:16 | 完整训练/测试设计、seeds、主指标、判定门槛；在新训练之前提交 |
| 神经评估与分析实现 | `a4ee54e71ccd8ac02fd91daeab4602343086120e`，10月18日13:08:33 | 实现已冻结设计；在新测试数据生成之前提交，训练当时仍在进行 |

这是有限样本结果指导下的新桥接设计，不把先前数学例子倒写成预注册。设计与实现的两次冻结时间必须分别陈述，不能声称所有评估代码都在训练开始前冻结。

训练从13:01:18.864开始，到13:09:26.296完成；外层 manifest 记录总耗时 **487.419秒**。3个新训练数据 panel，每 panel 两个跨 panel 不重复的 seed，并在同一 seed 内配对 GRU/TCN，共12个拟合。训练、验证、验证 target-pair seeds 分别为 `714159/714160/714161`、`715159/715160/715161`、`716159/716160/716161`；对应模型 seeds 为 `5101/5102`、`5201/5202`、`5301/5302`。

独立只读核查确认：

- 3个 panel 各4个 fits，全部完整，12个 checkpoint 已保存；42个 panel artifact 哈希与外层 manifest 一致。
- 训练前/所选 checkpoint 的24项因果前缀审计均通过。
- 六个 GRU/TCN seed 对的188对共同训练 epoch，其 minibatch 哈希逐一相同。两架构停止时间不同，配对的是共同 epoch 前缀。
- 执行23–58 epochs，所选 checkpoint 为19–50 epochs；全部由验证 patience 停止。所选 epoch 允许小于 min20：min20约束停止时机，而不是 checkpoint 候选范围。绝对最优验证 checkpoint 与 `min_delta` 控制的 patience reference 分开维护，因此停止 epoch 也可能恰是所选 epoch。
- 训练 manifest 的 `test_data_accessed` 为 false。原 trainer 未改动，其内部固定日期 `2026-10-11` 是历史兼容字段；本次真实时间以外层 UTC 时间戳为准。

来源：`outputs/neural_compatibility_training/manifest.json`。协议 SHA-256：`f75b73aaaf3d2f4cc4d684a1a15f807a042812c45c744c1a892b3f5d10f8feb1`。

## 2. 三个不同的校准对象

模拟数据为 \(y_t=\sigma_t\epsilon_t\)，神经模型只接收过去的 \(y\)，发布尺度 \(s_t>0\)。真实 \(\sigma\) 只用于离线诊断，定义 \(\delta_t=\log(s_t/\sigma_t)\)。令 \(F\) 是对应 Gaussian 或标准化 t5 创新的绝对值 CDF，\(z=F^{-1}(.9)\)。

| 对象 | 定义 | 解释 |
|---|---|---|
| P：总体混合 | 解 \(W^{-1}\sum_{j=1}^W F(q_Pe^{\delta_{t-j}})=.9\) | 在实现的尺度误差轨迹上计算外生混合分布；不是给定神经轨迹后的真实残差条件分布 |
| A：原历史 | 对 \(|y_i|/s_i=|\epsilon_i|e^{-\delta_i}\)，\(i=t-W,\ldots,t-1\)，计算指定经验秩 | 保留原始发行尺度与历史创新之间的全部依赖 |
| D：独立重绘历史库 | 固定原 \(\delta\) 与当前尺度，将每个历史创新替换为独立 \(\epsilon_i^{*(r)}\)，计算 \(|\epsilon_i^{*(r)}|e^{-\delta_i}\) 的同一经验秩，并平均64个 replicas | 每个历史年龄一个独立创新，不是从年龄混合分布 IID 抽样；不重新运行模型，不是可部署校准方法 |

A、D 均使用相邻秩的精确概率混合：W32 的 rank29/30 权重为0.3/0.7；W128 的 rank116/117 权重为0.9/0.1。**每个候选区间先单独计算风险、覆盖及矩，再按权重平均**；不对平均半宽评分。P 始终为90%总体分位数。

同一个重绘 replica 的完整时间序列复用于全部 fits、contexts、windows 和三个 Gaussian 世界，保留比较的共同随机数配对。重叠窗口共享同一条历史；不会每个端点另造一个互不相关的库。

当前创新被解析积分，计算覆盖率、宽度、条件期望 interval score。因为发布半宽不依赖当前创新，这个风险计算仍然成立。它不意味着历史创新在给定整条神经尺度误差轨迹后独立。主风险除以真实当前尺度 \(\sigma_t\)；同时保存原单位结果。t5 的 native 诊断使用正确 t5 分位数，以隔离尺度误差，与既有 Gaussian-native 读数分开。

## 3. 精确矩分解与局部近似

对 A 或 D 的每个候选秩，定义

\[
\xi_P=\delta_t+\log(q_P/z),\qquad
\eta=\log(q/q_P),\qquad \xi=\xi_P+\eta.
\]

保存一阶矩、二阶矩与交叉矩，检查精确恒等式

\[
E\xi^2=E\xi_P^2+E\eta^2+2E(\xi_P\eta),\quad
E(\xi_P\eta)=\operatorname{Cov}(\xi_P,\eta)+E\xi_P E\eta.
\]

协方差和均值乘积分别报告。即使 D 独立重绘，其条件分位数偏差仍可能随尺度轨迹改变，故不能假设全局协方差为零。

另比较 \(\delta_t-\bar\delta\) 及其二阶方差修正与 exact \(\xi_P\)，并比较 \(z^2f(z)\xi^2/\alpha\) 与精确超额风险。大误差下局部近似可以失败，所有 fits 保留，不按近似好坏筛选。

## 4. 完整评估网格

- 12个冻结拟合 × 4个世界 × contexts16/64 × windows32/128 = **192组**；只做同一 context 同时作为当前/归档尺度的对角组合。
- 每个世界64条长度640路径。三个 Gaussian 世界共享创新：恒定尺度1、事件尺度2、事件尺度0.5；第四个世界为标准化 t5、事件尺度2。事件为 \([256,384)\)，其余时段真尺度1。
- 原创新 seeds 为 Gaussian `2903000`、t5 `2904000`；重绘 seeds 为 `2905000`、`2906000`。每个重绘数组为64 replicas × 64 paths × 640 times。
- 保存 oracle、native、P、A、D、D前32和D后32共7种读数；覆盖率、标准化宽度/IS、原单位宽度/IS共5项指标。oracle 为真实分布精确分位数，没有有限分数库噪声。

| 区段 | 固定评价端点 |
|---|---|
| pre | 192, 208, 224, 240 |
| entry | 256, 264, 272, 280 |
| late | 320, 336, 352, 368 |
| recovery | 384, 400, 416, 432 |

这是每区段4个指定端点的平均，不是整个区段平均。最早端点前已具备128步真实分数及64步模型上下文，不广播人工 warmup 分数。

## 5. 主指标和预先规定的判断

主设定为 Gaussian 上升事件、context64、windows32/128、entry/recovery：12个 fits × 4种 window/phase 组合 =48个 cell。每个 cell 先平均64条路径和4个端点的标准化条件期望 IS，得到 \(\bar R_P,\bar R_A,\bar R_D\)，再计算

\[
G=\operatorname{mean}_{48}\left[|\bar R_P-\bar R_A|-|\bar R_D-\bar R_A|\right].
\]

正号表示有限独立历史库比总体近似更接近真实历史的**平均风险**，不等同于 D 是可部署方法或逐路径更好。若先逐路径取绝对误差，会改变本轮估计目标。

4000次 shared-path bootstrap（seed `2026101807`）对全部 fits/cells 同时重抽原路径，每次重新计算均值与绝对误差，报告95%区间。另做4000次共同 replica bootstrap（seed `2026101808`），固定原路径与P/A，只估计64次重绘的 Monte Carlo 不确定性。两者分开，均不涵盖训练流程总体不确定性。

只有以下全部通过，才说 D 在这一受限设定下实质改善平均风险近似：

1. \(G>0\)，且主 path interval 下界>0。
2. 相对 population MAE 降低至少10%；population MAE≤`1e-12` 时比例未定义，门槛失败。
3. 两个架构的平均 gain 均正，至少2/3个 panel 的平均 gain 正。
4. 重绘 MC SE / \(|G|\)≤0.1，且前后32个 replica 的 gain 差 / \(|G|\)≤0.2；分母数值保护为`1e-12`。

其他世界、contexts、区段和机制分解均为预定描述性边界检查，不追加多重比较确认主张。只有3个训练 panel；12个 fits、64条路径和64个 replicas 各自代表不同层次，不能相乘当独立训练样本。若未通过，不追加 replicas、不换主设定、不重新挑 seed 来获得通过结果。

## 6. 命令与不覆盖约束

以下命令从仓库根目录、已配置的 `.venv` 执行。前四条只核查输入或运行独立 toy checks，不生成研究测试数据：

```bash
.venv/bin/python scripts/train_neural_compatibility.py --check-only
.venv/bin/python scripts/evaluate_neural_compatibility.py --self-check
.venv/bin/python scripts/analyze_neural_compatibility.py --self-check
.venv/bin/python scripts/verify_finite_mechanism.py
```

核验当前已保存的全部神经结果，可运行下面这条只读命令。它重新检查来源哈希、192组指标、原历史分位数、矩分解及两类 bootstrap，打印结果但不写入或替换现有 verification JSON，也不生成新的训练或测试数据：

```bash
.venv/bin/python scripts/verify_neural_compatibility.py --check-only
```

完整顺序如下；**仅适用于三个神经输出目录尚不存在的独立实验副本**。现有主仓库保存正式结果，不应为“复现”删除它们。

```bash
.venv/bin/python scripts/train_neural_compatibility.py
.venv/bin/python scripts/evaluate_neural_compatibility.py
.venv/bin/python scripts/analyze_neural_compatibility.py
.venv/bin/python scripts/verify_neural_compatibility.py
```

最后一条默认保存 `outputs/neural_compatibility_verification.json`，该记录已经存在时会拒绝覆盖；再次核验请使用上面的 `--check-only`。

需要在独立副本保留原代码、完整冻结协议及已固定的有限样本 artifacts；driver 会核查有限实验通过门槛和哈希。训练只改输出目录，不能改原 `train_stage2.py` / `stage2_models.py`。评估与分析会检查自己的源文件及协议与 Git HEAD 一致；任何实现修改需另起版本并明确先后顺序。

输出分别在 `outputs/neural_compatibility_training/`、`outputs/neural_compatibility_evaluation/`、`outputs/neural_compatibility_analysis/`。评估/分析目录已存在时拒绝运行；训练拒绝覆盖无记录或不完整 panel。训练的 `--resume` 只允许核对并跳过完整 panel，然后完成未开始的 panel，不在 partial panel 中接着训练。

**不要把 `--resume` 当成只读验证命令。** 它会更新外层 manifest 的 invocation 时间记录；评估开始后再运行会改变已引用的训练 manifest 哈希。已完成的 artifacts 应通过只读哈希核查，不触动其 manifest。`--check-only` 不修改输出，但也不替代所有已保存训练 artifacts 的逐项核验。

评估会保存原测试路径、已发布尺度、192组指标/矩、每 replica 的区段风险，以及来源哈希；不保存全部 replica×path×time 中间张量。新测试生成前先核对训练完整性，在验证路径上检查 TCN64 快速推理与 reset-window 实现；不通过则回退原实现。运行结束再核对训练和冻结源码未变。

本轮使用本地 CPU、4个 Torch 线程，没有普通计费 HPC 作业。训练时长是已完成运行的实测值，不承诺其他机器相同时长。科学结果、失败边界和后续取舍在完整评估/分析后记录于[神经桥接结果](neural_compatibility_results.md)。
