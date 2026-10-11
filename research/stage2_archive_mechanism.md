# Stage 2：当前尺度与历史分数归档的交叉诊断

本补充是看到 Stage 2 主面板结果以后设计的 **exploratory 机制诊断**。它使用原来的全部 10 个 checkpoint，以及原来两组 Gaussian/t5、A3 D128 参考路径。没有重新训练、选择初始化或增加未见测试。这里的区间与分解都应在这个范围内解释。

交叉后可以看到，Gaussian entry 中 TCN 的 current 分量在两条路线里都比 archive 分量更依赖初始化；recovery 中 current 和 archive 分量反向相消。Student-t5 entry 的架构平均 current 差异接近零。因而主面板的不稳定排序不能归结为一个普遍的“历史归档污染”结论，也不支持直接设计针对某个通道的通用修复。

本次运行用时 236.61 秒，保留 1280 条格子性能均值、51,200 条逐 path 分量。所有检查通过：640 个 diagonal 性能均值与原五指标最大绝对差 `1.07e−14`，10,240 条 diagonal path interactions 最大绝对差 `3.77e−15`；160 个 crossed 遗忘界之后 event/control 半宽度最大差为 `0`。

## 诊断具体改变什么

对当前时点使用 context `a∈{16,64}` 的尺度 `s^a_t`，对过去每个结果使用当时 context `b∈{16,64}` 已发出的尺度 `s^b_i`。半宽度定义为：

`h[a,b,W,t] = s^a_t × Quantile{ |y_i| / s^b_i : i=t−W,…,t−1 }`，`W∈{32,128}`。

每个模型、每个 world 都保留全部 8 个交叉格子。`a=b` 恰好回到主面板的原始方法；`a≠b` 改变“当前尺度”与“历史分母”两个通道的组合。各 `b` 用自己在固定 validation path0 最后 128 个时点发出的 normalized scores 作 warmup。分位数排除当前结果，模型只读取 `y[t−L:t]`，事件标签和真实尺度只用于生成、评分和复用已经证明相同的上下文。

记每个格子的 phase 平均 `event−matched-control` 性能差为 `E(a,b,W)`，再记 `D(a,b)=E(a,b,128)−E(a,b,32)`。主面板的 diagonal interaction 是 `T=D(64,64)−D(16,16)`。它可以沿两条路线分解：

| 路线 | Current 分量 | Archive 分量 | 和 |
| --- | --- | --- | --- |
| 先改变 archive，再改变 current | `C@b64 = D(64,64)−D(16,64)` | `A@a16 = D(16,64)−D(16,16)` | `T` |
| 先改变 current，再改变 archive | `C@b16 = D(64,16)−D(16,16)` | `A@a64 = D(64,64)−D(64,16)` | `T` |

这是全部交叉格子的精确计算分解。两条路线的分量不同代表路径依赖，不能唯一归因给某个通道，也不能解释为因果中介比例。两个均值分量可以相加；相关分量的置信区间端点不能直接相加。

## 评分与不确定性

四个 phase 与主面板完全一致：entry `[384,416)`、recovery64 `[512,576)`、recovery192 `[512,704)`、late `[704,768)`。保存原来的全部五个指标：interval score、实现 coverage、width、oracle coverage deficit、oracle excess width。Student-t5 的尺度归一化与概率计算直接复用被冻结且 hash 验证的 evaluator，没有把 t5 当成 Gaussian 计算 oracle。

新增 log-width 仅检查 `log h = log s_a + log Q_b,W` 的恒等式。固定 `b` 后，current-context × window 的 log-width interaction 必须为零；这是代数性质，不是新的经验发现。

分量先在每条 path 内计算，再平均五对固定拟合。4000 次 bootstrap 只重抽 128 条 paired paths，同样的 path 索引保留组件和架构间的配对；95% 区间未作多重比较校正，仅作描述。每个 seed 的均值、path 区间、五个 seed 的范围与样本标准差也全部保存。五个初始化不构成训练数据重抽样，不作初始化总体显著性声明。

## 实际机制读数

下面是 interval-score interaction 的两条分解路径，均为五个冻结模型与 128 条配对 path 的平均。符号表示性能差分的交互方向，不表示绝对 interval score 优劣。两个分量各自都带有 current/archive 与 `W` 的交互，不能把它们误称为某个通道的独立主效应。

| Family | Model | Phase | Total | C@b64 | A@a16 | C@b16 | A@a64 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Gaussian | GRU | entry | −0.22802 | −0.16402 | −0.06401 | −0.17159 | −0.05643 |
| Gaussian | TCN | entry | −0.17516 | −0.11012 | −0.06504 | −0.12605 | −0.04910 |
| Gaussian | GRU | recovery64 | −0.03845 | +0.05733 | −0.09578 | +0.04021 | −0.07866 |
| Gaussian | TCN | recovery64 | −0.07122 | −0.01204 | −0.05918 | −0.02893 | −0.04228 |
| t5 | GRU | entry | −0.22941 | −0.12645 | −0.10296 | −0.13082 | −0.09859 |
| t5 | TCN | entry | −0.21184 | −0.13487 | −0.07697 | −0.13825 | −0.07359 |
| t5 | GRU | recovery64 | −0.01946 | +0.02684 | −0.04630 | +0.00564 | −0.02510 |
| t5 | TCN | recovery64 | −0.05846 | −0.01942 | −0.03905 | −0.03194 | −0.02652 |

Gaussian entry 中，TCN current 分量的五-seed 样本标准差在两条路线为 `0.405/0.409`，archive 分量为 `0.093/0.097`；GRU current 为 `0.034/0.034`，archive 为 `0.042/0.041`。这定位了本组拟合里较大的离散发生在哪个计算通道。TCN current 分量有正有负，单个 seed 可以改变总架构差异；archive 也会相消或放大它。分量彼此相关，这些标准差不能转换为“贡献了多少百分比的初始化方差”。

| Gaussian TCN entry seed | C@b64 | A@a16 | Total | C@b16 | A@a64 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4101 | −0.16568 | −0.00305 | −0.16874 | −0.16159 | −0.00714 |
| 4102 | −0.73123 | +0.02075 | −0.71048 | −0.77325 | +0.06277 |
| 4103 | −0.09743 | −0.01259 | −0.11003 | −0.10083 | −0.00920 |
| 4104 | +0.37232 | −0.15354 | +0.21877 | +0.32741 | −0.10863 |
| 4105 | +0.07144 | −0.17676 | −0.10532 | +0.07800 | −0.18332 |

这里没有删去 TCN4102/4104，也没有以其验证表现重新选择模型。主面板已经展示这种初始化敏感性；本诊断补充其跨通道结构，不能重新解释为未见测试的确认。

对 GRU−TCN 架构差异，第一条路线的描述性区间如下。它们条件于五对固定拟合，和原主要分析的 97.5% 区间用途不同。第二条路线和全部五指标的区间都保留在完整 CSV。

| Family / phase | Component | Estimate | Unadjusted 95% paired-path CI | Five-seed range |
| --- | --- | ---: | --- | --- |
| Gaussian entry | Total | −0.05287 | [−0.10203, −0.00629] | [−0.42866, +0.47620] |
| Gaussian entry | C@b64 | −0.05390 | [−0.08454, −0.02476] | [−0.53411, +0.56237] |
| Gaussian entry | A@a16 | +0.00103 | [−0.03383, +0.03700] | [−0.11463, +0.16686] |
| Gaussian recovery64 | Total | +0.03276 | [+0.01796, +0.04802] | [−0.01723, +0.11717] |
| Gaussian recovery64 | C@b64 | +0.06937 | [+0.05730, +0.08159] | [−0.00191, +0.14745] |
| Gaussian recovery64 | A@a16 | −0.03661 | [−0.05071, −0.02153] | [−0.13600, +0.04751] |
| t5 entry | Total | −0.01757 | [−0.06558, +0.02940] | [−0.36193, +0.58143] |
| t5 entry | C@b64 | +0.00842 | [−0.01761, +0.03471] | [−0.32919, +0.48613] |
| t5 entry | A@a16 | −0.02599 | [−0.06418, +0.01075] | [−0.15960, +0.10329] |
| t5 recovery64 | Total | +0.03900 | [+0.02343, +0.05444] | [+0.01822, +0.08388] |
| t5 recovery64 | C@b64 | +0.04626 | [+0.03536, +0.05686] | [−0.00866, +0.09545] |
| t5 recovery64 | A@a16 | −0.00726 | [−0.02456, +0.00959] | [−0.07562, +0.03819] |

Gaussian recovery64 的 GRU current 分量为正、archive 为负；TCN 两分量的均值为负，所以 GRU−TCN 的 current 差与 archive 差相消。两条路线中 GRU−TCN current 均值分别 `+0.06937/+0.06914`，archive 分别 `−0.03661/−0.03638`，但每个 seed 的分量仍会变动。t5 recovery 的 archive 架构差异则较小，第二条路线甚至为 `+0.00142`，说明不能把 Gaussian 的分量方向直接推广到重尾噪声。

Recovery192 也完整保留。Gaussian 的 GRU/TCN 总交互为 `−0.03151/−0.03212`，架构总差约 `+0.00061`，但第一条路线的 current 架构差为 `+0.02327`、archive 为 `−0.02266`；t5 相应总差为 `+0.00490`，current `+0.01541`、archive `−0.01051`。总差接近零可以由非零分量抵消造成，不能据总值接近零认定两个机制都消失。late 四组模型的全部指标和分量都为零，符合有限记忆边界。

固定 archive 的 current×window log-width interaction 最大绝对误差为 `1.55e−15`。其为零来自前述恒等式，而 interval score、实现 coverage 和截断 oracle 指标是非线性的，分量不必为零。

## 对 A+ 研究路线的作用与边界

本补充把“当前尺度”和“历史分母”混合在 diagonal interaction 中的问题拆开，并保留两条分解路径、初始化变化与重尾失败条件。它提升机制解释的可审查程度，但这些计算格子与恒等式本身不构成新的校准保证或已经验证有优势的算法。当前最有证据支持的故事是 **相位、噪声尾部、拟合状态共同决定两种记忆通道的组合**，而不是某种长记忆通用有害结论。

冲击退出后的 archive 分量并非恒定单向损害；初始化差异也不能仅用旧输入梯度大小解释。若要提出修复策略，必须在 train/validation 选择其规则，固定独立评估后同时检验 score、coverage 与 width 的代价。新策略还需核对 replay/reset/window mixing 等直接先行工作，避免把现有组件宣称为原创。本探索不能替代这一环节，也不能保证 A+。

## 复现与检查

```sh
.venv/bin/python scripts/run_stage2_archive_diagnostic.py --self-check
.venv/bin/python scripts/run_stage2_archive_diagnostic.py
```

脚本拒绝静默覆盖已有 manifest。新源码先提交再运行；开始和结束都核验原协议、训练源码、10 个 checkpoint、原 evaluator 及其依赖、原测试全部 artifact 的 SHA256。生成出的无冲击观测与真实尺度的 array hash 必须与原 Stage 2 记录相同，事件数组还须逐项等于原 reference。seed4101 的参考尺度直接复用已验证文件，其余四个 seed 重新计算；模型边界还独立重算。

Toy 检查把向量化交叉分位数与逐时点排序比较，检查未来结果不会进入当前 archive quantile、diagonal 等价性、两条分解路线及 log-width 恒等式。另一个只读审查核验 bootstrap 的 seed/path/metric 排列，独立逐 seed/metric 循环与矩阵计算的最大浮点差为 `1.71e−13`；共同 paired 效应在 GRU−TCN contrast 中相消。

对每个交叉格子，事件退出于 `e=512` 后，充分遗忘界为 `e+max(L_a,L_b+W)`。最晚界为 704；此界仅适用于本诊断中的固定长度上下文与有限滚动窗口。

完整输出在 [mechanism manifest](../outputs/stage2_archive_mechanism/manifest.json)、[全部交叉性能](../outputs/stage2_archive_mechanism/crossed_performance_by_seed.csv)、[逐 path 分量](../outputs/stage2_archive_mechanism/paired_component_effects.csv.gz)、[分量汇总](../outputs/stage2_archive_mechanism/mechanism_summary.csv)、[全部 individual seeds](../outputs/stage2_archive_mechanism/individual_seed_components.csv) 与 [逐格遗忘检查](../outputs/stage2_archive_mechanism/negative_controls.csv)。
