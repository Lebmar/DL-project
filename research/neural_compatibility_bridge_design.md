# 有限样本机制到神经尺度模型：一个小型桥接实验

日期：2026-10-18。状态：**设计建议，尚未冻结，未训练、未生成或读取新测试结果。** 是否实施由本轮有限样本机制验证决定。这里只建议独立的新文件与输出目录；Stage 2/3、FX、既有报告和 PPT 不改动。本设计不等同于原先搁置的 pulse-feature 元回归方案。

## 1. 要跨越的真正缺口

外生误差实验中可以固定整个相对尺度误差轨迹

\[
\delta_t=\log(s_t/\sigma_t)
\]

再独立产生创新。神经模型中，\(s_t\) 是过去观测的函数：单个 \(\epsilon_i\) 与当时已发布的 \(s_i\) 独立，但它会进入之后的尺度预测。因此，给定完整实现的 \(\delta\) 轨迹后，原历史分数一般不能视为独立样本。

下一轮首先问：**总体混合分位数、独立有限分数库、真实在线历史，三者对新的神经拟合给出多大不同？差异来自有限样本误差、当前误差与分位数误差的相关，还是局部展开的非线性余项？**

这比拟合一个小元回归更直接，也允许结果否定简单外生解释。已知模拟真尺度只用于离线诊断，不能据此声称获得可部署的金融诊断器或条件覆盖保证。

## 2. 建议的最小训练设计

推荐 **3 个新训练数据 panel × 每 panel 2 个新 seed × GRU/TCN = 12 个拟合**。这只够作机制桥接，不够可靠估计完整训练流程总体。若有限样本结果只支持更小探索，可在任何新训练之前改成 2 panel / 8 fit，并明确更弱的范围；不得读结果后挑 panel。

| Panel | 训练 seed | 验证 seed | 验证 target-pair seed | 嵌套的模型/采样 seed |
|---|---:|---:|---:|---|
| 1 | 714159 | 714160 | 714161 | 5101, 5102 |
| 2 | 715159 | 715160 | 715161 | 5201, 5202 |
| 3 | 716159 | 716160 | 716161 | 5301, 5302 |

这些 ID 在审查时未出现在已有协议中；正式冻结仍须核查冲突。每个 panel 内 GRU/TCN 用相同 seed 匹配 minibatch 前缀；跨 panel 不复用 seed，因此是 **nested seed pairs**，不能照搬 Stage 3 的 shared-seed crossed bootstrap。

完全复用 `train_stage2.py` 的原函数，不修改该冻结文件：256×1024 训练路径、64×1024 验证路径、8192 验证 target pairs、contexts 16/64 平衡采样；GRU hidden24、TCN channel12 / receptive field64；Adam .002、batch128×32、max60 / min20 / patience8；绝对最小平均验证 NLL 的 checkpoint。新 driver 只提供完整的新 protocol，并将 `trainer.OUT` 指向独立目录，使用原 Stage 3 adapter 的方式。

原 trainer 的内部 manifest 含历史固定日期 `2026-10-11`。新 driver 应在自己的外层 manifest 明确真实执行日期及该字段的历史兼容含义，不改冻结 trainer 去修日期，也不把旧日期当新实验时间。

建议路径：`configs/neural_compatibility_protocol.json`、`scripts/train_neural_compatibility.py`、`scripts/evaluate_neural_compatibility.py`、`scripts/analyze_neural_compatibility.py`、`outputs/neural_compatibility_{training,evaluation,analysis}/`。所有 source hashes、protocol 和新 driver 在训练前提交；已有结果只作为设计背景。

## 3. 小型、共同、全新测试网格

建议 64 条路径、长度640、事件起点256、持续128步，创新按方差标准化。先固定这四个世界，不扩展成大模型网格：

1. Gaussian，恒定真尺度1。
2. Gaussian，事件期间真尺度2，其余1。
3. Gaussian，事件期间真尺度0.5，其余1。
4. 标准化 t5，事件期间真尺度2，其余1。

前三个世界共享一组新 Gaussian 创新，便于观察事件效应；t5 是另一组独立创新。全部模型共享每个世界的路径。候选种子 Gaussian `2903000`、t5 `2904000`，只在新协议提交之后生成。模型训练仍是既有 Gaussian switching distribution；t5 是明确的 distribution-shift stress case，不把 Gaussian NLL 视为其正确分布似然。

所有模型产生 contexts16/64 的完整 past-only 已发布尺度，评价 windows32/128。只做对角 \(a=b\) 以控制预算；这一轮不重跑全部八格 current/archive factorial，因为问题是误差—分位数的依赖，而不是重复原分解。没有验证分数广播 warmup：最早比较点192，其前至少128个真实历史分数和64步模型上下文均可用。

为降低总体混合求根成本，桥接比较可固定16个端点：

| 区段 | 端点 |
|---|---|
| 事件前 | 192, 208, 224, 240 |
| 进入 | 256, 264, 272, 280 |
| 事件后半段 | 320, 336, 352, 368 |
| 恢复 | 384, 400, 416, 432 |

这些是固定端点平均，**不是完整 phase 平均**。如需完整 phase，须在冻结前决定，并统一三个对象的评价端点，不能用密集真实风险与稀疏近似风险混比。首要读数建议 Gaussian 上升事件、context64；其余世界、context16 和四区段全部保留为预定边界检查。

## 4. 同一批已发布尺度，比较三个不同的分位数对象

令 \(u_i=|\epsilon_i|e^{-\delta_i}=|y_i|/s_i\)。对每个模型、路径和端点，保持 \(s_t,\sigma_t,\delta_{t-W:t}\) 不变。

### P：总体混合

解

\[
\frac1W\sum_{j=1}^W F(q_P e^{\delta_{t-j}})=1-\alpha,\quad \alpha=.1.
\]

这是一种误差轨迹诊断。对神经模型，不能把它叫作原分数过程在完整轨迹条件下的真实分布。

### D：独立重绘历史创新，保持已发布误差轨迹

对每个 replica \(r\)，从相同创新分布独立生成 \(\epsilon_i^{*(r)}\)，且与整条原 \(\delta\) 轨迹独立，构造

\[
u_i^{*(r)}=|\epsilon_i^{*(r)}|e^{-\delta_i},\quad
q_D^{(r)}=Q_k(u_{t-W:t}^{*(r)}).
\]

每个历史年龄恰有一个新创新，故是独立但通常非同分布的连续历史库，**不是**从年龄混合分布 IID 抽 W 次。必须与有限样本研究中的 IID mixture-bank 单独命名。候选固定64个 replicas，可按两个独立32-replica组保存，以量化 Monte Carlo 精度；不能看到不显著就追加 reps。

同一 replica 的完整新创新路径用于所有模型、contexts、windows，以及共享 Gaussian 创新的三个世界。这样保留比较的 common-random-number 配对；重绘创新不得输入神经网络重新预测尺度，否则改变了正在隔离的对象。重叠窗口使用同一条 replica 历史，不能每个端点单独重绘后又声称保留时间依赖。

这是 **score-bank decoupling counterfactual**：消除归档创新与完整尺度误差轨迹的关联，同时保留实现的尺度轨迹与跨时间结构。它不是可实施的校准算法，也不是原数据过程的因果干预可识别性证明。

### A：真实历史

\[
q_A=Q_k(u_{t-W:t})
\]

使用原观测与原已发布尺度，不重绘、不回算 denominator。实际半宽为 \(h_A=s_tq_A\)。所有经验分位数首先沿用 \(k=\lceil.9(W+1)\rceil\)，然后是否加入相邻秩精确混合，按有限样本结果在新神经 protocol 冻结前一次决定。若两规则都运行，分别保存并计算每个候选区间的损失后混合，不对平均半宽评分。

P 的90%总体分位数与 ceiling rank 的有限样本差是有限 rank / sampling 项的一部分；不得将其误说成全部依赖效应。

## 5. 当前目标噪声可以精确积分，但历史依赖不能忽略

对 Gaussian 或标准化 t5 创新，计算每个已发布半宽的条件期望 IS、覆盖率和宽度；这比抽取同一个当前目标创新更稳定。只要半宽 past-measurable，且模拟创新是独立的，给定预测历史的这一计算仍成立。不能把这一事实扩大为“给定整条未来尺度轨迹后原历史分数独立”。

建议主损失使用真尺度标准化后的期望 IS：

\[
\bar R(a)=2a+\frac{2}{\alpha}E[(|\epsilon|-a)_+],\quad a=h/\sigma.
\]

同时报告原单位 \(\sigma\bar R(a)\)，避免标准化掩盖高波动时期的实际成本。标准化是模拟机制诊断，不能在真实数据中偷偷用未知真尺度。

保存 exact oracle \(a=z=F^{-1}(.9)\)、native \(a=ze^{\delta_t}\)、P/D/A 的风险。D 对每个 replica 的 IS 先计算、再平均；oracle 从真实分布的精确分位数得到，不能用有限样本真实尺度区间代替。

可用下式独立实现 Gaussian 积分并对数值积分抽样检查：

\[
E[(|Z|-a)_+]=2\phi(a)-2a\{1-\Phi(a)\}.
\]

对 \(\epsilon=cT_\nu\)，\(c=\sqrt{(\nu-2)/\nu}\)，令 \(b=a/c\)：

\[
E[(|\epsilon|-a)_+]
=2c\frac{\nu+b^2}{\nu-1}f_\nu(b)-2a\{1-F_\nu(b)\}.
\]

这里 \(f_\nu\) 是双侧 Student-t 的普通密度；与绝对值密度的 factor2 不得混淆。对称区间的 native Gaussian multiplier 与 t5 正确 multiplier 不同，本研究为隔离尺度误差机制，t5 native 诊断用其真分布 multiplier，并另行标注；不能与已有 Gaussian-native t5 实验读数直接替换。

## 6. 不依赖局部近似的精确分解

定义

\[
\xi_P=\log(s_tq_P/(\sigma_tz)),\qquad
\eta_A=\log(q_A/q_P),\qquad
\xi_A=\xi_P+\eta_A.
\]

于是

\[
E\xi_A^2=E\xi_P^2+E\eta_A^2+2E(\xi_P\eta_A)
\]

是精确代数恒等式。将交叉项进一步写为

\[
2E(\xi_P\eta_A)=2\operatorname{Cov}(\xi_P,\eta_A)+2E\xi_PE\eta_A,
\]

避免把 finite-rank bias 的均值作用全部叫作 covariance。D 用相同定义并对 replicas 平均；比较 A 与 D 的全部三项，不预设交叉项为0。即使重绘创新，异方差历史与尺度轨迹也可使 \(E[\eta_D\mid\delta]\) 随轨迹变化，所以全局 covariance 未必为0。

另外比较一阶 \(\delta_t-\mu_t\)、带 \(v_t\) 的二阶近似与 exact \(\xi_P\)。最后比较 \(C\xi^2\) 与 exact excess risk，报告非线性余项。如果神经尺度误差较大，局部展开可能失败而 exact P/D/A 诊断仍有价值；不得删除这些模型以制造小误差条件。

## 7. 读数与判断：先量化误差，不拟合新回归

每个保存 fit × context × window × world × 区段，先对预定端点与64条独立路径平均。首要比较是 calibration-induced risk change：

\[
\Delta_A=R_A-R_N,\quad \Delta_D=E_r R_D-R_N,\quad \Delta_P=R_P-R_N.
\]

报告 `Δ_A−Δ_D`（score-bank decoupling gap）、`Δ_D−Δ_P`（独立有限历史与总体差异）、P/D 对 A 的绝对误差，以及每个 fit 的完整符号和幅度。它们可能相消，不能用总误差小来声称每一机制都小。

不回归、不挑 feature、不学 cutoff。报告 NLL、native IS、相对误差均值/方差和 lag-1 persistence，作为透明参照。NLL 比较应在同一组新测试路径上计算，且不再用于 checkpoint selection；不能混用各 panel 不同验证集的原 NLL 作公平性能排序。

对于“有限独立库是否比总体近似更接近实际”，建议预先指定 Gaussian 上升事件、context64、windows32/128、进入与恢复这四类 cell，同权平均每 fit 的绝对逼近误差；主标量是 `MAE(P,A)−MAE(D,A)`。这是固定新拟合上的机制逼近精度，不是估计独立12个市场，也不是已知真波动不可得时的在线模型选择结果。其余 cases 保留为边界检查，不能在看到结果后替换主比较。

## 8. 不确定性、预先停止与推进条件

主区间可用2000次 paired path bootstrap：所有 fits、cases、methods 用相同原路径索引，D 的 replica 不独立计作新的神经拟合。另报两个32-replica组的读数差异与 Monte Carlo SE；如果该误差与机制差同量级，结论是当前64-replica精度不足，不自动扩大至显著。

3 个 panel 的总体 bootstrap 很不稳定。必须展示每 panel、每架构与全部12 fits；panel/within-panel seed/common-path 重抽最多作为敏感性分析，且属于 nested design。不能从12×64×64个计算记录推导训练总体样本数。所有结论明确条件于保存的新 fits 与指定模拟机制。

有限样本阶段的数值检验与条件定义通过后，可以启动这一桥接，即使其结论是“finite noise 占主导”；但若有限阶段仅展示极小扰动的形式恒等式，或独立/连续历史区分仍有实现问题，则先修机制实验，不用神经训练掩盖问题。

桥接完成后采用三种结果分流：

1. **D 显著改善 P 且 gap 剩余小：** 有限历史是主要新增解释；下一步才考虑不依赖真尺度的估计规则。不能声称神经特殊机制。
2. **A−D 与协方差/均值项明显，或方向跨 panel 改变：** 外生模型不足；后续问题应聚焦 endogenous scale–score dependence，不继续宣传单一过滤增益准则。
3. **exact 机制分解可复现但局部展开/模型排序预测不稳定：** 保留范围明确的解释性负结果；不要追加不同 amplitude、seed、context 直到得到预想方向。

这里“明显”“剩余小”尚未定义数值门槛，故本文件不是成功判定协议。正式协议应在有限样本结果给出的数值精度与效应尺度基础上，**训练前**明确绝对/相对阈值、零分母处理、唯一主比较、bootstrap seeds 和有限误差标准。未通过不等于证明机制不存在；通过也不自动意味着达到可投稿原创性。

## 9. 开始前必要的实现核查

- 旧 trainer / model sources、所有旧 results 的 hashes 不变；新目录拒绝静默覆盖。
- 新训练协议和12个 fits grid 完整；新测试、replica、bootstrap seeds 在结果访问前冻结。
- `s_t` 与 `q_t` 都排除 `y_t`；改变当前/未来观测不能改变已发布前缀。
- D 使用完全独立的创新流且绝不重跑尺度；constant exogenous error 时 A/D 的分布与有限 order-statistic control 相符。
- 求根残差、独立 scalar rolling-quantile replay、exact log decomposition、oracle excess risk 非负（数值容差）、Gaussian/t5 数值积分对照通过。
- 同一均匀 scale rescaling 的精确有限样本抵消通过；各区段端点明确包含/排除事件边界。
- TCN64 快速整段推理只在现有审计容差内使用，context16/GRU 用已验证 reset-window 路径；所有比较共用同一发行尺度缓存。
- 原始路径/真尺度/发行尺度 hashes、每 fit 诊断、Monte Carlo 结果与失败完整保存。本地 CPU 足够；无需普通计费 HPC 作业。
