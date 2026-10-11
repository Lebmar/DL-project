# 文献查重与候选研究缺口

核查日期：2026-10-11（Asia/Shanghai）。仅使用原始论文、作者代码和会议官方页面；不涉及微众数据。本备忘录是选题诊断，不是原创性认证，也不推断 A+ 的概率。

## 结论先行

“市场突变后区间失效”“短暂冲击与持久变化”“恢复速度”“残差状态识别”“重置或重放校准记忆”均已有直接先例。不能将其中任一项单独写成新贡献。

目前值得做小实验的切口是：**神经预测器的尺度/上下文记忆与校准器的历史分数记忆，怎样共同造成冲击退出后的区间超调，何时相互放大或抵消？**候选贡献应首先是有严格对照的机制分析，再根据结果决定是否需要新增方法。

在本次阅读范围内，未看到将“神经预测器上下文时间尺度 × 校准分数时间尺度”独立交叉干预，并在相同随机轨迹上分析冲击进入、退出和隐藏持续时间的完整实验。但是“未看到”不能等同于“没有人做过”。AdaptNC、RetroAdj 和最新神经检索校准工作是必须认真区分的近邻。

## 六项最直接相关研究

| 原始研究、状态 | 已经做了什么 | 对本项目的约束与尚可核查的区别 |
|---|---|---|
| Cho & Matteson，**Improving Conditional Coverage in Time-Series Foundation Models via Direct Volatility Modeling**，ICLR 2026 TSALM **workshop 已接收**。[论文](https://openreview.net/pdf?id=dZx0R4Cm8d)、[官方名单](https://tsalm-workshop.github.io/accepted-papers/) | 展示 TSFM 平均覆盖掩盖高波动条件下的漏覆盖；在原生不确定性尺度上作 GARCH 乘性修正，与自适应 conformal 比较。 | “TSFM 高波动失效＋GARCH 修正”高度重合。此次 OpenReview 直开触发浏览器验证，已核实原始论文的索引摘要与官方接收状态；不能据此断言其正文没有恢复实验。需要下载全文后补查上下文长度和 shock-exit 分析。 |
| Angelopoulos, Candès & Tibshirani，**Conformal PID Control for Time Series Prediction**，**NeurIPS 2023**。[论文](https://proceedings.neurips.cc/paper_files/paper/2023/file/47f2fad8c1111d07f83c91be7870f8db-Paper-Conference.pdf)、[作者代码](https://github.com/aangelopoulos/conformal-time-series) | P 跟踪分位数、I 累积覆盖误差、D 使用 scorecaster 预测未来分数。允许任意 scorecaster；实验含市场收益与 Transformer。 | “神经残差预测＋在线覆盖反馈”不是新框架。若新增神经模块，需要说明超出一般 scorecasting 的机制和必要性。本文并不保证每次突变后即时条件覆盖。 |
| El Halabi & Brandt，**Adaptive Conformal Inference Under Delayed Feedback: Coverage Guarantees and a Delay-to-Memory Diagnostic**，2026-09 **arXiv 预印本**。[v1 正文](https://arxiv.org/html/2609.07251v1) | §6.5 使用 baseline→shift→baseline 三段，均值与方差变化，测量覆盖跌落后的恢复时长；§6.6 比较 Markov 状态持续性。§6.4、7.3 用当前尺度/目标尺度 oracle 区分延迟导致的尺度错配。 | “恢复时间”“返回 baseline”“持续性”“尺度错配 oracle”均已做。可区别的候选是神经预测器记忆与校准记忆的交互，而非只重复改变残差分布。先从一步预测开始以免与该文延迟问题混淆。 |
| Shahbazi & Baheri，**Heterogeneous-Horizon Conformal Ensembles for Online Prediction Under Distribution Shift**，**Forecasting 2026, 8(5), 77**，2026-09-01 已发表。[原文](https://www.mdpi.com/2571-9394/8/5/77) | PACE 集成不同窗口、衰减和尺度的校准器。§4.4 比较自然事件持续时间；§5.5 控制 D=5…160，与快速 ACI 追踪器比较；论文明确这一 controlled sweep 并未比较 SAOCP。 | “持续时长改变校准方法优劣”已有结果。PACE 与 SAOCP 的同流控制比较是作者公开指出的后续实验，但仅补这个实验未必足够 A+；应结合神经本体机制。PACE 不提供类 SAOCP 的强自适应保证，代码为按需提供。 |
| Tumu, Singh & Mangharam，**AdaptNC: Adaptive Nonconformity Scores for Conformal Prediction under Distribution Shift**，2026-05 修订 **arXiv 预印本**。[v2 正文](https://arxiv.org/html/2602.01629v2) | 同时适应 score 参数与阈值；更新 score 后重放窗口，缓解 coverage shock；区分数据与 score 更新的 shift。v2 定理 5.5 还在相同专家初态下比较新旧 score 的反事实 replay 轨迹。机器人实验含 LSTM。 | “反事实重放”“旧分数与新参数不一致”“同步重校准”已有直接方法，连反事实轨迹比较也有先例。潜在区别需落在 frozen neural scale/context 的可分离双记忆干预、进入/退出不同作用及隐藏持续时间；仅迁到金融不够。 |
| Jun & Ohn，**Online Conformal Inference with Retrospective Adjustment for Faster Adaptation to Distribution Shift**，**Pattern Recognition 180, Part D, 114406 (2026)，正式期刊记录，期次 December 2026**。[出版社/DOI](https://doi.org/10.1016/j.patcog.2026.114406)、[公开 v2 正文](https://arxiv.org/html/2511.04275v2)、[作者代码](https://github.com/jungbinary/OnlineConformalRetroAdj) | RetroAdj 每步重新计算窗口内 LOO 预测残差；通过线性平滑器/KRR 高效更新。定义恢复时间，并比较窗口大小；附录也分析 NTK。 | “历史残差不更新导致适应慢”“回溯重算残差”已经有论证。本项目若使用固定权重神经模型、仅因 context 改变而产生尺度滞后，与重新拟合/LOO 的统计对象不同，但仍须明确引用并公平比较，不能泛称首次发现陈旧分数。 |

以上发表/预印本状态以原始页面可核实的信息为准；RetroAdj 已由出版社、arXiv 期刊引用及作者代码确认正式期刊记录，其所列 December 2026 是期次，不据此推断在线发表日。其余没有出版证明的 arXiv 工作不称作已发表论文。对全文未能完整核实的条目已明确限制。

## 其他必须纳入的近邻与基线

- **SAOCP / SF-OGD（ICML 2023）**：新专家在不同活跃时间区间运作，获得所有长度子区间的强自适应 regret；改变窗口/生命周期适应突变已有理论。[正式论文](https://proceedings.mlr.press/v202/bhatnagar23a.html)、[作者实现](https://github.com/salesforce/online_conformal)。不能把“多时间尺度集成”新命名后当成新方法。
- **ACI（NeurIPS 2021）与 AgACI（ICML 2022）**：前者在线调整失覆盖率，后者集成适应率；用它们作基线是必要起点。[ACI](https://papers.nips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html)、[AgACI](https://proceedings.mlr.press/v162/zaffran22a.html)。
- **Online conformal prediction with decaying step sizes（ICML 2024）**：附录 B 已有“decay+adapt”，连续漏覆盖或连续覆盖触发学习率重置。[正式论文](https://proceedings.mlr.press/v235/angelopoulos24a.html)。不能将漏覆盖触发重置本身当作创新。
- **DRACP（arXiv 2608.17079，预印本）**：密度比、局部、状态权重与自调在线控制器；附录 B 指明基模型是非神经 boosting、尺度 random forest、状态 GMM，已分析权重/控制器消融。我们不能仅声称“regime-aware calibration”，但可以聚焦神经预测器的内部尺度滞后。[正文](https://arxiv.org/html/2608.17079v1)、[作者代码](https://github.com/bogdanoancea/dracp)。
- **Deep switching state-space＋ACI/AgACI（arXiv 2512.03298，预印本）**：状态切换神经预测器、S4、MC-Dropout GRU 与统一残差 wrapper；研究状态变化下覆盖与宽度。因此“神经模型识别状态＋conformal”已有工作。[正文](https://arxiv.org/html/2512.03298v2)。
- **PID-Expert（arXiv 2610.02777，2026-10-02 预印本）**：在相同预测/分数/scorecaster 下集成 PID 配置，分析突变后局部覆盖—宽度，四类模拟与真实应用。[正文](https://arxiv.org/html/2610.02777v1)。新的模块不能只是在宽度和漏覆盖损失上集成快慢控制器。
- **HopCPT（NeurIPS 2023）与 RareCP（2026-05 arXiv 预印本）**：神经检索类似状态的历史误差、条件加权校准已经成熟；RareCP 的 hypernetwork 区分漂移与并存状态，使用四个预测 backbone 的相同预测存档，实验固定 64-step context。这既是神经模块的重要对照，也说明“预测存档复用”是实验规范而非创新。[HopCPT 原文](https://arxiv.org/abs/2303.12783)、[作者代码](https://github.com/ml-jku/HopCPT)、[RareCP 正文](https://arxiv.org/html/2605.08857v1)。

## 候选切口 A：神经尺度记忆与标准化分数记忆的机制交互

候选问题：在完全相同的数据轨迹和点预测上，预测器尺度的上下文记忆与分数窗口分别控制多少冲击后超调？二者的交互是否随事件持续时间、尺度跳变强度和尾部厚度系统改变？

先固定一步预测及平均值，定义 `e_t = y_t − mu_t`、因果尺度 `s_t = neural_scale(y_<t)`、逐时存档分数 `r_t = |e_t| / s_t`，区间半宽 `h_t = s_t * Q_{1−alpha}(r_<t)`。所有分数只能在结果出现后加入窗口。

实施两个不同性质的实验，不能混为一谈：

1. **可部署的 2×2 因子实验**：短/长神经上下文 × 短/长校准窗口；每个预测器使用自己历史因果尺度算分数，共用相同随机轨迹。优先使用同一已训练且能处理不同上下文的架构；若必须单独训练，要把优化随机性/参数量作为混杂因素报告。
2. **诊断用路径干预**：冻结预测存档，更换校准器；或把本体尺度替换为真实条件尺度、把分位数替换为 oracle。固定来自另一预测器的 `q_t` 属于非部署反事实，不具有原方法的有效性，须单列。它用于分解路径作用，不能拿来报告新方法性能。

分析进入期漏覆盖、退出后多余宽度面积、区间分数；用 `log h_t = log s_t + log q_t` 展示两项作用。对同一事件/随机轨迹做 paired 比较。交互项可写为 `M(long,long)−M(long,short)−M(short,long)+M(short,short)`，但正负符号必须由证据决定，不能预设“长记忆一定更坏”。模拟中 oracle 分布允许直接算每步条件覆盖，避免一个 100-step rolling window 把短冲击平均掉；真实数据没有真实波动状态，不应冒充同等 oracle 结论。

**证伪/降级条件：**改变本体记忆没有可复现影响；交互在配对误差范围内接近零；结果完全由分数窗口单独解释；强 PID/SAOCP 已消除目标现象；同样现象在简单 EWMA/GARCH 中成立而神经模型没有额外机制。出现后不能强行保留“神经双记忆放大”叙事，改写为边界/负结果或换切口。

## 候选切口 B：基于诊断的协同适应模块（必须等待 A 的证据）

只在 A 显示可解释失效后，尝试一个小型模块读取“神经原生尺度变化、近期标准化残差、尺度与残差的分歧”，选择校准时间尺度或控制收缩。训练/调参使用独立模拟轨迹，测试对模块隐藏 onset、duration 和真实状态，并保留持续变化下的覆盖约束。模块应与带同样特征的线性/规则方案比较，以说明神经表示是否必要。

这只是可执行研究假设。简单 gating、快慢集成、重置、残差重放都有前例。要成为贡献，需证明模块利用了 A 识别的具体交互，并在 held-out 变化持续时间和至少一种公开真实序列上改善覆盖—宽度或区间分数；不能只在固定事件长度的训练设置上赢。

**证伪条件：**无优于强基线的 paired 改善；只有 oracle 状态输入有效；改善来自放大所有区间；代价是持续高波动下漏覆盖明显恶化；线性/简单规则与神经模块相同表现；在新持续时间/厚尾条件下失效。

## 一个必须避免的“新方法”假象

若把所有历史残差除以同一个当前尺度 `s_t > 0`，再把分位数乘回 `s_t`，则

`s_t * Q(|e_i| / s_t) = Q(|e_i|)`。

它恰好退化为原始残差分位数，不能产生新的尺度协同机制。逐时保存 `|e_i|/s_i` 与使用新参数对每个历史特征条件重评分不同；后者还必须区分参数更新、context 更新及回溯样本内拟合，并与 AdaptNC/RetroAdj 对照。原始在线区间存档不得被回溯改写成“当时做出的预测”。

## 当前 proposal 可以使用的谨慎表述

> We will investigate whether the interaction between neural scale memory and conformal score memory explains transient-shock recovery inefficiency, using controlled interventions that separate the forecaster and calibration pathways. We will then test whether a lightweight causal adaptation module improves the coverage–width trade-off under unseen event durations.

现在可称“investigate/test”，不可称“first”“novel guarantee”“large probability of A+”。初步 MLP 方差预测器只证明研究协议可运行，不能被描述成 TSFM 效果。最终课程论文若没有真实训练神经模型或有关深度学习机制的证据，仅统计校准实验可能偏离深度学习课程重点。

## 继续核查清单

1. 获取 Cho & Matteson workshop 全文并检查所有消融；当前访问限制尚未解决。
2. 为 AdaptNC 的新参数 replay、RetroAdj 的 LOO、HopCPT/RareCP 的检索模型画对象对照：哪些权重固定，哪些历史预测/分数改变，是否有因果 out-of-sample 评估。
3. 在定稿前继续搜索 neural context truncation、test-time scale adaptation 与 conformal 两层动态系统的相互作用，记录真正的近邻，而非依据标题判原创。
4. 把未来的结果分为：模拟 oracle 诊断、模拟可部署算法、真实公开数据验证。所有调参只用较早或独立数据。

## 二次深查与明确边界（2026-10-11）

**发表状态核查：**[arXiv 元数据](https://arxiv.org/abs/2511.04275)、[出版社页面](https://doi.org/10.1016/j.patcog.2026.114406) 和 [作者代码](https://github.com/jungbinary/OnlineConformalRetroAdj) 均明确将 RetroAdj 指向 **Pattern Recognition 180, Part D, 114406 (2026)**。上表已按正式期刊记录更新，公开 arXiv v2 用于阅读正文。

### 最靠近双记忆候选的四项正文核查

| 研究 | 本次具体核查 | 可以守住的区别及不能声称的内容 |
|---|---|---|
| **NCC** | 阅读 [正文](https://arxiv.org/html/2412.18144v1) 与 [作者模型源码](https://raw.githubusercontent.com/complex-ai-lab/ncc/master/src/forecaster/end2endmodel.py)。控制器编码历史 score、覆盖误差、此前阈值；源码的 score 序列含 `(y,yhat,score)`，各历史输入受共同 `window_size` 控制，并加入窗口平均覆盖误差修正。正文消融关注损失、辅助 view、TTA 与 backbone。 | “神经历史编码＋在线校准”已是直接先验。读到的实验没有独立改变预测本体上下文与校准窗口的冲击进入/退出配对实验；该限定是**本次阅读范围的观察**，不是不存在先例的证明。 |
| **AdaptNC** | 再核查 §4、§5.3、Appendix B.1 与定理 5.5。其 score 几何参数变化与阈值反馈会相互影响；使用相同专家初态的新旧 score replay 轨迹，缓解更新诱发的 coverage shock。 | “耦合不稳定”“反事实 replay”“同步历史分数”都不能写成首次。本项目应只研究固定训练权重、因果输入上下文变化的尺度记忆，以及它与逐时存档分数窗口的交互。AdaptNC 的一般框架仍可能覆盖这一实例，不能以金融应用或 GRU 替换作为原创性证明。 |
| **RareCP** | 核查 [正文](https://arxiv.org/html/2605.08857v1) §3.5、§4.1 与消融。固定 backbone 预测存档；神经专家检索历史签名残差；query-conditioned hypernetwork 更新检索几何；实验的预测上下文统一为 64 步。 | 相同存档比较不同校准器、神经误差状态检索、区分漂移与共存状态均已有。读到的正文未见独立的 backbone 上下文 × 校准历史长度因子实验；检索 key 的重新变换与按历史预测时尺度归一化残差是不同操作，不能混称全部“分数重算”。 |
| **TSFM＋GARCH workshop** | 官方接收名单与原始摘要已核实。继续检索标题、作者主页及作者公开网页仓库，未找到可读取的替代全文；OpenReview 验证页面没有绕过。 | “原生神经尺度与外部波动修正”高度相近；目前只能引用摘要支持的模型与问题描述。**不能宣称它没有上下文、恢复或记忆交互实验。**全文未读仍是最终 novelty 审核的明确未决项。 |

### 当前最可守的假设

> 在冻结模型训练权重、固定点预测与相同随机轨迹的条件下，校准历史长度对冲击进入/退出效率的作用可能依赖预测器尺度响应；其交互的方向和幅度可能随架构、持续时长与尾部条件改变。我们将检验这种依赖是否存在、是否能被简单波动模型解释，以及它是否超过配对实验误差。

这是可证伪的问题，**不预设“两个长记忆相互放大”“GRU 比 MLP 更好”或“神经模型优于 EWMA/GARCH”**。根节点同步的初步结果显示 GRU 与 MLP 的交互方向不同，公开外汇 GRU 暂无清楚的 EWMA 优势；这些观察支持保留方向中性的问题，不支持普遍双记忆放大结论。此处仅记录团队同步的 pilot 状态，正式数值结论以可复现实验产物为准。

可部署因子实验保留各自因果尺度生成的分数；诊断干预另存结果，禁止把来自别的预测器的阈值路径当作具有覆盖保证的新算法。`log h = log s + log q` 是恒等式，可以辅助展示，**单靠这张图不能证明因果**。证据来自预先规定的独立干预、同轨迹配对以及 oracle 参照。

建议保留两种不可混写的归档：①每一时点真实发出的预测/尺度/区间，永久不改；②用于机制诊断的替代尺度、分数及重放路径。双重归档是可信评估与审计要求，**本身不是新的 conformal 方法或原创算法贡献**。

如果统一架构、训练种子与事件种子下交互不稳定，或 EWMA/GARCH 充分解释现象，就将最终贡献诚实改写为：**已有在线校准方法在不同预测器尺度响应下的边界与负结果，及避免将预测器效应误归因于校准器的复现实验协议**。若只有某些条件稳定，可报告限定条件中的交互；不能从少数情形推广到所有神经预测器。是否需要新模块，应等诊断结果，而不是作为 proposal 的承诺。

建议 proposal 使用以下表述：

> We will test whether forecaster scale response changes the effect of conformal score memory on adaptation after volatility changes. Paired interventions and separate factual and diagnostic archives will distinguish forecaster and calibration effects. We do not assume a universal interaction direction or a neural advantage; null and architecture-dependent outcomes will be reported as part of the study.

### Proposal 最关键的四篇直接先验（完整引文）

1. **Ruipu Li and Alexander Rodríguez (2025). _Neural Conformal Control for Time Series Forecasting._** Proceedings of the AAAI Conference on Artificial Intelligence, 39(17), 18439–18447；**已发表，2025-04-11**。[正式论文页](https://ojs.aaai.org/index.php/AAAI/article/view/34029)，DOI: 10.1609/aaai.v39i17.34029。[作者代码](https://github.com/complex-ai-lab/ncc)。它界定神经校准控制器已有贡献。
2. **Renukanandan Tumu, Aditya Singh, and Rahul Mangharam (2026). _AdaptNC: Adaptive Nonconformity Scores for Conformal Prediction under Distribution Shift._** **arXiv 预印本，2602.01629v2，2026-05-13 修订**；本次未核实正式出版记录。[原始记录](https://arxiv.org/abs/2602.01629)，[正文](https://arxiv.org/html/2602.01629v2)。它界定 score—阈值耦合、反事实 replay 与同步校准已有贡献。
3. **Jungbin Jun and Ilsang Ohn (2026). _Online conformal inference with retrospective adjustment for faster adaptation to distribution shift._** **Pattern Recognition, 180, Part D, 114406；正式期刊记录，期次 December 2026**。[出版社/DOI](https://doi.org/10.1016/j.patcog.2026.114406)，[公开正文 v2](https://arxiv.org/html/2511.04275v2)，[作者代码](https://github.com/jungbinary/OnlineConformalRetroAdj)。它界定回溯更新历史残差与恢复速度已有贡献。
4. **Manuel Heurich, Maximilian Granz, and Tim Landgraf (2026). _RareCP: Regime-Aware Retrieval for Efficient Conformal Prediction._** **arXiv 预印本，2605.08857v1，2026-05-09**；本次未核实正式出版记录。[原始记录](https://arxiv.org/abs/2605.08857)，[正文](https://arxiv.org/html/2605.08857v1)。它界定神经状态检索、漂移适应与统一预测存档已有贡献。

最终一页 proposal 可采用 **NCC、AdaptNC、RetroAdj、SAOCP** 四篇组合：前两篇划定神经控制及耦合/replay 的边界，RetroAdj 划定恢复与回溯残差的先验，SAOCP 提供强在线基线。若仅能容纳三篇，可保留 NCC、AdaptNC、SAOCP，并在正文简短注明恢复/回溯已有 RetroAdj 先例；RareCP/HopCPT 留在扩展文献说明中。

SAOCP 完整引文：**Aadyot Bhatnagar, Huan Wang, Caiming Xiong, and Yu Bai (2023). _Improved Online Conformal Prediction via Strongly Adaptive Online Learning._** Proceedings of the 40th International Conference on Machine Learning, **PMLR 202:2337–2363；已发表 ICML 2023**。[会议原始页面](https://proceedings.mlr.press/v202/bhatnagar23a.html)，[作者代码](https://github.com/salesforce/online_conformal)。强自适应 regret 不等同于每一时点条件覆盖保证。

金融动机另引 **Jason B. Cho and David S. Matteson (2026), _Improving Conditional Coverage in Time-Series Foundation Models via Direct Volatility Modeling_，ICLR 2026 TSALM workshop 已接收**，[原始 PDF](https://openreview.net/pdf?id=dZx0R4Cm8d)，[官方接收名单](https://tsalm-workshop.github.io/accepted-papers/)，并明确全文核查限制。

### 最后一次官方公开下载核查

2026-10-11 使用未登录的普通 HTTP GET 分别请求 [OpenReview 原始 PDF](https://openreview.net/pdf?id=dZx0R4Cm8d) 与 [官方 API 的公开 note 查询](https://api2.openreview.net/notes?id=dZx0R4Cm8d)。两者均返回 **HTTP 403 Forbidden**，没有取得 PDF 或 note 元数据；未使用验证绕过、认证凭据或权限替代入口，随后停止尝试。

因此仍无法全文核实该文的 **context length、shock entry/exit 实验、尺度响应 × 校准窗口交互消融**。proposal 应明确这是近邻审查的未决限制，不能将其未读内容写成“尚无研究”证据。当前可守的贡献表述仍是提出并检验特定机制假设，而不是保证这类诊断此前从未出现。
