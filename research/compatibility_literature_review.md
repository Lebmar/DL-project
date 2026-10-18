# 预测器—校准器适配性：补充文献核查与下一步边界

核查日期：2026-10-18。用途：为下一轮研究选择可证伪的问题；不是报告稿，也不是原创性认证。先阅读了 `research/literature_gap.md`，本页只补充与“原生区间排名被校准改变”最相关的证据。只使用原始论文、正式会议页和作者公开稿；预印本的理论没有在本次核查中逐行复证。

## 判断

不宜把“校准器和预测器存在交互”“常数尺度会抵消”“按最终区间效率选模型”作为新贡献。它们已有直接先例。现在更值得检验的是：**消除静态尺度偏差以后，预测器的时间响应误差能否定量解释有限窗口校准的增益或损失，并在独立轨迹上预测其方向？**

目前外汇排名变化是这个问题的探索性动机，不能直接证明动态机制。先做固定历史时期估计的单一尺度修正，再与滚动校准比较，可区分“原生尺度未校准”与“时间响应导致的额外效应”。任何校正系数必须只由评估日期之前的资料选定。

## 五项直接近邻

| 原始来源与核查范围 | 重合点 | 本项目应守住的边界 |
|---|---|---|
| **Bellotti (2020), Constructing normalized nonconformity measures based on maximizing predictive efficiency.** COPA / PMLR 128:41–54，已发表。[论文页](https://proceedings.mlr.press/v128/bellotti20a.html)、[PDF](https://proceedings.mlr.press/v128/bellotti20a/bellotti20a.pdf)。读 §1–2，尤其 PDF 第 5 页的缩放讨论。 | 直接优化归一化分数的均值/尺度参数以提高区间效率；明确普通预测损失与该目标不同，并明确常数倍分数缩放不改变 conformal 集合。 | 不能把常数消去恒等式或“为校准后的结果训练”当成新方法。本文是交换性、批量校准框架；本项目若有贡献，应是受控的在线时间响应规律。 |
| **Xu & Xie (2023), Sequential Predictive Conformal Inference for Time Series.** ICML / PMLR 202:38707–38727，已发表。[论文页](https://proceedings.mlr.press/v202/xu23r.html)、[PDF](https://proceedings.mlr.press/v202/xu23r/xu23r.pdf)。读原始摘要与引言、方法概述。 | 以历史分数预测未来分数的条件分位数，利用残差的时间依赖；在其正则条件下给出渐近条件覆盖。 | 若新模块只是“从历史残差预测校准阈值”，必须与 SPCI、既有 PID scorecasting 区分；不能称这一目标本身新颖。该条未作全文理论复证。 |
| **Schweizer et al. (2026), Distribution-Aware Conformal Prediction: A Framework for Generating Efficient Prediction Intervals for Time Series.** arXiv 2605.26569v1。[元数据](https://arxiv.org/abs/2605.26569)、[正文](https://arxiv.org/html/2605.26569v1)。读 §3–5 与模型附录。 | 系统比较预测器与分数的组合；含 Z-score 归一化、滚动校准、异方差和分布变化模拟，以及 LSTM、TCN、TFT。 | “统一预测器—分数组合实验”已经高度重合。文中 lookback 固定为 3，未在读到的范围内给出当前尺度响应与历史分数窗口的独立时间干预；这不是不存在其他先例的证明。页面注明投稿 JMLR，**不等于已被接收**。 |
| **Halkiewicz (2026), Rolling-Origin Conformal Prediction under Local Stationarity and Weak Dependence.** arXiv 2605.08422v1。[元数据](https://arxiv.org/abs/2605.08422)、[正文 §3.7](https://arxiv.org/html/2605.08422v1)。读主要假设、覆盖误差分解与波动缩放推论。 | 分解窗口估计噪声与漂移；§3.7 明确写出波动尺度估计误差，并指出理想纯尺度归一化会消除分数漂移，使更长历史减少分位数噪声。 | “归一化降低漂移、最佳窗口依赖尺度质量”已有明确论述。突变与固定有限窗口路径响应不自动满足其 Hölder 局部平稳假设。本文是预印本，其最优性结论不能直接套到我们的冲击实验。 |
| **Tumu, Singh & Mangharam (2026), AdaptNC.** arXiv 2602.01629v2。[正文](https://arxiv.org/html/2602.01629v2)。复核 §5.1–5.2、定理 5.5 及 replay 证明定义。 | 分数函数改变会扰动阈值动态；同一初始专家状态下的新旧分数反事实 replay 已有理论对象。 | 不能以“两个模块耦合”或“反事实重放”宣称首次。当前项目固定权重、改变因果输入响应的对象较具体；是否形成独立贡献，仍需一个可检验、可推广的定量结论。 |

## 两项金融应用提醒

- **Aich et al., TCP，arXiv 2507.05470v7，2026-09-20 修订**：[元数据](https://arxiv.org/abs/2507.05470)、[v7 正文](https://arxiv.org/html/2507.05470v7)。最新摘要已明确：覆盖与区间分数的排名不同，波动模型在多数所列配对区间分数比较中优于 TCP。不能使用早期 v5 的积极结论代替最新版，也不能把“conformal 在金融中未必优于波动模型”当作本项目首次发现。
- **Eugen Soloviov, Conformal Prediction for Financial Returns: Where Coverage Survives and Where It Breaks**：[作者公开稿](https://conformal.marketmaker.cc/)。读方法、模拟与结果说明。已有 GARCH/突变模拟、真实尺度 oracle、EWMA 归一化、突破后恢复、匹配覆盖下宽度比较。页面自称 “arXiv-ready”，本次没有核实正式 arXiv 记录或同行评审发表；应称作者公开研究稿。其模型主要是 boosting/EWMA，不能据此把一般现象重新包装为神经模型独有发现。

## 一个可计算的理论起点，而非已经成立的新定理

为隔离机制，先考虑非部署的理想化人口混合分布：

\[
Y_i=\exp(\epsilon u_i)Z_i,\qquad
s_i=\exp(\epsilon v_i),\qquad
G_t(q)=\frac1W\sum_{i=t-W}^{t-1}F\!\left(qe^{-\epsilon(u_i-v_i)}\right),
\]

其中 `F` 是 `|Z|` 的共同连续分布，目标分位数 `q0` 附近密度正且可微，`u,v` 是给定的有界路径，`W` 固定；`q_t` 满足 `G_t(q_t)=1−α`。在 `ε=0` 附近作隐函数的一阶展开，得到候选表达式：

\[
\log\frac{s_tq_t}{e^{\epsilon u_t}q_0}
=\epsilon\left[(v_t-u_t)-\frac1W\sum_{i=t-W}^{t-1}(v_i-u_i)\right]+O(\epsilon^2).
\]

这是本次核查的独立推导，不归于上列论文；尚需完整条件、余项控制、独立复核和数值检验。它说明更精确的候选对象是**当前对数尺度误差减去其历史平均**，不是预设“两种滞后必然相加”。常数误差在该表达式中消去，而且对正齐次分位数实际可精确消去。

这一人口分位数对象不同于真实窗口内只有 `W` 个随机分数的经验顺序统计量。它不包含量化噪声、离散秩偏差、由同一数据估计尺度带来的依赖，也不能直接声称条件覆盖。现有 rank-sensitivity 只能检查其中一项，不能替代其余分析。

**Oracle 边界：**如果当前真实条件分布及其精确分位数已知，真实分位数最小化对应的期望 pinball/interval loss；不能声称次优尺度普遍击败这一 oracle。若次优尺度击败“真实尺度 × 有限样本经验分位数”，比较的是带量化噪声的校准估计器，必须进一步解释方差、相关性或有限样本偏差。两种 oracle 不可混称。

## 建议的推进门槛

1. 先完成过去数据估计的固定尺度修正。若其足以消除外汇排名变化，应把原现象解释为尺度校准问题，不强行归因于双记忆。
2. 用事先给定的尺度响应核和已知创新分布检验上述人口表达式；同时计算真实经验分位数，以分开路径偏差与有限窗口噪声。包括常数倍尺度、精确尺度、延迟/平滑尺度三个必要对照。
3. 只有在一阶或更精确规律能对**未用于拟合的**变化时长、幅度与响应核预测效应方向/量级后，才考察冻结神经尺度的轨迹是否落在该规律内。若简单 EWMA 已解释，不强求“神经特有”。
4. 再决定是否需要新模块。必须比既有静态校正、SPCI/PID 类分数预测及强在线校准多解决一个明确问题；新增 gating、replay 或大模型本身不构成贡献。

这个方向有明确的研究步骤，但目前既无原创性结论，也无投稿成熟度结论。一个被严格验证的负结果或边界仍可改善课程项目；期刊目标应由新增证据决定，而非由实验数量决定。
