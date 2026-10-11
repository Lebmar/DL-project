# 研究贡献独立复核：从双记忆交互到尺度响应的训练不稳定性

复核日期：2026-10-11。材料范围：已有 Stage 2 训练、验证记忆诊断、主测试、交叉 archive 补充，以及 Stage 3 已冻结协议；没有生成、读取或选择新的测试结果。本文件是独立贡献审查，不是原创性认证或评分预测。

## 最强与最弱定位

**当前最强、证据可守的定位是一项受控的深度学习机制研究：固定预测权重与点预测，用同轨迹配对的 current-scale × historical-score denominator × calibration-window 交叉实验，揭示神经尺度响应与归档通道在冲击进入、退出中可以相消，且某些通道对初始化更敏感。**这比单纯比较 GRU/TCN 或调窗口更有内容，但 Stage 2 的交叉诊断属于看到主测试结果后的探索；初始化与架构结论仍限于这些拟合程序。

**当前最弱的定位是“首次发现相似 validation NLL 的神经模型校准差异大，所以提出快慢窗口/重放/重置的新方法”。**训练管线 underspecification、神经校准、反事实重放和多尺度控制均有明确先例；而我们尚未验证一个新方法。有限遗忘界、`log h = log s + log q`、两条差分路径分解和期望 interval-score 的标准推导，只能作为理解与检查工具。

现阶段可以继续冲击高质量课程研究，但还不能把 A+ 的关键门槛描述成已经达到。增加实验数量不会自动填补原创贡献；下一轮最有价值的是回答：**上述通道定位在独立训练数据重训后是否成立；除平均 NLL 外的验证响应诊断是否对其行为提供可复现信息。**

## 直接先例比较

“正文可读”表示本次已读取相关方法、实验或讨论段落，不意味着完成论文所有推导的证明审查。“未见”始终只表示本次查阅范围没有找到；不以搜索结果缺席证明首次。

| 原始研究与本次证据状态 | 作者已做的相关工作 | 本项目不能声称什么 | 可继续核查的区别 |
| --- | --- | --- | --- |
| **NCC，Li & Rodríguez，AAAI 2025。**[正式出版页](https://ojs.aaai.org/index.php/AAAI/article/view/34029)；[可读正文](https://arxiv.org/html/2412.18144v1) §Our Approach、Ablation Study；[作者实现](https://github.com/complex-ai-lab/ncc)。 | 用 GRU 编码历史误覆盖、阈值、score，并与多视图编码通过 attention 融合；控制损失和 conformalization 追踪覆盖；比较多个预测 backbone。 | 神经历史编码、可学习校准控制、利用残差模式适应 shift 不是新框架。 | 本次读到的消融集中于损失、多视图、TTA/backbone；未找到将冻结 forecaster 的当前尺度与逐时 archived denominator 独立交叉，并考察训练随机性的实验。该具体区别仍需最终查重。 |
| **AdaptNC，Tumu、Singh、Mangharam，2026 arXiv 预印本。**[v2 正文](https://arxiv.org/html/2602.01629v2) §4 Replay、§5.3、Theorem 5.5 与其证明可读。 | 同时适应 score 几何与阈值；score 更新后重放近期数据、重算量化阈值并重建专家状态；新旧 score replay 使用共同初态；实验包括 LSTM。 | 首次发现 score/threshold 耦合、旧 score 错配、反事实 replay 或同步重校准都站不住。 | 我们保持训练权重固定，仅改变因果输入 context 所导致的尺度路径；诊断区分当前尺度和历史发出时分母。这个实例可能被其一般框架覆盖，不能以换成金融或 GRU 当区别。 |
| **RetroAdj，Jun & Ohn。**[公开 v2 正文](https://arxiv.org/html/2511.04275v2) §3、§5、Appendix E 可读；[DOI](https://doi.org/10.1016/j.patcog.2026.114406)；[作者代码](https://github.com/jungbinary/OnlineConformalRetroAdj)。 | 用线性平滑器/KRR 的高效 LOO 更新回溯残差与预测；测量覆盖 recovery time、改变窗口，附录比较 NTK。arXiv 元数据和作者项目指向 Pattern Recognition 180, 114406 (2026)。 | 历史分数陈旧、恢复缓慢、回溯重评分、窗口适应权衡并非首次。 | 在线重拟合/LOO 的统计对象与固定神经尺度的逐时 issued-score archive 不同；目前研究诊断通道而非声称替代 RetroAdj 的通用性能优势。 |
| **SAOCP/SF-OGD，Bhatnagar et al.，ICML 2023。**[正式论文页](https://proceedings.mlr.press/v202/bhatnagar23a.html)；[正文 PDF](https://proceedings.mlr.press/v202/bhatnagar23a/bhatnagar23a.pdf) §3–5 可读；[作者代码](https://github.com/salesforce/online_conformal)。 | 多专家各有活跃区间，控制所有子区间的 strongly adaptive regret；SF-OGD 可独立使用。 | 新命名多时间尺度专家、重启窗口或一般局部适应不构成新方法；regret 也不能直接换写成任意时刻的条件覆盖。 | 项目已加入作者实现补充，但只有一个拟合 seed、两种参考噪声；不能据此声称强控制器已解决或未解决所有初始化现象。 |
| **Conformal PID，Angelopoulos、Candès、Tibshirani，NeurIPS 2023。**[正式会议页](https://proceedings.neurips.cc/paper_files/paper/2023/hash/47f2fad8c1111d07f83c91be7870f8db-Abstract-Conference.html)；[可读正文](https://arxiv.org/html/2307.16895) §2–3、Appendix D–F；[作者代码](https://github.com/aangelopoulos/conformal-time-series)。 | 分位数追踪、积分覆盖反馈与 scorecasting；已有 Transformer、市场序列实验。作者讨论 scorecaster 有价值也可能增加波动，设计依赖 base forecaster 的误差结构。 | 可预测残差加覆盖反馈不是新思想；“校准器好坏依赖预测器”也不是一般性新发现。 | 可争取的具体贡献是 frozen neural scale 的两个计算通道、phase-specific 差分和重训重复，并给出哪些验证诊断有用/无用的边界。 |
| **HopCPT，Auer et al.，NeurIPS 2023。**[正文 PDF](https://arxiv.org/pdf/2303.12783) §2 与 synthetic example 可读；[作者代码](https://github.com/ml-jku/HopCPT)。HTML 打开后部分段落解析失败，方法核查使用 PDF。 | Modern Hopfield 网络学习相似历史 error regime 的检索与加权分位数；交替大小误差 regime 的模拟已展示近期误差传播造成漏覆盖/低效。 | neural error-regime 检索、状态相关加权或高波动/低波动交替失效不是新现象。 | 当前项目不训练检索模块，而是探查 forecaster-scale/issued-score 的路径结构；context 长度不是其 retrieval memory 的同义词。 |
| **RareCP，2026 arXiv 预印本。**[正文](https://arxiv.org/html/2605.08857v1) §3.5、§4.1、Appendix D.3 可读。 | query-conditioned retrieval 几何、冻结专家后训练 gate；用共同 backbone prediction cache 比较方法，统一 64-step context，验证集调参。 | 共同预测存档、神经 residual retrieval、regime gate 或 cache 公平比较不是原创算法。 | 其模型和校准记忆可调；本次未找到我们的 current/archive/窗口完整交叉加训练数据重复。不能把“固定 context64”误解为它没有任何 memory 消融。 |
| **Cho & Matteson，ICLR 2026 TSALM workshop。**[官方接收名单](https://tsalm-workshop.github.io/accepted-papers/)已核实；[论文链接](https://openreview.net/pdf?id=dZx0R4Cm8d)本次仍返回 OpenReview browser-verification 页面，仅原始索引摘要可读。 | 摘要说明 TSFM 平均覆盖掩盖高波动失覆盖，以 GARCH 对原生 uncertainty scale 作乘性修正，并与 adaptive conformal 比较。 | TSFM 高波动失效加 GARCH 修正高度重合；不能宣称正文没有 context、seed 或 recovery 分析。 | **全文未读仍是最终贡献审核的未决项。**没有绕过访问限制；目前只用摘要支持上列问题/方法。 |

这张表用于限制和精炼叙事，不要求把所有方法都塞进四页报告或全部重新实现。应根据最终贡献保留最直接的对照，其他相关工作在参考文献/附录交代。

## “相近 NLL、不同 calibration”有更广泛先例

[D’Amour et al., *Underspecification Presents Challenges for Credibility in Modern Machine Learning*, JMLR 23(226), 2022](https://jmlr.org/papers/v23/20-1335.html)及[原文 PDF](https://www.jmlr.org/papers/volume23/20-1335/20-1335.pdf)已读 §2、§5–8、Fig.7 及 Discussion。作者说明：域内性能相近的训练管线可以产生部署行为差异，random seed 是探查方式；Fig.7 展示两个 fine-tuning seed 在域内相机上的校准近似一致，在留出相机上却不同。一般“validation 表现不足以约束 stress behavior”已有充分先例。

[*HARNet: A Convolutional Neural Network for Realized Volatility Forecasting*](https://arxiv.org/html/2205.07719) §5.1 原文可读，已经在金融波动预测里研究随机初始化与优化稳定性，比较 baseline HAR 初始化和随机初始化。因此泛称“首次注意到神经波动模型的 seed 不稳定”也不成立。HARNet 核查对象是优化/预测损失稳定性，并不据此推断其已经做过我们的 conformal 通道交叉。

**我的推断：**本项目有机会给 underspecification 增加一个具体的机制案例：它怎样进入 normalized-score calibration、为何 current 与 archive 的作用可相消、哪些可观测 scale-response 诊断超出 validation NLL。必须用独立重复建立这一具体性；仅把已知现象搬到另一个数据集不够强。

## Stage 2 到底支持了多少

[训练记录](stage2_training_notes.md)给出 GRU 的 mean validation NLL 范围 0.864387–0.865713，TCN 0.864238–0.874217。NLL 是 `log σ + y²/(2σ²)`，省略共同常数。TCN4102/4104 的值分别为 0.874217/0.865807，相差约 0.008410；对应 Gaussian entry interaction 是 −0.71048/+0.21877。

“相近”只能作为描述，**尚未证明 validation-equivalent**：没有预定实质等效 margin、配对路径等效性区间或控制 NLL 后的独立预测。两个 seed 的训练状态确有差别。8192 个 target pairs 来自64条验证路径、context会重叠，不能当8192次独立重训。

[主分析](stage2_results.md)显示架构 contrast 的固定模型路径区间很小，但 crossed seed/path 区间包含0；[archive 补充](stage2_archive_mechanism.md)定位 Gaussian TCN entry 的 current×window 分量在两条分解路线中都比 archive 分量更离散，recovery 则出现相消。这个定位是已有证据中最具体的部分；它没有唯一的因果中介百分比，没有证明通用神经优势，也没有证明旧输入梯度比例预测 outcome。[验证记忆诊断](stage2_memory_notes.md)里两个 TCN seed 的高 older-gradient fraction 与 interaction 正负不单调，已经提示“更多旧梯度=更差恢复”的简单叙事不成立。

## 下一轮的具体诊断与防过拟合设计

Stage 3 的[独立重训协议](../configs/stage3_replication_protocol.json)已经在新测试访问前冻结：3份独立 train/validation 数据×5个配对 seed×2架构，使用新测试噪声；[验证 response probe](../configs/stage3_response_probes.json)也已预先提交，包含既有256对梯度/置换和独立 Gaussian pulse 的 entry/recovery 尺度响应。维持这些协议即可，不因本审查更改已冻结 checkpoint、主端点或测试 case。

1. **先重复现象，再谈解释。**逐训练 panel 展示全5个seed、NLL、primary contrast 与两条 current/archive 分解；保留失败 panel 和 seed。三个训练数据重复对训练总体推断仍很少。共用 seed 索引使初始化/采样跨 panel 相关，不能把15对拟合叫15个完全独立训练管线。
2. **先作少量预定描述，暂不训练元模型。**画 NLL 与 outcome、pulse signed context contrast 与 current-component outcome 的散点；按架构分面并显示全部 panel。梯度、置换、signed/absolute pulse 响应不要自由组合成大量回归后挑最好结果。当前 Stage 3 的 feature–outcome 关联仍 exploratory；“事先定义 feature”不自动使所有事后关联检验成为 confirmatory。
3. **定义可观察量的对象。**pulse 诊断读取冻结模型和独立验证观测，事件时刻只用于构造和评分，不作为神经输入。它是人为 stress probe，可以诊断响应，不等于在线部署已知真实 shock 时刻。梯度局部、置换可能分布外，应和响应诊断分别报告。
4. **如果探索提示稳定关系，另冻一轮真正的预测检验。**在已探索的训练 panels 上固定一个最多2–3个特征的简单模型，并保留 NLL-only、architecture-only 与常数预测基线；新训练数据 panels 整组留出，不能把同一拟合的128条测试 path 分给训练/验证，不能随机切分同一个初始化的不同 phase。固定目标为某一路 current component 或两路均值，冻结后不换最相关目标。
5. **用预测误差验证增量信息。**预先比较 held-out training-panel 上 NLL-only 与 NLL+response 的 MAE/RMSE，保留所有拟合；至少报告每个 panel 的增益与失败，而非只给一个 pooled R²。关联、预测和干预是三种不同证据：预测改善仍不能单独证明响应特征是因果中介。现有3个 panel 不足以支持复杂拟合和可靠总体显著性。
6. **若要使用“等效 validation”一词，先设定 margin。**margin 应由业务/损失含义或独立验证设计确定，不能根据已见 outcome 挑能产生反例的 NLL 阈值；验证不确定性按路径聚合/重抽，保留两个 context 的损失。否则使用“平均验证 NLL 差距较小，但不保证校准交互一致”这种描述。

未来独立 prediction holdout 的候选范围可以保守：同一训练生成机制的新 train/validation seeds、所有原定初始化保留、共同新 Gaussian/t5 paths；先重复 D128/A3，再加一个预先指定的中间 duration 或 amplitude 作迁移压力检查。**本文件没有生成这些路径，没有给出其结果；具体数量、seeds、目标、模型、分析和停止规则需在访问前另行提交。**新随机路径只检验模拟机制重复，不能代替真实金融场景的外部验证；已看过的 ECB 2021–2026 period 仍只算探索数据。

若 response 诊断无法超出 NLL，或 current/archive 离散关系不能跨 panel 重现，也有应报告的结果：这些训练模型的局部梯度和 synthetic pulse 响应不足以解释校准交互；初始架构排序来自少量拟合、阶段与指标组合。诚实限定的失败边界比事后堆叠 gating/replay/controller 更可信，但能否形成 A+ 级原创贡献仍需由完整论证和课程评审判断。

## 可用于最终报告的贡献句

现在可写：

> We separate the current neural scale and the historical score-denominator pathways in a paired volatility-shock study. For the frozen fits studied here, their contributions can oppose each other during recovery, and the apparent architecture contrast is sensitive to training randomness. These exploratory findings motivate independent training-data replication and validation-only response diagnostics.

只有新的重复和预测检验支持时，才升级为“response diagnostics provide information beyond validation NLL”。避免 `first`、`universal`、`guaranteed conditional coverage` 或“新校准算法已提升金融预测”这样的结论。最值得凝练的课程贡献是一个**有明确先例边界、可证伪、可复现的机制结果及其失败条件**。
