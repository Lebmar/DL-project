# 下一轮 response-diagnostic 预测验证的设计审查

**当前建议：先完成 Stage 3 机制研究的报告，不立即启动新增32个拟合的元预测实验。**已有证据值得保留的是 current/archive 通道与初始化、phase 的具体关系；entry signed pulse feature 在架构内并没有清晰的跨训练 panel 重复关系。增加一个小回归和新模型数量本身不能提高原创性，也不能把 scatter 变成已经验证的预测信息。

本文件只审查设计，没有生成任何新数据、训练或运行元模型，也没有修改冻结源码。2026-10-11 先读取 frozen Stage 3 protocol 与 response feature definitions；在 `outputs/stage3_analysis/manifest.json` 确认为 complete **之后**，才读正式 Stage 3 report 和现有30行 entry features。以下未来参数只是建议，尚未形成已冻结实验协议。

## 当前证据为何不足以立即推进

[Stage 3 结果](stage3_results.md)中，Gaussian entry 的 GRU−TCN contrast 在三个训练 panel 都为负，但大小分别为 −0.31008、−0.25474、−0.03089；固定-fit 路径区间为负，crossed sensitivity 仍包含0。Recovery contrast 的 panel 方向不同。TCN current 分量的 seed 标准差同时高于 archive 与 GRU current 的组合为4/6，panel1没有完整复制原先的 current-dominant variability。它们已经形成有信息的重复与失败边界，无须预设再加一个预测模型就能获得方法优势。

尤其不能仅凭 pooled pulse–outcome scatter 开始回归。当前 GRU 的 signed pulse feature 大多为正、current component 为负；TCN 的均值与范围不同，feature 容易代理 architecture。TCN panel1 的 seed4105 为 `pulse −0.06455 / current +0.00899`，panel2 同 seed 为 `−0.08216 / +0.86128`；panel3 seed4102 为 `+0.00659 / −0.48438`。这些例子不构成相关系数或总体检验，但足以说明单调关系和异常点作用没有清晰跨 panel 重复。不能把一个 extreme TCN fit 支配的 pooled 拟合，当作普遍机制。

本审查没有另外计算或拟合回归、相关系数、leave-one-panel-out 元模型来挑选方案。若继续新轮，应作为对一个尚未支持的候选 diagnostic 的**预先固定预测反证实验**，接受其完全失败；不能把它称作 Stage 3 已证实关系的最终验证。

## 统计对象必须说清楚

训练元模型可用 Stage 3 的 `3 training panels × 5 seeds × 2 architectures = 30 fits`，即每架构15个拟合。它们不是30个独立训练样本：同 panel 共享训练/验证数据，同 seed 跨 panel 共享初始权重规则与 sampling indices；全部标签还共享 Stage 3 的测试路径。

三个 training panels 对复杂选择、可靠 hyperparameter CV 或总体规律判定太少。数值上可以拟合只有两条连续 feature 的固定 ridge，但拟合成功不意味着获得足够独立信息。随机将这30行分成训练/验证，会把同一训练数据和 seed 的相关行分到两边；随机划分同一个拟合的128条 paths，更不是新-fit泛化检验。不能用现有目标标签选择 polynomial degree、ridge penalty、outlier trimming、feature interaction 或最佳 calibration component。

未来目标应始终是**某个新冻结拟合在一个指定 simulator case 中的 current-component path 平均值**。这不是金融收益预测，也不是未知市场 regime 的在线预警。元模型预测的是训练后可诊断的区间交互行为；即使成功，也不证明 pulse feature 是 causal mediator。

## NLL 与 pulse 的可比性

当前 `validation_nll` 来自每 panel 自己的 validation observations，且用于 checkpoint selection。原始值包含该组观测的难度与模型差异；例如 GRU 在 panel2 约0.796、panel3约0.840，不能把这一个跨 panel 差解释为后者训练更差。加 architecture dummy 或把30行合并 standardize，不会自动消除不同 validation data 的影响。

若未来实施，最佳补法是在新协议中先固定一份 **共同、独立的 diagnostic data 和 target-pair 集合**，对所有 Stage 3 旧 fits 与未来新 fits 都用相同观测计算 NLL。所有 checkpoint 保持不变，它不用于重新选 checkpoint。建议64条 length1024 的原 training-distribution paths、8192个共同 past-only target pairs，contexts16/64 的同权平均 NLL，沿用现有省略共同 Gaussian 常数的定义。三项 seeds、generator、采样规则须在生成前提交。

同理，pulse measurement 建议在下一轮固定为一组共同64条 length768 的 Gaussian paired pulse paths，沿用 onset256、duration128、amplitude3 和原 signed entry contrast 定义，对所有旧/新 fits 使用相同路径。这样减少 panel-specific probe noise，不能自由挑选最相关的一组 probe seeds。该 feature 读取的是受控合成实验，不要求部署中知道事件标签。

这属于**新轮预定的共同 diagnostic feature**，不是改写 Stage 3 既有 frozen features。不能再将其声称为完全沿用原 Stage 3 scatter 的结果。只做 panel 内 NLL centering 也可消除部分共同偏移，但新 panel 若仅有两个 seeds、训练 panel 有五个 seeds，群体均值的噪声与定义不同；它还要求同时收集新 panel 全部 fits，变成 cohort-level diagnostic。本审查优先推荐共同数据，避免以这个 centering 换写个人模型的预测问题。

## 如果仍要做，保持一个最小、明确的预测问题

只固定以下一个目标：Gaussian、D128、A3、onset384、`entry=[384,416)`、interval score、`current_at_archive64` 的 event-minus-matched-control 值。令 `d(a,b)=mean_IS_event(a,b,W128)−mean_IS_event(a,b,W32)−对应control差`，标签为 `d(64,64)−d(16,64)`。保持 archive 分母 context64 不变，改变 current context16/64；不转而挑另一条分解路线、diagonal J、recovery 或 coverage。

每个 fit 先平均固定的全部独立 path，得到一行标签；meta training 使用30行，不能展开为每个 path 一行样本。训练标签可用已完成的 Stage 3 Gaussian D128 case。它们已经用于探索，未来协议必须写明 `Stage3-informed`。未来 held-out labels 使用共同新 Gaussian D128 A3 配对 paths，不能复用 Stage 3 test seeds，也不把共享 duration entry 当额外样本。建议仍保留128条 paths，并在精度不足时报告区间，而非读结果后加样本直至显著。

建议四个预定模型：

| 模型 | 输入 | 用途 |
| --- | --- | --- |
| Constant | 截距 | 简单下限对照 |
| Architecture-only | 截距、GRU/TCN indicator | 消除 feature 仅代理架构的解释 |
| Baseline | 截距、architecture、共同 diagnostic NLL | **主基线** |
| Augmented | 截距、architecture、共同 NLL、共同 pulse signed entry contrast | 唯一 candidate |

主要比较必须是 **Augmented 对 Baseline**，不能只用无 architecture 的 NLL-only 模型作对照。所有模型共用训练 labels 和固定 preprocessing。

一个可执行但保守的参数建议是：两个连续 features 在30个训练 fits 中先减去各自 architecture 的训练均值，再用全部训练 residuals 的标准差 standardize；未来输入只用这些已保存的均值/标准差。目标用训练30行的全局 mean/std standardize。截距与 architecture coefficient 不处罚，连续系数用固定 ridge objective `mean(square_error) + 1.0 * sum(beta_continuous**2)`。`lambda=1.0` 是明确固定的收缩选择，不声称最优；**不要**用三个 panel 做 CV 后选择 lambda。只用线性 main effects，不加 architecture×pulse、NLL×pulse、截断、筛样或 nonlinear search。

共同 feature 的训练 residual std 若≤`1e−12`，将该列固定为0、系数0并记录退化；若 pulse feature 退化，该 candidate 不能支持增量信息结论。目标 std 若≤`1e−12`，预测问题退化，终止该轮假设而非更换目标。模型 coefficients、scalers 与全部30个训练 fit IDs 必须在 held-out neural training/labels 访问前保存并提交；预测时不重新 fit scaler。

## 8×2×2 的 holdout：两种 seed 设计不可混用

八个新的 training/validation data panels、每 panel 两个新初始化/采样 seeds、两架构，共32个 held-out neural fits，比随机划分已有30行好得多。所有拟合使用原 NN 训练管线，checkpoint 仍只按各自独立 validation 选择；全部32个 fits 保留，不删除 NLL 较差、training response极端或误预测的模型。

若只有两个 seed IDs 在八 panel 重复使用，这是 **8 panels × 2 shared seeds 的 crossed design**，不是16个独立初始化。同一架构、同 seed 的初始权重/采样规则跨 panel 共享；仅两个初始化太少，不能据此推广初始化总体。分析重抽 seed 时必须给每个被抽的 panel 使用相同 seed draws。

更推荐在保持32 fits预算的情况下，为每个 panel 预先指定两个**彼此及跨 panel 均不同**的新 seed IDs；每个 panel/seed 内 GRU与TCN仍共享 sampler seed 配对。这样是 eight data panels × two nested seed pairs，包含16个独立生成规则的新 seed pairs。它仍不是32 iid行；每 panel 共用训练数据，两架构成对。新分析必须写 nested design，不可套用 Stage 3 的 shared-seed advanced indexing或要求所有 panel seed IDs 相同的 grid函数。具体 IDs 在新协议中一次指定，不能按结果挑选。

预算更小的方案可做四 panel×三新seed×两架构（24 fits），但 training-data holdout只剩四组，panel总体不确定性更大；不应称同等强度验证。若目标主要是训练数据泛化，八 panel/32 fits 比在少数 panel 增加许多 seeds 更合适。共同128条 test paths用于所有32 fits，架构与 panel 比较都保留共同路径配对。

## 预测误差、bootstrap 与失败条件

预定主要预测误差为每个 fit 的 path-mean target 上的 squared error。每 panel 对两个 seed与两个架构同权平均，再对八 panel同权平均。主要增益定义

\[
\Delta=MSE_{Baseline}-MSE_{Augmented},
\]

正号表示 candidate改善。附报两架构、每 panel 的 `Δ`、MAE、全部32个固定预测与误差，不能用 pooled R² 的高低单独判断。Outlier 会真实影响 squared-error loss，应完整保留；如果它支配结果，报告影响，不事后 winsorize。

Bootstrap 必须重算 path-mean label之后再计算预测误差，并使用相同 draws比较 baseline/augmented。测试 paths在所有 fits之间共用，必须对所有模型同时重抽同一 path index。固定模型的 path interval只覆盖当前拟合与共同 diagnostic/meta training固定后的测试路径随机性，不能代替 new-training-pipeline population interval。

若使用 shared two-seed design，建议另报 `panel × shared seed × common path` 三维配对 bootstrap，seed draw跨所有抽到的 panel共享；因为 seed只有两个，其区间只作敏感性检查。若改用推荐的 nested seeds，panel整组重抽、组内成对 seed重抽、共同 test path重抽，不能将architecture独立抽走。八个 panel仍少，percentile区间只是近似；分别展示仅固定fits/path、panel/path和nested seed/path的范围，并明确都条件于已经固定的meta training coefficients，不涵盖只有三组的元训练数据不确定性。不要从bootstrap重复次数4000推导独立training样本数量。

具体建议把 **training panel 整组 + common path** 的两维配对 bootstrap作为主要区间，每个被抽的panel先完整保留其两个seed/两个architecture，候选与基线共用同一个panel/path draw。对共享two-seed设计，这个区间条件于那两个固定seed；对跨panel唯一nested seeds，panel包含其预先抽定的seed对。三维crossed/nested-seed resampling只作另一个敏感性范围，避免只有两seed却将其总体区间当作可靠主证据。Fixed-fit/path区间再单列，说明测试噪声贡献。所用主区间应在新协议中明确一次固定。

建议在协议中选一个 primary comparison、95% paired interval、4000 resamples、固定 RNG seeds；MAE和分组读数均为描述性，无第二轮hyperparameter search。若要有实际改善标准，可预先固定 `relative MSE gain≥10%`；基线 MSE接近0时这个比例无定义，应报告原始Delta和失败，不更换分母。

一个刻意保守的成功门槛可要求：总体Delta为正且主要预定区间不含0、改善达到预定实质门槛、两个架构各自point Delta都为正、至少六个八panel point gains为正、leave-one-panel-out的全部point Delta仍为正。这些分组/LOO是固定 robustness gates，不是六次独立显著性检验。仅fixed-fit path interval通过时，措辞限于这批保存的 fits，不得升级成初始化/训练总体结论。

以下任一情况均应判为**该轮未通过预设成功门槛**，保留完整失败结果并停止该candidate的当前轮；未通过不等于证明不存在任何预测信息：

- 增益为负、主要区间包含0或未达到预定实质门槛。
- 增益主要来自一个 extreme fit/panel，删去该 panel就翻号；或仅一个 architecture获益、另一architecture退化。
- Architecture-only 已解释绝大部分变化，加入 pulse 相对 architecture+NLL没有稳定改善。
- 新panel features离开训练范围导致外推，或共同 NLL/pulse重新测量后原scatter关系不在；报告范围与所有预测，不剪掉越界fit。
- 改变目标、路线、probe amplitude、penalty、seed、样本数量后才得到正结果。此时必须另标 exploratory，不能保留原 confirmatory标签。

## 顺序与结论边界

若之后执行：先提交目标、共同 diagnostic数据规则、meta class/preprocessing/lambda、holdout NN/seeds/grid、loss/bootstrap/失败规则；再测量旧30 fits共同features并一次拟合meta，冻结系数；接着按协议训练新32 fits、算features、保存每个fit的meta预测；最后才生成/读取新target paths并评分。每一步来源哈希和已完成时间可审计。任何步骤失败不允许静默切换方案。新NN继续保留原trainer不改动，新增driver/分析须使用新文件。

即使全部门槛通过，也只能说：在同一受控生成机制、新训练数据与指定Gaussian pulse case下，固定诊断组合比architecture+共同NLL更准确地预测了该current-component均值。Probe自身和outcome都围绕人工pulse，成功不是未知真实市场shock的预测或金融部署优势，不是因果中介证据，也不是新校准算法的覆盖保证。

以当前Stage3读数，我优先建议把 **独立训练重复的部分成功与失败、current/archive相消和初始化边界** 写清楚，并让学生理解完整机制，再决定是否值得为这个尚未稳定的pulse候选新增验证。这个判断属于研究取舍，不是对A+的保证或否定。
