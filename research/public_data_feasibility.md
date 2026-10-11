# 公开金融数据与概率神经模型可行性核查

核查日期：2026-10-11（Asia/Shanghai）。只使用官方原始来源。本文件不包含、也不使用微众数据、派生特征或比赛结果；未访问 HPC、未使用任何账号凭据。

落地状态更新：完整 ECB ZIP 已下载并固定，包含 1999-01-04 至 2026-10-09 的 7,111 个公布日；四个选定币种完整且为正。三种子 GRU 和公开基线已运行。审查建议中的验证期校准起点、真实未来标签干预、clipping 终态列名、参数/checkpoint hash 和许可链接已修正；日期块分析由 `scripts/analyze_public_fx.py` 实现并通过独立数值复核。下文保留样例核查及历史审查过程。

## 当前结论

1. **立即可用的数据方案：ECB 官方日度外汇参考率 + 自行生成的模拟序列。** 完整历史已下载并解析，不需账户或私人 API key。它是参考汇率数据，不是成交价格、日内 realized volatility 或可执行交易报价。
2. **VOLARE 保留为有条件的升级方案。** 论文和官网存在，统计量与研究主题契合；但当前官网需要登录，免费访问需要申请，尚未拿到数据包和明确的数据使用许可。不能写成“已下载、无门槛开放数据”。
3. **不使用从 FRED 获取的训练/评估数据。** 当前 FRED 官方条款包含机器学习开发/训练禁用条款。需要相同变量时，直接检查和使用原始生产者的数据与许可。ECB 是可立即执行的不同来源。
4. **主神经模型已从头训练小型 GRU；Chronos-Bolt Tiny 是可选扩展。** 若使用预训练模型，固定 2024 年权重版本，评估选其发布之后；真实预训练语料仍有不确定性。HPC GPU 规格仍未知，当前方案不依赖它。

## 数据方案 A：ECB 参考汇率（已验证）

官方入口：[Euro foreign exchange reference rates](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html)。页面提供完整 CSV ZIP、完整 XML、最近 90 日 XML；每日工作日约 16:00 CET 发布（TARGET 休市日除外），基础币种为 EUR。

| 项目 | 已核查事实 / 实验处理建议 |
|---|---|
| 样例获取 | `https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist-90d.xml` |
| 样例范围 | 本次返回 2026-07-13 至 2026-10-09，65 个观测日，70,586 bytes |
| 样例字段 | XML 日期 `time`、货币 `currency`、汇率 `rate`；当前每个观测日 29 种货币 |
| 历史 CSV API | `https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?startPeriod=1999-01-01&endPeriod=1999-01-15&format=csvdata` |
| 历史范围验证 | 初始小请求返回 10 行；后续完整下载包含 7,111 个日期，1999-01-04 至 2026-10-09。USD/JPY/GBP/CHF 四列已审计，完整且为正；其他货币不纳入本先导 |
| CSV 主要字段 | `TIME_PERIOD`, `OBS_VALUE`, `OBS_STATUS`, `FREQ`, `CURRENCY`, `CURRENCY_DENOM`, `UNIT` |
| 候选序列 | USD、JPY、GBP、CHF、CAD、AUD 对 EUR；最终预先固定货币集合，审计缺失后决定 |
| 许可 | [ECB copyright](https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html)：信息可免费复用，复制准确、注明 ECB 来源；若转换成收益率等，明确说明修改。不能把 ECB 署名论文的特殊授权规则误用于统计数据 |
| 引用 | European Central Bank, Euro foreign exchange reference rates，货币/序列标识、来源链接、取数日期；同时注明计算 `log(rate_t / rate_{t-1})` |

本次小样例保存在 `/Users/lebmar/Documents/ChatGPT/DL-project/tmp/public_data_check/`，其中 `ecb_sample_metadata.json` 记录取数 URL、日期范围、大小和 SHA-256；未大规模下载历史数据。

适合作为**下一观测日 log return 的预测区间**任务。若用 `r_t²` 或过去若干日的平方收益平均值，应称为日度风险代理，不应称为基于日内价格的 realized variance。实验必须先固定预测对象：预测 return 的分布与预测 volatility 的分布是两个不同任务。

日历处理：保留原始真实日期，只在观测日序列上预测“下一观测日”。不要为周末填零收益；货币退出、RUB 2022 年以后停止发布等不能随意补全。EUR 作为所有货币的共同分母造成相关性，不能将多个货币当作独立重复实验来夸大显著性。

## 数据方案 B：VOLARE（需申请，许可待确认）

官方 [VOLARE](https://volare.unime.it/) 是 React 网站；本次读取其官方公开前端资源，下载页明确需要登录，网站提供 [免费访问申请表](https://forms.gle/diCUQmZDpa35Jd778)。本轮没有注册账号、填写表单或尝试访问认证数据端点。

[作者数据论文](https://arxiv.org/abs/2602.19732)描述股票、外汇、期货的日度 realized variance/covariance 数据。可见数据目录股票起始通常为 2015-01-02，五个可视化外汇序列约始于 2009-09-25，期货约始于 2009-09-28；论文分析样本止于 2026-01-30，**不等于本日可下载包的实际终点**。官网 FAQ 声称每月第二天更新，实际包版本未验证。

候选字段包含 `rv1`, `rv5`, `rv5_ss`, `bv1`, `bv5`, `rsp*`, `rsn*`, `rq*`, `rk`，以及 OHLC；包内 CSV 与 README 预计提供实际时间范围、字段定义。原始 tick 数据不开放。最终字段名和有效行数仍需包内 README 核查。

**许可未知：** arXiv 论文采用 CC BY 4.0，不自动授权数据集；站点“free access”也不能直接推断允许公开再分发、训练神经网络和发布派生数据。取得账号/包后应先读取数据条款；如果不够明确，可由用户决定向维护者询问。此处只是识别后续依赖，不阻止先用 ECB 与模拟数据推进。

## 排除或暂缓的来源

- **FRED：** [现行官方法律条款](https://fred.stlouisfed.org/legal)在 Prohibited Use 和 API 条款都包含 ML 系统开发/训练限制；不下载其序列用于本项目。FRED 的“Public Domain: Citation Requested”标签不能覆盖服务层面的限制。
- **S&P500 via FRED：** [series notes](https://fred.stlouisfed.org/series/SP500)只提供十年日度历史，并有第三方版权声明；不作为默认无条件公开数据。
- **美联储 Board H.10 原站：** [数据入口](https://www.federalreserve.gov/releases/h10/default.htm)、[DDP](https://www.federalreserve.gov/datadownload/Choose.aspx?rel=H10)和 [Board disclaimer](https://www.federalreserve.gov/disclaimer.htm)可作为独立候选；Board 网站材料默认公共领域、要求引用来源。但 DDP 正在转型，H.10 是每周公布上周日度数据；若直接使用需处理发布时滞，不能以观测日期冒充实时可用日期。本轮未下载其完整 XML ZIP。
- **Stooq：** 未取得清晰的官方数据许可和可重复的获取验证，暂不列为可执行主来源。不能因网上教程曾支持 CSV 就认为 2026 年仍无账户限制或有公开发布授权。

## 轻量模型选择

### 1. 从头训练小型概率 RNN（建议主实验）

[DeepAR 原始论文](https://arxiv.org/abs/1704.04110)与 [GluonTS 官方 PyTorch API](https://ts.gluon.ai/stable/api/gluonts/gluonts.torch.model.deepar.html)可作基线来源。API 默认是两层、hidden size 40、Student-t 输出；我们可设置更小的容量，或直接用 PyTorch 实现 GRU 的 Student-t/分位数头。这样的实验能明确控制训练数据、网络输入、更新方式和种子，不受未知预训练污染影响。

“GRU + 分位数头”不是 DeepAR 的完全复现，应独立命名并引用其实际方法；Student-t 输出也只是建模假设，须与不依赖该假设的分位数/CQR 方案对照。训练/验证/测试时间段必须严格分开，scaler 只在训练段拟合，在线更新只能使用已到达的真实值。

### 2. Chronos-Bolt Tiny（建议预训练扩展）

[Amazon 官方模型卡](https://huggingface.co/amazon/chronos-bolt-tiny)：约 9M 参数、公开非 gated、Apache-2.0、输出分位数，支持 CPU/MPS/CUDA。模型卡的 API 示例可用于小规模推理；**本轮仅核验文档、模型配置和文件元数据，尚未安装或运行权重**。

建议固定 revision：`6904c2c38d12fb281d38cbf589c9171bfc050066`。本轮 [HF 官方文件 API](https://huggingface.co/api/models/amazon/chronos-bolt-tiny/tree/6904c2c38d12fb281d38cbf589c9171bfc050066?recursive=true&expand=true)返回权重的上传提交日期 2024-11-13、大小 34,622,352 bytes、LFS SHA-256 `75068728d376d2bec670379eeef4bfb4d24c0cfe24d957451f8d19b447030a32`。元数据副本已保存到样例目录。

固定版本 config 为 context 2048、原生预测长度 64、分位数 `[0.1,0.2,...,0.9]`。因此 **q10–q90 是 80% 区间**；没有原生 q05/q95，不能错误标为 90%。若目标覆盖率为 90%，应显式校准或选择支持这些分位数的主模型。

训练数据的全部构成、逐来源时间截止：**未知**。官方模型卡说训练近 100 billion 个观测，但不能由此得出金融数据一定未见。固定的 2024 权重与 2025 年后测试时段可排除读取未来观测的时间污染；早期事件分析只能标为可能重叠的历史诊断。公开上传日期也不能证明某个早期 benchmark 未被用于模型选择。

### 备选：TimesFM 2.5（目前无需采用）

[Google 官方模型卡](https://huggingface.co/google/timesfm-2.5-200m-pytorch)公开 Apache-2.0、200M 权重，列出 GiftEvalPretrain、Wikimedia（Nov 2023 截止）、Google Trends（2022 年底截止）、合成/增强数据。**这些截止只适用于所列来源，不是模型全局截止**。最新 2025 模型不足以保证 2025 年数据未见。相比 Tiny，它不是当前 pilot 的必要依赖，资源/版本核验留到需要更广模型验证时。

## 真实评估的最低边界

下面是研究设计建议，不是课程硬性规定，也不是已经完成的实验：

1. 一个可执行起点：从头训练模型用 1999–2022 训练、2023–2024 验证；最终 2025-01-01 至 2026-10-09 测试（每个货币实际起始以审计为准）。预训练模型固定上面的 2024 revision；未来若更新权重，重新定义干净测试段。
2. 预测时刻在 ECB 当日参考率发布之后，预测下一观测日。只使用届时已到达的价格、残差、区间命中情况。适应模块的超参数用验证段确定；测试段允许按预先声明规则在线更新，不能事后调节直到测试性能好看。
3. 先做一步预测，减少延迟标签和重叠目标问题。多步需要登记预测 origin 和实际反馈到达时刻；不能提前用 `t+h` 结果更新 `t+1` 的模型。
4. 真实数据的“短冲击/持续变化”可用于事后评价分组，但不能作为模型输入；若用未来窗口定义事件，明确该标签是 hindsight evaluation。更稳妥是先固定基于历史数据的触发规则，把完整测试时段全部保留。
5. 模拟里让短冲击与持续变化拥有相同初始幅度和共享前缀，持续时间是生成真值，只供评价。路径尚未分岔前，任何因果模型无法知道将来会不会持续；不能要求模型提前识别，更不能偷偷输入真实状态或恢复日期。
6. 同时报覆盖率、平均区间宽度、interval score、超界聚集程度，以及预先定义的事件后恢复指标。宽度不能单独评价；长样本平均覆盖也不能替代短期可靠性。金融真实状态的“oracle”界宽未知，不要把模拟 oracle 指标移植成事实。
7. 真实货币相关，按共同日期成块评估/重采样，报告每货币结果；模拟用多个独立种子。不能把重叠时间窗视为独立样本。

## 下一步可立即执行

- 先用已验证 ECB 数据通路，正式获取预先固定的少量货币，冻结截止日期、记录元数据与许可。
- 在自生成序列上构建一步预测任务与短冲击/持续波动变化的共同前缀测试。
- 训练小型概率 RNN，再小规模运行固定版本 Chronos-Bolt Tiny。是否扩大模型数量由 pilot 结果和 HPC 实际资源决定。
- VOLARE 账号、包内条款与预训练数据逐来源审计属于后续确认项，不能把它们写成已完成成果。

## 真实 ECB + GRU pilot 独立代码审查（2026-10-11 后续）

对象：`scripts/run_public_fx_pilot.py` 与 `outputs/public_fx/run_manifest.json`；审查时脚本 SHA-256 为 `fdb9f15927285f1cd3cf8858a71f6c08c8b6dbeba5eb9541a9d359a83c7c53f6`。本审查没有修改根节点脚本。

### 已验证的事实

- 根节点已取得正式 ECB 全历史 ZIP，7111 个汇率观测日，1999-01-04 至 2026-10-09；四种货币 USD、JPY、GBP、CHF 无缺失并为正。原始包 SHA-256 与 manifest 一致。之前“仅小样例”的状态已由本次正式 pilot 更新。
- 当前实现是 **Gaussian likelihood 的小型 GRU**，不是 Student-t DeepAR，也不是预训练模型；参数 1266，三个训练种子。训练时同一模型混用 16/64 长度，比较时共享权重。
- 实际训练目标日期 1999-04-05 至 2018-12-31（每货币 5055 个窗），验证 2019-01-02 至 2020-12-31（每货币 512），测试 2021-01-04 至 2026-10-09（每货币 1479）。此前提出的 2025 年后测试段是预训练污染控制建议，当前从头训练模型可以使用更早的测试段。
- 189312 条预测记录的 `(date, asset, seed, predictor, context, calibrator, window)` 组合没有重复，manifest 的样本范围、参数数量和源代码 hash 均吻合。记录行数包含模型种子、上下文和校准器重复，不能称为 189312 个独立观测。

### 核心因果检查：通过

1. `build_arrays` 的 64 长度窗以 `r[t-1]` 结束，目标是 `r[t]`；标准差仅用 2018 年末及以前的数据。训练随机打乱训练窗没有使用测试目标，验证集只用于 checkpoint 选择。
2. 神经模型推理仅使用历史窗，推理时没有 batch normalization 或跨测试样本注意力；把全部测试窗一起预测本身不构成泄漏。
3. 滚动分数只读取 `[t-window:t]`。即使提前算出整个 `normalized` 数组，当期区间也不读未来元素。ACI 在给出当期 width 后才读取 `y[t]` 并更新 alpha。EWMA 同样先写当期 sigma，再读当期收益更新下一步。
4. 独立运行了**真正后缀干预**：把 2024-01-02 及以后的原始收益大幅修改后，目标日期不晚于该日的历史输入、训练 scaler、冻结模型预测完全不变；单独修改未来目标标签，native/rolling/ACI 在第一个被改标签到达以前以及该日生成的区间完全不变。该实测支持上述代码审查。

### 需要修正或明确的边界

- **校准起点：** 审查时 ACI 从训练历史开始递推，归档包含最终已拟合模型对训练样本的 in-sample 残差。它没有读取未来测试值，但会造成初始化/状态比较不清楚。根节点已经计划把校准起点限制到 2019 年验证段；GRU 与 EWMA 应采用相同校准起点和 warm-up。修复后重跑并更新 hash。
- **验证回放的含义：** 最佳 checkpoint 使用整个 2019–2020 验证段选择，随后用冻结模型回放验证段来初始化 2021 测试。这可以作为测试前校准准备，但不能称为该模型在 2019 年当时真实部署的在线预测，也不能宣称满足独立、可交换的 split-conformal 保证。
- **原脚本的 suffix 注释不成立：** 审查时所谓 suffix intervention 只是相同输入与 clone 的比较，没有扰动未来；建议删除或替换为真正后缀干预。上面的独立检查确实扰动了原始后缀与未来标签，因此可作为有意义的补充证据。
- **完整历史 clipping count：** `clipping_total_history` 在所有预测行中存的是整段运行后的总次数，包含该行日期之后的诊断信息。它不参与当前预测，因此没有模型泄漏；但不能作为在线输入或当期可用诊断。宜移到运行汇总，或改名明确表示 final count。
- **相关性：** 本脚本只有逐资产/种子描述性汇总，没有置信区间，因此没有 IID 显著性错误。文件开头“date-block resampling retains dependence”的措辞应对应后续真实实现；当前此脚本没有执行重采样。后续 bootstrap 应共同重采样同一日期块，先平均模型种子，不能把相同金融路径的种子运行当作独立金融样本。
- **manifest 还应补充：** calibration start/warm-up/reset、ACI gamma=.01 和 alpha clipping 范围、随机种子及当前 checkpoint hash；固定原始截止日期。当前日期是硬编码，复用缓存时 `retrieved_date` 应区分原始取数与本次运行日期。
- **许可链接：** manifest 所用 `/services/copyright/html/index.en.html` 本次 web tool 无法打开；已核实的规范链接是 `https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html`。
- **研究结论限制：** 90% native 区间来自 Gaussian 假设，不代表实际覆盖为 90%。这轮没有真实市场状态标签、GARCH 与强自适应基线、独立后续盲测，不能据此宣称已解决短冲击/持续变化识别或已证明原创方法优越。
