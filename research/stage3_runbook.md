# Stage3：独立训练数据上的机制复制

本轮由 Stage2 已观察到的初始化敏感性与 current/archive 相消启发，属于预先固定的新复制；不是把 Stage2 重新命名为盲测，也不是新算法评估。协议在训练和新测试生成前提交为 `c91daf3`；额外验证响应特征在新测试访问前提交为 `425ff3d`。

全部30个新拟合与4组新测试现已完成，结果见 [Stage3实际报告](stage3_results.md)。Gaussian D128 A3 的entry架构contrast在三个训练面板均为负，但crossed训练面板/共享seed/path敏感性区间包含0；recovery没有稳健重复。同一Gaussian entry条件下，TCN current种子标准差在6/6个面板×路线组合大于GRU current，只有4/6大于自身archive；这是相关组合的描述，不能称普遍current主导。

[四页报告工作稿PDF（草稿）](../output/pdf/working_report.pdf)与[LaTeX源码](../report/working_report.tex)供学生理解和修订，尚未提交。当前下一步是收敛机制叙事并获取学生反馈；32-fit meta-prediction扩展未启动。

## 固定的设计

3个新的训练/验证面板，每个面板用5个初始化及训练采样种子，配对训练 GRU 和 TCN，共30个新拟合。模型结构、特征、优化程序、上下文16/64与早停规则直接复用未修改的 Stage2 源码。训练与验证的数据种子分别为 `(314259,314260)`、`(314359,314360)`、`(314459,314460)`；验证目标采样种子为314261/314361/314461。

新测试固定 Gaussian 与单位方差 t5、冲击幅度3、持续32/128、128条路径。基础种子1902000，第二个噪声家族加1000；同家族两个持续时间共享整条基准噪声，因此它们的 entry32 是同一条件，不能作为独立重复。当前尺度上下文与归档分母上下文保留完整交叉8格，报告两条分解路径。没有按验证或测试排名删掉模型。

每个 seed 索引跨训练面板共享初始化及小批次采样程序；GRU/TCN 共享数据、目标与批次前缀。训练面板与 seed 是交叉结构，15组拟合不是15份独立训练数据。主要区间只重抽测试 path、条件于这15对拟合。3个训练面板与5个 seed 的 crossed 重抽只作敏感性展示；面板数少，不能据此宣布总体架构显著优势。

## 验证特征

固定每个面板最先保存的256个验证目标，用原梯度与旧48块置换诊断；另用独立64-path Gaussian 验证脉冲读取 entry/recovery64 的尺度响应。模型只看到过去观测，事件位置只用于构造和汇总探针。该验证探针的随机种子、阶段、幅度及特征在新测试访问前固定。

这些特征与测试行为的关联只是探索性描述，不是已训练的预测器或模型选择准则。验证 NLL 的差距需要实际报告；没有预设等效性界或做等效性检验，不能称所有模型 validation-equivalent。

## 执行与复现

当前保存的结果可直接核查，无需重新训练或生成测试：

```sh
.venv/bin/python scripts/verify_stage3_artifacts.py
```

259项核查已通过，覆盖冻结协议与源码、全部30个checkpoint及验证选择、配对训练批次、validation目标、eval/probe/analysis来源和输出哈希、完整网格、逐path分解和统计检查。Stage3的960个crossed遗忘界、480个diagonal五指标检查通过；遗忘界后半宽最大误差0。D32/128的entry在逐path interaction、components及两类性能表中完全一致，不能将这两组重复计数。

本地CPU实际耗时如下；Python 3.12.14、Torch 2.14.1，完整依赖见 [requirements-lock.txt](../requirements-lock.txt)。不同硬件与库版本可能有数值和耗时差异。

| 步骤 | 耗时 | 输出 |
| --- | ---: | --- |
| 3面板、30拟合训练 | 935.88秒 | `outputs/stage3_training/` |
| 仅验证集响应探针 | 63.73秒 | `outputs/stage3_response_probes/` |
| 4情景、完整factorial评估 | 598.53秒 | `outputs/stage3_evaluation/` |
| 固定拟合统计与图表 | 3.82秒 | `outputs/stage3_analysis/` |

在完整锁定环境和干净输出位置按顺序执行：

```sh
.venv/bin/python scripts/evaluate_stage3.py --self-check
.venv/bin/python scripts/analyze_stage3.py --self-check
.venv/bin/python scripts/train_stage3_repeats.py
.venv/bin/python scripts/stage3_response_probes.py
.venv/bin/python scripts/evaluate_stage3.py
.venv/bin/python scripts/analyze_stage3.py
.venv/bin/python scripts/verify_stage3_artifacts.py
```

新 evaluator 与分析源码也必须先提交再读取相应新数据。各步骤拒绝覆盖已有 manifest。重现运行应另建 checkout 或先备份 `outputs/stage3_training/`、`stage3_response_probes/`、`stage3_evaluation/`、`stage3_analysis/`。训练 wrapper 的 `--resume` 只会核查并跳过已完整登记的面板，不会自动覆盖部分训练。

训练 wrapper 临时把原 trainer 的输出目录指向各 panel；源码和算法均不改。每个 panel 保留自己的 input/frozen protocol、checkpoint、epoch history、validation target pairs 和 manifest；顶层 manifest 保存原源码 hash、wrapper hash、协议 hash 与每个 panel manifest hash。测试和分析再核验这些来源。

TCN的结构感受野恰好64，evaluator仅对context64使用同数学上下文的完整因果卷积前向；GRU与TCN16仍逐窗口重置。原始零观测padding保持一致。该执行优化在toy全时点与每个新TCN的validation片段上先和原函数比较：预置`rtol=1e−6, atol=2e−6`，未通过就自动使用原逐窗函数；实际15个新TCN全部通过，最大绝对尺度差`2.861e−6`，相对项也包含在阈值中。源码在新test访问前提交，不按test选执行方式。

不同duration的事件前向统一使用协议固定的最大duration所决定的相同计算形状，只复制实际受影响的上下文。它避免执行形状引起共同entry的舍入差异；toy、旧冻结权重及新结果的共同前缀检查通过。事件位置仅用于路径构造、phase/oracle评分与等价上下文缓存，未进入模型或分位数特征。

主要输出是5760条native/rolling性能均值、7680条crossed格子均值、61,440条逐path diagonal interaction，以及368,640条逐path六分量记录。六分量为total、两条路线的四个current/archive分量，以及third-order路径差；后者满足`C@b64−C@b16=A@a64−A@a16`。原五指标全部保留，log-width抵消仅作代数检查，不作新发现。

## 贡献边界

较早的 underspecification 研究已经揭示 iid 验证表现与压力行为之间的区别；本项目需要说明具体通道诊断的增量知识，不能仅宣布随机种子很重要。[当前贡献审查](contribution_review.md)列出直接先例和仍缺失的证据。新复制支持收窄到阶段、训练面板和分解路线依赖的经验结论；recovery失败重复与panel1的archive较大变异都完整保留。三面板和五个共享seed不支持初始化/训练数据总体的显著优势声明。

有限分位数秩也随W变化：W32与W128分别使用第30/117顺序统计量；这不是纯粹的历史长度变化。Exchangeability下的理想秩参照不构成本实验相依、非平稳scores的覆盖保证，见 [有限秩备注](quantile_rank_notes.md)。有限遗忘界与log-width恒等式同样不能包装为新定理或真实市场恢复保证。

第一阶段真实ECB时期已经用于探索；Stage2主比较、事后archive补充和Stage3新复制按各自历史保留，未追溯改称同一批盲测。只使用公开与自生成数据，不读取微众比赛数据及派生特征、样本、结果，不保存HPC凭据。AI工具已获课程许可，学生仍承担个人作者责任并须理解全部假设、代码和结果。Proposal截止10月16日23:59；报告、PDF slides与8–10分钟视频截止12月11日23:59，具体格式见 [课程要求](course_requirements.md)。
