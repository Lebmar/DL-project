# Stage3：独立训练数据上的机制复制

本轮由 Stage2 已观察到的初始化敏感性与 current/archive 相消启发，属于预先固定的新复制；不是把 Stage2 重新命名为盲测，也不是新算法评估。协议在训练和新测试生成前提交为 `c91daf3`；额外验证响应特征在新测试访问前提交为 `425ff3d`。

## 固定的设计

3个新的训练/验证面板，每个面板用5个初始化及训练采样种子，配对训练 GRU 和 TCN，共30个新拟合。模型结构、特征、优化程序、上下文16/64与早停规则直接复用未修改的 Stage2 源码。训练与验证的数据种子分别为 `(314259,314260)`、`(314359,314360)`、`(314459,314460)`；验证目标采样种子为314261/314361/314461。

新测试固定 Gaussian 与单位方差 t5、冲击幅度3、持续32/128、128条路径。基础种子1902000，第二个噪声家族加1000；同家族两个持续时间共享冲击前噪声，因此它们的 entry32 是同一条件，不能作为独立重复。当前尺度上下文与归档分母上下文保留完整交叉8格，报告两条分解路径。没有按验证或测试排名删掉模型。

每个 seed 索引跨训练面板共享初始化及小批次采样程序；GRU/TCN 共享数据、目标与批次前缀。训练面板与 seed 是交叉结构，15组拟合不是15份独立训练数据。主要区间只重抽测试 path、条件于这15对拟合。3个训练面板与5个 seed 的 crossed 重抽只作敏感性展示；面板数少，不能据此宣布总体架构显著优势。

## 验证特征

固定每个面板最先保存的256个验证目标，用原梯度与旧48块置换诊断；另用独立64-path Gaussian 验证脉冲读取 entry/recovery64 的尺度响应。模型只看到过去观测，事件位置只用于构造和汇总探针。该验证探针的随机种子、阶段、幅度及特征在新测试访问前固定。

这些特征与测试行为的关联只是探索性描述，不是已训练的预测器或模型选择准则。验证 NLL 的差距需要实际报告；没有预设等效性界或做等效性检验，不能称所有模型 validation-equivalent。

## 执行与复现

在完整锁定环境和干净输出位置按顺序执行：

```sh
.venv/bin/python scripts/train_stage3_repeats.py
.venv/bin/python scripts/stage3_response_probes.py
.venv/bin/python scripts/evaluate_stage3.py
.venv/bin/python scripts/analyze_stage3.py
```

新 evaluator 与分析源码也必须先提交再读取相应新数据。各步骤拒绝覆盖已有 manifest。重现运行应另建 checkout 或先备份 `outputs/stage3_training/`、`stage3_response_probes/`、`stage3_evaluation/`、`stage3_analysis/`。训练 wrapper 的 `--resume` 只会核查并跳过已完整登记的面板，不会自动覆盖部分训练。

训练 wrapper 临时把原 trainer 的输出目录指向各 panel；源码和算法均不改。每个 panel 保留自己的 input/frozen protocol、checkpoint、epoch history、validation target pairs 和 manifest；顶层 manifest 保存原源码 hash、wrapper hash、协议 hash 与每个 panel manifest hash。测试和分析再核验这些来源。

## 贡献边界

较早的 underspecification 研究已经揭示 iid 验证表现与压力行为之间的区别；本项目需要证明具体通道诊断有什么增量知识，而不能仅宣布随机种子很重要。[当前贡献审查](contribution_review.md)列出直接先例和仍缺失的证据。只有新复制完成后，才能评估 Stage2 的具体机制是否保留、需要收窄还是失败。
