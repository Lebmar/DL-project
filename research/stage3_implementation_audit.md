# Stage 3 训练重复与 validation response probes 的独立实现审核

本文件由不负责 Stage 3 训练/评分写出程序的审核 agent 撰写；只读代码与已经完成的训练文件，不修改 checkpoint、training manifest 或原源码，也不主动执行任何 writer。日期：2026-10-11。最终审核状态为 **全部30个模型训练来源、sampler/early stopping、selected validation NLL 以及全部30个 response probes 复算通过**。另完成分析程序的无真实测试数据 toy 审核。本审核不读取新的 replication test paths 或结果。

## 训练输出与冻结协议

审核 [train_stage3_repeats.py](../scripts/train_stage3_repeats.py)、被原样复用的 [train_stage2.py](../scripts/train_stage2.py) 和 [stage3_replication_protocol.json](../configs/stage3_replication_protocol.json)：

- 原 trainer 的数据文件、manifest、checkpoint、每模型 history、汇总 CSV 均通过模块全局 `OUT` 形成路径。driver 在 `trainer.main()` 前将 `trainer.OUT` 指向 `outputs/stage3_training/panelN`，并通过 `sys.argv` 指定该 panel 的 `input_protocol.json`；`finally` 恢复这两个全局值。没有把新拟合写回 Stage 2 目录的隐藏固定路径。
- 原 trainer 的 `main` 只有显式调用时才运行。导入不读取 test 文件。driver 不读取 Stage 2 的性能结果，也没有基于 validation/test 排名删除模型。
- 父协议在首次训练前 frozen、已提交；driver 验证父协议与自身的字节内容等于 HEAD 中已提交版本，并核对原 trainer/models/evaluator/baseline source hashes。生成的 panel protocol 包含完整 training config 与父协议 SHA256；原 trainer 将原字节复制为 `frozen_protocol.json`，其 checkpoint 记录该 panel protocol hash。
- 三个 panel 的模型、contexts、generator、优化、CPU 配置都沿用原定义；变化仅为 training data seed、validation data seed 和 validation pair seed。训练/验证随机流各自不同。额外的 panel 编号只用于来源标识。
- `--resume` 只跳过已经记录且完整、manifest hash/config/checkpoint hashes 正确的 panel。已有部分文件或未记录的 panel manifest 会拒绝静默重训。整个流程串行调用 trainer，因此全局目录重定向不会在两个训练调用之间竞态。

审核时的协议 SHA256 为 `f57018b06e92688ccc92905d84adfb9fedc07ee7a238e4df9622ccb21f3280d6`；driver SHA256 为 `3898736b30610a49006f915c9e978e73db7dd9ae4f3de97162505eb28aadc34e`。原 trainer 的 SHA256 保持 `d93d1ecbb6ce30ccba88c2566be2632d769db6c2968717c3f6c36f24fbc49850`。

## 配对、checkpoint 选择和抽样单位

每个 seed/architecture 调用前重新设定 Torch seed，并独立重新创建以相同模型 seed 初始化的 NumPy batch sampler。后者与模型架构消耗的 Torch RNG 无关，所以跨架构共享每 epoch 的 contexts/path/target 批次前缀。每 epoch 恰好 16 个 context16 与 16 个 context64 批次。两个架构 early stopping 可以在不同 epoch 结束，因此并不保证相同优化步数，也不能据此将表现差异全部归因于架构本身。

checkpoint 选择使用两种 context 的 validation NLL 的绝对最小值；`min_delta=.0001` 只用于 patience reference。`min_epochs=20` 是允许停止训练的最早 epoch，不是禁止保存早于 20 的最佳 checkpoint。Panel1 的 TCN 4101 可以合法地在 epoch22 停止、保留 epoch14。

三个数据 panel × 五个初始化/sampling seed 是**交叉设计**。相同 seed 跨 panel 共用初始权重生成规则和采样索引；同一 panel 共用训练/验证数据。15 对模型不能当成 15 个独立同分布的训练重抽样。冻结协议要求 panel、seed 两维分别重抽并保留配对，而且只有三个 training panel、五个 seed，任何关于训练总体的推断都很弱。测试模型还会共用路径；主要 path 区间必须先平均全部固定拟合，再抽取独立配对 path，不能把 model×path 当成独立样本。本文件只检查设计与实现约定，没有生成总体显著性结论。

## Response probes 的信息边界与索引

审核 [stage3_response_probes.py](../scripts/stage3_response_probes.py)、[stage3_response_probes.json](../configs/stage3_response_probes.json) 和复用的 [stage2_memory_diagnostics.py](../scripts/stage2_memory_diagnostics.py)。实际配置固定 context64/recent16/permutation seed920110、Gaussian pulse onset256/duration128/amplitude3、64 条 length768 路径。

程序只重生每个 panel 的 validation observations，读取既有 validation target pairs 和冻结 checkpoint；额外 controlled pulse 使用 `2718000+1000*panel` 的独立 Gaussian 流。它是预定义的模型响应探针，不能将该标签扩展成一个未见过的 replication test 或金融部署数据。

预测使用 `[t−context,t)` 的过去观测、每个目标独立重置模型状态。输出片段 `[onset,exit+64)` 的 array offset0 是绝对时点256，event/control 发出尺度完全相等；冲击最早影响 t257 的预测。entry `[0:32]` 对应绝对时点256–287；recovery64 `[128:192]` 对应384–447，退出时第一个预测仍包含冲击中的过去观测。最大 context64 在 t448 后仅含重新相同的 event/control 观测，因而发出的尺度重新相等。

梯度/置换复用原 `diagnose_model` 函数：first256 saved validation target pairs，context16 与 context64 的最后16观测一致；把完整较旧48观测块在这256个目标之间置换，recent16 保持；profile 的 lag1 为最新观测，older fraction 是平均绝对梯度曲线的 older17–64 占比。目标可以来自相同路径且重叠，不是256条独立序列。梯度是局部敏感性，置换可能制造分布外上下文，均不构成长记忆的因果证明。

`diagnose_model` 将模型切到 eval、在内存中关闭参数梯度，但不写 checkpoint，也无 optimizer update；后续 pulse 仅做推断。脚本 import `predict_scales` 不会执行 evaluator 的 `main()`；没有新 replication test 的加载路径。

初次审查提出两个来源 guard 补充：完整 saved pair arrays 应与 panel manifest 的 pair hashes 比对；旧 `stage2_memory_diagnostics.py` 应核对已有 Stage 2 memory source hash。根 agent 在首次 probes 执行前提交 `e197a66` 补齐这些核对，同时增加硬编码 config 对应的 assertions。再次只读代码已确认此修订只增加 guards，没有改变上述计算定义。修订后 probes source SHA256 为 `13194430189c5ab27d569e2178c2d82cc778486a3e371dfb7ffd459d743f56d1`。

## 全部训练模型的独立数值核查

对三个 panel 的全部30个模型逐一核查（panel1 先完成，panel2/3 后追加）：

- 从冻结配置重生完整训练与验证数组及8192个 validation path/target pairs，各 array hash 与 manifest 一致；`input_protocol`/`frozen_protocol`/checkpoint 的协议引用一致，全部 checkpoint 位于各自独立 panel 目录。三个 training arrays 与三个 validation arrays 分别有三个不同 SHA256。
- 对每个 seed 独立重生所有执行 epoch 的 NumPy sampler、context schedule 和 byte-level digest，与两架构各自 history 的**全部** epoch digest 一致，超过共同前缀的单架构 epoch 也核查。
- selected epoch 与全部保存的 history 中 validation 最小值一致；独立逐 epoch 重算 `min_delta` patience counter 和最早允许停止点，与实际停止一致。
- 重新加载30个 selected checkpoint，重新计算两个 contexts 的8192个 validation target losses。30个模型重算的最大绝对误差均为 `0`，不是只核对 manifest 内自报结果。全部模型均按 validation patience 停止，没有触及 max60。
- 对 panel1 GRU/TCN seed4101 额外用 controlled pulse 的前4条路径比对全长预测与 `[256:448]` chunk；t256 因果前缀、t448后的 finite-context equality、raw context 正确切片与复用的 gradient/permutation 计算全部通过。该小数组核查不读取 replication test。

| panel | model | seed | 执行 epochs | selected epoch | selected NLL 重算误差 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | GRU | 4101 | 34 | 26 | 0 |
| 1 | TCN | 4101 | 22 | 14 | 0 |
| 1 | GRU | 4102 | 28 | 20 | 0 |
| 1 | TCN | 4102 | 27 | 19 | 0 |
| 1 | GRU | 4103 | 21 | 13 | 0 |
| 1 | TCN | 4103 | 21 | 13 | 0 |
| 1 | GRU | 4104 | 39 | 31 | 0 |
| 1 | TCN | 4104 | 26 | 18 | 0 |
| 1 | GRU | 4105 | 31 | 23 | 0 |
| 1 | TCN | 4105 | 31 | 23 | 0 |
| 2 | GRU | 4101 | 30 | 27 | 0 |
| 2 | TCN | 4101 | 36 | 28 | 0 |
| 2 | GRU | 4102 | 40 | 32 | 0 |
| 2 | TCN | 4102 | 31 | 23 | 0 |
| 2 | GRU | 4103 | 29 | 21 | 0 |
| 2 | TCN | 4103 | 42 | 34 | 0 |
| 2 | GRU | 4104 | 20 | 8 | 0 |
| 2 | TCN | 4104 | 37 | 29 | 0 |
| 2 | GRU | 4105 | 29 | 21 | 0 |
| 2 | TCN | 4105 | 29 | 21 | 0 |
| 3 | GRU | 4101 | 52 | 44 | 0 |
| 3 | TCN | 4101 | 39 | 31 | 0 |
| 3 | GRU | 4102 | 40 | 32 | 0 |
| 3 | TCN | 4102 | 41 | 33 | 0 |
| 3 | GRU | 4103 | 36 | 28 | 0 |
| 3 | TCN | 4103 | 36 | 28 | 0 |
| 3 | GRU | 4104 | 35 | 27 | 0 |
| 3 | TCN | 4104 | 24 | 16 | 0 |
| 3 | GRU | 4105 | 24 | 16 | 0 |
| 3 | TCN | 4105 | 24 | 16 | 0 |

完整 training master SHA256 为 `bfc3c80c60d7ef3d7f48508c37615a1acf8531db94f85e1560664ef280d5ed05`。独立审核重生的数据也确认没有读取 Stage 3 test 流。

## 全部30个 probes 的独立复算

在 probes writer 完成后，核对 complete/model_rows30、全部输入 config/source/training hashes、两个输出 CSV hashes、每个模型的 checkpoint 引用与 validation NLL。然后重新生成每 panel 的256个原目标 raw contexts 和64条完整 controlled-pulse 路径，核对 validation/pulse 数组来源。

本审核没有调用 probes `main()` 或复用 `diagnose_model` 的 feature 聚合，而直接对 raw contexts 用 `torch.autograd.grad` 重算 lag profile，再按定义重算 old-block permutation、context16/64 log/scale 差异与全部 moments/quantiles/clamp fractions；全部30模型和64 lags均核查。Pulse 则用已经过因果/chunk 独立检查的 `predict_scales` 在全部64条路径重新预测，再直接重算 entry/recovery 两context的均值和 signed/absolute contrast，共八个 pulse features/model。

| 复算范围 | 最大绝对误差 |
| --- | ---: |
| 30 models × 64 lags 的 gradient profile | 9.98e−17 |
| 全部30模型的依赖/置换/context features | 2.22e−16 |
| 全部30模型、64条 pulse path 的八个 response features | 9.71e−17 |

用于完整复算的 Torch 线程数和参数冻结顺序与冻结程序一致。初次少线程小检查产生约 `9.31e−10` 的局部 floating-point 差，随后在原四线程设置下完成上述全量比较；没有把线程导致的数值舍入差判作模型或索引错误，也没有改动任何输出。Probes manifest SHA256 为 `3fcf61e721c4fc53f1c02f3a136af0b5d1ab7ceca9f1f5e906de97fe5e913dd6`。

## 分析程序的无真实测试数据审核

额外只读审核 [analyze_stage3.py](../scripts/analyze_stage3.py)，其源码 SHA256 为 `c755f579e9d49f671d51df0f346e43640f7c5b774998394b6d0a69ece9e64a11`。没有加载 Stage 3 evaluation manifest/CSV；仅运行 `--self-check` 与直接传入人为构造数据的 pure `analyze()` 函数，没有调用写出主程序。

Grid 以显式 panel/seed/path Cartesian index 重建，拒绝重复、缺失与 off-grid identities。GRU−TCN 在相同 panel/seed/path 上先做差；primary bootstrap 先平均15个固定拟合，再共同重抽 path。Crossed sensitivity 的 panel、seed、path advanced indexing 分别 broadcast，**同一个 seed draw 用于每个抽到的 panel**，没有错误地为不同 panel 独立抽 seed。Component seed standard deviation 在先平均每条 path 的固定 fit 值后跨五个 seed 计算；不是把 path 当 seed。两种分解路线与 third-order identity 均逐 tensor 检查，entry D32/D128 相同由因果前缀核对。

独立端到端 toy 使用 `3 panels × 5 seeds × 7 paths × 5 metrics`、全部四 case/four phase/six components，随机打乱行，再构造80次 resamples。用另一段标量循环重算 bootstrap，而非只运行 source 自带检查。两 primary phases 的 conditional 97.5% endpoints 最大误差 `1.67e−16`，crossed endpoints 最大误差 `2.22e−16`；primary mean方向、leave-one-panel-out 均值与每 panel 两架构 current/archive seed std 均一致。独立 toy 的 decomposition 最大误差 `8.88e−16`，diagonal和共享 entry error 均为0。

这确认了抽样/配对/grid实现，不构成实际 Stage 3 科学结论。Actual evaluation/analysis 的来源与结果还应由其自身 provenance verifier 和单独 readout 核对。本审核没有读这些新测试结果，也不会据模型的 validation 排名修改分析或筛掉任何拟合。
