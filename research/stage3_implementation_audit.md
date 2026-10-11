# Stage 3 训练重复与 validation response probes 的独立实现审核

本文件由不负责 Stage 3 训练/评分写出程序的审核 agent 撰写；只读代码与已经完成的训练文件，不修改 checkpoint、training manifest 或原源码，也不主动执行任何 writer。日期：2026-10-11。当前审核状态为 **代码审查与 panel1 的 10 个模型通过；其余训练完成后继续补齐**。本审核不读取新的 replication test paths 或结果。

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

## 当前已完成的独立数值核查

Panel1 的完整10模型：

- 从冻结配置重生完整训练与验证数组及8192个 validation path/target pairs，各 array hash 与 manifest 一致；`input_protocol`/`frozen_protocol`/checkpoint 的协议引用一致，全部 checkpoint 位于独立 panel1 目录。
- 对每个 seed 独立重生所有执行 epoch 的 NumPy sampler、context schedule 和 byte-level digest，与两架构各自 history 的**全部** epoch digest 一致，超过共同前缀的单架构 epoch 也核查。
- selected epoch 与全部保存的 history 中 validation 最小值一致；独立逐 epoch 重算 `min_delta` patience counter 和最早允许停止点，与实际停止一致。
- 重新加载10个 selected checkpoint，重新计算两个 contexts 的8192个 validation target losses。10个模型重算的最大绝对误差均为 `0`，不是只核对 manifest 内自报结果。
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

Panel2/3 尚在训练，本文件不提前称全部30模型通过。待 training master complete 和 probes complete 后，继续核查其来源、NLL、sampler/early-stop 与实际 probe features；已有训练写出程序由根 agent 控制，本审核不会另起一次 writer。
