# 本轮怎么理解、核查和复现

本轮只改变一个部分：**从历史scores中取哪个分位数秩**。所有30个神经网络权重、输入路径、current/archive上下文、两阶段位置均沿用原Stage3。没有调用HPC重新训练；本地CPU约7.4分钟完成evaluation，约2秒完成analysis。

## 建议的20分钟阅读顺序

1. 用5分钟读[结果前两节](rank_sensitivity_results.md)，先回答“原来的32和128窗口为什么并非恰好相同的90%秩水平”。
2. 用5分钟对照两张图，区分蓝色的固定模型路径区间和灰色的训练面板/共享seed敏感性区间。灰色含0意味着不能把平均点估计直接当成普遍架构排序。
3. 用5分钟理解随机秩期望：先对每个候选区间评分，再平均分数。用观测1、半宽0.5与2的例子算出6.1与3.1为什么不同。
4. 用5分钟读两个“敏感细节”：TCN接近0的entry均值变号，t5 recovery变化在固定模型区间更明显。报告中必须保留它们。

你应能用自己的话解释这五个问题：

- 为什么`ceil((W+1)×.9)`不一定对应精确的90%理想覆盖？
- 为什么W32的候选秩29/30需要0.30/0.70，而W128的116/117需要0.90/0.10？
- 为什么对随机区间“先平均宽度后评分”会改变研究对象？
- 为什么相同的原规则点估计仍可出现略不同的bootstrap区间端点？
- 为什么本轮主结论保留方向，仍不能说“秩规则不重要”或“GRU普遍优于TCN”？

## 已保存输出是什么

| 文件 | 用途 |
|---|---|
| `outputs/rank_sensitivity/rank_candidate_phase_metrics.npz` | 所有模型/family/world/current/archive/window/候选rank/phase/path/metric的原始阶段均值，形状30×2×2×2×2×2×2×2×128×5；轴标签和模型身份同时保存 |
| `outputs/rank_sensitivity/paired_component_effects.csv.gz` | 两规则、两phase、两family、全部模型的六个分量，184,320行；total即J |
| `outputs/rank_sensitivity/cell_performance_by_seed.csv` | 八格、两world、两规则及全部模型的3840个绝对表现均值 |
| `outputs/rank_sensitivity/negative_controls.csv` | 两候选rank的960条未来前缀/有限记忆核验 |
| `outputs/rank_sensitivity/ceiling_reproduction.json` | 原规则与冻结Stage3的完整再现误差 |
| `outputs/rank_sensitivity_analysis/summary.csv` | 所有规则及变化、架构及差异、分量、指标的1080行结果 |
| `outputs/rank_sensitivity_analysis/panel_summary.csv` | 按训练数据面板分组的结果 |
| `outputs/rank_sensitivity_analysis/individual_fits.csv` | 所有15个拟合或架构配对的结果；没有只展示好seed |
| `outputs/rank_sensitivity_analysis/leave_one_panel_out.csv` | 去掉每个训练数据面板后的描述性均值 |
| 两目录的`manifest.json` | 输入/输出/source哈希、配置、检查与耗时记录 |

随机秩的五个指标都是各候选区间对应指标的期望。尤其oracle coverage deficit与excess width中的`max(...,0)`发生在平均之前；它们不是把随机化后的平均概率/宽度放入正部函数。这一固定定义写在协议中。

## 只核查实现，不改写结果

在仓库根目录运行以下两条。它们只创建很小的synthetic arrays，既不重新访问真实评估路径，也不写入结果目录：

```sh
.venv/bin/python scripts/run_rank_sensitivity.py --self-check
.venv/bin/python scripts/analyze_rank_sensitivity.py --self-check
```

完整evaluation和analysis已成功完成。默认命令发现结果目录已存在会拒绝覆盖，这是有意的。需要从头复现时，应在冻结commit的独立checkout中运行，并使用一致的Python依赖；不要删除或覆盖当前证据。冻结commit尚未包含本轮输出，因此独立checkout可从保存的Stage3模型/路径定义重建本轮全部结果。两条完整命令按顺序执行：

```sh
.venv/bin/python scripts/run_rank_sensitivity.py
.venv/bin/python scripts/analyze_rank_sensitivity.py
```

独立checkout需要先配置与原本地`.venv`一致的依赖，或使用原环境Python的绝对路径。脚本会核对自身及协议的git冻结状态、原训练/评估来源、checkpoint哈希、原测试数组哈希，最后重新验证输入没有变化。不同硬件/软件产生的浮点差异若超过既定容差，应记录原因并停下检查，不能靠放宽容差让结果通过。

## 本次来源记录

| 项目 | 值 |
|---|---|
| freeze_commit | `df8db3ea91eed4f1ca90c5cfadc2f68d45e923ac` |
| protocol_sha256 | `18c108f62f23c266c03a5c325c1f79db6dad42466766a9ad3dbd0fd7ea5870bc` |
| evaluation_source_sha256 | `7ac20014316c88b540ac707b64d8a6881b0b1aef394d0bb681d84973f5c73543` |
| analysis_source_sha256 | `c6c21adc4d4e2d64407d3dd2ade4deae1ced0ae2adf7845c66b98e94dd44f4ee` |
| evaluation_manifest_sha256 | `f508b7984a2b6ae08fc7803edda37d302378e0e6750a3c9482a86f5bb0d26b21` |
| analysis_manifest_sha256 | `8723e9a21fe9f58c99381ac5e230e24661d89e157a8ff83259070f5e336201b6` |

这些哈希标识本次已经完成的数值版本。新的中文解释文件不会反向改变原Stage3或本轮数值manifests。原Stage3主要结论继续保留，本轮应在其后以“事后秩规则敏感性”标注。
