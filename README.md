# Neural Memory and Interval Calibration

DSAA 2012 Deep Learning 课程项目。问题是：**神经预测器的尺度响应如何改变在线校准在波动冲击进入、退出时的效果？**只使用公开 ECB 汇率与自生成模拟数据。

已完成统一训练的 GRU/TCN 五种子配对比较与30种独立测试情景。主比较在给定五组拟合时呈现小幅阶段差异，但跨初始化敏感性区间包含0，种子点估计会反向，**尚不能声称稳定的架构优势或 A+ 级别原创贡献**。第一阶段公开汇率上 GRU 也没有清楚胜过 EWMA。完整结果保留，不预设长上下文有害或神经模型一定占优。

## 从这里开始

- [一页 proposal 草稿](output/pdf/project_proposal.pdf) · [LaTeX 源码与官方模板](proposal/)
- [第二阶段实际结论与完整网格](research/stage2_results.md)
- [当前尺度与历史档案的交叉机制诊断](research/stage2_archive_mechanism.md)
- [评分曲率的基础解释与oracle诊断](research/score_geometry.md) · [直接先例与贡献审查](research/contribution_review.md)
- [中文项目解释：如何讲清楚设计与统计量](research/project_explainer_zh.md)
- [第二阶段冻结协议](research/stage2_protocol.md) · [训练核查](research/stage2_training_notes.md) · [验证记忆诊断](research/stage2_memory_notes.md) · [复现入口](research/stage2_runbook.md)
- [第一阶段结论与 A+ 差距](research/feasibility_report.md)
- [下一阶段实验关卡与时间安排](research/next_experiments.md)
- [课程要求映射](research/course_requirements.md)
- [文献与原创性边界](research/literature_gap.md)
- [公开数据核查](research/public_data_feasibility.md) · [实验审查](research/experiment_design_audit.md)
- [作者源码强基线说明](research/strong_baseline_notes.md) · [HPC 交接](research/hpc_runbook.md)

## 已运行的实验

| 脚本 | 目的 | 输出 |
|---|---|---|
| `scripts/run_pilot.py` | 手工摘要 MLP 的配对模拟诊断 | `outputs/pilot/` |
| `scripts/run_gru_simulation.py` | 真正的 GRU，3 种子×128 共享路径，context×calibration | `outputs/gru_simulation/` |
| `scripts/run_archive_intervention.py` | 独立改变当前尺度与归档分母，诊断归因 | `outputs/archive_intervention/` |
| `scripts/run_public_fx_pilot.py` | ECB 公开数据上的从头训练 GRU 与 EWMA | `outputs/public_fx/` |
| `scripts/run_strong_calibration.py` | 固定作者源码 SF-OGD/SAOCP，复用 issued 预测 | `outputs/strong_calibration/` |
| `scripts/analyze_public_fx.py` | 配对日期块统计，保留跨资产依赖 | `outputs/public_fx/` |
| `scripts/train_stage2.py` | 同数据/特征/批次、验证早停，10个GRU/TCN拟合 | `outputs/stage2_training/` |
| `scripts/stage2_baselines.py` | 训练集拟合GARCH、固定EWMA与校准参数 | `outputs/stage2_controls/` |
| `scripts/stage2_memory_diagnostics.py` | 仅验证集梯度/旧历史置换诊断 | `outputs/stage2_memory/` |
| `scripts/evaluate_stage2.py` | 冻结30情景，配对反事实与有限记忆检查 | `outputs/stage2_evaluation/` |
| `scripts/analyze_stage2.py` | 两项主比较与全部描述性网格，不重训 | `outputs/stage2_analysis/` |
| `scripts/stage2_strong_supplement.py` | 引用组单种子ACI/SF-OGD/SAOCP在线对照 | `outputs/stage2_strong_supplement/` |
| `scripts/run_stage2_archive_diagnostic.py` | 全五种子current/archive交叉，明确探索范围 | `outputs/stage2_archive_mechanism/` |

每个实验目录包含运行设置、source/checkpoint hash 与限制说明。大逐行 CSV 以 `.csv.gz` 保存到 Git，展开后与原始 CSV 字节一致。运行脚本会生成原 CSV；分析脚本支持两种格式。

第三阶段已在训练前固定[独立训练数据重复协议](configs/stage3_replication_protocol.json)与[验证尺度响应探针](configs/stage3_response_probes.json)：3批新的训练/验证数据，每批5对GRU/TCN种子，共30个新拟合。新测试只做已明确假设的复制，不能把旧结果变成新方法的盲测证据。[第三阶段执行与统计说明](research/stage3_runbook.md)，完成后另行发布结果。

![第二阶段预设比较](outputs/stage2_analysis/primary_architecture_contrast.png)

![ECB 公开实验](outputs/public_fx/public_fx_comparison.png)

## 本地复现

本次环境为 Python 3.12.14、Torch 2.14.1；完整版本见 `requirements-lock.txt`。第一阶段训练每个脚本约20秒；第二阶段10个模型训练约302秒，主评估约1032秒，强控制器补充约444秒，均在本地CPU完成。第二阶段从头运行顺序与已有输出备份方式见 [复现入口](research/stage2_runbook.md)。

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
.venv/bin/python scripts/run_pilot.py
.venv/bin/python scripts/run_gru_simulation.py
.venv/bin/python scripts/run_archive_intervention.py
.venv/bin/python scripts/run_public_fx_pilot.py
.venv/bin/python scripts/run_strong_calibration.py
.venv/bin/python scripts/analyze_public_fx.py
.venv/bin/python scripts/verify_artifacts.py
.venv/bin/python scripts/verify_stage2_artifacts.py
```

不同硬件/库版本可能产生小的数值差异；冻结数据 archive 可避免重新下载时历史区间变动。SF-OGD/SAOCP 使用作者源码的明确适配，原始 Apache-2.0 许可与版权头保留，详见其输出目录的 `vendor/`。未改变算法的更新或专家聚合。

## 解释与使用边界

区间目标是 90% 覆盖，指标同时报告宽度和区间分数。模拟真尺度只用于评估/诊断，算法输入不含未来持续时间或事件标签。真实数据的 2021–2026 段已经用于探索，不能再称盲测。日期块区间条件于已拟合模型，不能自动证明任意漂移下有效。

本项目不读取、复制或使用微众比赛数据、派生特征、样本、结果，也不保存 HPC 凭据。ECB 数据来源、单位与许可说明见 [data/public/README.md](data/public/README.md)。学生于 2026-10-11 确认课程允许 AI 工具；学生仍承担个人作者责任。Proposal 为审核修改的草稿。
