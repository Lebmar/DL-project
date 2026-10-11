# 第二阶段复现入口

## 先核查已发布结果

项目内已有冻结模型和结果，通常无需重新训练：

```bash
.venv/bin/python scripts/verify_artifacts.py
.venv/bin/python scripts/verify_stage2_artifacts.py
```

第二个命令核对10个模型、冻结源码、协议、完整30情景、逐路径配对输出和有限记忆负控制。分析、验证记忆诊断和强校准补充各有自己的 `manifest.json`。

## 从头运行

建议在另一个 checkout 重建，不覆盖用于报告的原结果。脚本在发现已有 manifest 时会拒绝静默替换；有意重建时先把六个 Stage2 输出目录一起移动到该 checkout 的 `tmp/stage2_saved/`。这是备份移动，无需删除原成果。

按以下顺序运行，先训练和验证准备，再访问测试：

```bash
.venv/bin/python scripts/train_stage2.py --protocol configs/stage2_protocol.json
.venv/bin/python scripts/stage2_baselines.py --fit-protocol configs/stage2_protocol.json
.venv/bin/python scripts/stage2_memory_diagnostics.py
.venv/bin/python scripts/stage2_strong_supplement.py --prepare-only
.venv/bin/python scripts/evaluate_stage2.py
.venv/bin/python scripts/analyze_stage2.py
.venv/bin/python scripts/stage2_strong_supplement.py --evaluate
.venv/bin/python scripts/verify_stage2_artifacts.py
```

各脚本的参数默认指向本 checkout 的冻结配置和输出目录。评估/分析入口需要其准确源码已提交到Git；克隆仓库后使用发布源码即可满足，临时改代码后需先提交并重新冻结相应实验。训练与验证准备不访问测试；主评估无法用于测试选优。

本次训练耗时约302秒，主评估和作者控制器补充耗时见各 manifest，均在本地CPU运行。作者控制器按独立路径分4进程执行，原数学、顺序与暖启动不变，已用合成路径核对与串行结果逐点一致。不同硬件/运行库可有浮点差异；严格比较时使用 `requirements-lock.txt` 的版本。

## 读取表格

| 输出 | 含义 |
|---|---|
| `stage2_training/model_summary.csv` | 10个模型的训练停止与验证选择 |
| `stage2_memory/model_memory_summary.csv` | 固定验证样本的旧输入依赖诊断 |
| `stage2_evaluation/performance_by_seed.csv` | 各情景、方法、种子、阶段、世界的整体表现 |
| `stage2_evaluation/paired_interactions.csv.gz` | 逐配对路径的事件减无事件 context×window 交互 |
| `stage2_analysis/primary_contrasts.csv` | 两个主端点，97.5%条件区间与种子敏感性 |
| `stage2_analysis/secondary_interactions.csv` | 全30情景与全部指标，95%描述区间 |
| `stage2_strong_supplement/phase_summary.csv` | 引用组的单种子在线控制器对照 |

区间 score 的单位是模拟收益单位；coverage 为比例。绝对表现和交互量回答不同问题。不要对交互输出再减一次 control，也不要把多个拟合模型共享的路径当作独立样本。
