# 16批独立路径验证：HPC执行记录

2026-10-19。遵循[冻结方案](long_window_validation_protocol.md)，所有12个旧模型均保留；本轮没有训练、模型选择、报告或PPT工作。评估、bootstrap及完整核验通过Slurm计算节点执行，日志保留在服务器。

## 固定计算与保存内容

- `evaluate_long_window_validation.py`：16批各64路径，5情景、2窗口、12模型，共1920组。每组保存6种方法、16个时点、64个路径的风险/半宽/覆盖率及精确分解；D保存64份独立重抽的逐端点与逐阶段均值，X使用同批全部63个其他路径。
- `analyze_long_window_validation.py`：固定的12000次整批bootstrap和4000次批内独立replica bootstrap；先池化再取绝对值，保留H1/H2不同的评价目标，以及H3的Student-t敏感性检查。失败或精度不足不追加数据。
- `verify_long_window_validation.py`：独立排序与SciPy计算，检查来源、随机种子、保存结果和统计规则。核验覆盖范围与计算数量在核验JSON明确披露，不能把抽样核验表述成所有二级银行的全面重放。

独立重抽噪声按`[replica,path,time]`生成，同批共享replica编号，不同批独立。P的eta及经验CDF偏离量定义为0；这些是总体参照量。协议中的oracle下界只适用于风险分量，不适用于覆盖率等异量纲指标。精度门槛的分母为`max(abs(对应G),1e-12)`。这些执行解释在首次新数据生成前记录。

输出目录为`outputs/long_window_validation`、`outputs/long_window_validation_analysis`及`outputs/long_window_validation_verification.json`。原始创新、重抽银行、尺度缓存和逐路径数组完整保留HPC；代码、索引、哈希manifest、统计摘要和核验记录同步Git。大体积原始数组不默认加入Git，恢复完整重放时从HPC复制相应目录，并按manifest核对SHA256。

## 提交与查看

先在本地审查、运行小样例检查并提交全部源码；再以Git bundle同步到HPC的干净工作树。正式执行前脚本检查协议、源码和模型哈希，拒绝覆盖已有输出。若执行失败，先保留失败目录和日志，查明原因；同种子重算属于恢复，不允许换种子、增样本或挑选有利结果。

```bash
ssh dsaa2012-hpc
cd "$HOME/projects/DL-project"
mkdir -p logs
/opt/slurm/bin/sbatch hpc/long_window_validation.sbatch
```

使用普通CPU分区`i64m512u`及同名QoS，4 CPU、16 GB内存、最多3小时。已确认该账户的Slurm关联含此QoS；这不代表课程账户余额已核实或普通队列免费。没有使用高优先级队列，也没有把正式计算拆成debug作业。队列依据：[学校官方说明](https://docs.hpc.hkust-gz.edu.cn/docs/hpc12/slurm/queue/)。

状态依次写入`logs/long-validation-JOBID.status`：SELF_CHECKS、EVALUATION、ANALYSIS、INDEPENDENT_VERIFICATION、COMPLETE/FAILED。stdout/stderr在同名`.out/.err`，只查询简短状态及必要的错误尾部；不持续读取长日志。任务完成后保存`sacct`状态、退出码、实际资源/用时，并核验输出再解释结果。

本文件初次创建时尚未提交作业；实际提交与完成记录以之后的运行记录和输出manifest为准。
