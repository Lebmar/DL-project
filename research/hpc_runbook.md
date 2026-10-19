# HPC 二期：项目环境与运行入口

更新于2026-10-19。课程账户已成功登录学校 HPC 二期，使用 Slurm 调度。原先缺少服务器入口的问题已经解决。连接使用学校官方主机 hpc2login.hpc.hkust-gz.edu.cn；本机已配置专用 SSH 别名 dsaa2012-hpc，私钥只在本机 ~/.ssh/ 中，仓库不保存密码、私钥或课程账户名。

## 当前默认：HPC优先与低交互运行

2026-10-19用户明确要求后续计算尽可能放到HPC，并接受排队和较慢执行。这一偏好替代此前本地CPU优先的安排：

- 训练、批量模拟/评估、bootstrap和完整结果重放核验在Slurm计算节点运行。本地只做编辑、轻量静态/小样例检查及必要的结果浏览。
- 不因排队而自动回退本地大量计算。登录节点仅传输、安装、检查文件和提交作业；正式研究不拆成反复debug作业。
- 保持各研究冻结的CPU/GPU与线程设定；HPC不等于必须用GPU。本轮长窗口验证仍为CPU、4线程。
- 先提交源码，再以Git bundle同步干净快照及已保存模型；检查远端工作树，禁止覆盖未提交修改。首次新数据生成前冻结执行器。
- 完整stdout/stderr写入logs，输出按实验/作业独立保存；只回传作业编号、状态变化、失败摘要和最终结果，不逐epoch输出。
- 默认提交后查一次状态，之后在后续研究节点检查；需要等待时低频检查。不为等待持续占用对话，也不擅自创建定时自动化。队列无变化不反复叙述。
- 选择账户可用队列和QoS，设定时间/CPU/内存上限；保留作业资源、环境版本与来源。课程余额未核实，不把普通队列称为免费。

计算位置本身不决定对话token消耗；减少重复推理、长日志回传和频繁查询才是降低交互开销的直接措施。

2026-10-19连接检查：SSH成功、项目Python入口存在、账户作业列表为空、CPU/GPU分区可见。分区可见不等于已核实每个分区的账户准入或剩余额度。本次仅检查与同步，不提交新研究作业。

## 连接与目录

在这台 Mac 上：

```bash
ssh dsaa2012-hpc
```

本机必须保持校园网或学校 VPN 可达。该别名配置属于本机，并不随 Git 仓库传播。其他电脑应按学校说明使用自己的已授权账户设置连接。

HPC 上的项目目录为 $HOME/projects/DL-project；独立 Python 环境为 $HOME/.conda/envs/dl-project。使用 Git bundle 从本地已提交快照复制，保留真实提交历史。远端 origin 当前是上传的本地 bundle，不配置 GitHub 写入凭据。后续源码更新由本地审查、提交后再同步。

非交互 SSH 会话可能没有 Slurm 的 PATH；使用 /opt/slurm/bin/ 下的命令，避免将“找不到 squeue”误判为未安装调度器。

## 环境设置

学校共享基础环境实测为 Python 3.11.9、CPU PyTorch 2.3.0；另有 PyTorch 2.0.1+cu118 的旧 GPU 环境。项目使用独立前缀，避免修改共享软件。

本次选择 Python 3.12.14 与 PyTorch 2.14.1+cu126，其余项目库按照 requirements-lock.txt 安装。CUDA 本地版本后缀与 Linux 的额外 NVIDIA 依赖会单独记录，不能把 macOS 的锁文件当作完全相同的二进制环境。

```bash
cd "$HOME/projects/DL-project"
bash hpc/bootstrap_env.sh
```

脚本仅在学校已核实的 HPC 二期目录结构下工作。它使用 conda-forge 创建用户环境、PyTorch 官方 CUDA12.6 源安装 torch，再安装项目锁定依赖，并执行 pip check。无需改变系统驱动、共享 Conda 环境或全局软件源。

第一次 GPU 探针作业13162894已在免费debug分区完成，Slurm记录 COMPLETED/0:0，实际运行3秒；分配NVIDIA A40，显存49140MiB，驱动575.57.08。旧共享GPU环境完成CUDA矩阵运算。随后项目专用环境的GRU/TCN验证与保存结果核验也已通过，见下表。

驱动575处于CUDA12兼容范围；PyPI默认的CUDA13包不适合直接套用，因此明确选择cu126构建。驱动兼容表不能代替真实运行检查。

## 项目运行验证

先在仓库根目录创建日志目录，再提交两个短作业：

```bash
cd "$HOME/projects/DL-project"
mkdir -p logs
/opt/slurm/bin/sbatch hpc/smoke_gpu.sbatch
/opt/slurm/bin/sbatch hpc/verify_saved.sbatch
```

- GPU smoke：debug队列，1张GPU、4核、8GB内存、最多5分钟。只用专门生成的少量随机数据，检查GRU/TCN×context16/64的forward、backward、Adam更新、同设备重复性及同权重CPU/GPU预测误差；不读取研究测试数据。
- CPU saved audit：debug队列，4核、8GB内存、最多10分钟。先做CPU smoke，再核查三个已完成阶段的保存结果。核查保存结果不等于从头重训。

结果文件写入 outputs/hpc_readiness/，带Slurm作业编号并拒绝覆盖；stdout/stderr写入已忽略的logs/。工具返回作业编号只说明提交成功，必须同时检查最终状态、退出码和JSON报告。

```bash
/opt/slurm/bin/squeue -u "$USER"
/opt/slurm/bin/sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocTRES -P
```

debug是学校明确提供的免费调试分区，最长30分钟；本次只用短时环境验证。正式长实验需按其用途选择适合的普通队列并记录资源与耗时，不把正式计算拆成无限调试作业。登录节点只用于安装、传输、查看和提交；训练与大规模评估走调度器。

## 本次实际完成的检查与费用规则

| 作业 | 用途 | Slurm最终状态 | 分配资源与实际时间 |
|---|---|---|---|
| 13162894 | 初始CUDA矩阵探针 | COMPLETED，0:0 | debug，1A40+2CPU，3秒 |
| 13163212 | 项目环境GPU smoke | COMPLETED，0:0 | debug，1A40+4CPU，40秒 |
| 13163213 | CPU smoke及保存结果核查 | COMPLETED，0:0 | debug，4CPU，14秒 |

GPU检查完成全部GRU/TCN×context16/64组合的前向、反向、Adam更新和同设备固定种子精确重复。同权重CPU/GPU尺度预测最大绝对差为3.58×10⁻⁷，低于预设容差。CPU检查通过第一、二、三阶段52+130+259=441项保存结果验证。Slurm总时长含解释器启动等开销，不能作为纯模型benchmark。完整[核验manifest](../outputs/hpc_readiness/manifest.json)、[作业记录](../outputs/hpc_readiness/slurm_jobs.psv)与两份smoke JSON已保存；完整安装版本在[hpc/requirements-linux-cu126.txt](../hpc/requirements-linux-cu126.txt)。

学校[账单说明](https://docs.hpc.hkust-gz.edu.cn/docs/charges/billing-payment/)明确：目前学生无需个人缴费，由所属教师通过校内PBMS统一结算；[账号申请说明](https://docs.hpc.hkust-gz.edu.cn/docs/application/account-application/)指出费用计入导师账户。**这不等于平台无限免费**；课程账号具体预算和额度仍未核实。

以下为2026-10-18历史记录，不作为当前价格或执行偏好：当时三个作业均为官方免费debug，没有提交普通付费计算，当时新的FX正式实验继续使用本地CPU。普通低优先级A40定价为0.50元/卡/小时，CPU为0.02元/核/小时，按实际运行秒数计；例如1A40+4CPU约0.58元/小时，最终以学校账单为准。Slurm输出的`billing=4`是调度资源权重，不是已扣4元。存储另有免费额度及超额收费规则，不能仅从计算作业记录推断账户完整账单。

## 与已冻结科学实验的关系

Stage2/Stage3的正式结果仍来自原本地CPU环境，原脚本、协议、checkpoint和输出不因HPC配置而修改。GPU smoke验证软件可运行，不证明GPU重训可逐位复现CPU结果。新GPU训练须另存完整协议、依赖版本、设备、随机种子和数值容差，并用独立输出目录。

当前冻结训练脚本明确使用CPU；没有为了使用GPU就静默改变它们的设备或协议。HPC可用于独立CPU运行、新实验的GPU实现或更大重复，但每项正式结果都须明确其运行环境与来源。

## 官方依据

- [HPC二期登录与连接](https://docs.hpc.hkust-gz.edu.cn/docs/hpc12/connection/)
- [HPC二期队列、资源与免费debug限制](https://docs.hpc.hkust-gz.edu.cn/docs/hpc12/slurm/queue/)
- [一期与二期资源定价](https://docs.hpc.hkust-gz.edu.cn/docs/charges/prices-phaseI%26II/)
- [Slurm作业提交与管理](https://docs.hpc.hkust-gz.edu.cn/docs/hpc12/slurm/job2/)
- [PyTorch官方安装说明](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA驱动兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

本次项目环境检查已经完成；后续新计算需另存作业状态和来源，不能将这次smoke通过当作所有未来实验自动通过。
