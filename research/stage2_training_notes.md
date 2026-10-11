# Stage 2 matched neural training notes

日期：2026-10-11。冻结协议 `stage2-v1.1` 的全部 10 个拟合已完成；这份记录只使用训练、验证、checkpoint 和人工因果干预输入，未读取或评估测试数据。

## Frozen run and reproducibility

执行命令：

```sh
.venv/bin/python scripts/train_stage2.py --protocol configs/stage2_protocol.json
```

终端 exit code 0。`outputs/stage2_training/manifest.json` 为 `status=complete`、`completed_models=10`、`matching_minibatches_verified=true`。总 wall time 为 301.721 秒。环境为 Python 3.12.14、NumPy 2.5.4、PyTorch 2.14.1，CPU、4 个 Torch threads、deterministic algorithms。未改变冻结配置、未重训、未扩大 60 epoch 上限。

```text
protocol_sha256: aa2bda29ba79b13df1b4ccba614bea1994670adf09133e359471d9d8dc2b9ef6
train_stage2.py: d93d1ecbb6ce30ccba88c2566be2632d769db6c2968717c3f6c36f24fbc49850
stage2_models.py: c93e127dd43a3f9f70fed3fbd4c63bed8fa0c74b261c33dbd0a327889bd10f2e
training observations: 0ad0d89fdbbe923755f49c6820734ea8bdff83b8aef205816bd202c72a6b4b98
validation observations: fa7bc688adadca0408c189236f6c32946c66a45c3b1b3db5483571b04ece6b6a
validation pair paths: cbb8977f22e72ad7270a150bf9dfa713ec52edcb34380209bff1600d213913a8
validation pair targets: 46b1dd3ee53ff1750ef3154d7bf0d47dd721f6333f16cd479750511594be852d
```

已对当前协议、保存的 `frozen_protocol.json`、两个源文件、所有 checkpoint 的 SHA256 独立检查；协议字节完全一致，源码与保存的 manifest 一致。以冻结 seed 重新生成训练和验证数据及验证目标对，以上数据哈希全部重现。

## Matching and causal boundary

训练数据为 seed 314159 的 256 × 1024 条观测；验证数据为独立 seed 314160 的 64 × 1024 条观测。验证目标对使用 seed 314161，共 8192 次共同抽样，两个上下文都使用相同 `(path,t)` 与 `y_t`。目标时间均为 64–1023；输入严格是 `y[t-context:t]`，不包含 `y_t`。生成器仅返回 observations，不向神经模型提供真实尺度、事件标签或分布 oracle。

两个架构都使用逐时点的 `(raw/3, log1p(abs(raw)))` 特征、固定零均值 Gaussian scale head、log σ 截断 `[-3,3]`。没有额外动态归一化。GRU hidden 24，有 2041 个参数；TCN channels 12，有 1897 个参数（比 GRU 少 144，即 7.06%）。TCN 的五层卷积 kernel 为 `[4,3,3,3,3]`、dilation 为 `[1,2,4,8,16]`，只使用左侧 padding，结构感受野为 `1+3+2*(2+4+8+16)=64`。

每个架构与 seed 都只有一个共同权重 checkpoint，训练中混合上下文 16/64；每 epoch 各 16 个 batch，共 32 个 batch，batch size 128。Adam lr=.002，gradient norm clip=5。各 seed 的 NumPy minibatch sampler 被独立重置为同一 model seed，因此不受架构初始化或 Torch RNG 消耗量影响。

独立审查重新构造了每个模型全部已执行 epoch 的 batch schedule、path indices 和 target indices，完整重现每 epoch 的 `minibatch_epoch_sha256`，且每 epoch 上下文分配均为 16/16。GRU/TCN 每 seed 的共同已执行前缀分别有 36、20、26、32、28 个 epoch，所有哈希相同。冻结 early stopping 允许两架构结束 epoch 不同，因此 selected checkpoint 的总优化步数不强行相等；这里的 matching 指共同数据、目标、输入、采样前缀和选择规则。

## Validation selection and convergence

每 epoch 在共同验证目标上分别计算 context16/64 NLL，checkpoint 只按两者算术平均的绝对最小值选择。NLL 为 `log σ + y²/(2σ²)`，省略所有模型共同的 `0.5*log(2π)` 常数；这些数值不是真实市场单位下的误差或覆盖率。单位尺度基准的同一验证 NLL 为 1.473557472.

`min_delta=1e-4` 仅控制 patience reference 更新，不会过滤 checkpoint 的微小改善。patience 计数从第 1 epoch 开始，停止只在 epoch≥20 且 stale≥8 后发生，最大 60 epoch。因此最小验证 checkpoint 可以早于第 20 epoch（如 GRU 4102 第 17 轮、TCN 4102 第 11 轮），这与冻结规则一致。

| Model | Seed | Parameters | Executed epochs | Selected epoch | NLL ctx16 | NLL ctx64 | Mean NLL | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GRU | 4101 | 2041 | 36 | 28 | 0.867247112 | 0.863119517 | 0.865183314 | 10.92 |
| TCN | 4101 | 1897 | 38 | 30 | 0.873458080 | 0.872072861 | 0.872765470 | 67.94 |
| GRU | 4102 | 2041 | 25 | 17 | 0.866675254 | 0.862916980 | 0.864796117 | 9.22 |
| TCN | 4102 | 1897 | 20 | 11 | 0.876339614 | 0.872093935 | 0.874216774 | 34.58 |
| GRU | 4103 | 2041 | 37 | 29 | 0.865631219 | 0.863142733 | 0.864386976 | 11.19 |
| TCN | 4103 | 1897 | 26 | 18 | 0.871782314 | 0.866802238 | 0.869292276 | 44.41 |
| GRU | 4104 | 2041 | 38 | 30 | 0.866844971 | 0.862280644 | 0.864562808 | 12.10 |
| TCN | 4104 | 1897 | 32 | 24 | 0.867633812 | 0.863979626 | 0.865806719 | 54.92 |
| GRU | 4105 | 2041 | 31 | 23 | 0.868132524 | 0.863293421 | 0.865712972 | 9.57 |
| TCN | 4105 | 1897 | 28 | 20 | 0.866726253 | 0.861750647 | 0.864238450 | 46.79 |

全部 10 个模型的停止原因为 `validation_patience`，没有触及第 60 epoch，也没有 best epoch 位于第 60 轮的截断案例。所有 selected checkpoint 都比第 1 轮有明显验证改善；末轮 NLL 有波动，因此保存较早最优验证 checkpoint 是预定行为。达到固定 patience 表示按既定预算停止，不能证明达到全局最优或排除更长训练会改善。

五个 GRU 的最小 mean validation NLL 为 0.864387–0.865713，均值 0.864928；TCN 为 0.864238–0.874217，均值 0.869264。TCN 的 seed 波动较大，4105 的值低于相配 GRU。以上是训练/验证收敛描述，不能据此声称某架构在新测试事件、真实金融数据或初始化总体上占优。

## Audits and interpretation limits

训练脚本在初始化与 selected checkpoint 都进行了人工 suffix intervention 与 prefix endpoint 检查：改变位置 37 及以后的输入不影响此前预测，完整序列的位置 36 输出与只给长度 37 前缀的 endpoint 输出一致。已保存的 selected checkpoint suffix 最大误差均为 0，prefix endpoint 最大误差为 8.94e-8，低于 1e-6 tolerance。人工干预独立于课程数据和模拟测试集，不是预测性能测试。

验证的 8192 对是 64 条验证轨迹内共同抽样，并非 8192 条独立时间序列。五个 model seed 使用同一固定 train/validation 数据，主要反映初始化与优化随机性；没有训练数据重抽样。架构参数量接近但不完全相等，计算成本、padding 和训练过程也不同，所以随后 architecture contrast 应表述为这两个冻结拟合程序之间的差异，而不能泛化为 architecture alone 的普遍因果效应。

结构 RF64 或给出 context64 不等于模型真的学习了 64 日记忆；有效记忆与归档分母机制需要由后续预先指定的 prefix/context 或 archive intervention 诊断区分。训练 Gaussian scale head 在 heavy-tail/SV 测试下存在预先规定的模型失配，这属于泛化实验范围，不能事后为改善测试结果更改模型。

测试阶段只使用这些冻结 checkpoint；不根据测试图或 primary contrast 返回调参。本 notes 不包含测试结论。primary inference 按协议条件于这五对固定模型，对独立测试 path 配对 bootstrap；crossed seed/path bootstrap 只作 sensitivity，并另报告五个 seed 的效果与 leave-one-seed-out。

## Independent second audit

另一个只读审查进程独立重载了全部 10 个 checkpoint，重新计算冻结 context16/64 validation NLL，与各自 selected history row 的误差均为 0，并确认 selected epoch 是已执行历史中的绝对最低平均 NLL。逐 epoch 重建 patience counter 与停止时机，全部符合冻结协议。

第二次 synthetic intervention 扩展到序列长度 65/96/129、cut 1/16/37/64；全部模型的 suffix 影响误差为 0，prefix endpoint 最大误差为 1.19e-7，低于 1e-6 tolerance。这次检查只使用训练/验证及人工输入，未读取测试数据。
