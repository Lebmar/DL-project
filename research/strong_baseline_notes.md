# 官方 SF-OGD / SAOCP 强在线校准基线先导

日期：2026-10-11。已完成可运行先导；本轮没有修改 `run_public_fx_pilot.py`，没有新增神经训练、微众数据或 HPC 访问。

## 实现与来源

作者仓库：[salesforce/online_conformal](https://github.com/salesforce/online_conformal)，对应 Bhatnagar et al., **Improved Online Conformal Prediction via Strongly Adaptive Online Learning**, ICML 2023。[论文](https://arxiv.org/abs/2302.07869)

固定 commit：`95790a9b9e4061166e6d7a8a13282e019b1a8663`。作者库同时提供 SF-OGD 与 SAOCP，README 明确允许 `model=None`、外部自定义 score 的接口。官方实现文件和 SHA-256 记录在 `outputs/strong_calibration/run_manifest.json`。

本项目保留作者 Apache-2.0 LICENSE、原始版权头和源码快照；核查完整 pinned repo 未发现 NOTICE 文件。源码快照位于 `outputs/strong_calibration/vendor/online_conformal/`。我创建的临时嵌套 `.git` 已删除，固定版本记录在 `PINNED_COMMIT.txt`。

**这是明确的源码适配，不是完整官方 package 安装。** 官方包依赖 Merlion forecaster 等较重库，当前仅需要外部分数更新。脚本通过 AST 加载原始 `BasePredictor`、`ScaleFreeOGD`、`_OGD`、`SAOCP` 和完整 utilities 定义，跳过原始 import 与未使用 Enb ensemble 类；Merlion type annotation 由禁止执行的绑定提供，误用 forecaster 路径会报错。没有改写梯度、阈值投影、专家生命周期、聚合权重或初始化代码。所有关键原始文件运行前按 pinned SHA-256 校验。详细适配说明保存在 vendor 的 `ADAPTATION.md`。

脚本：`scripts/run_strong_calibration.py`。执行方式：

```sh
.venv/bin/python scripts/run_strong_calibration.py
```

默认运行两种方法，也可使用 `--methods sfogd`。依赖已经存在的公共 returns、GRU checkpoints 与已归档 issued forecasts，不下载新金融数据。

## 严格时间与分数协议

- 输入测试预测来自 `outputs/public_fx/predictions.csv` 的 GRU/native 行，仅取已经发出的 `mu` 与 `sigma`；三个 seed、16/64 两种上下文、USD/JPY/GBP/CHF 四种货币。
- 为预热，读取已经保存的 GRU checkpoint 与公开收益，调用根节点 `build_arrays` / `predict` 重建 2019–2020 验证段；同时重建并逐行核对 **每组 1479 行、24 组、共 35496 条已发神经预测** 的 actual/mu/sigma，最大绝对误差 `4.44e-16`。
- 非一致性分数 `S_t = abs(actual_t - issued_mu_t) / issued_sigma_t`；区间半宽为 `q_t * issued_sigma_t`，名义覆盖率 90%。这与现有 GRU rolling/ACI 使用同一类型的归一化残差。
- 每个方法/模型种子/上下文/货币拥有独立状态。严格先 `predict()` 得到 q、生成区间，再看到当前 true score 并 `update()`。不把真实未来值传给阈值预测。
- 官方初始 q=0。`max_scale` 固定为 **2019 年归一化分数最大值 × sqrt(3)**，采用作者构造函数中的默认 scale 估计形式，但估计日期仅限 2019；随后按顺序回放 2019–2020 初始化状态。2019 只是离线初始化，不报告为当时已部署的在线验证。
- 2020 diagnostics 只保存观察，没有用其选择参数。SAOCP lifetime=8 为作者默认值。没有在 2021+ 测试段选择参数、方法、学习率、scale 或最佳 seed；在线看到测试标签后按原算法更新属于事先声明的设定。
- GRU checkpoint 本身在完整 2019–2020 验证期选出，因此验证回放是测试前准备。2020 diagnostics 不是独立于模型选择的盲测，也不能用于宣称严格独立 split-conformal 保证。

## score bound 与保证的限制

**`max_scale` 是估计的预期尺度，不是金融归一化残差的硬上界。** 固定估计不能保证以后极端事件不超过它。测试共 32 条方法/种子/上下文记录超出估计 scale（两个方法重复计数，同一真实市场路径也重复 seed），不应当成 32 个独立市场事件。

没有截断测试分数，也没有动态更新这个 scale。原始 SF-OGD/SAOCP 阈值可随更新变化；原始算法在非负阈值上的投影保持不变。由于分数未满足已知固定边界、金融序列有时间依赖、拟合尺度可能错设，本项目**不据此引用理论覆盖或 regret 保证**。将超界次数和每组 scale 写入初始化/结果文件，是为了使限制可审查。

如果之后做 scale/lifetime 的敏感性分析，应先在验证段预定网格和选法，再使用新的保留时段；不能根据本轮测试结果决定“最优参数”并重新称盲测。

## 数值验证与输出

运行约 19.8 秒，生成 **70992 条强基线区间**，每个方法复用同一 35496 条已发 GRU 预测。结果 key 无重复、数值有限；原始输入 CSV hash、checkpoint hash、根节点源码 hash、适配脚本 hash 都已存档。

真正的未来标签干预检查：以一条代表性测试流的前 300 项为例，修改第 200 项开始的所有未来真实分数；原始与修改后的 q 在第一个变更标签到达前及当项生成时完全一致，随后状态确实发生变化。SF-OGD、SAOCP 两者都通过。

输出：

| 文件 | 含义 |
|---|---|
| `predictions.csv` | 逐日区间、q、分数、命中、interval score、超 scale 标记 |
| `reconstructed_validation_forecasts.csv` | 公开数据与 checkpoint 重建的 2019–2020 forecasts |
| `validation_diagnostics.csv` | 2020 顺序回放描述性表现，未用于调参 |
| `initialization.json` | 每组 scale、q0、测试初始 q、预热范围与分数数量 |
| `summary_by_asset_seed.csv` | 按资产/seed/context/method 的描述性汇总 |
| `run_manifest.json` | 固定 source provenance、SHA、重建检查、干预检查与限制 |

## 本轮描述性结果

下面按四种货币与三个 seed 等权平均，不是 IID 显著性检验。width 的单位是百分比 log-return；“更窄”不自动等于更好，coverage 与 interval score 必须同时看。

| Context | 方法 | 覆盖率 | 平均宽度 | Interval score |
|---:|---|---:|---:|---:|
| 16 | SF-OGD | 0.899820 | 1.421286 | 1.924211 |
| 16 | SAOCP | 0.909624 | 1.573791 | 2.021765 |
| 64 | SF-OGD | 0.899594 | 1.436046 | 1.927540 |
| 64 | SAOCP | 0.909342 | 1.578817 | 2.022745 |

这轮可说明强基线已实际接入，且 SF-OGD 的整体平均覆盖接近名义值。不能说明它在短冲击恢复窗口也可靠，也不能由这张表推断原创方法有效。两个上下文的 scale 分别由各自验证残差确定；如要隔离“上下文记忆”机制，仍需进一步固定校准预算或开展公平消融。

所有货币共享 EUR，seed 只是对同一金融路径的重复训练。后续统计应共同按日期块重采样，先处理 seed 重复；本脚本不生成 IID confidence interval。当前 2021+ 数据已经探索，新增强基线结果也不能重新当作新的盲测结果。
