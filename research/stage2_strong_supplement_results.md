# Stage 2 single-seed strong-calibration supplement

日期：2026-10-11。完整运行耗时 443.90 秒，exit code 0。仅使用冻结 model seed4101 的 GRU/TCN × context16/64，在预先指定 Gaussian/t5、duration128、amplitude3 两个 reference 上比较 native、rolling32/128、clipped ACI128（γ=.02）、作者 SF-OGD、作者 SAOCP。全部方法和 event/control 世界均保留，未根据测试结果选择参数。

以下表格为 event 世界的描述性汇总：每个 cell 对四个固定的 architecture/context 配置等权平均，每个配置包含 128 条独立测试轨迹。只用于展示覆盖与区间效率的权衡；四个配置不是四个独立训练种子，不报告初始化总体显著性。完整文件还保留 recovery192、late、control、每个单独配置和 event-minus-control。

| Family | Phase | Method | Mean IS | Coverage | Mean width |
|---|---|---|---:|---:|---:|
| gaussian | entry | native | 13.9749 | 0.8043 | 8.2225 |
| gaussian | entry | rolling32 | 14.9499 | 0.8770 | 11.2469 |
| gaussian | entry | rolling128 | 14.2465 | 0.7981 | 8.2063 |
| gaussian | entry | aci128 | 14.5391 | 0.8439 | 9.8823 |
| gaussian | entry | sfogd | 15.5531 | 0.8936 | 12.3586 |
| gaussian | entry | saocp | 16.1142 | 0.9138 | 13.5863 |
| gaussian | recovery64 | native | 4.9685 | 0.9443 | 4.5091 |
| gaussian | recovery64 | rolling32 | 5.2260 | 0.9164 | 4.4546 |
| gaussian | recovery64 | rolling128 | 5.2267 | 0.9464 | 4.7815 |
| gaussian | recovery64 | aci128 | 5.1206 | 0.9211 | 4.4114 |
| gaussian | recovery64 | sfogd | 5.6796 | 0.9030 | 4.6732 |
| gaussian | recovery64 | saocp | 6.1469 | 0.9409 | 5.5893 |
| student_t5 | entry | native | 15.1891 | 0.8008 | 7.6525 |
| student_t5 | entry | rolling32 | 16.0559 | 0.8784 | 11.2254 |
| student_t5 | entry | rolling128 | 15.2366 | 0.8129 | 8.1182 |
| student_t5 | entry | aci128 | 15.6478 | 0.8500 | 9.8475 |
| student_t5 | entry | sfogd | 16.7847 | 0.8925 | 12.5655 |
| student_t5 | entry | saocp | 18.0177 | 0.9109 | 14.1964 |
| student_t5 | recovery64 | native | 5.1197 | 0.9305 | 4.1864 |
| student_t5 | recovery64 | rolling32 | 5.5627 | 0.9150 | 4.4192 |
| student_t5 | recovery64 | rolling128 | 5.4660 | 0.9418 | 4.6802 |
| student_t5 | recovery64 | aci128 | 5.3849 | 0.9174 | 4.2808 |
| student_t5 | recovery64 | sfogd | 5.8990 | 0.9024 | 4.5613 |
| student_t5 | recovery64 | saocp | 6.4730 | 0.9327 | 5.5427 |

冲击初段中，SAOCP、SF-OGD 和 rolling32 提高了实现覆盖率，也增加平均宽度；例如 Gaussian entry 的 SAOCP 覆盖 .9138、宽度 13.5863，而 native 覆盖 .8043、宽度 8.2225。对应 IS 为 16.1142 与 13.9749。较好的阶段平均覆盖率本身不能证明更好的条件预测分布；proper interval score 同时计入宽度与漏覆盖损失。上述数值应连同每个配置、对照世界及其它阶段一起解释，不能根据这些单 seed、两个事件组合泛化算法胜负。

## Reproducibility and execution-only revision

在测试引用读取前，script 与最初 validation-only preparation 分别于 `4fff13f` / `f92033a` 提交。随后仅优化 supplement wrapper 的执行：按四个独立 path chunk 启动进程，调用原有 `stage2_baselines.calibrate_paths`，每条 path 的 warmup、spec、predict/update 序列保持一致，不共享状态。人工串行/并行检查显示 SF-OGD、SAOCP 的 q、halfwidth、scores、covered、valid 逐点完全一致，合并后的原 metadata 完全一致。

旧 preparation 保存在 `validation_calibration_specs_before_parallel.json`，新版本及说明见 `validation_preparation_archive.json`。重新准备后，四个 calibration specs、validation observation/hash、validation prediction hashes 和 checkpoint hashes 与旧记录完全相同；只改变 supplement source hash 与耗时记录。并行版本与重冻结 preparation 在 `8c14e79` 提交后才读取 ready references。未修改训练、模型、baseline 源码或协议。

```sh
.venv/bin/python scripts/stage2_strong_supplement.py --self-check
.venv/bin/python scripts/stage2_strong_supplement.py --prepare-only
.venv/bin/python scripts/stage2_strong_supplement.py --evaluate
```

已有输出会拒绝悄悄覆盖；复现应使用新的 `--output` 目录并保留旧记录。warmup 为每个 predictor 对共同 validation path0 最后128观测的 causally issued normalized scores；target pre384 顺序反馈。SF-OGD/SAOCP 的作者代码保持原数学与 Apache-2.0 来源记录。每一时点先给定 sigma/q，再观察 y 和更新。

## Integrity and score-bound scope

49,152 条 path rows 无重复，384 groups 均有 128 个 path；覆盖率位于 [0,1]，所有指标有限，IS≥width。逐 path 重算的 summary 一致。主 panel 的 seed4101 native/rolling32/rolling128 共192个汇总组与补充结果一致：IS/width 最大绝对差 5.33e-15，coverage 最大差 2.22e-16。固定 reference 数值抽查与 frozen checkpoint 的最大 scale 差为 1.82e-7。

Gaussian 的每个 predictor/event/control 全131,072个 normalized scores 均未超过相应 validation score maximum。t5 event 世界的 GRU16/GRU64/TCN16/TCN64 超出次数为 7/11/4/3，control 为 5/10/2/3。这是每个 predictor/world 的全轨迹计数，包含未评分 pre-event 段；没有将相同分数在六个方法中重复累加。phase-specific 次数也在 metadata 保存。

所有测试分数均未超过该次运行的 `sqrt(3)*validation_max_score` max_scale heuristic。这不能确立硬支持界：Gaussian/t5 分布都有无界支持，max_scale 只由独立 validation 数据估计，分数没有截断，未主张 bounded-score 理论保证。ACI 为 clipped practical variant；native Gaussian quantile 在 t5 上存在模型失配。SF-OGD/SAOCP 保留额外状态，因此 rolling 的 exit+192 erasure 边界不适用于这些作者算法或 ACI。

- [逐 path 指标](/Users/lebmar/Documents/ChatGPT/DL-project/outputs/stage2_strong_supplement/path_metrics.csv.gz)
- [每个配置与阶段的完整汇总](/Users/lebmar/Documents/ChatGPT/DL-project/outputs/stage2_strong_supplement/phase_summary.csv)
- [配对 event-minus-control 汇总](/Users/lebmar/Documents/ChatGPT/DL-project/outputs/stage2_strong_supplement/paired_event_minus_control_summary.csv)
- [算法 metadata 与超界次数](/Users/lebmar/Documents/ChatGPT/DL-project/outputs/stage2_strong_supplement/calibrator_metadata.json)
- [provenance manifest](/Users/lebmar/Documents/ChatGPT/DL-project/outputs/stage2_strong_supplement/manifest.json)
