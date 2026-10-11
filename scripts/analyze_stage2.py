"""Analyze the prospectively frozen Stage 2 paired interaction panel.

Primary uncertainty conditions on five fixed fits. Paths, not individual
observations or model/path combinations, are the resampling unit. All frozen
cases and all secondary metrics are retained; there is no test-based tuning.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "outputs/stage2_evaluation"
OUT = ROOT / "outputs/stage2_analysis"
METRICS = ["interval_score", "coverage", "width", "coverage_deficit",
           "excess_width_to_oracle"]
PHASES = ["entry", "recovery64", "recovery192", "late"]
PALETTE = {"gru": "#285a8a", "tcn": "#b56b2b", 16: "#285a8a", 64: "#b56b2b"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_inputs():
    protocol_path = ROOT / "configs/stage2_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    training_path = ROOT / "outputs/stage2_training/manifest.json"
    training = json.loads(training_path.read_text())
    evaluation = json.loads((IN / "manifest.json").read_text())
    assert protocol["frozen"] and training["status"] == evaluation["status"] == "complete"
    assert sha(protocol_path) == training["protocol_sha256"] == evaluation["protocol_sha256"]
    assert sha(training_path) == evaluation["training_manifest_sha256"]
    assert training["matching_minibatches_verified"] and len(training["models"]) == 10
    for name, value in training["source_sha256"].items():
        assert sha(ROOT / "scripts" / name) == value, name
    for entry in training["models"]:
        assert sha(ROOT / entry["checkpoint"]) == entry["checkpoint_sha256"]
    for name, value in evaluation["sources"].items():
        assert sha(ROOT / "scripts" / name) == value, name
    assert sha(ROOT / "outputs/stage2_controls/models.json") == evaluation["controls_sha256"]
    for name, value in evaluation["artifacts"].items():
        assert sha(IN / name) == value, name
    assert evaluation["cases"] == 30
    return protocol, training, evaluation


def path_weights(paths, replicates, seed):
    rng = np.random.default_rng(seed)
    indices = rng.integers(paths, size=(replicates, paths))
    weights = np.zeros((replicates, paths), dtype=float)
    np.add.at(weights, (np.arange(replicates)[:, None], indices), 1.0 / paths)
    return indices, weights


def interval(samples, level):
    tail = (1-level)/2
    return np.quantile(samples, [tail, 1-tail], axis=0)


def effect_tensor(frame, seeds, paths):
    """Rows are paired model seeds; columns are identically indexed paths."""
    assert not frame.duplicated(["seed", "path"]).any()
    expected = pd.MultiIndex.from_product([seeds, range(paths)], names=["seed", "path"])
    indexed = frame.set_index(["seed", "path"])
    assert len(indexed) == len(expected) and set(indexed.index) == set(expected)
    result = indexed.reindex(expected)[METRICS].to_numpy().reshape(len(seeds), paths, len(METRICS))
    assert np.isfinite(result).all()
    return result


def tiny_self_check():
    _, weights = path_weights(7, 200, 191)
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-15)
    # Adding the same arbitrary path effect to both architectures cancels.
    common = np.arange(7, dtype=float)[None, :]
    gru = common + np.array([1., 3.])[:, None]
    tcn = common + np.array([0., 0.])[:, None]
    difference = gru-tcn
    np.testing.assert_array_equal(difference.mean(axis=0), np.full(7, 2.))
    np.testing.assert_allclose(weights @ difference.mean(axis=0), 2, atol=1e-14)
    # Fixed-seed path resampling must not resample the two fitted seeds.
    lo, hi = interval(weights @ difference.mean(axis=0), .975)
    np.testing.assert_allclose([lo, hi], [2, 2], atol=1e-14)
    return {"passed": True, "paired_architecture_direction_verified": True,
            "conditional_resampling_keeps_fitted_seeds_fixed": True,
            "weights_sum_to_one": True}


def analyze(frame, protocol):
    config, primary = protocol["test"], protocol["primary"]
    seeds = protocol["training_config"]["model_seeds"]
    n, b = config["paths"], primary["bootstrap_replicates"]
    path_indices, primary_weights = path_weights(n, b, primary["bootstrap_seed"])
    _, secondary_weights = path_weights(n, b, primary["bootstrap_seed"]+2)
    rng_cross = np.random.default_rng(primary["bootstrap_seed"]+1)
    seed_indices = rng_cross.integers(len(seeds), size=(b, len(seeds)))
    keys = ["family", "duration", "amplitude", "architecture", "phase"]
    tensors, secondary_rows, individual_rows = {}, [], []
    expected_keys = set(product(config["families"], config["durations"], config["amplitudes"],
                                ["gru", "tcn", "ewma"], PHASES))
    for key, group in frame.groupby(keys, sort=True):
        group_seeds = [0] if key[3] == "ewma" else seeds
        tensor = effect_tensor(group, group_seeds, n)
        tensors[key] = tensor
        estimate = tensor.mean(axis=(0, 1))
        bounds = interval(secondary_weights @ tensor.mean(axis=0), .95)
        for i, metric in enumerate(METRICS):
            per_seed = tensor[:, :, i].mean(axis=1)
            secondary_rows.append({**dict(zip(keys, key)), "metric": metric,
                "estimate": estimate[i], "ci_low": bounds[0, i], "ci_high": bounds[1, i],
                "interval_level": .95, "paths": n, "fixed_fits": len(group_seeds),
                "seed_min": per_seed.min(), "seed_max": per_seed.max(),
                "interval_scope": "unadjusted descriptive; conditional on fixed fits"})
            for j, seed in enumerate(group_seeds):
                individual_rows.append({**dict(zip(keys, key)), "metric": metric,
                                        "seed": seed, "estimate": per_seed[j]})
    assert set(tensors) == expected_keys, "Missing or off-grid frozen case/architecture/phase"
    primary_rows, primary_seeds, leave_one_out, contrasts = [], [], [], []
    for family in config["families"]:
        for duration in config["durations"]:
            for amplitude in config["amplitudes"]:
                for phase in PHASES:
                    gru = tensors[(family, duration, amplitude, "gru", phase)]
                    tcn = tensors[(family, duration, amplitude, "tcn", phase)]
                    difference = gru-tcn
                    bounds = interval(secondary_weights @ difference.mean(axis=0), .95)
                    for i, metric in enumerate(METRICS):
                        contrasts.append({"family": family, "duration": duration, "amplitude": amplitude,
                            "phase": phase, "metric": metric, "estimate": difference[:, :, i].mean(),
                            "ci_low": bounds[0, i], "ci_high": bounds[1, i], "interval_level": .95,
                            "interval_scope": "unadjusted descriptive; conditional on five paired fixed fits"})
                    is_primary = (family == primary["family"] and duration == primary["duration"]
                                  and amplitude == primary["amplitude"] and phase in primary["phases"])
                    if not is_primary:
                        continue
                    i = METRICS.index(primary["metric"])
                    effect = difference[:, :, i]
                    per_seed = effect.mean(axis=1)
                    lo, hi = interval(primary_weights @ effect.mean(axis=0), primary["interval_level"])
                    crossed = effect[seed_indices[:, :, None], path_indices[:, None, :]].mean(axis=(1, 2))
                    cross_lo, cross_hi = interval(crossed, primary["interval_level"])
                    row = {"phase": phase, "estimate": effect.mean(), "ci_low": lo, "ci_high": hi,
                        "gru_interaction": gru[:, :, i].mean(), "tcn_interaction": tcn[:, :, i].mean(),
                        "crossed_sensitivity_low": cross_lo, "crossed_sensitivity_high": cross_hi,
                        "seed_min": per_seed.min(), "seed_max": per_seed.max(), "paths": n,
                        "fixed_seed_pairs": len(seeds), "interval_level": primary["interval_level"]}
                    primary_rows.append(row)
                    for j, seed in enumerate(seeds):
                        primary_seeds.append({"phase": phase, "seed": seed, "estimate": per_seed[j],
                            "gru_interaction": gru[j, :, i].mean(), "tcn_interaction": tcn[j, :, i].mean()})
                        leave_one_out.append({"phase": phase, "omitted_seed": seed,
                                             "estimate": np.delete(effect, j, axis=0).mean()})
    return {"primary_contrasts": pd.DataFrame(primary_rows),
            "primary_individual_seeds": pd.DataFrame(primary_seeds),
            "primary_leave_one_out": pd.DataFrame(leave_one_out),
            "secondary_interactions": pd.DataFrame(secondary_rows),
            "secondary_architecture_contrasts": pd.DataFrame(contrasts),
            "individual_seed_effects": pd.DataFrame(individual_rows)}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.labelcolor": "#222222",
        "text.color": "#222222", "axes.edgecolor": "#777777", "savefig.dpi": 180})


def plots(tables, curves, protocol):
    style()
    primary = tables["primary_contrasts"].set_index("phase")
    seeds = tables["primary_individual_seeds"]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for j, phase in enumerate(protocol["primary"]["phases"]):
        row = primary.loc[phase]
        ax.hlines(j, row.ci_low, row.ci_high, color=PALETTE["gru"], linewidth=3)
        ax.scatter([row.estimate], [j], color=PALETTE["gru"], s=65, zorder=3)
        each = seeds.loc[seeds.phase == phase, "estimate"].to_numpy()
        ax.scatter(each, j+np.linspace(-.13, .13, len(each)), color="#888888", s=24, marker="x")
    ax.axvline(0, color="#444444", linewidth=.8)
    ax.set_yticks([0, 1], ["Entry (32 steps)", "Recovery (64 steps)"])
    ax.invert_yaxis()
    ax.set_xlabel("GRU interaction − TCN interaction (interval-score units)")
    ax.set_title("Prespecified Gaussian pulse: amplitude 3, duration 128")
    fig.text(.02, .01, "Blue: 97.5% paired-path bootstrap interval, five fixed seed pairs. Grey: individual seed effects.\n"
             "128 independent paired paths; intervals use a Bonferroni adjustment for two primary phases.", fontsize=9)
    fig.tight_layout(rect=(0, .15, 1, 1))
    fig.savefig(OUT / "primary_architecture_contrast.png")
    plt.close(fig)

    secondary = tables["secondary_interactions"]
    contrast = tables["secondary_architecture_contrasts"]
    families = protocol["test"]["families"]
    amplitudes = protocol["test"]["amplitudes"]
    durations = protocol["test"]["durations"]
    row_keys = [(f, a) for f in families for a in amplitudes]
    matrices = []
    for phase in ("entry", "recovery64"):
        for architecture in ("gru", "tcn", "gru-minus-tcn"):
            selected = contrast if architecture == "gru-minus-tcn" else secondary[secondary.architecture == architecture]
            selected = selected[(selected.phase == phase) & (selected.metric == "interval_score")]
            lookup = selected.set_index(["family", "amplitude", "duration"])["estimate"]
            matrices.append(np.array([[lookup.loc[(f, a, d)] for d in durations] for f, a in row_keys]))
    limit = max(float(np.abs(matrices).max()), 1e-12)
    cmap = LinearSegmentedColormap.from_list("signed_blue_orange", ["#285a8a", "#ffffff", "#b56b2b"])
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2), sharex=True, sharey=True)
    for ax, matrix, j in zip(axes.flat, matrices, range(6)):
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=TwoSlopeNorm(0, -limit, limit))
        for r in range(len(row_keys)):
            for c in range(len(durations)):
                ax.text(c, r, f"{matrix[r,c]:+.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(matrix[r,c]) > .65*limit else "#222222")
        ax.set_xticks(range(len(durations)), durations)
        ax.set_yticks(range(len(row_keys)), [f"{f.replace('student_t5','t5').replace('sv_gaussian','SV')} | A={a:g}" for f, a in row_keys])
        ax.set_title(["GRU", "TCN", "GRU − TCN"][j % 3]+" | "+["Entry", "Recovery 64"][j//3])
        if j >= 3:
            ax.set_xlabel("Pulse duration (steps)")
    fig.colorbar(image, ax=axes, shrink=.75, label="Event-minus-control context × window interaction (IS units)")
    fig.suptitle("Complete frozen grid: 3 families × 2 amplitudes × 5 durations", y=.98)
    fig.text(.06, .015, "Cell means across five fixed fits and 128 paired paths. Secondary results are descriptive; full intervals are in CSV.\n"
             "Positive GRU − TCN values indicate a larger interaction, not a ranking of absolute forecast performance.", fontsize=9)
    fig.subplots_adjust(left=.18, right=.80, top=.9, bottom=.14, hspace=.25, wspace=.12)
    fig.savefig(OUT / "full_grid_interactions.png")
    plt.close(fig)

    gaussian = curves[(curves.family == "gaussian") & (curves.window.isin([32, 128]))]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    onset = protocol["test"]["onset"]
    for row, architecture in enumerate(["gru", "tcn"]):
        for context in [16, 64]:
            for window in [32, 128]:
                selected = gaussian[(gaussian.architecture == architecture) & (gaussian.context == context)
                                    & (gaussian.window == window)].pivot(index="time", columns="world", values=["coverage", "width"])
                x = selected.index.to_numpy()-onset
                label = f"context {context}, window {window}"
                kwargs = {"color": PALETTE[context], "linestyle": "-" if window == 32 else "--", "linewidth": 1.2}
                axes[row, 0].plot(x, selected["coverage"]["event"], label=label, **kwargs)
                axes[row, 1].plot(x, selected["width"]["event"]/selected["width"]["control"], label=label, **kwargs)
        axes[row, 0].axhline(.9, color="#444444", linewidth=.8, linestyle=":")
        axes[row, 1].axhline(1, color="#444444", linewidth=.8, linestyle=":")
        for col in range(2):
            axes[row, col].axvspan(0, 128, color="#dedede", alpha=.4)
            axes[row, col].axvline(128+192, color="#555555", linestyle=":", linewidth=.8)
            axes[row, col].set_title(architecture.upper()+" | "+["Empirical coverage", "Mean width: event/control"][col])
        axes[row, 0].set_ylim(0, 1.03)
    axes[0, 0].legend(fontsize=8, ncol=2, loc="lower right")
    axes[1, 0].set_xlabel("Steps since pulse onset")
    axes[1, 1].set_xlabel("Steps since pulse onset")
    fig.text(.025, .015, "Gaussian A=3, D=128; means over five fixed fits and 128 paths. Shading marks the pulse.\n"
             "Dotted vertical line: exit + 192, the sufficient finite-context rolling-memory erasure bound. No temporal CI shown.", fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, 1))
    fig.savefig(OUT / "reference_gaussian_curves.png")
    plt.close(fig)


def md_table(frame, columns, precision=4):
    lines = ["| "+" | ".join(columns)+" |", "| "+" | ".join(["---"]*len(columns))+" |"]
    for _, row in frame.iterrows():
        values = [f"{row[c]:.{precision}f}" if isinstance(row[c], (float, np.floating)) else str(row[c]) for c in columns]
        lines.append("| "+" | ".join(values)+" |")
    return "\n".join(lines)


def write_report(tables, performance, protocol, evaluation):
    primary = tables["primary_contrasts"]
    seed_table = tables["primary_individual_seeds"]
    loo = tables["primary_leave_one_out"]
    claims = []
    for _, row in primary.iterrows():
        if row.ci_low > 0:
            status = "在预定的条件区间中为正"
        elif row.ci_high < 0:
            status = "在预定的条件区间中为负"
        else:
            status = "条件区间包含 0，不能据此排除零差异"
        sub = loo[loo.phase == row.phase]
        claims.append(f"- `{row.phase}` 的 GRU − TCN interaction 为 {row.estimate:+.5f}，97.5% 区间 [{row.ci_low:+.5f}, {row.ci_high:+.5f}]，{status}。五个 seed 的均值范围 [{row.seed_min:+.5f}, {row.seed_max:+.5f}]；leave-one-seed-out 范围 [{sub.estimate.min():+.5f}, {sub.estimate.max():+.5f}]。")
    summary = primary.rename(columns={"estimate": "GRU−TCN", "gru_interaction": "GRU interaction",
        "tcn_interaction": "TCN interaction", "ci_low": "97.5% low", "ci_high": "97.5% high",
        "crossed_sensitivity_low": "crossed low", "crossed_sensitivity_high": "crossed high"})
    text = f"""# Stage 2：统一训练后的独立配对测试

日期：2026-10-11。全部 10 个冻结拟合和全部 30 个预先指定测试 case 均已完成。分析没有返回训练、修改 checkpoint 或挑选测试中的最优 case。当前结果检验的是两个具体拟合程序的尺度响应与滚动分数窗口的交互；它本身不构成 A+ 保证，也不是已经证明有优势的新方法。

## 预定主要结论

{chr(10).join(claims)}

{md_table(summary, ['phase', 'GRU interaction', 'TCN interaction', 'GRU−TCN', '97.5% low', '97.5% high', 'crossed low', 'crossed high'])}

这里的 interaction 已经是 event-minus-matched-control 的 2×2 差分：`[(context64,W128−context16,W128)−(context64,W32−context16,W32)]event − 同一项control`。主 contrast 再取 GRU interaction − TCN interaction。正号只表示 GRU 的该交互更大，不能等同于 GRU 的绝对区间分数更差，更不能直接推出“长记忆有害”。Interval score 越小越好，但 interaction 是数个性能差值的组合。

主要 case 预先固定为 Gaussian、amplitude 3、duration 128；entry 为前 32 个冲击时点，recovery64 为退出后的前 64 个时点。区间使用 4000 次 paired-path percentile bootstrap，先平均五对**固定**已拟合模型，再重抽 128 条独立配对 path。两端点用 97.5% marginal interval 作 Bonferroni 调整；这是 bootstrap 近似，不是有限样本严格覆盖定理。表中的 crossed seed/path 区间只是敏感性检查，不能从五个初始化推断初始化总体显著性。训练数据固定，模型 seed 并不代表重抽训练数据。

![预定 primary contrast](../outputs/stage2_analysis/primary_architecture_contrast.png)

## 全部初始化与全网格

{md_table(seed_table, ['phase', 'seed', 'gru_interaction', 'tcn_interaction', 'estimate'])}

完整 secondary 文件包含全部 `3 noise families × 5 durations × 2 amplitudes × 4 phases × 5 metrics`，并分别保留 GRU、TCN、EWMA interaction，以及全部 GRU − TCN contrast。95% secondary 区间未做多重检验校正，只用于描述。不同 duration 和 amplitude 共用每个 family 的相同噪声路径；不能把所有 case 再合并为独立样本。

![完整 30-case grid](../outputs/stage2_analysis/full_grid_interactions.png)

- [全部 architecture-specific interaction 与区间](../outputs/stage2_analysis/secondary_interactions.csv)
- [全部 GRU − TCN contrast 与区间](../outputs/stage2_analysis/secondary_architecture_contrasts.csv)
- [全部 individual-seed effects](../outputs/stage2_analysis/individual_seed_effects.csv)
- [全部 leave-one-seed-out primary estimates](../outputs/stage2_analysis/primary_leave_one_out.csv)
- [全部绝对表现汇总：event 与 control、native/rolling32/rolling128](../outputs/stage2_analysis/performance_summary.csv)

## 恢复边界与指标含义

共检查 {evaluation['negative_control_checks']} 个神经模型 context/calibrator/case 组合。退出 192 个时点以后，event 与 matched-control 的最大绝对宽度误差为 `{evaluation['maximum_negative_control_width_error']:.3g}`。这是固定长度上下文 16/64 与过去分数 rolling32/128 的充分记忆消失边界验证；它不适用于保留无限历史或额外状态的 EWMA、GARCH、ACI、SF-OGD、SAOCP，也不保证真实市场的路径重新重合。因果前缀、过去分数分位数、上下文缓存与独立重算边界检查已通过。

![Gaussian reference 动态轨迹](../outputs/stage2_analysis/reference_gaussian_curves.png)

Coverage 是真实已实现包含率；coverage deficit 使用已知模拟分布的区间包含概率。Gaussian/t5 的 oracle 指定清楚；SV 的条件概率和 oracle width 条件于模拟潜在状态，**不是**观测历史下可部署预测的条件概率。图中的均值平均五个固定拟合后再平均 128 条 path，未把 640 个 model/path 组合当成独立样本。曲线没有逐时点置信带，不能据单个时点的起伏宣称显著变化。

## 深度学习与有效记忆诊断

GRU 与 TCN 使用同一训练/验证观测、相同目标与输入特征、共同批次采样前缀、同一验证选择规则。参数量为 2041 与 1897，early stopping 使优化总步数可以不同，因此比较仍然是这两个冻结拟合程序；不能把任何差异全部归因于架构本身。10 个模型都按 validation patience 结束，未触及 60 epoch 上限。

"""
    memory_path = ROOT / "outputs/stage2_memory/manifest.json"
    if memory_path.exists():
        memory = json.loads(memory_path.read_text())
        assert memory["status"] == "complete"
        assert memory["protocol_sha256"] == sha(ROOT / "configs/stage2_protocol.json")
        assert memory["training_manifest_sha256"] == sha(ROOT / "outputs/stage2_training/manifest.json")
        assert memory["source_sha256"] == sha(ROOT / "scripts/stage2_memory_diagnostics.py")
        for name, value in memory["output_sha256"].items():
            assert sha(memory_path.parent / name) == value
        diagnostics = pd.read_csv(memory_path.parent / "model_memory_summary.csv")
        rows = []
        for architecture, group in diagnostics.groupby("architecture"):
            rows.append({"architecture": architecture,
                "older>16 gradient fraction min": group.older_than16_gradient_mass_fraction.min(),
                "older>16 gradient fraction max": group.older_than16_gradient_mass_fraction.max(),
                "older permutation mean |Δlogσ| min": group.older_permutation_mean_abs_delta_log_scale.min(),
                "older permutation mean |Δlogσ| max": group.older_permutation_mean_abs_delta_log_scale.max()})
        text += md_table(pd.DataFrame(rows), list(rows[0]))+"\n\n"
        text += "诊断只读取冻结验证集的前 256 个既有目标对。older48 block 的梯度质量、置换前后 logσ 差异与 context16/64 差异说明模型对较旧输入的局部敏感性；梯度不是长记忆的因果证明，置换可产生分布外上下文。这些目标可能来自相同路径并且上下文重叠，未声称 256 条独立样本的置信区间。完整图与记录见 [validation memory diagnostics](../outputs/stage2_memory/manifest.json)。\n\n"
    text += """## 与 A+ 目标之间还缺什么

这一步补上了第 1 阶段 MLP/GRU 比较中的主要混杂：统一数据、输入、checkpoint 选择规则、五个 seed 配对与独立测试，并保留零效应或反向效应。它提高了研究结论的可信度；课程 A+ 仍然要求可辨识、充分论证的原创贡献，不能用完成训练次数、漂亮图或单个区间不含 0 来替代。

接下来适合用这批冻结预测器做 current-scale 与 historical-score archive 的独立交叉干预，分离“当前尺度响应”和“归档分母污染”两个来源。该补充将受到当前结果启发，应明确标记为 exploratory；若要再作 confirmatory claim，需先固定新对照和新随机种子，再生成新的独立测试。还需要核对直接相关研究中的 replay、reset、neural controller 等已有方法，避免把已有组件当成新贡献。强 calibration 补充和真实公开金融数据的结果应单独展示其范围；当前 ECB 测试时期已经探索过，不能追溯改称未见过的最终 holdout。

如尝试一个有依据的校准策略，其选择必须使用 train/validation 信息，最终结论应同时报告 interval score、coverage、width，以及失败条件和代价。当前主实验没有训练或验证这样一个新策略，所以本报告不声称性能提升。AI 使用已获课程许可；学生仍须理解全部假设、代码与实验并按课程要求承担署名与成果责任。

## 复现与来源

```sh
.venv/bin/python scripts/analyze_stage2.py --self-check
.venv/bin/python scripts/analyze_stage2.py
```

分析脚本在读取真实第二阶段测试输出前已提交。manifest 校验冻结协议、训练源码、10 个 checkpoint、控制模型、evaluation 全部 artifact SHA256 后才计算统计；记录输入 SHA256、分析源码 SHA256 和 bootstrap seed/rules。分析不会重新生成测试或改变模型。[analysis manifest](../outputs/stage2_analysis/manifest.json) 和 [frozen protocol](../configs/stage2_protocol.json) 是具体来源。
"""
    (ROOT / "research/stage2_results.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(tiny_self_check()))
        return
    protocol, training, evaluation = verified_inputs()
    if (OUT / "manifest.json").exists():
        raise FileExistsError("Analysis manifest exists; refusing silent overwrite")
    OUT.mkdir(parents=True, exist_ok=True)
    script_sha = sha(Path(__file__))
    frame = pd.read_csv(IN / "paired_interactions.csv.gz")
    assert len(frame) == evaluation["interaction_rows"]
    tables = analyze(frame, protocol)
    performance = pd.read_csv(IN / "performance_by_seed.csv")
    group = ["family", "duration", "amplitude", "architecture", "context", "window", "phase", "world"]
    means = performance.groupby(group)[METRICS].mean()
    means["fixed_fits"] = performance.groupby(group).seed.nunique()
    means["paths"] = protocol["test"]["paths"]
    tables["performance_summary"] = means.reset_index()
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False)
    curves = pd.read_csv(IN / "reference_curves.csv.gz")
    plots(tables, curves, protocol)
    write_report(tables, means.reset_index(), protocol, evaluation)
    assert sha(Path(__file__)) == script_sha, "Analysis source changed during run"
    manifest = {"status": "complete", "self_check": tiny_self_check(), "analysis_source_sha256": script_sha,
        "protocol_sha256": sha(ROOT / "configs/stage2_protocol.json"),
        "training_manifest_sha256": sha(ROOT / "outputs/stage2_training/manifest.json"),
        "evaluation_manifest_sha256": sha(IN / "manifest.json"),
        "input_sha256": {name: sha(IN / name) for name in ["paired_interactions.csv.gz", "performance_by_seed.csv", "reference_curves.csv.gz"]},
        "primary_bootstrap": {"seed": protocol["primary"]["bootstrap_seed"], "replicates": 4000,
            "level": .975, "unit": "paired path; mean across five fixed seed pairs before resampling",
            "adjustment": "two prespecified endpoint Bonferroni; percentile bootstrap approximation",
            "same_path_draws_used_for_both_primary_phases": True},
        "crossed_sensitivity": {"seed": protocol["primary"]["bootstrap_seed"]+1,
            "unit": "seed-pair and common paired path resampled independently; sensitivity only"},
        "secondary_bootstrap": {"seed": protocol["primary"]["bootstrap_seed"]+2, "replicates": 4000,
            "level": .95, "scope": "unadjusted descriptive; no joint testing or initialization population claim"},
        "negative_control_maximum_width_error": evaluation["maximum_negative_control_width_error"],
        "row_counts": {name: len(table) for name, table in tables.items()},
        "artifact_sha256": {p.name: sha(p) for p in OUT.iterdir() if p.is_file()},
        "report_sha256": sha(ROOT / "research/stage2_results.md")}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(tables["primary_contrasts"].to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
