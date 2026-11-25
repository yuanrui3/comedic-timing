"""CPU-only synthetic checks for the 2026-11-18 mathematical research memo."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.special import ndtr
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261118
REPLICATIONS = 120_000
PROTECTED = {
    "runs/corrected-pauses-v1/metrics.json":
        "5991384df05895e17de474e88778cac303f000899466ab5e897a5baa5956aa34",
    "evaluation/human-practice-v1/START.html":
        "7be3ae02bcaea10053e843b62d70645192eb69d48acefc374d2182cc1aa6c866",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def run(out: Path) -> None:
    # Never mix new source with an existing result directory.
    out.mkdir(parents=True, exist_ok=False)
    before = {name: digest(ROOT / name) for name in PROTECTED}
    assert before == PROTECTED, "Protected study artifacts changed before audit"
    source = Path(__file__).read_bytes()
    (out / "mathematical_audit.py").write_bytes(source)
    plan = ROOT / "docs/mathematical-audit-plan.md"
    (out / "plan.md").write_bytes(plan.read_bytes())
    rng = np.random.default_rng(SEED)
    checks: list[str] = []

    # Order statistics: count of estimates below target must be 2,...,8.
    errors = np.sort(rng.normal(0, 10, (REPLICATIONS, 10)), axis=1)
    exact_median = 1 - 2 * (math.comb(10, 0) + math.comb(10, 1)) / 2**10
    empirical_median = float(np.mean((errors[:, 1] <= 0) & (errors[:, 8] >= 0)))
    assert abs(empirical_median - exact_median) < .002
    ensemble = []
    for bias in range(0, 51, 5):
        p = float(ndtr(-bias / 10))
        exact_truth = float(binom.cdf(8, 10, p) - binom.cdf(1, 10, p))
        empirical_truth = float(np.mean(
            (errors[:, 1] + bias <= 0) & (errors[:, 8] + bias >= 0)))
        assert abs(empirical_truth - exact_truth) < .004
        ensemble.append({"shared_bias_ms": bias,
                         "truth_coverage_exact": exact_truth,
                         "truth_coverage_simulated": empirical_truth,
                         "model_median_coverage_exact": exact_median,
                         "model_median_coverage_simulated": empirical_median})
    checks.append("Order-statistic Monte Carlo agrees with exact binomial coverage")

    # Perfect replication of latent script scores: explicit dependence example.
    group_scores = rng.uniform(size=(REPLICATIONS, 11))
    repeated_scores = np.repeat(group_scores[:, :10], 4, axis=1)
    naive_k = math.ceil((40 + 1) * .95)
    q_naive = np.sort(repeated_scores, axis=1)[:, naive_k - 1]
    clustered_sim = float(np.mean(group_scores[:, 10] <= q_naive))
    clustered_exact = 10 / 11
    assert abs(clustered_sim - clustered_exact) < .004
    assert np.all(q_naive == np.max(group_scores[:, :10], axis=1))
    quantiles = []
    for n in (10, 19, 20, 40):
        k = math.ceil((n + 1) * .95)
        quantiles.append({"independent_calibration_groups": n, "rank": k,
                          "finite_empirical_quantile": k <= n,
                          "coverage_continuous_iid": k / (n + 1) if k <= n else 1.0})
    checks.append("Duplicated-score quantile equals 10-group maximum, not 40 iid scores")

    # Projection effects on a finite frame, without filtering or changing clips.
    n_frame, fs = 160, 16_000
    t = (np.arange(n_frame) - (n_frame - 1) / 2) / fs
    phases = 2 * np.pi * np.arange(64) / 64
    basis = np.column_stack([np.ones(n_frame), t / max(abs(t))])
    q, _ = np.linalg.qr(basis)
    frequencies = np.unique(np.r_[np.geomspace(1, 1000, 301),
                                  [10, 20, 40, 60, 100, 200, 500]])
    retention = []
    for frequency in frequencies:
        signals = np.sin(2 * np.pi * frequency * t[:, None] + phases[None, :])
        centered = signals - signals.mean(axis=0)
        detrended = signals - q @ (q.T @ signals)
        input_energy = np.sum(signals**2, axis=0)
        linear_ratios = np.sum(detrended**2, axis=0) / input_energy
        ac_ratio = float(np.sum(centered**2) / np.sum(signals**2))
        linear_ratio = float(np.sum(detrended**2) / np.sum(signals**2))
        assert 0 <= linear_ratio <= ac_ratio + 1e-12 <= 1 + 1e-12
        retention.append({"frequency_hz": float(frequency),
                          "mean_removed_pooled_energy_fraction": ac_ratio,
                          "linear_removed_pooled_energy_fraction": linear_ratio,
                          "linear_removed_pooled_db": float(10*np.log10(linear_ratio)),
                          "linear_phase_min_db": float(10*np.log10(min(linear_ratios))),
                          "linear_phase_max_db": float(10*np.log10(max(linear_ratios)))})
    assert np.linalg.norm(basis - q @ (q.T @ basis)) < 1e-12
    checks.append("Nuisance projection annihilates its basis and cannot increase energy")

    # Profile regression improvement equals projected matched-template statistic.
    x = rng.normal(size=n_frame) + basis @ np.array([.8, -.4])
    s = np.tanh(600*t)
    xp, sp = x - q @ (q.T @ x), s - q @ (q.T @ s)
    gain_formula = float((sp @ xp)**2 / (sp @ sp))
    full_design = np.column_stack([basis, s])
    full_residual = x - full_design @ np.linalg.lstsq(full_design, x, rcond=None)[0]
    gain_regression = float(xp @ xp - full_residual @ full_residual)
    assert np.isclose(gain_formula, gain_regression, atol=1e-10)
    checks.append("Profile-likelihood numerator matches direct nuisance regression")

    # Generated endpoint rectangles, including overlap: do not silently clamp.
    endpoints = np.sort(rng.uniform(-100, 100, (5000, 2, 2)), axis=-1)
    s_low, s_high = endpoints[:, 0, 0], endpoints[:, 0, 1]
    e_low, e_high = endpoints[:, 1, 0], endpoints[:, 1, 1]
    corners = np.stack([e_low-s_low, e_low-s_high,
                        e_high-s_low, e_high-s_high], axis=1)
    assert np.allclose(corners.min(axis=1), e_low-s_high)
    assert np.allclose(corners.max(axis=1), e_high-s_low)
    candidate = np.linspace(0, 60, 601)
    worst_error = np.maximum(abs(candidate), abs(candidate-60))
    assert min(worst_error) == 30
    checks.append("Interval gap extrema match all corners; indistinguishable-pair bound is tight")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.9))
    blue, orange, gray = "#126782", "#b34f16", "#52616b"
    axs[0].plot([v["shared_bias_ms"] for v in ensemble],
                [100*v["truth_coverage_exact"] for v in ensemble],
                color=orange, lw=2.4, label="Actual synthetic boundary")
    axs[0].axhline(100*exact_median, color=blue, lw=2,
                   label="Population model median")
    axs[0].scatter([v["shared_bias_ms"] for v in ensemble],
                   [100*v["truth_coverage_simulated"] for v in ensemble],
                   color=orange, s=16, zorder=4)
    axs[0].set(xlabel="Shared model bias (ms)", ylabel="Interval coverage (%)",
               ylim=(-3, 104), title="A  Agreement can be confidently wrong")
    axs[0].legend(loc="center right", frameon=False, fontsize=9)
    bars = axs[1].bar([0, 1, 2], [100*39/41, 100*clustered_exact, 100],
                      color=[gray, orange, blue], width=.65)
    axs[1].set_xticks([0, 1, 2], ["40 independent\ncalibration scores",
                                  "10 scores, each\ncopied four times",
                                  "10 groups, proper\n95% rank (+infinity)"])
    axs[1].set(ylabel="New-group / new-score coverage (%)", ylim=(80, 104),
               title="B  Repeats do not create independence")
    axs[1].axhline(95, color=gray, ls=":", lw=1)
    axs[1].bar_label(bars, labels=["95.12%", "90.91%", "100%\n(vacuous)"],
                     padding=4, fontsize=9)
    f = [v["frequency_hz"] for v in retention]
    axs[2].semilogx(f, [10*np.log10(v["mean_removed_pooled_energy_fraction"])
                        for v in retention], color=blue, lw=2, label="Remove mean")
    axs[2].semilogx(f, [v["linear_removed_pooled_db"] for v in retention],
                    color=orange, lw=2, label="Remove line")
    axs[2].fill_between(f, [v["linear_phase_min_db"] for v in retention],
                       [v["linear_phase_max_db"] for v in retention],
                       color=orange, alpha=.12, label="Line removal: phase range")
    axs[2].set(xlabel="Pure-tone frequency (Hz), 10 ms frame",
               ylabel="Retained signal energy (dB)", ylim=(-100, 3),
               title="C  Detrending can also erase a signal")
    axs[2].legend(loc="lower right", frameon=False, fontsize=9)
    for ax in axs:
        ax.grid(axis="y", alpha=.16)
    fig.suptitle("Mathematical audit | Synthetic examples only", fontsize=18,
                 fontweight="bold", x=.035, ha="left", y=1.01)
    fig.text(.035, -.025,
             "No original recordings, detector timestamps, or human answers are used. "
             "Panel A: 10 Gaussian models, within-ensemble SD 10 ms; second/ninth order interval.",
             fontsize=10, color=gray)
    fig.tight_layout(w_pad=2.5)
    for ext in ("png", "svg"):
        fig.savefig(out / f"mathematical-audit.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    after = {name: digest(ROOT / name) for name in PROTECTED}
    assert after == before
    summary = {
        "kind": "synthetic mathematical checks; not natural-speech validation",
        "seed": SEED, "replications": REPLICATIONS,
        "ensemble": {"models": 10, "noise_sd_ms": 10,
                     "order_statistics_one_based": [2, 9],
                     "exact_model_median_coverage": exact_median,
                     "median_interval_width_ms": float(np.median(errors[:, 8]-errors[:, 1])),
                     "cases": ensemble},
        "clusters": {"independent_groups": 10, "copies_per_group": 4,
                     "naive_rank": naive_k, "actual_coverage_exact": clustered_exact,
                     "actual_coverage_simulated": clustered_sim,
                     "if_40_really_independent_coverage": 39/41,
                     "group_quantiles": quantiles},
        "projection": {"sample_rate_hz": fs, "frame_samples": n_frame,
                       "phases": len(phases), "retention": retention,
                       "regression_gain_formula": gain_formula,
                       "regression_gain_direct": gain_regression},
        "checks": checks,
        "protected_hashes_before": before, "protected_hashes_after": after,
    }
    dump(out / "summary.json", summary)
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(),
                  "python": platform.python_version(), "numpy": np.__version__,
                  "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
                  "source_sha256": hashlib.sha256(source).hexdigest(),
                  "plan_sha256": digest(plan),
                  "outputs": {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()},
                  "no_audio_or_human_response_inputs": True,
                  "protected_files_read_only_for_integrity_hashing": list(PROTECTED)}
    dump(out / "provenance.json", provenance)
    print(json.dumps({"out": str(out), "checks_passed": len(checks),
                      "model_median_coverage": exact_median,
                      "coverage_with_30ms_bias": ensemble[6],
                      "naive_cluster_coverage": clustered_sim,
                      "protected_artifacts_unchanged": after == before}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "runs/mathematical-audit-v1")
    run(parser.parse_args().out)
