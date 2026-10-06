"""Phase 1.3 analysis + Phase 3 decision gate.

Consumes the per-clip monitor time series produced by any adapter and emits the
full pre-registered result set: thresholds, Time-to-Alarm, Kaplan-Meier curves,
paired tests with Holm correction, false-alarm rates, and the gate verdict.

Two threshold regimes are reported side by side:

* `per_frame_matched`  - the pre-registered PRIMARY. 99th percentile of the
  control-calibration score distribution, equalising per-frame false alarms.
* `clip_matched_5pct`  - a SECONDARY added before any real inference run,
  equalising the clip-level false-alarm rate at 5%, which is the operationally
  meaningful calibration for a monitor that raises one alarm per episode.

The Phase 3 verdict is decided by the primary regime alone. The secondary and
the censoring-robust RMTA statistics are reported so that a degenerate median
(which happens whenever fewer than half the clips alarm) cannot silently hide a
real effect.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import config as C
from . import stats as S
from .probes import fit_ridge_predictor, ridge_loss

N_BOOT = 2000

REGIMES = (
    ("per_frame_matched", "PRIMARY (pre-registered): per-frame FPR = 1%"),
    ("clip_matched_5pct", "SECONDARY: clip-level FPR = 5%"),
)


def _l1_scores(Z: np.ndarray, W: np.ndarray) -> np.ndarray:
    return np.stack([ridge_loss(Z[i].astype(np.float64), W) for i in range(Z.shape[0])])


def add_l1_baseline(scores: dict, n_folds: int = 5) -> None:
    """Attach baseline L1, cross-fitted on control_calib (prereg Sec. 2.2)."""
    Zc = scores["control_calib"]["Z"]
    n = Zc.shape[0]
    folds = np.arange(n) % n_folds
    oof = np.full((n, Zc.shape[1]), np.nan)
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        W = fit_ridge_predictor(Zc[tr].astype(np.float64))
        oof[te] = _l1_scores(Zc[te], W)
    scores["control_calib"]["loss_L1"] = oof
    W_full = fit_ridge_predictor(Zc.astype(np.float64))
    for split in scores:
        if split != "control_calib":
            scores[split]["loss_L1"] = _l1_scores(scores[split]["Z"], W_full)


def _thresholds(calib: dict, regime: str) -> dict:
    if regime == "per_frame_matched":
        return {m: S.threshold(calib[m]) for m in C.MONITORS}
    if regime == "clip_matched_5pct":
        return {m: S.threshold_clip_fpr(calib[m], 0.05) for m in C.MONITORS}
    raise ValueError(regime)


def _run_regime(anom: dict, calib: dict, ev: dict, conds: np.ndarray,
                regime: str) -> dict:
    thresholds = _thresholds(calib, regime)
    per_monitor, dts, events = {}, {}, {}

    for m in C.MONITORS:
        dt, e = S.alarm_times(anom[m], thresholds[m])
        dts[m], events[m] = dt, e
        rec = S.summarise(dt, e)
        rec["threshold"] = thresholds[m]
        rec["rmta"] = S.rmta(dt, e)
        rec["rmta_ci95"] = S.rmta_bootstrap_ci(dt, e, n_boot=N_BOOT)
        rec["clip_fpr_control_eval"] = S.clip_false_alarm_rate(ev[m], thresholds[m])
        rec["clip_fpr_control_calib"] = S.clip_false_alarm_rate(calib[m], thresholds[m])
        rec["by_condition"] = {
            c: dict(S.summarise(dt[conds == c], e[conds == c]),
                    rmta=S.rmta(dt[conds == c], e[conds == c]))
            for c in C.CONDITIONS
        }
        per_monitor[m] = rec

    comparisons, pvals = {}, {}
    for pr in C.PROBES:
        for b in C.BASELINES:
            key = pr + "_vs_" + b
            w = S.wilcoxon_paired(dts[pr], dts[b])
            w["logrank"] = S.logrank(dts[pr], events[pr], dts[b], events[b])
            w["rmta"] = S.paired_rmta_diff_test(dts[pr], events[pr], dts[b],
                                                events[b], n_boot=N_BOOT)
            w["median_gain_frames"] = (per_monitor[b]["median_dt"]
                                       - per_monitor[pr]["median_dt"])
            w["rmta_gain_frames"] = -w["rmta"]["rmta_diff"]
            w["by_condition"] = {
                c: S.wilcoxon_paired(dts[pr][conds == c], dts[b][conds == c])
                for c in C.CONDITIONS
            }
            comparisons[key] = w
            pvals[key] = w["p_one_sided"]

    for k, v in S.holm(pvals).items():
        comparisons[k]["p_holm"] = v
        comparisons[k]["significant"] = bool(v < C.ALPHA)
    for k, v in S.holm({k: c["rmta"]["p_one_sided_boot"]
                        for k, c in comparisons.items()}).items():
        comparisons[k]["p_holm_rmta"] = v

    early_warners = [
        pr for pr in C.PROBES
        if all(comparisons[pr + "_vs_" + b]["median_gain_frames"] >= C.MIN_MEDIAN_GAIN
               and comparisons[pr + "_vs_" + b]["p_holm"] < C.ALPHA
               for b in C.BASELINES)
    ]
    rmta_warners = [
        pr for pr in C.PROBES
        if all(comparisons[pr + "_vs_" + b]["rmta_gain_frames"] >= C.MIN_MEDIAN_GAIN
               and comparisons[pr + "_vs_" + b]["p_holm_rmta"] < C.ALPHA
               for b in C.BASELINES)
    ]

    km = {}
    for m in C.MONITORS:
        t, s = S.kaplan_meier(dts[m], events[m])
        km[m] = {"times": t.tolist(), "survival": s.tolist()}

    return {
        "regime": regime,
        "thresholds": thresholds,
        "per_monitor": per_monitor,
        "comparisons": comparisons,
        "kaplan_meier": km,
        "early_warners_median": early_warners,
        "early_warners_rmta": rmta_warners,
        "median_degenerate": all(
            per_monitor[m]["median_dt"] >= C.CENSOR_DT for m in C.MONITORS),
    }


def analyse(scores: dict, out_dir: str, model_name: str, note: str = "") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    add_l1_baseline(scores)

    anom, calib, ev = scores["anomaly"], scores["control_calib"], scores["control_eval"]
    conds = anom["condition"]

    regimes = {name: _run_regime(anom, calib, ev, conds, name) for name, _ in REGIMES}
    primary = regimes["per_frame_matched"]

    results = {
        "model": model_name,
        "note": note,
        "n_anomaly": int(anom["Z"].shape[0]),
        "n_control_calib": int(calib["Z"].shape[0]),
        "n_control_eval": int(ev["Z"].shape[0]),
        "latent_dim": int(anom["Z"].shape[2]),
        "n_boot": N_BOOT,
        "regimes": regimes,
        "gate": {
            "decided_by": "per_frame_matched / median Time-to-Alarm (pre-registered)",
            "early_warners": primary["early_warners_median"],
            "verdict": ("PROCEED_TO_PHASE_4" if primary["early_warners_median"]
                        else "NEGATIVE_RESULT"),
            "criterion": ("median gain >= {} frame vs BOTH baselines and "
                          "Holm-adjusted one-sided p < {}".format(
                              C.MIN_MEDIAN_GAIN, C.ALPHA)),
            "primary_median_degenerate": primary["median_degenerate"],
            "secondary_rmta_early_warners": primary["early_warners_rmta"],
            "clip_matched_median_early_warners":
                regimes["clip_matched_5pct"]["early_warners_median"],
            "clip_matched_rmta_early_warners":
                regimes["clip_matched_5pct"]["early_warners_rmta"],
        },
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    _write_report(results, out_dir)
    _plot_km(results, out_dir)
    return results


def _write_report(r: dict, out_dir: str) -> None:
    L = ["# Results - {}".format(r["model"])]
    if r["note"]:
        L.append("\n> {}\n".format(r["note"]))
    L.append("\n`n_anomaly={}`, `n_control_calib={}`, `n_control_eval={}`, "
             "`D={}`, `n_boot={}`\n".format(
                 r["n_anomaly"], r["n_control_calib"], r["n_control_eval"],
                 r["latent_dim"], r["n_boot"]))

    for name, desc in REGIMES:
        g = r["regimes"][name]
        L.append("\n---\n\n## Regime: `{}`\n\n*{}*\n".format(name, desc))

        L.append("### Time-to-Alarm per monitor\n")
        L.append("| monitor | median dt | RMTA | RMTA 95% CI | detection rate "
                 "| clip FPR (held-out) | threshold |")
        L.append("|---|---|---|---|---|---|---|")
        for m in C.MONITORS:
            d = g["per_monitor"][m]
            L.append("| `{}` | {:.1f} | {:.2f} | [{:.2f}, {:.2f}] | {:.3f} | {:.3f} | {:.4g} |".format(
                m, d["median_dt"], d["rmta"], d["rmta_ci95"][0], d["rmta_ci95"][1],
                d["detection_rate"], d["clip_fpr_control_eval"], d["threshold"]))
        if g["median_degenerate"]:
            L.append("\n> **Median is degenerate in this regime** (every monitor is "
                     "right-censored at the median, i.e. detection rate < 50%). "
                     "Read the RMTA columns, which remain well defined.\n")

        L.append("\n### Paired comparisons (probe vs baseline)\n")
        L.append("| comparison | median gain | RMTA gain | RMTA 95% CI | p one-sided "
                 "| p two-sided | p Holm | p Holm (RMTA) | log-rank p | early warner |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for k, c in g["comparisons"].items():
            lo, hi = c["rmta"]["ci95"]
            L.append("| `{}` | {:+.1f} | {:+.2f} | [{:+.2f}, {:+.2f}] | {:.3g} | {:.3g} "
                     "| {:.3g} | {:.3g} | {:.3g} | {} |".format(
                         k, c["median_gain_frames"], c["rmta_gain_frames"], -hi, -lo,
                         c["p_one_sided"], c["p_two_sided"], c["p_holm"],
                         c["p_holm_rmta"], c["logrank"]["p"],
                         "YES" if c["significant"] else "no"))

        L.append("\n### Per-condition RMTA\n")
        L.append("| monitor | " + " | ".join(C.CONDITIONS) + " |")
        L.append("|---" * (len(C.CONDITIONS) + 1) + "|")
        for m in C.MONITORS:
            row = ["{:.2f}".format(g["per_monitor"][m]["by_condition"][c]["rmta"])
                   for c in C.CONDITIONS]
            L.append("| `{}` | ".format(m) + " | ".join(row) + " |")

    g = r["gate"]
    L.append("\n---\n\n## Phase 3 gate\n")
    L.append("**Verdict: {}**\n".format(g["verdict"]))
    L.append("- Decided by: {}".format(g["decided_by"]))
    L.append("- Criterion: {}".format(g["criterion"]))
    L.append("- Early warners (pre-registered primary): {}".format(
        g["early_warners"] or "none"))
    L.append("- Primary median degenerate: {}".format(g["primary_median_degenerate"]))
    L.append("\nSecondary evidence (reported, does not decide the gate):\n")
    L.append("- RMTA early warners, per-frame-matched: {}".format(
        g["secondary_rmta_early_warners"] or "none"))
    L.append("- Median early warners, clip-matched: {}".format(
        g["clip_matched_median_early_warners"] or "none"))
    L.append("- RMTA early warners, clip-matched: {}\n".format(
        g["clip_matched_rmta_early_warners"] or "none"))
    with open(os.path.join(out_dir, "REPORT.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def _plot_km(r: dict, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {"loss_L0": ("k", "-"), "loss_L1": ("0.45", "--"),
              "probe_A": ("#1f77b4", "-"), "probe_B": ("#d62728", "-"),
              "probe_C": ("#2ca02c", "-")}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (name, desc) in zip(axes, REGIMES):
        g = r["regimes"][name]
        for m in C.MONITORS:
            t = np.asarray(g["kaplan_meier"][m]["times"])
            s = np.asarray(g["kaplan_meier"][m]["survival"])
            col, ls = styles[m]
            ax.step(t, s, where="post", color=col, linestyle=ls, lw=2, label=m)
        ax.set_xlabel("frames since anomaly injection (dt)")
        ax.set_title(desc, fontsize=10)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("P(anomaly still undetected)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Time-to-detection - {}".format(r["model"]), fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "survival_curves.png"), dpi=160)
    plt.close(fig)
