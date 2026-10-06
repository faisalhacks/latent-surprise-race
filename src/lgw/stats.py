"""Phase 1.3 - thresholds, Time-to-Alarm, survival analysis (prereg Sec. 3)."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps

from . import config as C

EVAL = slice(C.EVAL_START, C.EVAL_END)


def threshold(scores_calib: np.ndarray) -> float:
    """99th percentile over control-calibration clips x eval frames.

    Equalises the per-frame false-alarm rate across monitors, which is what
    makes the Time-to-Alarm comparison fair.
    """
    vals = np.asarray(scores_calib, dtype=np.float64)[:, EVAL].ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("no finite calibration scores in the evaluation window")
    return float(np.percentile(vals, 100.0 * (1.0 - C.FPR_TARGET)))


def alarm_times(scores: np.ndarray, thr: float):
    """-> (dt, event). dt in {0..CENSOR_DT}; event=False means right-censored."""
    s = np.asarray(scores, dtype=np.float64)[:, EVAL]
    exceed = np.isfinite(s) & (s > thr)
    any_alarm = exceed.any(axis=1)
    first = np.argmax(exceed, axis=1).astype(np.int64)
    dt = np.where(any_alarm, first, C.CENSOR_DT)
    return dt, any_alarm


def clip_false_alarm_rate(scores_eval: np.ndarray, thr: float) -> float:
    _, event = alarm_times(scores_eval, thr)
    return float(event.mean())


def wilcoxon_paired(dt_probe: np.ndarray, dt_base: np.ndarray) -> dict:
    """Paired signed-rank test. Censored clips enter at CENSOR_DT (conservative)."""
    d = np.asarray(dt_probe, float) - np.asarray(dt_base, float)
    n_eff = int(np.sum(d != 0))
    out = {
        "n": int(d.size), "n_nonzero": n_eff,
        "median_diff": float(np.median(d)),
        "mean_diff": float(np.mean(d)),
        "frac_probe_earlier": float(np.mean(d < 0)),
        "frac_tied": float(np.mean(d == 0)),
    }
    if n_eff == 0:
        out.update(p_one_sided=1.0, p_two_sided=1.0, statistic=float("nan"))
        return out
    r_less = sps.wilcoxon(dt_probe, dt_base, alternative="less")
    r_two = sps.wilcoxon(dt_probe, dt_base, alternative="two-sided")
    out.update(statistic=float(r_two.statistic),
               p_one_sided=float(r_less.pvalue),
               p_two_sided=float(r_two.pvalue))
    return out


def kaplan_meier(dt: np.ndarray, event: np.ndarray):
    """KM estimate of P(still undetected at time t). Returns (times, surv)."""
    dt = np.asarray(dt, np.int64); event = np.asarray(event, bool)
    times = np.arange(0, C.CENSOR_DT + 1)
    n = dt.size
    at_risk, surv, s = n, [], 1.0
    for t in times:
        d = int(np.sum((dt == t) & event))
        c = int(np.sum((dt == t) & ~event))
        if at_risk > 0 and d > 0:
            s *= 1.0 - d / at_risk
        surv.append(s)
        at_risk -= (d + c)
    return times, np.asarray(surv)


def logrank(dt1, e1, dt2, e2) -> dict:
    """Two-sample log-rank test.

    Secondary, censoring-aware test. Note the two monitor streams come from the
    same clips, so the independence assumption is violated and this is reported
    as corroborative only; the paired Wilcoxon is primary.
    """
    dt1 = np.asarray(dt1, np.int64); dt2 = np.asarray(dt2, np.int64)
    e1 = np.asarray(e1, bool); e2 = np.asarray(e2, bool)
    n1, n2 = dt1.size, dt2.size
    O1 = E1 = V = 0.0
    for t in range(0, C.CENSOR_DT + 1):
        if n1 <= 0 and n2 <= 0:
            break
        d1 = float(np.sum((dt1 == t) & e1)); d2 = float(np.sum((dt2 == t) & e2))
        d, n = d1 + d2, float(n1 + n2)
        if d > 0 and n > 1 and n1 > 0 and n2 > 0:
            O1 += d1
            E1 += d * n1 / n
            V += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
        n1 -= int(d1 + np.sum((dt1 == t) & ~e1))
        n2 -= int(d2 + np.sum((dt2 == t) & ~e2))
    if V <= 0:
        return {"chi2": float("nan"), "p": 1.0, "observed_1": O1, "expected_1": E1}
    chi2 = (O1 - E1) ** 2 / V
    return {"chi2": float(chi2), "p": float(sps.chi2.sf(chi2, 1)),
            "observed_1": O1, "expected_1": E1}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjustment (prereg Sec. 3, multiplicity)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def summarise(dt: np.ndarray, event: np.ndarray) -> dict:
    dt = np.asarray(dt, float)
    return {
        "median_dt": float(np.median(dt)),
        "mean_dt": float(np.mean(dt)),
        "q25_dt": float(np.percentile(dt, 25)),
        "q75_dt": float(np.percentile(dt, 75)),
        "detection_rate": float(np.mean(event)),
        "n": int(dt.size),
    }


# ---------------------------------------------------------------------------
# Secondary analyses added 2026-10-06, before any V-JEPA 2 inference run.
# Motivation (see DEVIATIONS.md): the pre-registered primary is a median over a
# right-censored distribution, which is degenerate whenever fewer than half the
# clips alarm inside the window; and a 1% per-frame threshold compounds to a
# double-digit clip-level false-alarm rate over a 15-frame window.
# ---------------------------------------------------------------------------

def threshold_clip_fpr(scores_calib: np.ndarray, target: float = 0.05) -> float:
    """Threshold giving a clip-level false-alarm rate of `target` on calibration.

    Operationally the relevant calibration for a monitor that raises one alarm
    per episode, as opposed to matching the per-frame rate.
    """
    s = np.asarray(scores_calib, dtype=np.float64)[:, EVAL]
    filled = np.where(np.isfinite(s), s, -np.inf)
    per_clip_max = filled.max(axis=1)
    per_clip_max = per_clip_max[np.isfinite(per_clip_max)]
    if per_clip_max.size == 0:
        raise ValueError("no finite calibration scores in the evaluation window")
    return float(np.percentile(per_clip_max, 100.0 * (1.0 - target)))


def rmta(dt: np.ndarray, event: np.ndarray, horizon: int = C.CENSOR_DT) -> float:
    """Restricted mean time to alarm, E[min(T, horizon)], from the KM curve.

    Censoring-robust: well defined at any detection rate, unlike the median.
    Lower is earlier.
    """
    times, surv = kaplan_meier(dt, event)
    return float(np.sum(surv[times < horizon]))


def rmta_bootstrap_ci(dt: np.ndarray, event: np.ndarray, n_boot: int = 2000,
                      seed: int = 0, horizon: int = C.CENSOR_DT):
    """Percentile bootstrap CI for RMTA over clips."""
    rng = np.random.default_rng(seed)
    n = dt.size
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = rmta(dt[idx], event[idx], horizon)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_rmta_diff_test(dt_a, ev_a, dt_b, ev_b, n_boot: int = 2000,
                          seed: int = 0, horizon: int = C.CENSOR_DT) -> dict:
    """Paired bootstrap over clips for RMTA(a) - RMTA(b). Negative = a earlier."""
    dt_a, dt_b = np.asarray(dt_a), np.asarray(dt_b)
    ev_a, ev_b = np.asarray(ev_a), np.asarray(ev_b)
    point = rmta(dt_a, ev_a, horizon) - rmta(dt_b, ev_b, horizon)
    rng = np.random.default_rng(seed)
    n = dt_a.size
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)   # same resample for both: preserves pairing
        diffs[b] = (rmta(dt_a[idx], ev_a[idx], horizon)
                    - rmta(dt_b[idx], ev_b[idx], horizon))
    p_one = float(np.mean(diffs >= 0.0))
    return {
        "rmta_diff": point,
        "ci95": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
        "p_one_sided_boot": max(p_one, 1.0 / n_boot),
    }
