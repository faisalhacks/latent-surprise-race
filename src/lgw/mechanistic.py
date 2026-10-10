"""Phase 4.3 - what is represented in the lag window.

If a geometry probe alarms before the prediction loss does, the frames in
between are the interesting ones: the representation has registered something
the output error has not. This module asks what that something is, by fitting
linear probes to the latents in exactly those frames.

Two things make a probe like this easy to over-read, and both are handled here:

* **Frame leakage.** Frames from one clip are highly correlated. Splitting them
  randomly across train and test lets the probe memorise clip identity and
  report a near-perfect score that means nothing. All cross-validation is
  grouped by clip.
* **No reference point.** An accuracy is uninterpretable without knowing what
  chance looks like. Every result is reported next to the majority-class rate
  *and* a label-shuffled control, which must land at chance; if it does not,
  the split is leaking and the real score should not be believed.

The module is written so it runs identically on whichever lag window the gate
produces, including an empty one -- in which case it reports that, rather than
silently probing the wrong frames.
"""
from __future__ import annotations

import numpy as np

from . import config as C


def lag_window_slices(dt_fast: np.ndarray, ev_fast: np.ndarray,
                      dt_slow: np.ndarray, ev_slow: np.ndarray,
                      require_both: bool = False) -> list:
    """Per clip, the absolute frame range strictly between two alarms.

    `[15 + dt_fast, 15 + dt_slow)` for clips where the fast monitor alarmed and
    did so strictly earlier. If `require_both`, clips where the slow monitor
    never alarmed are dropped; otherwise its alarm is taken at the censoring
    bound, which is the honest reading of "it had not fired yet".
    """
    out = []
    for i in range(len(dt_fast)):
        if not ev_fast[i]:
            out.append(None)
            continue
        if not ev_slow[i]:
            if require_both:
                out.append(None)
                continue
            hi_dt = C.CENSOR_DT + 1
        else:
            hi_dt = dt_slow[i]
        lo, hi = C.EVAL_START + int(dt_fast[i]), C.EVAL_START + int(hi_dt)
        out.append((lo, hi) if hi > lo else None)
    return out


def collect_frames(Z: np.ndarray, slices: list, labels: np.ndarray):
    """-> (X (n_frames, D), y (n_frames,), groups (n_frames,) clip indices)."""
    xs, ys, gs = [], [], []
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        lo, hi = sl
        hi = min(hi, Z.shape[1])
        if hi <= lo:
            continue
        xs.append(Z[i, lo:hi])
        ys.append(np.repeat(labels[i], hi - lo) if np.ndim(labels[i]) == 0
                  else np.tile(labels[i], (hi - lo, 1)))
        gs.append(np.full(hi - lo, i))
    if not xs:
        D = Z.shape[2]
        return (np.zeros((0, D)), np.zeros((0,)), np.zeros((0,), dtype=int))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(gs)


def _cv_scores(X, y, groups, task, n_splits, seed, shuffle_labels=False):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y)
    if shuffle_labels:
        rng = np.random.default_rng(seed)
        uniq = np.unique(groups)
        perm = rng.permutation(uniq)
        mapping = dict(zip(uniq, perm))
        # permute labels at the CLIP level, not the frame level: shuffling
        # frames within a clip would leave the clip-identity signal intact and
        # the control would not control for anything
        first = {g: y[groups == g][0] for g in uniq}
        y = np.array([first[mapping[g]] for g in groups])

    n_groups = len(np.unique(groups))
    k = int(min(n_splits, n_groups))
    if k < 2:
        return float("nan")
    scores = []
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        if task == "classification" and len(np.unique(y[tr])) < 2:
            continue
        est = (LogisticRegression(max_iter=2000, C=1.0)
               if task == "classification" else Ridge(alpha=1.0))
        model = make_pipeline(StandardScaler(), est)
        model.fit(X[tr], y[tr])
        scores.append(float(model.score(X[te], y[te])))
    return float(np.mean(scores)) if scores else float("nan")


def linear_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 task: str = "classification", n_splits: int = 5,
                 seed: int = 0) -> dict:
    """Clip-grouped cross-validated linear probe, with its own controls."""
    rec = {"n_frames": int(X.shape[0]),
           "n_clips": int(len(np.unique(groups))) if groups.size else 0,
           "task": task}
    if X.shape[0] < 10 or rec["n_clips"] < 4:
        rec.update(score=float("nan"), chance=float("nan"),
                   shuffled=float("nan"),
                   note="too few frames or clips in the lag window to probe")
        return rec

    rec["score"] = _cv_scores(X, y, groups, task, n_splits, seed)
    rec["shuffled"] = _cv_scores(X, y, groups, task, n_splits, seed,
                                 shuffle_labels=True)
    if task == "classification":
        vals, counts = np.unique(y, return_counts=True)
        rec["chance"] = float(counts.max() / counts.sum())
        rec["n_classes"] = int(len(vals))
    else:
        rec["chance"] = 0.0           # R^2 of predicting the mean
    rec["lift_over_chance"] = rec["score"] - rec["chance"]
    rec["control_ok"] = bool(abs(rec["shuffled"] - rec["chance"]) < 0.15)
    if not rec["control_ok"]:
        rec["note"] = ("label-shuffled control did not land at chance; the "
                       "split is leaking and the score is not trustworthy")
    return rec


def analyse_lag_window(Z: np.ndarray, conditions: np.ndarray, slices: list,
                       traj: np.ndarray | None = None, seed: int = 0) -> dict:
    """Probe what the latents encode in the lag window.

    `Z` (N, T, D); `conditions` (N,) strings; `slices` from
    `lag_window_slices`; `traj` (N, T, 2) disc centres if available.
    """
    out = {"n_clips_with_lag": int(sum(s is not None for s in slices))}
    lengths = [hi - lo for s in slices if s is not None for lo, hi in [s]]
    out["lag_frames_median"] = float(np.median(lengths)) if lengths else 0.0
    out["lag_frames_total"] = int(sum(lengths))

    codes = {c: i for i, c in enumerate(C.CONDITIONS)}
    y_cond = np.array([codes.get(c, -1) for c in conditions])
    X, y, g = collect_frames(Z, slices, y_cond)
    out["which_anomaly"] = linear_probe(X, y, g, "classification", seed=seed)

    if traj is not None:
        # velocity at each frame, the quantity the physics anomalies change
        vel = np.diff(traj, axis=1, prepend=traj[:, :1])
        for name, arr in (("disc_x", traj[:, :, 0]), ("disc_y", traj[:, :, 1]),
                          ("vel_x", vel[:, :, 0]), ("vel_y", vel[:, :, 1])):
            xs, ys, gs = [], [], []
            for i, sl in enumerate(slices):
                if sl is None:
                    continue
                lo, hi = sl
                hi = min(hi, Z.shape[1], arr.shape[1])
                if hi <= lo:
                    continue
                xs.append(Z[i, lo:hi]); ys.append(arr[i, lo:hi])
                gs.append(np.full(hi - lo, i))
            if xs:
                out[name] = linear_probe(np.concatenate(xs),
                                         np.concatenate(ys),
                                         np.concatenate(gs),
                                         "regression", seed=seed)
    return out


def format_report(res: dict) -> str:
    L = ["# Mechanistic analysis of the lag window", ""]
    L.append("Clips with a lag window: {}".format(res["n_clips_with_lag"]))
    L.append("Median lag length: {:.1f} frames ({} frames total)".format(
        res["lag_frames_median"], res["lag_frames_total"]))
    L.append("")
    L.append("| target | task | score | chance | shuffled control | lift | control ok |")
    L.append("|---|---|---|---|---|---|---|")
    for k, v in res.items():
        if not isinstance(v, dict):
            continue
        L.append("| `{}` | {} | {:.3f} | {:.3f} | {:.3f} | {:+.3f} | {} |".format(
            k, v.get("task", "-"), v.get("score", float("nan")),
            v.get("chance", float("nan")), v.get("shuffled", float("nan")),
            v.get("lift_over_chance", float("nan")),
            "yes" if v.get("control_ok") else "NO"))
    notes = [(k, v["note"]) for k, v in res.items()
             if isinstance(v, dict) and "note" in v]
    if notes:
        L.append("")
        for k, n in notes:
            L.append("- `{}`: {}".format(k, n))
    return "\n".join(L) + "\n"
