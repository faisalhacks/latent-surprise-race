"""Render every number in the paper from a results.json.

No figure, table or numeric sentence in the manuscript is written by hand. This
script reads a run's `results.json` and emits LaTeX into `paper/generated/`:

    macros.tex    setup constants and the gate verdict, as \\newcommand
    table_*.tex   the result tables
    findings.tex  the numeric prose of the results section, derived from data
    fig_survival.png

If no run has been analysed yet, every macro renders as PENDING, the tables
become visible placeholders and the paper compiles with a DRAFT banner. That is
deliberate: a paper skeleton that silently contains invented numbers is worse
than one that is obviously incomplete.

Usage:
    python scripts/make_paper.py [path/to/results.json]
    python scripts/make_paper.py --list
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lgw import config as C                       # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "paper", "generated")
PENDING = r"\textsc{pending}"

PRETTY = {
    "loss_L0": r"$\mathcal{L}_0$ (V-JEPA 2 own loss)",
    "loss_L1": r"$\mathcal{L}_1$ (ridge, info-matched)",
    "probe_A": r"$\mathrm{A}$: curvature",
    "probe_B": r"$\mathrm{B}$: repr.\ distance",
    "probe_C": r"$\mathrm{C}$: covariance shift",
}
COND_PRETTY = {
    "gravity_flip": "gravity inv.",
    "friction_high": "friction",
    "occlusion": "occlusion",
    "frame_drop": "frame drop",
}


def esc(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"),
                 ("%", r"\%"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def find_results(explicit: str | None) -> str | None:
    if explicit:
        return explicit if os.path.exists(explicit) else None
    cands = [p for p in glob.glob(os.path.join(ROOT, "results", "*", "analysis",
                                               "results.json"))
             + glob.glob(os.path.join(ROOT, "results", "*", "results.json"))
             if "surrogate" not in p and "archdry" not in p]
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


# ---------------------------------------------------------------- placeholders
def write_pending() -> None:
    m = [r"\newcommand{\lgwHasResults}{0}"]
    for k in ("lgwModel", "lgwNAnom", "lgwNCalib", "lgwNEval", "lgwDim",
              "lgwNBoot", "lgwVerdict", "lgwEarlyWarners", "lgwRmtaWarners",
              "lgwMedianDegenerate", "lgwBaselinesAtChance", "lgwVacuous",
              "lgwSource", "lgwNote"):
        m.append(r"\newcommand{\%s}{%s}" % (k, PENDING))
    _w("macros.tex", "\n".join(m))
    ph = (r"\begin{center}\fbox{\parbox{0.8\linewidth}{\centering\vspace{4pt}"
          r"\textsc{pending}\\[2pt]\footnotesize No inference run has been "
          r"analysed yet. Regenerate with \texttt{python scripts/make\_paper.py}"
          r"\vspace{4pt}}}\end{center}")
    for f in ("table_tta.tex", "table_cmp.tex", "table_cond.tex",
              "table_secondary.tex"):
        _w(f, ph)
    _w("findings.tex",
       "No inference run has been analysed yet, so this section is empty by "
       "construction rather than by omission. Running the harness repopulates "
       "every number in it automatically.")
    print("no results found -> wrote PENDING placeholders")


def _w(name: str, body: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(body + "\n")


# ---------------------------------------------------------------------- tables
def table_tta(g: dict) -> str:
    L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Monitor & median $\Delta t$ & RMTA & 95\% CI & detect. & "
         r"clip FPR & lift \\", r"\midrule"]
    for m in C.MONITORS:
        d = g["per_monitor"][m]
        lo, hi = d["rmta_ci95"]
        flag = r"\,$\dagger$" if d["fires_at_chance"] else ""
        L.append(r"{} & {:.1f} & {:.2f} & [{:.2f}, {:.2f}] & {:.3f} & {:.3f} & "
                 r"{:+.3f}{} \\".format(PRETTY[m], d["median_dt"], d["rmta"],
                                        lo, hi, d["detection_rate"],
                                        d["clip_fpr_control_eval"],
                                        d["lift_over_fpr"], flag))
        if m == "loss_L1":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def table_cmp(g: dict) -> str:
    L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Comparison & median gain & RMTA gain & 95\% CI & $p$ (Holm) & "
         r"$p$ (Holm, RMTA) \\", r"\midrule"]
    for k, c in g["comparisons"].items():
        lo, hi = c["rmta"]["ci95"]
        L.append(r"\texttt{{{}}} & {:+.1f} & {:+.2f} & [{:+.2f}, {:+.2f}] & "
                 r"{} & {} \\".format(
                     esc(k), c["median_gain_frames"], c["rmta_gain_frames"],
                     -hi, -lo, _p(c["p_holm"]), _p(c["p_holm_rmta"])))
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def _p(v: float) -> str:
    if v < 1e-4:
        return r"$<10^{-4}$"
    return "{:.3g}".format(v)


def table_cond(g: dict) -> str:
    cols = "l" + "r" * len(C.CONDITIONS)
    L = [r"\begin{tabular}{%s}" % cols, r"\toprule",
         "Monitor & " + " & ".join(COND_PRETTY[c] for c in C.CONDITIONS) + r" \\",
         r"\midrule"]
    for m in C.MONITORS:
        vals = [g["per_monitor"][m]["by_condition"][c]["rmta"]
                for c in C.CONDITIONS]
        L.append("{} & ".format(PRETTY[m])
                 + " & ".join("{:.2f}".format(v) for v in vals) + r" \\")
        if m == "loss_L1":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def table_secondary(g: dict) -> str:
    L = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Monitor & median $\Delta t$ & RMTA & detect. & clip FPR \\",
         r"\midrule"]
    for m in C.MONITORS:
        d = g["per_monitor"][m]
        L.append(r"{} & {:.1f} & {:.2f} & {:.3f} & {:.3f} \\".format(
            PRETTY[m], d["median_dt"], d["rmta"], d["detection_rate"],
            d["clip_fpr_control_eval"]))
        if m == "loss_L1":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


# -------------------------------------------------------------------- findings
def findings(r: dict) -> str:
    g = r["regimes"]["per_frame_matched"]
    pm = g["per_monitor"]
    gate = r["gate"]
    S = []

    base_lines = []
    for b in C.BASELINES:
        d = pm[b]
        base_lines.append(
            r"\texttt{{{}}} alarmed on {:.1%} of anomalous clips against a "
            r"held-out clip-level false-alarm rate of {:.1%} (lift {:+.3f})"
            .format(esc(b), d["detection_rate"], d["clip_fpr_control_eval"],
                    d["lift_over_fpr"]).replace("%", r"\%"))
    S.append("At the pre-registered operating point, " + "; ".join(base_lines)
             + ".")

    at_chance = gate["baselines_firing_at_chance"]
    if at_chance:
        S.append(
            r"\textbf{{Baseline(s) {} do not detect anomalies above their own "
            r"false-alarm rate.}} Any comparison against them is therefore "
            r"uninformative about H1: a probe can only be shown to warn "
            r"\emph{{earlier}} than a signal that warns at all.".format(
                ", ".join(r"\texttt{" + esc(b) + "}" for b in at_chance)))
    else:
        S.append("Both baselines detect anomalies above chance, so the "
                 "comparison against them is meaningful.")

    if g["median_degenerate"]:
        S.append(
            r"The pre-registered primary metric is \emph{degenerate} in this "
            r"run: fewer than half the clips alarm inside the 15-frame window "
            r"for every monitor, so the median Time-to-Alarm is pinned at the "
            r"censoring value and cannot separate them whatever the true "
            r"effect. The RMTA columns carry the evidence instead.")

    best, best_gain = None, -1e9
    for pr in C.PROBES:
        gains = [g["comparisons"][pr + "_vs_" + b]["rmta_gain_frames"]
                 for b in C.BASELINES]
        if min(gains) > best_gain:
            best, best_gain = pr, min(gains)
    if best is not None:
        c0 = g["comparisons"][best + "_vs_loss_L0"]
        lo, hi = c0["rmta"]["ci95"]
        S.append(
            r"The strongest probe by RMTA is \texttt{{{}}}, whose worst-case "
            r"gain against either baseline is {:+.2f} frames; against "
            r"\texttt{{loss\_L0}} its RMTA gain is {:+.2f} frames "
            r"(95\% CI [{:+.2f}, {:+.2f}], Holm-adjusted $p={}$).".format(
                esc(best), best_gain, c0["rmta_gain_frames"], -hi, -lo,
                _p(c0["p_holm_rmta"])))

    ew = gate["early_warners"]
    if ew:
        S.append(r"Under the frozen gate, {} {} an Early Warner.".format(
            ", ".join(r"\texttt{" + esc(p) + "}" for p in ew),
            "is" if len(ew) == 1 else "are"))
    else:
        S.append("No probe satisfies the frozen Early Warner criterion "
                 "(a median gain of at least one frame against both baselines "
                 "at Holm-adjusted $p<0.01$).")
    if gate["verdict_is_vacuous"]:
        S.append(r"\textbf{This PROCEED verdict is vacuous} and must not be "
                 r"used to advance to Phase 4: the probe cleared the gate only "
                 r"against a baseline that fires at chance.")

    sec = r["regimes"]["clip_matched_5pct"]
    S.append(
        r"Re-deriving thresholds so the clip-level false-alarm rate is 5\% "
        r"rather than matching the per-frame rate {} the picture: the Early "
        r"Warner set becomes {}.".format(
            "does not change" if (sec["early_warners_median"]
                                  == g["early_warners_median"]) else "changes",
            ", ".join(r"\texttt{" + esc(p) + "}"
                      for p in sec["early_warners_median"]) or "empty"))
    return "\n\n".join(S)


# ------------------------------------------------------------------------ main
def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--list" in sys.argv:
        for p in glob.glob(os.path.join(ROOT, "results", "*", "**",
                                        "results.json"), recursive=True):
            print(os.path.relpath(p, ROOT))
        return

    path = find_results(args[0] if args else None)
    os.makedirs(OUT, exist_ok=True)
    if path is None:
        write_pending()
        return

    with open(path, encoding="utf-8") as f:
        r = json.load(f)
    g = r["regimes"]["per_frame_matched"]
    gate = r["gate"]

    macros = [
        r"\newcommand{\lgwHasResults}{1}",
        r"\newcommand{\lgwModel}{\texttt{%s}}" % esc(str(r["model"])),
        r"\newcommand{\lgwNAnom}{%d}" % r["n_anomaly"],
        r"\newcommand{\lgwNCalib}{%d}" % r["n_control_calib"],
        r"\newcommand{\lgwNEval}{%d}" % r["n_control_eval"],
        r"\newcommand{\lgwDim}{%d}" % r["latent_dim"],
        r"\newcommand{\lgwNBoot}{%d}" % r.get("n_boot", 2000),
        r"\newcommand{\lgwVerdict}{\textsc{%s}}" % esc(
            gate["verdict"].replace("_", " ").lower()),
        r"\newcommand{\lgwEarlyWarners}{%s}" % (
            ", ".join(r"\texttt{" + esc(p) + "}"
                      for p in gate["early_warners"]) or "none"),
        r"\newcommand{\lgwRmtaWarners}{%s}" % (
            ", ".join(r"\texttt{" + esc(p) + "}"
                      for p in gate["secondary_rmta_early_warners"]) or "none"),
        r"\newcommand{\lgwMedianDegenerate}{%s}" % (
            "yes" if gate["primary_median_degenerate"] else "no"),
        r"\newcommand{\lgwBaselinesAtChance}{%s}" % (
            ", ".join(r"\texttt{" + esc(b) + "}"
                      for b in gate["baselines_firing_at_chance"]) or "none"),
        r"\newcommand{\lgwVacuous}{%s}" % (
            "yes" if gate["verdict_is_vacuous"] else "no"),
        r"\newcommand{\lgwSource}{\texttt{%s}}" % esc(
            os.path.relpath(path, ROOT).replace("\\", "/")),
        r"\newcommand{\lgwNote}{%s}" % esc(str(r.get("note", ""))),
    ]
    _w("macros.tex", "\n".join(macros))
    _w("table_tta.tex", table_tta(g))
    _w("table_cmp.tex", table_cmp(g))
    _w("table_cond.tex", table_cond(g))
    _w("table_secondary.tex", table_secondary(r["regimes"]["clip_matched_5pct"]))
    _w("findings.tex", findings(r))

    src = os.path.join(os.path.dirname(path), "survival_curves.png")
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(OUT, "fig_survival.png"))

    print("rendered paper/generated/ from {}".format(
        os.path.relpath(path, ROOT)))
    print("  verdict={}  early warners={}  vacuous={}".format(
        gate["verdict"], gate["early_warners"] or "none",
        gate["verdict_is_vacuous"]))


if __name__ == "__main__":
    main()
