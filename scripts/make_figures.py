"""Generate the paper's result-independent figures.

  fig_dataset.png  - the benchmark: one row per condition, injection marked
  fig_windows.png  - the causal construction: what frame t may see, and why the
                     L0 context needs a second encoder pass

Both are regenerated from the frozen dataset code, so they cannot drift from
what the experiment actually ran.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                   # noqa: E402
from matplotlib.patches import Rectangle          # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lgw import config as C                       # noqa: E402
from lgw.datagen import make_clip                 # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "generated")
# 16 and 20 are held frames in the frame_drop condition, so the column set
# makes that condition visible rather than implying it looks like the control.
SHOW_FRAMES = (12, 15, 16, 20, 24)
LABEL = {
    None: "control",
    "gravity_flip": "gravity inverted",
    "friction_high": "friction $\\to$ 0.8",
    "occlusion": "occlusion (30\\%)",
    "frame_drop": "frame drop (hold)",
}


def fig_dataset() -> str:
    rows = [None] + list(C.CONDITIONS)
    fig, axes = plt.subplots(len(rows), len(SHOW_FRAMES),
                             figsize=(1.35 * len(SHOW_FRAMES), 1.35 * len(rows)))
    for r, cond in enumerate(rows):
        frames, _, held = make_clip(7, cond)
        for c, t in enumerate(SHOW_FRAMES):
            ax = axes[r, c]
            ax.imshow(frames[t], cmap="gray", vmin=0, vmax=255,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.6)
                s.set_color("#bbbbbb")
            if t >= C.INJECT_FRAME and cond is not None:
                for s in ax.spines.values():
                    s.set_color("#d62728")
                    s.set_linewidth(1.6)
            if t in held:
                ax.text(0.5, 0.06, "held", transform=ax.transAxes, fontsize=6.5,
                        ha="center", va="bottom", color="#ffd166")
            if r == 0:
                ax.set_title("$t={}$".format(t), fontsize=9, pad=4)
            if c == 0:
                ax.set_ylabel(LABEL[cond].replace("$\\to$", "to").replace("\\%", "%"),
                              fontsize=8, rotation=0, ha="right", va="center",
                              labelpad=8)
    fig.suptitle("Anomalies injected at frame {} (red border = post-injection)"
                 .format(C.INJECT_FRAME), fontsize=9, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    p = os.path.join(OUT, "fig_dataset.png")
    fig.savefig(p, dpi=220)
    plt.close(fig)
    return p


def fig_windows() -> str:
    """Schematic of the two windows used at frame t."""
    t = 21
    n_tub = C.T_CTX // C.TUBELET
    first = t - C.T_CTX + 1
    fig, ax = plt.subplots(figsize=(7.6, 1.95))

    def row(y, pairs, label, colors):
        for j, (lo, hi) in enumerate(pairs):
            ax.add_patch(Rectangle((j, y), 0.9, 0.62, facecolor=colors[j],
                                   edgecolor="#333333", linewidth=0.8))
            ax.text(j + 0.45, y + 0.31, "{},{}".format(lo, hi), ha="center",
                    va="center", fontsize=7.2,
                    color="white" if colors[j] == "#d62728" else "#111111")
        ax.text(-0.22, y + 0.31, label, ha="right", va="center", fontsize=8.5)

    full = [(first + 2 * j, first + 2 * j + 1) for j in range(n_tub)]
    ctx = full[:-1] + [(t - 2, t - 2)]
    row(1.15, full, "$W_t$  (target pass)",
        ["#cfd8e3"] * (n_tub - 1) + ["#d62728"])
    row(0.15, ctx, "$W'_t$  (context pass)",
        ["#7fa8d4"] * (n_tub - 1) + ["#e8e8e8"])

    ax.annotate("", xy=(n_tub - 0.55, 1.12), xytext=(n_tub - 0.55, 0.80),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.1))
    ax.text(n_tub - 0.45, 0.95, " predict", fontsize=8, va="center")
    ax.text(n_tub / 2.0, 2.05, "frame index $t = {}$".format(t), ha="center",
            va="center", fontsize=8.5, style="italic")
    ax.set_xlim(-3.3, n_tub + 1.0)
    ax.set_ylim(-0.12, 2.3)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    p = os.path.join(OUT, "fig_windows.png")
    fig.savefig(p, dpi=220)
    plt.close(fig)
    return p


def dataset_macros(n: int = 250) -> str:
    """Measure what the occlusion condition actually does to the disc.

    A 35x35 square anchored on the path of a 10 px disc does not 'partially
    block' it, and the paper should say what it really tests.
    """
    import numpy as np

    base = C.CONDITIONS.index("occlusion") * C.N_PER_CONDITION
    hidden = []
    for i in range(base, base + n):
        anom, _, _ = make_clip(i, "occlusion")
        ctrl, _, _ = make_clip(i, None)
        for t in range(C.INJECT_FRAME, C.N_FRAMES):
            vis_c = float(ctrl[t].sum())
            hidden.append(1.0 - float(anom[t].sum()) / max(vis_c, 1e-9))
    h = np.asarray(hidden).reshape(n, -1)
    full = float((h > 0.99).mean())
    part = float(((h > 0.01) & (h <= 0.99)).mean())
    all_hidden_clips = float((h > 0.99).all(axis=1).mean())

    def pct(x):
        return "{:.1f}\\%".format(100.0 * x)

    lines = [
        r"\newcommand{\lgwOccFullyHidden}{%s}" % pct(full),
        r"\newcommand{\lgwOccPartial}{%s}" % pct(part),
        r"\newcommand{\lgwOccFullyHiddenClips}{%s}" % pct(all_hidden_clips),
        r"\newcommand{\lgwOccluderArea}{%s}" % pct(
            (int(round((C.OCCLUDER_AREA_FRAC ** 0.5) * C.RES)) ** 2) / C.RES ** 2),
    ]
    p = os.path.join(OUT, "dataset_macros.tex")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return p


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for p in (fig_dataset(), fig_windows(), dataset_macros()):
        print("wrote", os.path.relpath(p))


if __name__ == "__main__":
    main()
