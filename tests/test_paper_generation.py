"""The paper generator must never assert a check it did not perform.

A missing diagnostic key in an older results.json means 'not recorded'. If it
silently defaults to a benign value, the generated prose will claim the check
passed. That bug existed and these tests exist because of it.
"""
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
SPEC = os.path.join(HERE, "..", "scripts", "make_paper.py")


def _load():
    spec = importlib.util.spec_from_file_location("make_paper", SPEC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_paper"] = mod
    spec.loader.exec_module(mod)
    return mod


mp = _load()


def _monitor(det, fpr, lift=None):
    rec = {"detection_rate": det, "clip_fpr_control_eval": fpr,
           "median_dt": 14.0, "rmta": 12.0, "rmta_ci95": [11.0, 13.0],
           "by_condition": {}}
    if lift is not None:
        rec["lift_over_fpr"] = lift
    return rec


def test_lift_is_recomputed_when_absent():
    rec = _monitor(0.062, 0.080)                 # no stored lift
    assert mp.lift_of(rec) == pytest.approx(-0.018)


def test_stored_lift_is_preferred():
    rec = _monitor(0.5, 0.1, lift=0.4)
    assert mp.lift_of(rec) == pytest.approx(0.4)


def test_missing_diagnostic_does_not_become_a_passing_claim():
    """The exact failure: an old JSON must still flag an at-chance baseline."""
    pm = {"loss_L0": _monitor(0.062, 0.080),     # below chance
          "loss_L1": _monitor(0.346, 0.060)}
    gate = {}                                     # key absent, as in old runs
    assert mp.at_chance_baselines(gate, pm) == ["loss_L0"]


def test_stored_at_chance_list_is_respected():
    pm = {"loss_L0": _monitor(0.9, 0.05, lift=0.85),
          "loss_L1": _monitor(0.9, 0.05, lift=0.85)}
    assert mp.at_chance_baselines({"baselines_firing_at_chance": []}, pm) == []


def test_healthy_baselines_are_not_flagged():
    pm = {"loss_L0": _monitor(0.80, 0.05),
          "loss_L1": _monitor(0.75, 0.05)}
    assert mp.at_chance_baselines({}, pm) == []


def test_latex_escaping_of_monitor_names():
    assert mp.esc("probe_A_vs_loss_L0") == r"probe\_A\_vs\_loss\_L0"
    assert mp.esc("30%") == r"30\%"          # escaped, not stripped
    assert mp.esc("a&b") == r"a\&b"


def test_pending_removes_a_stale_survival_figure(tmp_path, monkeypatch):
    """A figure must not outlive the run it came from.

    Otherwise a survival curve from an earlier run sits in the PDF next to
    PENDING tables and reads as a result.
    """
    monkeypatch.setattr(mp, "OUT", str(tmp_path))
    stale = tmp_path / "fig_survival.png"
    stale.write_bytes(b"not really a png")
    mp.write_pending()
    assert not stale.exists()


def test_pending_macros_cover_everything_main_tex_uses(tmp_path, monkeypatch):
    """Every \\lgw macro the manuscript uses must be defined with no results.

    Otherwise the draft fails to compile exactly when there are no results,
    which is when it is most likely to be built.

    This runs the generators that always run, rather than maintaining a list of
    exceptions -- an exclusion list silently rots every time a generator is
    added, and the point of the test is to catch exactly that.
    """
    import importlib.util
    import re

    monkeypatch.setattr(mp, "OUT", str(tmp_path))
    mp.write_pending()
    mp.pilot_macros(pilot_dir=str(tmp_path / "no_such_dir"))
    mp.precision_table(path=str(tmp_path / "no_such_file.json"))

    spec = importlib.util.spec_from_file_location(
        "make_figures", os.path.join(HERE, "..", "scripts", "make_figures.py"))
    mf = importlib.util.module_from_spec(spec)
    sys.modules["make_figures"] = mf
    spec.loader.exec_module(mf)
    monkeypatch.setattr(mf, "OUT", str(tmp_path))
    mf.dataset_macros(n=2)        # tiny: we only need the macro names

    defined = set()
    for f in tmp_path.glob("*.tex"):
        defined |= set(re.findall(r"\\newcommand\{\\(lgw\w+)\}", f.read_text()))
    used = set()
    paper = os.path.join(HERE, "..", "paper")
    for root, _, files in os.walk(paper):
        if "generated" in root:
            continue
        for fn in files:
            if fn.endswith(".tex"):
                with open(os.path.join(root, fn), encoding="utf-8") as f:
                    used |= set(re.findall(r"\\(lgw[A-Za-z]+)", f.read()))
    used -= {"lgwHasResults"}     # set by write_pending as a \newcommand too
    missing = used - defined
    assert not missing, "macros used by the paper but missing in PENDING mode: {}".format(
        sorted(missing))
