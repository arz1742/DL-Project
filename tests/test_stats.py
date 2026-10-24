"""tests/test_stats.py — paired bootstrap, Holm correction and CLIPScore formulas."""

import os
import sys
import importlib.util

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sig = _load("significance")
cs = _load("clipscore")


def test_bootstrap_detects_real_gap_and_not_noise():
    rng = np.random.default_rng(0)
    b = rng.normal(0.4, 0.2, 1000)
    real = sig.paired_bootstrap(b + 0.05 + rng.normal(0, 0.05, 1000), b, n_boot=2000)
    noise = sig.paired_bootstrap(b + rng.normal(0, 0.05, 1000), b, n_boot=2000)
    assert real["p"] < 0.01 and real["lo"] > 0
    assert noise["lo"] < 0 < noise["hi"]


def test_holm_is_monotone_and_capped():
    adj = sig.holm([0.01, 0.04, 0.03, 0.5])
    assert abs(adj[0] - 0.04) < 1e-12 and all(a <= 1 for a in adj)
    assert adj[0] <= adj[2] <= adj[1] <= adj[3]


def test_clipscore_formulas():
    img = cs._norm(np.array([[1.0, 0.0], [0.0, 1.0]]))
    cap = cs._norm(np.array([[1.0, 0.0], [1.0, 0.0]]))
    s = cs.clip_s(cap, img)
    assert np.allclose(s, [2.5, 0.0])                     # perfect match, orthogonal -> 0
    refs = np.stack([cap, cap], axis=1)                   # references identical to the caption
    r = cs.refclip_s(s, cap, refs)
    assert np.isclose(r[0], 2 * 2.5 * 1 / 3.5) and r[1] == 0.0
