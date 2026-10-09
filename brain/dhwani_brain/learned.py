# -*- coding: utf-8 -*-
"""Learned candidate chooser: pure NumPy inference of a small DeepSets network.

    per candidate  e_i = phi(c_i)                 2-layer ReLU MLP, shared weights
    pooled         [mean_i e_i, max_i e_i]        over PRESENT candidates (order-free)
    context        h = psi([g, pooled])           2-layer ReLU MLP on the global vector
    choose         logit_i = rho([e_i, h])        masked softmax over ALLOWED candidates
    sensor         dwell_i = sigmoid(tau([e_i, h]))  DWELL_SCAN if > 0.5 for the chosen one
    confidence     Platt scaling of p(slot) x p(dwell choice), fitted on validation mines

The .npz holds only float32 arrays plus a uint8 JSON blob ("meta"), so it loads on numpy 1.13
under python 2.7 without pickle. No GRU: the per-decision features already carry the history
that matters (time since contact, odometry, previous goal, trail), and a recurrent state would
be one more thing that can silently desynchronise when the executive restarts.
"""
from __future__ import division, print_function, absolute_import

import json
import os

import numpy as np

LAYERS = ("phi1", "phi2", "psi1", "psi2", "rho1", "rho2", "tau1", "tau2")


def default_weights_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "policy.npz")


def _relu(x):
    return np.maximum(x, 0.0)


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))            # = 1 / (1 + exp(-x)), cannot overflow


class LearnedPolicy(object):
    def __init__(self, path=None):
        path = path or default_weights_path()
        z = np.load(path)
        files = set(z.files)
        self.w = {}
        for name in LAYERS:
            if name + "_W" not in files:
                raise ValueError("%s is missing layer %s" % (path, name))
            self.w[name] = (z[name + "_W"].astype(np.float32), z[name + "_b"].astype(np.float32))
        self.c_mean = z["c_mean"].astype(np.float32)
        self.c_std = np.maximum(z["c_std"].astype(np.float32), 1e-3)
        self.g_mean = z["g_mean"].astype(np.float32)
        self.g_std = np.maximum(z["g_std"].astype(np.float32), 1e-3)
        self.calib_a = float(z["calib_a"]) if "calib_a" in files else 1.0
        self.calib_b = float(z["calib_b"]) if "calib_b" in files else 0.0
        self.meta = json.loads(z["meta"].tobytes().decode("utf-8")) if "meta" in files else {}
        self.n_params = int(sum(W.size + b.size for W, b in self.w.values()))
        self.path = path

    def _lin(self, name, x):
        W, b = self.w[name]
        return x.dot(W) + b

    def forward(self, C, present, allowed, G):
        """C: K x NC, present/allowed: K bool, G: NG. Returns (probs K, dwell_prob K)."""
        C = (np.asarray(C, dtype=np.float32) - self.c_mean) / self.c_std
        G = (np.asarray(G, dtype=np.float32) - self.g_mean) / self.g_std
        e = _relu(self._lin("phi2", _relu(self._lin("phi1", C))))            # K x H
        pm = present.astype(np.float32)[:, None]
        n = max(float(pm.sum()), 1.0)
        mean = (e * pm).sum(axis=0) / n
        mx = np.where(present[:, None], e, -np.inf).max(axis=0) if present.any() else np.zeros(e.shape[1], np.float32)
        h = _relu(self._lin("psi2", _relu(self._lin("psi1", np.concatenate([G, mean, mx])))))
        eh = np.concatenate([e, np.repeat(h[None, :], e.shape[0], axis=0)], axis=1)
        logit = self._lin("rho2", _relu(self._lin("rho1", eh)))[:, 0]
        dwell = self._lin("tau2", _relu(self._lin("tau1", eh)))[:, 0]
        logit = np.where(allowed, logit, -np.inf)
        if not allowed.any():
            return np.zeros(len(allowed), np.float32), _sigmoid(dwell).astype(np.float32)
        m = logit[allowed].max()
        ex = np.where(allowed, np.exp(logit - m), 0.0)
        probs = ex / ex.sum()
        return probs.astype(np.float32), _sigmoid(dwell).astype(np.float32)

    def confidence(self, p):
        p = min(max(float(p), 1e-6), 1.0 - 1e-6)
        z = self.calib_a * np.log(p / (1.0 - p)) + self.calib_b
        return float(_sigmoid(z))

    def act(self, C, present, allowed, G):
        """Greedy action (slot, dwell). The confidence is calibrated on the probability of the
        WHOLE action: p(slot) x p(dwell choice), since the rules can differ in either part."""
        probs, dwell = self.forward(C, present, allowed, G)
        k = int(np.argmax(probs))
        dw = bool(dwell[k] > 0.5)
        p_joint = float(probs[k]) * (float(dwell[k]) if dw else 1.0 - float(dwell[k]))
        return k, dw, self.confidence(p_joint), probs, dwell
