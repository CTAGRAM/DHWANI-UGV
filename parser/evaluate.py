#!/usr/bin/env python3
"""Evaluate the exported models with the NumPy runtime and write eval.json.

    python evaluate.py

For each model (tinybert4, berttiny), on data/val.jsonl and data/test_handwritten.jsonl:
  per-field accuracy (12 mission fields + understood), exact match (all 12 mission fields),
  accept/reject statistics at the model's threshold, the task confusion matrix, and for the
  hand-written set every failure with expected and predicted JSON. Also Mac latency.
Exact match ignores confidence, understood and text. "end_to_end_correct" counts an example as
right when the gold is understood and the parser accepts it with every field right, or the gold is
not understood (UNSUPPORTED / GOTO without a letter) and the parser does not accept it.
"""
import collections
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dhwani_parser import MissionParser, MISSION_FIELDS, TASKS  # noqa: E402

LATENCY_TEXT = ("Go through this mine and find any person alive, report where they are, "
                "and come back if the link becomes unstable")


def load(path):
    with open(os.path.join(HERE, path)) as f:
        return [json.loads(l) for l in f]


def evaluate(p, rows, keep_failures):
    field_ok = collections.Counter()
    conf_mat = collections.OrderedDict((g, collections.OrderedDict((t, 0) for t in TASKS))
                                       for g in TASKS)
    n = len(rows)
    em = e2e = accepted = accepted_ok = gold_und = covered = unsup_accepted = 0
    rejected_correct = []
    failures = []
    for r in rows:
        g = r["label"] if "label" in r else r      # test file: mission JSON at top level
        m = p.parse(r["text"])
        bad = [k for k in MISSION_FIELDS if m[k] != g[k]]
        for k in MISSION_FIELDS + ["understood"]:
            field_ok[k] += int(m[k] == g[k])
        conf_mat[g["task"]][m["task"]] += 1
        full = not bad
        em += int(full)
        if m["understood"]:
            accepted += 1
            accepted_ok += int(full)
            unsup_accepted += int(g["task"] == "UNSUPPORTED")
        if g["understood"]:
            gold_und += 1
            covered += int(m["understood"] and full)
            e2e += int(m["understood"] and full)
        else:
            e2e += int(not m["understood"])
        if keep_failures:
            if bad:
                exp = collections.OrderedDict((k, g[k]) for k in MISSION_FIELDS)
                exp["understood"] = g["understood"]
                failures.append(collections.OrderedDict([
                    ("text", r["text"]), ("wrong_fields", bad), ("expected", exp),
                    ("predicted", m)]))
            elif m["understood"] != g["understood"]:
                rejected_correct.append({"text": r["text"], "confidence": m["confidence"],
                                         "task": m["task"]})
    res = collections.OrderedDict([
        ("n", n),
        ("exact_match", round(em / float(n), 4)),
        ("per_field_accuracy", collections.OrderedDict(
            (k, round(field_ok[k] / float(n), 4)) for k in MISSION_FIELDS + ["understood"])),
        ("accepted", accepted),
        ("accepted_fully_correct", round(accepted_ok / float(max(1, accepted)), 4)),
        ("coverage_of_understood_gold", round(covered / float(max(1, gold_und)), 4)),
        ("unsupported_accepted", unsup_accepted),
        ("end_to_end_correct", round(e2e / float(n), 4)),
        ("task_confusion_rows_gold_cols_pred", conf_mat)])
    if keep_failures:
        res["failures"] = failures
        res["fully_correct_but_rejected_by_threshold"] = rejected_correct
    return res


def latency(p, texts):
    per = []
    for t in texts:
        p.parse(t)
        t0 = time.perf_counter()
        p.parse(t)
        per.append((time.perf_counter() - t0) * 1e3)
    for _ in range(5):
        p.parse(LATENCY_TEXT)
    fixed = []
    for _ in range(50):
        t0 = time.perf_counter()
        p.parse(LATENCY_TEXT)
        fixed.append((time.perf_counter() - t0) * 1e3)
    return collections.OrderedDict([
        ("device", "Mac (%s, %s), numpy %s, single process" % (platform.machine(),
                                                               platform.platform(), np.__version__)),
        ("median_ms_example_sentence_50_calls", round(float(np.median(fixed)), 2)),
        ("median_ms_over_handwritten_set", round(float(np.median(per)), 2)),
        ("p95_ms_over_handwritten_set", round(float(np.percentile(per, 95)), 2))])


def main():
    val = load("data/val.jsonl")
    test = load("data/test_handwritten.jsonl")
    out = collections.OrderedDict([
        ("generated", time.strftime("%Y-%m-%d %H:%M:%S")),
        ("runtime", "dhwani_parser.py (pure NumPy); the torch checkpoints are only used by parity_test.py"),
        ("exact_match_definition", "all 12 mission fields equal: " + ", ".join(MISSION_FIELDS)),
        ("models", collections.OrderedDict())])
    for name in ("tinybert4", "berttiny"):
        mdir = os.path.join(HERE, "models", name)
        t0 = time.perf_counter()
        p = MissionParser(mdir)
        load_s = time.perf_counter() - t0
        cfg = p.cfg
        entry = collections.OrderedDict([
            ("base_model", cfg["base_model"]),
            ("float32_parameters", int(sum(np.load(os.path.join(mdir, "weights.npz"))[k].size
                                           for k in np.load(os.path.join(mdir, "weights.npz")).files))),
            ("weights_npz_mb", round(os.path.getsize(os.path.join(mdir, "weights.npz")) / 1e6, 1)),
            ("threshold", p.threshold), ("threshold_rule", cfg["threshold_rule"]),
            ("best_epoch", cfg["best_epoch"]), ("load_s_mac", round(load_s, 2)),
            ("val", evaluate(p, val, False)),
            ("handwritten_test", evaluate(p, test, True)),
            ("latency_mac", latency(p, [r["text"] for r in test]))])
        out["models"][name] = entry
        v, h = entry["val"], entry["handwritten_test"]
        print("%-10s val EM %.4f | handwritten EM %.4f  e2e %.4f  accepted %d (%.3f correct)  "
              "unsupported accepted %d | Mac median %.1f ms"
              % (name, v["exact_match"], h["exact_match"], h["end_to_end_correct"], h["accepted"],
                 h["accepted_fully_correct"], h["unsupported_accepted"],
                 entry["latency_mac"]["median_ms_example_sentence_50_calls"]))
    parity = os.path.join(HERE, "checkpoints", "parity.json")
    if os.path.exists(parity):
        out["parity_numpy_vs_torch"] = json.load(open(parity))
    rover = os.path.join(HERE, "rover_check_output.json")
    if os.path.exists(rover):
        rc = json.load(open(rover))
        for name, m in rc["models"].items():          # same inputs through the Mac runtime
            p = MissionParser(os.path.join(HERE, "models", name))
            same = [json.dumps(p.parse(r["text"])) == json.dumps(r) for r in m["parsed"]]
            m["identical_json_to_mac"] = "%d of %d" % (sum(same), len(same))
        rc["note"] = ("compatibility check on the rover (Jetson Nano, ROS running), not the formal "
                      "benchmark; timings are per parse() call after 5 warm-up calls")
        out["rover_compatibility_check"] = rc
    with open(os.path.join(HERE, "eval.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote eval.json")


if __name__ == "__main__":
    main()
