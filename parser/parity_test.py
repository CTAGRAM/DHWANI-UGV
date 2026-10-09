#!/usr/bin/env python3
"""Parity: NumPy runtime (dhwani_parser.py) vs the trained torch model, on the Mac.

For each model, 100 examples (50 from data/val.jsonl, 50 from data/test_handwritten.jsonl; the
test set is only used as input text here, nothing is tuned):
  * max |logit difference| over all 8 sentence heads and all token logits must be < 1e-3
  * the decoded mission JSON must be identical (confidence compared to 1e-4, it is rounded)
The NumPy side runs exactly as on the rover (real length padded to a multiple of 8, erf-approx
GELU); the torch side runs on CPU in float32 with no padding.

    python parity_test.py            # exit code 0 = pass
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dhwani_parser import MissionParser, decode_mission, HEADS  # noqa: E402
from train_parser import ParserModel  # noqa: E402

LOGIT_TOL = 1e-3


def texts():
    val = [json.loads(l)["text"] for l in open(os.path.join(HERE, "data", "val.jsonl"))][:50]
    test = [json.loads(l)["text"]
            for l in open(os.path.join(HERE, "data", "test_handwritten.jsonl"))][:50]
    return val + test


def run(name, rows):
    mdir = os.path.join(HERE, "models", name)
    npp = MissionParser(mdir)
    tm = ParserModel.from_checkpoint(os.path.join(HERE, "checkpoints", name + ".pt"), mdir)
    worst, json_mismatch, conf_worst = 0.0, [], 0.0
    for text in rows:
        words, widx, sent_np, tok_np = npp.forward(text)
        ids, widx2, words2 = npp.tok.encode(text)
        with torch.no_grad():
            s_t, t_t = tm(torch.tensor([ids]), torch.ones(1, len(ids), dtype=torch.long))
        s_t = {k: v[0].numpy() for k, v in s_t.items()}
        t_t = t_t[0].numpy()
        d = max([float(np.max(np.abs(sent_np[k] - s_t[k]))) for k in HEADS]
                + [float(np.max(np.abs(tok_np - t_t)))])
        worst = max(worst, d)
        a = npp.parse(text)
        b = decode_mission(text, words2, widx2, s_t, t_t, npp.threshold)
        conf_worst = max(conf_worst, abs(a["confidence"] - b["confidence"]))
        a2 = dict(a, confidence=None)
        b2 = dict(b, confidence=None)
        if json.dumps(a2) != json.dumps(b2) or abs(a["confidence"] - b["confidence"]) > 1e-4:
            json_mismatch.append((text, a, b))
    ok = worst < LOGIT_TOL and not json_mismatch
    print("%-10s n=%d  max|logit diff|=%.2e  max|conf diff|=%.1e  JSON mismatches=%d  %s"
          % (name, len(rows), worst, conf_worst, len(json_mismatch), "PASS" if ok else "FAIL"))
    for text, a, b in json_mismatch[:5]:
        print("  MISMATCH", text, "\n   numpy", json.dumps(a), "\n   torch", json.dumps(b))
    return ok, worst, len(json_mismatch)


def main():
    rows = texts()
    results = {}
    for name in ("tinybert4", "berttiny"):
        if not os.path.exists(os.path.join(HERE, "checkpoints", name + ".pt")):
            print(name, "no checkpoint, skipped")
            continue
        ok, worst, mism = run(name, rows)
        results[name] = {"pass": ok, "max_abs_logit_diff": worst, "json_mismatches": mism,
                         "n": len(rows)}
    with open(os.path.join(HERE, "checkpoints", "parity.json"), "w") as f:
        json.dump(results, f, indent=2)
    return 0 if results and all(r["pass"] for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
