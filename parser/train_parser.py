#!/usr/bin/env python3
"""Fine-tune a small BERT + heads for the DHWANI mission parser (Mac, torch + MPS).

    python train_parser.py --model tinybert4        # huawei-noah/TinyBERT_General_4L_312D
    python train_parser.py --model berttiny         # google/bert_uncased_L-2_H-128_A-2

Trains on data/train.jsonl, evaluates exact match on data/val.jsonl after every epoch and keeps
the best epoch, chooses the confidence threshold on val, then exports
models/<model>/{weights.npz, vocab.txt, config.json} for the NumPy runtime (dhwani_parser.py).
The torch checkpoint goes to checkpoints/<model>.pt (used only by parity_test.py; not deployed).
data/test_handwritten.jsonl is never read here.

Pooling: masked mean over all real tokens ([CLS] and [SEP] included) of the last layer. Chosen
over [CLS] because TinyBERT-General's [CLS]/pooler was never trained for sentence tasks and a mean
is the more stable sentence summary for 2-4 layer encoders fine-tuned on 20k examples.
"""
import argparse
import collections
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dhwani_parser import Tokenizer, HEADS, BIO, MISSION_FIELDS, decode_mission  # noqa: E402

MODELS = {
    "tinybert4": {"repo": "huawei-noah/TinyBERT_General_4L_312D", "lr": 1e-4, "epochs": 10,
                  "weights": "pytorch_model.bin"},
    "berttiny": {"repo": "google/bert_uncased_L-2_H-128_A-2", "lr": 5e-4, "epochs": 24,
                 "weights": "model.safetensors"},
}
MAX_LEN = 64
TARGET_PRECISION = 0.97


def base_dir_for(name, base):
    """A local directory with config.json + vocab.txt + weights, downloading if needed."""
    if base and os.path.isdir(base):
        return base
    from huggingface_hub import hf_hub_download
    spec = MODELS[name]
    path = None
    for f in ("config.json", "vocab.txt", spec["weights"]):
        path = hf_hub_download(spec["repo"], f)
    return os.path.dirname(path)


class ParserModel(nn.Module):
    def __init__(self, base_dir, dropout=0.1, bert_config=None):
        super(ParserModel, self).__init__()
        from transformers import BertModel
        if bert_config is not None:
            self.bert = BertModel(bert_config, add_pooling_layer=False)
        else:
            self.bert = BertModel.from_pretrained(base_dir, add_pooling_layer=False,
                                                  attn_implementation="eager")
        h = self.bert.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleDict((k, nn.Linear(h, len(v))) for k, v in HEADS.items())
        self.tok = nn.Linear(h, len(BIO))

    @classmethod
    def from_checkpoint(cls, ckpt_path, model_dir):
        """Rebuild the trained torch model from checkpoints/<m>.pt + models/<m>/config.json."""
        from transformers import BertConfig
        sd = torch.load(ckpt_path, map_location="cpu")
        c = json.load(open(os.path.join(model_dir, "config.json")))
        cfg = BertConfig(vocab_size=c["vocab_size"], hidden_size=c["hidden_size"],
                         num_hidden_layers=c["num_hidden_layers"],
                         num_attention_heads=c["num_attention_heads"],
                         intermediate_size=c["intermediate_size"], hidden_act="gelu",
                         layer_norm_eps=c["layer_norm_eps"],
                         max_position_embeddings=sd["bert.embeddings.position_embeddings.weight"].shape[0],
                         attn_implementation="eager")
        m = cls(None, bert_config=cfg)
        m.load_state_dict(sd)
        return m.eval()

    def forward(self, ids, mask):
        hid = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(hid.dtype)
        pooled = self.drop((hid * m).sum(1) / m.sum(1))
        sent = collections.OrderedDict((k, head(pooled)) for k, head in self.heads.items())
        return sent, self.tok(self.drop(hid))


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def encode(tok, ex):
    ids, widx, words = tok.encode(ex["text"])
    tags = ex["tags"]
    assert len(words) == len(tags), ex["text"]
    tag_ids, prev = [], -1
    for wi in widx:
        if wi < 0:
            tag_ids.append(-100)
            continue
        t = tags[wi]
        if wi != prev:
            tag_ids.append(BIO.index(t))
        else:
            tag_ids.append(0 if t == "O" else BIO.index("I-" + t[2:]))
        prev = wi
    heads = [HEADS[h].index(ex["label"][h]) for h in HEADS]
    return ids, tag_ids, heads


def batches(encoded, bs, shuffle, rng):
    idx = list(range(len(encoded)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [encoded[j] for j in idx[i:i + bs]]
        L = max(len(c[0]) for c in chunk)
        ids = torch.zeros(len(chunk), L, dtype=torch.long)
        mask = torch.zeros(len(chunk), L, dtype=torch.long)
        tags = torch.full((len(chunk), L), -100, dtype=torch.long)
        for r, (a, t, _) in enumerate(chunk):
            ids[r, :len(a)] = torch.tensor(a)
            mask[r, :len(a)] = 1
            tags[r, :len(t)] = torch.tensor(t)
        heads = torch.tensor([c[2] for c in chunk], dtype=torch.long)
        yield idx[i:i + bs], ids, mask, tags, heads


@torch.no_grad()
def predict(model, tok, rows, device, bs=128):
    """Runs the torch model and the shared post-processing. Returns list of mission dicts."""
    model.eval()
    enc = [tok.encode(r["text"]) for r in rows]
    out = [None] * len(rows)
    order = sorted(range(len(rows)), key=lambda i: len(enc[i][0]))
    for s in range(0, len(order), bs):
        chunk = order[s:s + bs]
        L = max(len(enc[i][0]) for i in chunk)
        ids = torch.zeros(len(chunk), L, dtype=torch.long)
        mask = torch.zeros(len(chunk), L, dtype=torch.long)
        for r, i in enumerate(chunk):
            ids[r, :len(enc[i][0])] = torch.tensor(enc[i][0])
            mask[r, :len(enc[i][0])] = 1
        sent, tokl = model(ids.to(device), mask.to(device))
        sent = {k: v.float().cpu().numpy() for k, v in sent.items()}
        tokl = tokl.float().cpu().numpy()
        for r, i in enumerate(chunk):
            n = len(enc[i][0])
            out[i] = decode_mission(rows[i]["text"], enc[i][2], enc[i][1],
                                    {k: v[r] for k, v in sent.items()}, tokl[r, :n], 0.0)
    return out


def fully_correct(pred, gold):
    return all(pred[k] == gold[k] for k in MISSION_FIELDS)


def choose_threshold(preds, golds):
    """Smallest threshold whose accepted val examples are >= 97% fully correct."""
    table = []
    for t in [i / 100.0 for i in range(0, 100)] + [0.995, 0.999]:
        acc = [fully_correct(p, g) for p, g in zip(preds, golds)
               if p["task"] != "UNSUPPORTED" and p["confidence"] >= t
               and not (p["task"] == "GOTO_WAYPOINT" and p["waypoint"] is None)]
        prec = sum(acc) / float(len(acc)) if acc else 1.0
        table.append((t, len(acc), prec))
    chosen = next(t for t, n, p in table if p >= TARGET_PRECISION)
    return chosen, table


def export(model, tok_vocab, out_dir, meta):
    sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    cfg = model.bert.config
    W = {"emb.word": sd["bert.embeddings.word_embeddings.weight"],
         "emb.pos": sd["bert.embeddings.position_embeddings.weight"][:MAX_LEN],
         "emb.type0": sd["bert.embeddings.token_type_embeddings.weight"][0],
         "emb.ln.g": sd["bert.embeddings.LayerNorm.weight"],
         "emb.ln.b": sd["bert.embeddings.LayerNorm.bias"]}
    names = {"q": "attention.self.query", "k": "attention.self.key", "v": "attention.self.value",
             "ao": "attention.output.dense", "i": "intermediate.dense", "o": "output.dense"}
    for i in range(cfg.num_hidden_layers):
        p = "bert.encoder.layer.%d." % i
        for short, long_ in names.items():
            W["l%d.%s.w" % (i, short)] = sd[p + long_ + ".weight"].T
            W["l%d.%s.b" % (i, short)] = sd[p + long_ + ".bias"]
        W["l%d.ln1.g" % i] = sd[p + "attention.output.LayerNorm.weight"]
        W["l%d.ln1.b" % i] = sd[p + "attention.output.LayerNorm.bias"]
        W["l%d.ln2.g" % i] = sd[p + "output.LayerNorm.weight"]
        W["l%d.ln2.b" % i] = sd[p + "output.LayerNorm.bias"]
    for h in HEADS:
        W["head.%s.w" % h] = sd["heads.%s.weight" % h].T
        W["head.%s.b" % h] = sd["heads.%s.bias" % h]
    W["tok.w"] = sd["tok.weight"].T
    W["tok.b"] = sd["tok.bias"]
    W = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in W.items()}
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "weights.npz"), **W)
    with open(tok_vocab) as src, open(os.path.join(out_dir, "vocab.txt"), "w") as dst:
        dst.write(src.read())
    config = collections.OrderedDict([
        ("hidden_size", cfg.hidden_size), ("num_hidden_layers", cfg.num_hidden_layers),
        ("num_attention_heads", cfg.num_attention_heads),
        ("intermediate_size", cfg.intermediate_size), ("layer_norm_eps", cfg.layer_norm_eps),
        ("hidden_act", "gelu (erf)"), ("max_len", MAX_LEN), ("pooling", "masked_mean"),
        ("vocab_size", cfg.vocab_size), ("heads", HEADS), ("bio", BIO)])
    config.update(meta)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    return sum(v.size for v in W.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--base", default=None, help="local dir with the base checkpoint")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    spec = MODELS[args.model]
    epochs = args.epochs or spec["epochs"]
    lr = args.lr or spec["lr"]
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    base = base_dir_for(args.model, args.base)
    vocab = os.path.join(base, "vocab.txt")
    tok = Tokenizer(vocab, MAX_LEN)
    train = load_jsonl(os.path.join(HERE, "data", "train.jsonl"))
    val = load_jsonl(os.path.join(HERE, "data", "val.jsonl"))
    enc_train = [encode(tok, r) for r in train]
    for r, e in zip(train, enc_train):
        assert len(e[0]) <= MAX_LEN, r["text"]

    model = ParserModel(base).to(device)
    no_decay = ("bias", "LayerNorm.weight")
    params = [{"params": [p for n, p in model.named_parameters()
                          if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
              {"params": [p for n, p in model.named_parameters()
                          if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(params, lr=lr)
    steps = epochs * int(math.ceil(len(enc_train) / float(args.bs)))
    warm = int(0.06 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / float(warm)) if s < warm
        else max(0.0, (steps - s) / float(max(1, steps - warm))))

    ckpt_dir = os.path.join(HERE, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = os.path.join(ckpt_dir, args.model + ".pt")
    best = (-1.0, None)
    history = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for _, ids, mask, tags, heads in batches(enc_train, args.bs, True, rng):
            ids, mask, tags, heads = ids.to(device), mask.to(device), tags.to(device), heads.to(device)
            sent, tokl = model(ids, mask)
            loss = sum(F.cross_entropy(sent[h], heads[:, j]) for j, h in enumerate(HEADS))
            loss = loss + F.cross_entropy(tokl.reshape(-1, len(BIO)), tags.reshape(-1),
                                          ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        preds = predict(model, tok, val, device)
        em = np.mean([fully_correct(p, r["label"]) for p, r in zip(preds, val)])
        history.append({"epoch": ep, "train_loss": round(tot / nb, 4), "val_exact_match": round(float(em), 4),
                        "elapsed_s": round(time.time() - t0, 1)})
        print(json.dumps(history[-1]), flush=True)
        if em > best[0]:
            best = (em, ep)
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    preds = predict(model, tok, val, device)
    golds = [r["label"] for r in val]
    thr, table = choose_threshold(preds, golds)
    acc_at = [row for row in table if row[0] == thr][0]
    print("best epoch", best[1], "val EM", round(best[0], 4), "threshold", thr,
          "accepted", acc_at[1], "precision", round(acc_at[2], 4))
    meta = collections.OrderedDict([
        ("model_name", args.model), ("base_model", spec["repo"]), ("threshold", thr),
        ("threshold_rule", "smallest t with >= 97% of accepted val examples fully correct"),
        ("val_exact_match", round(float(best[0]), 4)), ("best_epoch", best[1]),
        ("epochs", epochs), ("lr", lr), ("batch_size", args.bs), ("seed", args.seed),
        ("train_examples", len(train)), ("val_examples", len(val)),
        ("threshold_table", [[t, n, round(p, 4)] for t, n, p in table if t in
                             (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, thr)]),
        ("history", history), ("device", str(device))])
    n = export(model.cpu(), vocab, os.path.join(HERE, "models", args.model), meta)
    print("exported", n, "float32 parameters to models/%s" % args.model)


if __name__ == "__main__":
    main()
