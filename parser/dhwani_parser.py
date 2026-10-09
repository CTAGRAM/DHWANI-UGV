# -*- coding: utf-8 -*-
"""DHWANI natural-language mission parser.

Runs on the rover: Python 3.6, numpy 1.13, no torch. Pure NumPy BERT encoder + sentence heads +
BIO slot tagger + deterministic post-processing that always emits the mission JSON of
docs/INTERFACES.md section 1.

    from dhwani_parser import MissionParser
    p = MissionParser("models/tinybert4")
    m = p.parse("go through this mine and find anyone alive")

Command line:  python3 dhwani_parser.py MODEL_DIR "instruction text"
"""
from __future__ import division, print_function

import collections
import json
import math
import os
import re
import sys
import time
import unicodedata

import numpy as np

# ----------------------------------------------------------------------------------------------
# Label sets and mission defaults (docs/INTERFACES.md section 1)
# ----------------------------------------------------------------------------------------------
TASKS = ["SEARCH_SURVIVORS", "MAP_AREA", "GOTO_WAYPOINT", "RETURN_HOME", "HALT", "UNSUPPORTED"]
HEADS = collections.OrderedDict([
    ("task", TASKS),
    ("report_person", [False, True]),
    ("approach_person", [False, True]),
    ("on_person_found", ["REPORT_AND_CONTINUE", "REPORT_AND_RETURN", "STAY"]),
    ("on_link_loss", ["CONTINUE_THEN_RETURN", "RETURN_NOW", "CONTINUE", "HOLD"]),
    ("return_when_done", [False, True]),
    ("first_branch", ["ANY", "LEFT", "RIGHT", "STRAIGHT"]),
    ("speed", ["NORMAL", "SLOW"]),
])
BEHAVIOUR = list(HEADS.keys())[1:]
BIO = ["O", "B-DUR", "I-DUR", "B-DIST", "I-DIST", "B-BATT", "I-BATT", "B-WP", "I-WP"]
KEYS = ["task", "report_person", "approach_person", "on_person_found", "on_link_loss",
        "return_when_done", "first_branch", "speed", "waypoint", "max_duration_min",
        "max_distance_m", "min_battery_pct", "confidence", "understood", "text"]
MISSION_FIELDS = KEYS[:12]          # the fields compared for exact match
RANGES = {"max_duration_min": (1, 240), "max_distance_m": (5, 2000), "min_battery_pct": (5, 90)}

# The SEARCH_SURVIVORS line of the interfaces doc is the full default record; every other task
# overrides only what the doc lists for it. UNSUPPORTED is made inert like HALT (return false).
GLOBAL_DEFAULTS = collections.OrderedDict([
    ("report_person", True), ("approach_person", True),
    ("on_person_found", "REPORT_AND_CONTINUE"), ("on_link_loss", "CONTINUE_THEN_RETURN"),
    ("return_when_done", True), ("first_branch", "ANY"), ("speed", "NORMAL")])
TASK_OVERRIDES = {
    "SEARCH_SURVIVORS": {},
    "MAP_AREA": {"approach_person": False},
    "GOTO_WAYPOINT": {"on_link_loss": "RETURN_NOW", "return_when_done": False},
    "RETURN_HOME": {"return_when_done": False},
    "HALT": {"return_when_done": False},
    "UNSUPPORTED": {"return_when_done": False},
}


def task_defaults(task):
    """Behaviour fields + waypoint + numbers for a task when the instruction is silent."""
    d = collections.OrderedDict(GLOBAL_DEFAULTS)
    d.update(TASK_OVERRIDES[task])
    d["waypoint"] = None
    d["max_duration_min"] = None
    d["max_distance_m"] = None
    d["min_battery_pct"] = None
    return d


# ----------------------------------------------------------------------------------------------
# BERT uncased tokenizer (BasicTokenizer + WordPiece), reimplemented from vocab.txt
# ----------------------------------------------------------------------------------------------
def _is_whitespace(ch):
    if ch in (" ", "\t", "\n", "\r"):
        return True
    return unicodedata.category(ch) == "Zs"


def _is_control(ch):
    if ch in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(ch).startswith("C")


def _is_punctuation(ch):
    cp = ord(ch)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(cp):
    return ((0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or (0x20000 <= cp <= 0x2A6DF)
            or (0x2A700 <= cp <= 0x2B73F) or (0x2B740 <= cp <= 0x2B81F)
            or (0x2B820 <= cp <= 0x2CEAF) or (0xF900 <= cp <= 0xFAFF)
            or (0x2F800 <= cp <= 0x2FA1F))


class Tokenizer(object):
    """Uncased BERT tokenizer that also records which whitespace word each wordpiece came from."""

    def __init__(self, vocab_path, max_len=64):
        self.vocab = {}
        with open(vocab_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                tok = line.rstrip("\n")
                if tok not in self.vocab:
                    self.vocab[tok] = i
        self.max_len = max_len
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.pad_id = self.vocab["[PAD]"]
        self.unk_id = self.vocab["[UNK]"]

    @staticmethod
    def clean(text):
        out = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or _is_control(ch):
                continue
            if _is_whitespace(ch):
                out.append(" ")
            elif _is_cjk(cp):
                out.append(" " + ch + " ")
            else:
                out.append(ch)
        return unicodedata.normalize("NFC", "".join(out))

    def words(self, text):
        """Whitespace words of the cleaned text; BIO tags and slot spans live on these."""
        return self.clean(text).split()

    @staticmethod
    def basic(word):
        w = unicodedata.normalize("NFD", word.lower())
        w = "".join(c for c in w if unicodedata.category(c) != "Mn")
        toks, cur = [], []
        for ch in w:
            if _is_punctuation(ch):
                if cur:
                    toks.append("".join(cur))
                    cur = []
                toks.append(ch)
            else:
                cur.append(ch)
        if cur:
            toks.append("".join(cur))
        return toks

    def wordpiece(self, token):
        if len(token) > 100:
            return ["[UNK]"]
        pieces, start = [], 0
        while start < len(token):
            end, cur = len(token), None
            while start < end:
                sub = token[start:end]
                if start > 0:
                    sub = "##" + sub
                if sub in self.vocab:
                    cur = sub
                    break
                end -= 1
            if cur is None:
                return ["[UNK]"]
            pieces.append(cur)
            start = end
        return pieces

    def encode(self, text):
        """Returns (ids with [CLS]/[SEP], word index per id (-1 for specials), words)."""
        words = self.words(text)
        ids, widx = [], []
        for wi, w in enumerate(words):
            for bt in self.basic(w):
                for p in self.wordpiece(bt):
                    ids.append(self.vocab.get(p, self.unk_id))
                    widx.append(wi)
        ids, widx = ids[:self.max_len - 2], widx[:self.max_len - 2]
        return [self.cls_id] + ids + [self.sep_id], [-1] + widx + [-1], words


# ----------------------------------------------------------------------------------------------
# Deterministic slot value parsing: spans -> minutes, metres, percent, waypoint letter
# ----------------------------------------------------------------------------------------------
_NUM_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
              "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
              "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
              "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
              "fourty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000}
_FRACTIONS = {"half": 0.5, "quarter": 0.25}
_VAGUE = {"few", "several", "some", "many"}
_NATO = {"alpha": "A", "alfa": "A", "bravo": "B", "charlie": "C", "delta": "D", "echo": "E",
         "foxtrot": "F", "golf": "G", "hotel": "H", "india": "I", "juliet": "J", "juliett": "J",
         "kilo": "K", "lima": "L", "mike": "M", "november": "N", "oscar": "O", "papa": "P",
         "quebec": "Q", "romeo": "R", "sierra": "S", "tango": "T", "uniform": "U", "victor": "V",
         "whiskey": "W", "whisky": "W", "xray": "X", "yankee": "Y", "zulu": "Z"}


def _span_tokens(text):
    t = text.lower().replace("per cent", "percent").replace("x-ray", "xray")
    t = re.sub(r"(\d),(\d{3})", r"\1\2", t)            # 1,000 -> 1000
    t = re.sub(r"(\d)\s*/\s*(\d)", r"\1/\2", t)        # 1 / 2 -> 1/2
    t = re.sub(r"[^a-z0-9.%/ ]", " ", t)                # hyphens, quotes, commas -> space
    t = re.sub(r"(\d)([a-z%])", r"\1 \2", t)            # 200m -> 200 m, 20% -> 20 %
    t = re.sub(r"([a-z%])(\d)", r"\1 \2", t)            # hr30 -> hr 30
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)           # dots that are not decimal points
    t = re.sub(r"(?<!\d)/|/(?!\d)", " ", t)
    return t.split()


def _dur_unit(t):
    if t in ("s", "sec", "secs") or t.startswith("second"):
        return 1.0 / 60.0
    if t in ("m", "mn", "mns") or t.startswith("min") or t.startswith("mnt"):
        return 1.0
    if t in ("h", "hr", "hrs", "hrz") or t.startswith("hour") or t.startswith("hou"):
        return 60.0
    return None


def _dist_unit(t):
    if t in ("km", "kms", "kmtr", "kmtrs") or t.startswith("kilo"):
        return 1000.0
    if t in ("m", "mt", "mts") or t.startswith("met") or t.startswith("mtr"):
        return 1.0
    if t in ("ft", "feet", "foot"):
        return 0.3048
    return None


def _batt_unit(t):
    if t in ("%", "pct", "pc") or t.startswith("perc"):
        return 1.0
    return None


def _read_number(toks, i, unit_fn):
    """Read a number starting at toks[i]. Returns (value, next_index, from_fraction) or None."""
    n = len(toks)
    t = toks[i]
    frac = False
    if re.match(r"^\d+(\.\d+)?$", t):
        v, i = float(t), i + 1
    elif re.match(r"^\d+/\d+$", t):
        a, b = t.split("/")
        if float(b) == 0:
            return None
        v, i, frac = float(a) / float(b), i + 1, True
    elif t in _NUM_WORDS or t in _SCALES:
        total, cur = 0.0, 0.0
        while i < n:
            w = toks[i]
            if w in _NUM_WORDS:
                cur += _NUM_WORDS[w]
            elif w == "hundred":
                cur = (cur or 1.0) * 100.0
            elif w == "thousand":
                total += (cur or 1.0) * 1000.0
                cur = 0.0
            elif w == "and" and i + 1 < n and toks[i + 1] in _NUM_WORDS:
                pass
            else:
                break
            i += 1
        v = total + cur
    elif t in ("a", "an", "one") and i + 1 < n and toks[i + 1] in _FRACTIONS:
        return _read_number(toks, i + 1, unit_fn)
    elif t in ("a", "an") and i + 1 < n and (unit_fn(toks[i + 1]) is not None):
        v, i = 1.0, i + 1
    elif t in _FRACTIONS:
        v, i, frac = _FRACTIONS[t], i + 1, True
        if i < n and toks[i] == "of":
            i += 1
        if i < n and toks[i] in ("a", "an", "the"):
            i += 1
        return v, i, frac
    elif t == "couple":
        v, i = 2.0, i + 1
        if i < n and toks[i] == "of":
            i += 1
        return v, i, frac
    else:
        return None
    # "one and a half", "2 and half", "one and a quarter"
    if i + 1 < n and toks[i] == "and" and toks[i + 1] in _FRACTIONS:
        v, i = v + _FRACTIONS[toks[i + 1]], i + 2
    elif i + 2 < n and toks[i] == "and" and toks[i + 1] in ("a", "an") and toks[i + 2] in _FRACTIONS:
        v, i = v + _FRACTIONS[toks[i + 2]], i + 3
    return v, i, frac


def _parse_quantity(text, unit_fn, default_mult):
    """Sum of number x unit groups in a span, e.g. '1 hr and 30 mins' -> 90 (minutes)."""
    toks = _span_tokens(text)
    if any(t in _VAGUE for t in toks):
        return None, False
    total, pending, last_mult, used, only_frac = 0.0, None, None, False, True
    i = 0
    while i < len(toks):
        mult = unit_fn(toks[i])
        if mult is not None and not (toks[i] in ("a", "an")):
            if pending is not None:
                total += pending * mult
                pending = None
            elif not used:
                total += mult             # bare unit: "hour and a half"
                only_frac = False
            used, last_mult = True, mult
            i += 1
            continue
        r = _read_number(toks, i, unit_fn)
        if r is None:
            i += 1
            continue
        v, i, frac = r
        only_frac = only_frac and frac
        if pending is not None:
            total += pending * (last_mult or default_mult)
            used = True
        pending = v
    if pending is not None:
        if used and pending < 1.0 and last_mult is not None:
            total += pending * last_mult  # "an hour and a half"
        else:
            total += pending * default_mult
        used = True
    if not used:
        return None, False
    return total, only_frac


def parse_duration(text):
    v, _ = _parse_quantity(text, _dur_unit, 1.0)
    return v


def parse_distance(text):
    v, _ = _parse_quantity(text, _dist_unit, 1.0)
    return v


def parse_battery(text):
    v, only_frac = _parse_quantity(text, _batt_unit, 1.0)
    if v is not None and only_frac and v <= 1.0:
        v = v * 100.0                     # "half battery" -> 50
    return v


def parse_waypoint(text):
    for t in re.split(r"[^a-z]+", text.lower().replace("x-ray", "xray")):
        if len(t) == 1:
            return t.upper()
        if t in _NATO:
            return _NATO[t]
    return None


def clean_number(field, v):
    """Round, make integral values ints, and apply the doc's range (out of range -> None)."""
    if v is None:
        return None
    v = round(float(v), 2)
    lo, hi = RANGES[field]
    if not (lo <= v <= hi):
        return None
    return int(v) if v == int(v) else v


SLOT_FIELDS = [("DUR", "max_duration_min", parse_duration),
               ("DIST", "max_distance_m", parse_distance),
               ("BATT", "min_battery_pct", parse_battery)]


def spans_from_word_tags(word_tags):
    """word_tags: list of BIO strings, one per whitespace word. Returns [(type, start, end_incl)]."""
    spans, cur = [], None
    for i, t in enumerate(word_tags):
        if t == "O":
            if cur:
                spans.append(tuple(cur))
            cur = None
            continue
        kind, typ = t.split("-", 1)
        if kind == "B" or cur is None or cur[0] != typ:
            if cur:
                spans.append(tuple(cur))
            cur = [typ, i, i]
        else:
            cur[2] = i
    if cur:
        spans.append(tuple(cur))
    return spans


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def decode_mission(text, words, widx, sent_logits, tok_logits, threshold):
    """Turn raw head outputs into the mission dict. Shared by the runtime, training and eval.

    sent_logits: dict head -> 1-D logits. tok_logits: (n_ids, 9) for the real (unpadded) ids.
    """
    probs = collections.OrderedDict((h, softmax(np.asarray(sent_logits[h], dtype=np.float64)))
                                    for h in HEADS)
    pred = {h: HEADS[h][int(np.argmax(p))] for h, p in probs.items()}
    conf = float(min(np.max(p) for p in probs.values()))
    task = pred["task"]

    # first wordpiece of each whitespace word carries that word's tag
    word_tags = ["O"] * len(words)
    seen = set()
    tag_ids = np.argmax(np.asarray(tok_logits), axis=-1)
    for pos, wi in enumerate(widx):
        if wi < 0 or wi in seen:
            continue
        seen.add(wi)
        word_tags[wi] = BIO[int(tag_ids[pos])]
    values = {"waypoint": None, "max_duration_min": None, "max_distance_m": None,
              "min_battery_pct": None}
    for typ, a, b in spans_from_word_tags(word_tags):
        span = " ".join(words[a:b + 1])
        if typ == "WP":
            if values["waypoint"] is None:
                values["waypoint"] = parse_waypoint(span)
            continue
        for t, field, fn in SLOT_FIELDS:
            if t == typ and values[field] is None:
                values[field] = clean_number(field, fn(span))

    if task == "UNSUPPORTED":
        fields = task_defaults(task)
    elif task == "HALT":
        fields = task_defaults(task)
        fields["max_duration_min"] = values["max_duration_min"]   # "wait here for 5 minutes"
    else:
        fields = collections.OrderedDict((h, pred[h]) for h in BEHAVIOUR)
        fields.update(values)
    understood = (task != "UNSUPPORTED" and conf >= threshold
                  and not (task == "GOTO_WAYPOINT" and fields["waypoint"] is None))
    out = collections.OrderedDict()
    out["task"] = task
    for k in KEYS[1:12]:
        out[k] = fields[k]
    out["confidence"] = round(conf, 4)
    out["understood"] = bool(understood)
    out["text"] = text
    return out


# ----------------------------------------------------------------------------------------------
# Pure NumPy BERT encoder
# ----------------------------------------------------------------------------------------------
_SQRT_HALF = 0.7071067811865476


def erf(x):
    """Abramowitz & Stegun 7.1.26; max abs error 1.5e-7 (numpy 1.13 has no erf)."""
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
                + 0.254829592) * t) * np.exp(-a * a)
    return s * y


def gelu(x):
    return 0.5 * x * (1.0 + erf(x * _SQRT_HALF))


def layer_norm(x, g, b, eps):
    mu = x.mean(axis=-1, keepdims=True)
    d = x - mu
    var = (d * d).mean(axis=-1, keepdims=True)
    return d / np.sqrt(var + eps) * g + b


class MissionParser(object):
    def __init__(self, model_dir, threshold=None):
        with open(os.path.join(model_dir, "config.json")) as f:
            self.cfg = json.load(f)
        c = self.cfg
        self.max_len = int(c.get("max_len", 64))
        self.tok = Tokenizer(os.path.join(model_dir, "vocab.txt"), self.max_len)
        self.threshold = float(c["threshold"] if threshold is None else threshold)
        self.H = int(c["hidden_size"])
        self.nh = int(c["num_attention_heads"])
        self.dh = self.H // self.nh
        self.eps = float(c.get("layer_norm_eps", 1e-12))
        z = np.load(os.path.join(model_dir, "weights.npz"))
        W = dict((k, z[k].astype(np.float32)) for k in z.files)
        z.close()
        self.word = W["emb.word"]
        self.pos_type = W["emb.pos"][:self.max_len] + W["emb.type0"][None, :]
        self.emb_ln = (W["emb.ln.g"], W["emb.ln.b"])
        self.layers = []
        for i in range(int(c["num_hidden_layers"])):
            p = "l%d." % i
            qkv_w = np.ascontiguousarray(np.concatenate(
                [W[p + "q.w"], W[p + "k.w"], W[p + "v.w"]], axis=1))
            qkv_b = np.concatenate([W[p + "q.b"], W[p + "k.b"], W[p + "v.b"]])
            self.layers.append((qkv_w, qkv_b, W[p + "ao.w"], W[p + "ao.b"],
                                W[p + "ln1.g"], W[p + "ln1.b"], W[p + "i.w"], W[p + "i.b"],
                                W[p + "o.w"], W[p + "o.b"], W[p + "ln2.g"], W[p + "ln2.b"]))
        self.heads = collections.OrderedDict(
            (h, (W["head.%s.w" % h], W["head.%s.b" % h])) for h in HEADS)
        self.tok_head = (W["tok.w"], W["tok.b"])

    def encode_hidden(self, ids, n_real):
        """ids: padded id list (length multiple of 8); first n_real are real tokens."""
        L, H, nh, dh = len(ids), self.H, self.nh, self.dh
        h = self.word[np.asarray(ids)] + self.pos_type[:L]
        h = layer_norm(h, self.emb_ln[0], self.emb_ln[1], self.eps)
        bias = np.zeros((1, 1, L), dtype=np.float32)
        bias[0, 0, n_real:] = -1e9
        scale = np.float32(1.0 / math.sqrt(dh))
        for (qkv_w, qkv_b, ao_w, ao_b, g1, b1, i_w, i_b, o_w, o_b, g2, b2) in self.layers:
            qkv = h.dot(qkv_w) + qkv_b                                   # (L, 3H)
            q = qkv[:, :H].reshape(L, nh, dh).transpose(1, 0, 2)          # (nh, L, dh)
            k = qkv[:, H:2 * H].reshape(L, nh, dh).transpose(1, 2, 0)     # (nh, dh, L)
            v = qkv[:, 2 * H:].reshape(L, nh, dh).transpose(1, 0, 2)      # (nh, L, dh)
            att = softmax(np.matmul(q, k) * scale + bias)                 # (nh, L, L)
            ctx = np.matmul(att, v).transpose(1, 0, 2).reshape(L, H)
            h = layer_norm(ctx.dot(ao_w) + ao_b + h, g1, b1, self.eps)
            ff = gelu(h.dot(i_w) + i_b).dot(o_w) + o_b
            h = layer_norm(ff + h, g2, b2, self.eps)
        return h

    def forward(self, text):
        """Returns (words, widx, sentence logits dict, token logits for real ids)."""
        ids, widx, words = self.tok.encode(text)
        n = len(ids)
        L = min(self.max_len, int(math.ceil(n / 8.0)) * 8)
        padded = ids + [self.tok.pad_id] * (L - n)
        h = self.encode_hidden(padded, n)
        pooled = h[:n].mean(axis=0)                                       # masked mean pool
        sent = collections.OrderedDict((k, pooled.dot(w) + b) for k, (w, b) in self.heads.items())
        tok = h[:n].dot(self.tok_head[0]) + self.tok_head[1]
        return words, widx, sent, tok

    def parse(self, text):
        if not self.tok.words(text):
            out = collections.OrderedDict([("task", "UNSUPPORTED")])
            for k, v in task_defaults("UNSUPPORTED").items():
                out[k] = v
            out["confidence"] = 0.0
            out["understood"] = False
            out["text"] = text
            return out
        words, widx, sent, tok = self.forward(text)
        return decode_mission(text, words, widx, sent, tok, self.threshold)


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 1
    t0 = time.time()
    p = MissionParser(argv[1])
    t1 = time.time()
    m = p.parse(" ".join(argv[2:]))
    t2 = time.time()
    print(json.dumps(m, indent=2))
    print("load %.2f s, parse %.1f ms" % (t1 - t0, (t2 - t1) * 1e3), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
