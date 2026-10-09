# DHWANI mission parser

Turns an operator's English instruction into the mission JSON of [docs/INTERFACES.md](../docs/INTERFACES.md) §1,
offline, on the rover's CPU, in pure NumPy (Python 3.6, numpy 1.13; no torch, no GPU).

```python
from dhwani_parser import MissionParser
p = MissionParser("models/berttiny")          # or models/tinybert4
m = p.parse("Go through this area and find any person alive, report where they are, "
            "and come back if the link becomes unstable")
# {"task": "SEARCH_SURVIVORS", ..., "on_link_loss": "RETURN_NOW", ..., "confidence": 0.99,
#  "understood": true, "text": "..."}
```

Command line: `python3 dhwani_parser.py models/berttiny "go to point B and come back"`.
On the rover, everything is in `~/dhwani_parser/` (code + both models, md5-verified).

**Weights.** The trained `weights.npz` files (56.5 MB and 17.3 MB) are too large for git. They are
attached to the [v1.0 release](https://github.com/CTAGRAM/DHWANI-UGV/releases/tag/v1.0) as
`tinybert4-weights.npz` and `berttiny-weights.npz`; save each one in its model folder as
`weights.npz`. The release notes list their MD5 sums.

**Speed.** The rover times in the table below use the Nano's system NumPy 1.13. With the NumPy
1.19 OpenBLAS build the dashboard gateway loads (one thread, `OPENBLAS_CORETYPE=ARMV8`),
TinyBERT's median drops from 99 ms to 45 ms on a parked rover with the full ROS stack, and is
52 ms with two extra busy cores standing in for driving load ([../bench/results](../bench/results)).

## Which model to use

| | TinyBERT-4L-312D (`models/tinybert4`) | BERT-Tiny (`models/berttiny`) |
|---|---|---|
| float32 parameters / `weights.npz` | 14.1 M / 56.5 MB | 4.3 M / 17.3 MB |
| val exact match (2,000 generated) | **97.75 %** | 95.55 % |
| hand-written test exact match (192) | **98.4 %** (189/192) | 97.4 % (187/192) |
| confidence threshold (val rule) | 0.00 (gate inactive, see below) | 0.75 |
| **rover**, median per `parse()` call, 25-token example sentence | **185.8 ms** | **24.8 ms** |
| rover, `parse("stop")` | 60.6 ms | 9.2 ms |
| rover, model load | 0.50 s | 0.23 s |
| Mac, median per call, same sentence | 1.9 ms | 0.45 ms |

The "tens of milliseconds" goal is met by BERT-Tiny only. TinyBERT is about 7x slower on the
rover because numpy there links the generic single-threaded ATLAS BLAS (measured 2.1 GFLOPS on a
32x312x1200 matmul, with ROS load around 4). TinyBERT is the more accurate model. Its 186 ms is still
fine for a typed command, but it misses the "tens of ms" goal.

## Measured numbers and where they were measured

| what | value | device |
|---|---|---|
| NumPy vs torch parity, 100 inputs per model (50 val + 50 hand-written) | max abs logit diff 1.24e-5 (tinybert4), 1.34e-5 (berttiny); decoded JSON identical 100/100 | Mac CPU, torch 2.14 float32 |
| Rover compatibility run (10 hand-written sentences, 5 warm-up, 50 timed calls, both models) | works under Python 3.6.9 / numpy 1.13.3; JSON identical to the Mac for 10/10 sentences per model; whole run 15.5 s wall, 15.1 s CPU | Jetson Nano (`jetracer`), ROS up, loadavg 4.3 |
| Training | tinybert4: 10 epochs, 371 s, best epoch 9; berttiny: 24 epochs, 242 s, best epoch 23 | Mac, torch MPS |
| Evaluation | everything in `eval.json` is from the NumPy runtime (`dhwani_parser.py`), not from torch | Mac CPU, numpy 2.5.3 |

The rover timings are a compatibility check, not the formal benchmark.

### Accuracy detail (`eval.json` has per-field accuracy, confusion matrices and every failure)

| | tinybert4 val | tinybert4 hand-written | berttiny val | berttiny hand-written |
|---|---|---|---|---|
| exact match (all 12 mission fields) | 97.75 % | 98.44 % | 95.55 % | 97.40 % |
| task accuracy | 98.95 % | 98.44 % | 98.40 % | 98.44 % |
| accepted (`understood` = true) | 1,832 | 158 | 1,790 | 158 |
| accepted and fully correct | 98.0 % | 98.7 % | 97.0 % | 97.5 % |
| UNSUPPORTED wrongly accepted | 7 of 160 | 2 of 34 | 8 of 160 | 2 of 34 |
| val rows with a never-trained phrasing (482) / without (1,518) | 95.6 % / 98.4 % | | 91.7 % / 96.8 % | |

The failures on the hand-written set are all of them:

- **Both models:** "RTB" is read as UNSUPPORTED instead of RETURN_HOME. They also accept
  "search for gold in the mine" and "pick up the injured worker and carry him out" as
  SEARCH_SURVIVORS, which should be UNSUPPORTED. TinyBERT does this with confidence 0.999 and 0.978.
- **BERT-Tiny also:** "search for survivors for five hours" gets max_duration_min 60 (should be
  null, since 300 min is out of range). "search for one and a half hours" gets 30 instead of 90.

**Read the 98 % with care.** I wrote the test set before any generator code existed. It is held
out at the sentence level: 0 test sentences appear in train/val verbatim or after lowercasing and
stripping punctuation, and no test sentence of 4+ words appears inside a generated sentence. But
the same author then wrote the generator, so the test mostly contains
phrasing styles the generator also anticipates. 12 short test commands (e.g. "return to base",
"hold your position") also occur as parts of longer generated sentences. So the score measures
robustness to new sentences in familiar styles. It does not measure accuracy on real operators,
which will be lower. The honest next step is to log what people actually type into the dashboard
and label 100-200 of those as a fresh test set.

## How it works

1. **Tokenizer.** BERT uncased BasicTokenizer + WordPiece, reimplemented from `vocab.txt`. It gives
   the same ids as the HuggingFace tokenizer on all 22,201 train/val/test sentences plus odd Unicode.
   Each wordpiece remembers its whitespace word, so slot spans map back to the text. Input is
   `[CLS] ... [SEP]`, truncated to 64 ids, padded up to a multiple of 8 with masked attention.
2. **Encoder.** Pure NumPy BERT: embeddings, LayerNorm (eps 1e-12), masked multi-head attention
   with fused QKV, and GELU using the Abramowitz-Stegun 7.1.26 erf (|error| <= 1.5e-7).
3. **Pooling: masked mean** over all real tokens, [CLS]/[SEP] included. Chosen over [CLS] because
   TinyBERT-General's [CLS]/pooler was never trained for sentence tasks, and a mean is the steadier
   sentence summary for 2- and 4-layer encoders.
4. **Heads.** Eight linear sentence heads on the pooled vector: task (6), report_person (2),
   approach_person (2), on_person_found (3), on_link_loss (4), return_when_done (2),
   first_branch (4), speed (2). One linear token head does BIO over
   O, B/I-DUR, B/I-DIST, B/I-BATT, B/I-WP. A word takes the tag of its first wordpiece.
5. **Post-processing** (deterministic, `decode_mission`):
   - Spans become minutes / metres / percent / letter. Digits and words are handled ("thirty",
     "forty-five", "one hundred and fifty", "half an hour", "an hour and a half",
     "a quarter of an hour", "1 hr and 30 mins", "1.5 km", "half a kilometre", "200m",
     "20 per cent", "half battery"). NATO letters work too ("point bravo" gives B).
   - Values are range-checked against the doc (duration 1-240 min, distance 5-2000 m,
     battery 5-90 %); out of range becomes null. Whole numbers are emitted as ints.
   - Task defaults are filled in and the exact key order of the doc is emitted, so the output is
     well-formed by construction.
6. **Confidence** is the minimum over the 8 sentence heads of the max softmax probability.
   `understood = task != UNSUPPORTED and confidence >= threshold and not (GOTO_WAYPOINT without a
   letter)`.
   - **Threshold rule:** the smallest t at which at least 97 % of accepted val examples are fully
     correct. For TinyBERT that is t = 0.00: 98.0 % of val is already correct with no gating, so its
     gate never fires and confidence is informational only. For BERT-Tiny it is t = 0.75.
   - The two confident false accepts above show that confidence does not catch semantic near-misses.
     The dashboard should show the parsed mission to the operator before it is sent.

## Labelling conventions (how the parser reads docs/INTERFACES.md)

Defaults when the instruction is silent. The SEARCH_SURVIVORS line of the doc is taken as the full
default record, and each other task overrides only what the doc lists:

| task | report | approach | on_person_found | on_link_loss | return_when_done | branch | speed |
|---|---|---|---|---|---|---|---|
| SEARCH_SURVIVORS | true | true | REPORT_AND_CONTINUE | CONTINUE_THEN_RETURN | true | ANY | NORMAL |
| MAP_AREA | true | **false** | REPORT_AND_CONTINUE | CONTINUE_THEN_RETURN | true | ANY | NORMAL |
| GOTO_WAYPOINT | true | true | REPORT_AND_CONTINUE | **RETURN_NOW** | **false** | ANY | NORMAL |
| RETURN_HOME, HALT | true | true | REPORT_AND_CONTINUE | CONTINUE_THEN_RETURN | **false** | ANY | NORMAL |
| UNSUPPORTED | same as HALT; numbers and waypoint always null; understood false | | | | | | |

Choices the doc leaves open:

- **UNSUPPORTED is made inert** (return_when_done false). **HALT** forces the default
  behaviour fields and keeps only a duration ("wait here for 5 minutes" gives
  max_duration_min 5).
- **GOTO_WAYPOINT without a valid letter** ("go to the waypoint") keeps task GOTO_WAYPOINT with
  waypoint null and understood false, because the doc says the waypoint is required. Named places
  ("go to the canteen") are UNSUPPORTED.
- **A waypoint named inside a search or map** ("find the miner near point C") is kept in `waypoint`
  as an area of interest. The executive may ignore it.
- **SEARCH vs MAP:** the goal decides. Finding or checking for people is SEARCH, including a bare
  "search the mine" and "check the tunnel for survivors". Exploring, mapping, surveying or
  inspecting with people only in "if you see someone..." clauses is MAP ("go and check the right
  tunnel").
- **on_person_found:** "come back as soon as you find one" is REPORT_AND_RETURN. "stay with them"
  and "wait near the survivor" are STAY.
- **on_link_loss:** "come back if the link drops" is RETURN_NOW. "keep going / continue even if
  signal is lost" is CONTINUE. "stop and wait / hold position" is HOLD. "keep going and come back
  after" is CONTINUE_THEN_RETURN.
- **Other behaviour cues:**
  - "don't approach / keep your distance" sets approach false; "go near them" sets it true.
  - "ignore people / don't report" sets report and approach false.
  - "don't come back / stay there when done" sets return false; for GOTO, "and come back" sets it true.
  - "slowly / carefully / don't go too fast / at low speed" means SLOW.
  - "left/right/straight/middle branch first" sets first_branch. "right now", "go straight to
    point C" and "come straight back" do not.
- **Numbers count only as limits.** "search for 30 minutes", "within 200 m" and "come back at 20 %
  battery" are filled in. Context numbers are not: "last seen 2 hours ago", "battery is at 80 %
  now", "panel 4".
- **Treated as UNSUPPORTED:** manual driving ("turn left", "move forward 2 metres"), manipulation
  ("open the door", "dig"), questions, chit-chat and system or destructive commands.
- **Plain Indian English and a few Hinglish words** are covered: "go and check", "do the needful",
  "reverse back" (RETURN_HOME), "wapas aao", "ruko", "dheere".

## Data

- **`data/test_handwritten.jsonl`: 192 hand-written instructions, never used for training, model
  selection or threshold choice.**
  - Mix: SEARCH 67, MAP 29, GOTO 24, RETURN 19, HALT 19, UNSUPPORTED 34.
  - Styles: long multi-clause orders, terse commands, lowercase without punctuation, typos, numbers
    as digits and words, waypoints, off-topic requests and near-misses.
  - Every field is written out by hand in `data/test_handwritten_src.tsv`, and
    `data/build_test_set.py` only expands the codes (`--review` prints each non-default field).
  - The .tsv was created at 05:12 and `gen_data.py` at 05:27.
- **`data/train.jsonl` (20,000) and `data/val.jsonl` (2,000) come from `gen_data.py`** (template
  grammar, deterministic; re-running reproduces the files byte for byte).
  - Mix: SEARCH 32 %, MAP 18 %, GOTO 16 %, RETURN 13 %, HALT 13 %, UNSUPPORTED 8 %.
  - Variety: many short HALT/RETURN commands, synonyms, clause order, fillers, casing, typos, SMS
    spellings and context sentences with distractor numbers.
  - Every slot span is parsed back by the runtime's value parser and must give the intended value.
  - Val never repeats a train sentence.
  - **Phrase hold-out:** about 1 in 10 non-canonical alternatives of each large phrase list (152
    phrasings, listed in `data/heldout_phrasings.txt`) is never used for train and appears only in
    val. 482 val rows contain one. Without this, val is too easy to calibrate a threshold on. The
    cost is that those phrasings were never trained.

## Files

| file | purpose |
|---|---|
| `dhwani_parser.py` | the runtime: tokenizer, NumPy BERT, heads, post-processing, `MissionParser` (Python 3.6 / numpy 1.13) |
| `models/{tinybert4,berttiny}/` | `weights.npz` (float32), `vocab.txt`, `config.json` (architecture, label sets, threshold, training history) |
| `gen_data.py` | generator for train/val |
| `train_parser.py` | fine-tuning on MPS, best-epoch selection by val exact match, threshold, export |
| `parity_test.py` | NumPy vs torch parity (needs `checkpoints/*.pt`, which are Mac-only and not deployed) |
| `evaluate.py` | writes `eval.json` |
| `eval.json` | all metrics, confusion matrices, every hand-written failure with expected vs predicted JSON, parity, rover run |
| `rover_check.py`, `rover_check_output.json` | the rover compatibility run and its output |

Reproduce (Mac, `PY` = the layaenv Python of the interfaces doc):

```
python3 data/build_test_set.py && python3 gen_data.py
$PY train_parser.py --model berttiny     # add --base DIR to use an already-downloaded base model
$PY train_parser.py --model tinybert4
$PY parity_test.py && $PY evaluate.py
```

Deploy: copy `dhwani_parser.py`, `rover_check.py` and `models/` (with the weights from the
GitHub Release) to `~/dhwani_parser/` on the rover, for example with
`scp -r dhwani_parser.py rover_check.py models jetson@<rover-ip>:~/dhwani_parser/`, then run
`python3 rover_check.py` there.

## Known limits

- **Accuracy on real operators is unmeasured** (see "Read the 98 % with care" above).
- **Out-of-scope detection** is only as good as the UNSUPPORTED examples. A task verb followed by
  an unusual object ("search for gold in the mine") can be accepted with high confidence.
- **The TinyBERT threshold is 0** under the specified rule, and it takes 186 ms per call on the
  rover. Prefer BERT-Tiny when latency matters, or recalibrate on real sentences.
- **One value per slot.** The first span of each slot type is used. A tour over several waypoints
  ("go to A then B") keeps only A. Vague amounts ("a few minutes") give null. Input beyond 62
  wordpieces is cut off.
- **English only**, plus the few Hinglish words above. The model is uncased, so "go to a" (the
  letter A) relies on context.
- **Changing a convention** above means editing the generator and retraining (about 10 min on the
  Mac for both models), then re-running `parity_test.py` and `evaluate.py`.
