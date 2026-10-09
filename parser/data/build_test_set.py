"""Convert the hand-written test_handwritten_src.tsv into test_handwritten.jsonl (full mission JSON).

Every field is written out by hand in the .tsv; this script only expands the codes. With --review it
prints, for each item, the fields that differ from the per-task defaults so each override can be
checked by eye.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = {"S": "SEARCH_SURVIVORS", "M": "MAP_AREA", "G": "GOTO_WAYPOINT", "R": "RETURN_HOME",
        "H": "HALT", "U": "UNSUPPORTED"}
BOOL = {"T": True, "F": False}
FOUND = {"RC": "REPORT_AND_CONTINUE", "RR": "REPORT_AND_RETURN", "ST": "STAY"}
LINK = {"CR": "CONTINUE_THEN_RETURN", "RN": "RETURN_NOW", "C": "CONTINUE", "H": "HOLD"}
BRANCH = {"ANY": "ANY", "L": "LEFT", "R": "RIGHT", "STR": "STRAIGHT"}
SPEED = {"N": "NORMAL", "SL": "SLOW"}


def num(s):
    if s == "-":
        return None
    v = float(s)
    return int(v) if v == int(v) else v


def convert(line):
    text, codes = line.rstrip("\n").split("\t")
    c = codes.split()
    assert len(c) == 12, line
    task = TASK[c[0]]
    wp = None if c[8] == "-" else c[8]
    m = {
        "task": task,
        "report_person": BOOL[c[1]],
        "approach_person": BOOL[c[2]],
        "on_person_found": FOUND[c[3]],
        "on_link_loss": LINK[c[4]],
        "return_when_done": BOOL[c[5]],
        "first_branch": BRANCH[c[6]],
        "speed": SPEED[c[7]],
        "waypoint": wp,
        "max_duration_min": num(c[9]),
        "max_distance_m": num(c[10]),
        "min_battery_pct": num(c[11]),
        "confidence": 1.0,
        "understood": task != "UNSUPPORTED" and not (task == "GOTO_WAYPOINT" and wp is None),
        "text": text,
    }
    return m


def main():
    src = os.path.join(HERE, "test_handwritten_src.tsv")
    out = os.path.join(HERE, "test_handwritten.jsonl")
    rows = []
    with open(src) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            rows.append(convert(line))
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote", len(rows), "items to", out)
    if "--review" in sys.argv:
        sys.path.insert(0, os.path.dirname(HERE))
        from dhwani_parser import task_defaults
        for r in rows:
            d = task_defaults(r["task"])
            diff = {k: r[k] for k in d if r[k] != d[k]}
            print("%-16s %-60s %s" % (r["task"][:16], r["text"][:60], diff))


if __name__ == "__main__":
    main()
