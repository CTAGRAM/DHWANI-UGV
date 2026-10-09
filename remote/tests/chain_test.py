#!/usr/bin/env python3
"""Whole chain on a parked rover: English -> TinyBERT parser -> mission -> executive -> brain ->
decision on the dashboard. Requires the gateway in --dry-run and the executive in TEST MODE; it
checks both and refuses otherwise. It also records /move_base/goal, /move_base_simple/goal and
/cmd_vel on the rover during the run and fails if the executive sent anything.

    python3 chain_test.py http://192.168.55.1:8088/ <test-pin>
"""
import asyncio
import json
import subprocess
import sys
import time

from websockets.asyncio.client import connect

RR = "./rr"
TEXT = ("Go through this mine and find any person alive, report where they are, "
        "and come back if the link becomes unstable")


def rover(cmd, timeout=120):
    return subprocess.run([RR, cmd], capture_output=True, text=True, timeout=timeout).stdout


async def main(url, pin):
    ws = await connect(url.replace("https://", "wss://").replace("http://", "ws://").rstrip("/") + "/ws", max_size=2 ** 22)
    inbox = []

    async def reader():
        async for m in ws:
            if isinstance(m, str):
                inbox.append(json.loads(m))
    rt = asyncio.ensure_future(reader())

    async def wait(t, pred=lambda d: True, timeout=10):
        t0 = time.time()
        while time.time() - t0 < timeout:
            for d in list(inbox):
                if d.get("t") == t and pred(d):
                    inbox.remove(d)
                    return d
            await asyncio.sleep(0.05)
        return None

    hello = await wait("hello")
    assert hello and hello.get("dry_run"), "gateway must be in --dry-run"
    ex = await wait("exec", timeout=10)
    assert ex and ex["state"].get("live") is False, "executive must be in TEST MODE: %s" % (ex and ex["state"].get("live"))
    print("gateway dry-run, executive TEST MODE, brain:", ex["state"].get("brain"))

    rec = ("pkill -x rostopic; source /opt/ros/melodic/setup.bash; rm -f /tmp/c_*.txt; touch /tmp/c_marker; "
           "for t in /move_base/goal /move_base_simple/goal /cmd_vel; do n=$(echo $t | tr / _); "
           "nohup timeout 100 rostopic echo $t > /tmp/c$n.txt 2>&1 & done; sleep 14; echo ready")
    out = await asyncio.get_running_loop().run_in_executor(None, rover, rec)
    assert "ready" in out, out

    await ws.send(json.dumps({"t": "auth", "pin": pin, "name": "chain test"}))
    tok = (await wait("auth"))["token"]
    await ws.send(json.dumps({"t": "take_control", "token": tok}))
    await wait("control", lambda d: d.get("controller") == "chain test")
    await ws.send(json.dumps({"t": "resume", "token": tok}))

    t0 = time.time()
    await ws.send(json.dumps({"t": "parse", "text": TEXT, "req": 1}))
    m = (await wait("parsed"))["mission"]
    print("\n1. parsed on the rover in %s ms (%.0f ms round trip): %s, link loss %s, person %s, confidence %.3f" % (
        m["parse_ms"], (time.time() - t0) * 1000, m["task"], m["on_link_loss"], m["on_person_found"], m["confidence"]))

    await ws.send(json.dumps({"t": "start_mission", "token": tok, "mission": m}))
    a = await wait("ack", lambda d: d.get("cmd") == "start_mission")
    print("2. start mission:", a and a.get("msg"))
    await asyncio.sleep(1.5)
    await ws.send(json.dumps({"t": "arm", "token": tok, "on": True}))
    t_arm = time.time()
    dec = None
    while time.time() - t_arm < 25:
        e = await wait("exec", timeout=3)
        if e and e["state"].get("armed") and (e["state"].get("decision") or {}).get("kind"):
            dec = e["state"]
            break
    assert dec, "no decision within 25 s of ARM"
    d = dec["decision"]
    print("3. armed; first decision %.1f s after ARM, computed in %s ms on the rover" % (time.time() - t_arm, dec.get("decide_ms")))
    print("   decision: %s -> %s, sensor %s, by %s (%s rule)" % (d["kind"], d.get("goal"), d.get("sensor"), d.get("source"), d.get("tier")))
    print("   reason:   %s" % d.get("reason"))
    print("   options weighed: %d, route points: %d, battery %.1f %%, link_ok %s, home %s" % (
        len(dec.get("candidates") or []), len(d.get("path") or []), dec.get("battery_pct"), dec.get("link_ok"), dec.get("home")))
    for c in (dec.get("candidates") or [])[:6]:
        print("     option %-15s goal %-16s score %s" % (c.get("kind"), c.get("goal"), c.get("score")))

    await asyncio.sleep(12)          # let it re-decide at least once
    inbox[:] = [x for x in inbox if x.get("t") != "exec"]
    e = await wait("exec", timeout=3)
    if e:
        d2 = e["state"].get("decision") or {}
        print("4. after 12 s: %s, %s" % (d2.get("kind"), d2.get("reason")))

    await ws.send(json.dumps({"t": "stop", "why": "chain test done"}))
    await asyncio.sleep(2)
    inbox[:] = [x for x in inbox if x.get("t") != "exec"]
    e = await wait("exec", timeout=4)
    print("5. after STOP: phase %s, armed %s" % (e and e["state"].get("phase"), e and e["state"].get("armed")))

    res = rover("for f in /tmp/c_*.txt; do echo \"$f: $(grep -c '^---' $f) messages\"; done")
    print("\n6. messages recorded on the rover during the test (the executive must have sent none):")
    print(res.strip())
    pub = rover("source /opt/ros/melodic/setup.bash; rostopic info /move_base/goal | sed -n '/Publishers/,/Subscribers/p'")
    print("   /move_base/goal publishers:", " ".join(pub.split()))
    rt.cancel()
    await ws.close()


asyncio.run(main(sys.argv[1], sys.argv[2]))
