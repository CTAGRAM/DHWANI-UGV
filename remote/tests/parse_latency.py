#!/usr/bin/env python3
"""Plain-English instructions through the public link: what the rover understood and how long it took."""
import asyncio, json, sys, time
from websockets.asyncio.client import connect
TESTS = [
    "Go through this mine and find any person alive, report where they are, and come back if the link becomes unstable",
    "search the left tunnel for survivors for 20 minutes",
    "map this area and return when battery drops below 30 percent",
    "go slowly and look for people, if you lose signal just stop and wait",
    "find the missing miner and stay with him if you find him",
    "kindly go to point B",
    "come back home",
    "halt now",
    "what is the weather today",
    "explore everything you can reach but dont go further than 100 metres",
]
async def main(url):
    async with connect(url.replace("https://", "wss://").replace("http://", "ws://").rstrip("/") + "/ws") as ws:
        rows = []
        for i, t in enumerate(TESTS):
            t0 = time.time()
            await ws.send(json.dumps({"t": "parse", "text": t, "req": i}))
            while True:
                d = json.loads(await ws.recv()) if True else None
                if isinstance(d, dict) and d.get("t") == "parsed" and d.get("req") == i:
                    break
            m = d["mission"]; rt = (time.time() - t0) * 1000
            extra = {k: m[k] for k in ("waypoint", "max_duration_min", "max_distance_m", "min_battery_pct") if m.get(k) is not None}
            print("%-4.0f ms total | %5.1f ms on rover | %-16s understood=%-5s conf=%.3f link_loss=%-20s person=%-19s first=%s speed=%s %s\n       \"%s\"" % (
                rt, m.get("parse_ms", -1), m["task"], m["understood"], m["confidence"], m["on_link_loss"], m["on_person_found"],
                m["first_branch"], m["speed"], extra, t))
asyncio.run(main(sys.argv[1]))
