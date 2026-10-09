#!/bin/bash
# Does the new AI slow the rover's own stack down? Measure lidar and odometry rates and CPU while
# (1) nothing extra runs, (2) the parser runs back to back (far worse than real use: it runs once
# per mission), (3) the brain decides back to back (real use: about once every few seconds).
source /opt/ros/melodic/setup.bash
export ROS_MASTER_URI=http://localhost:11311 ROS_IP=127.0.0.1
OUT=${1:-/tmp/ros_impact.txt}
: > "$OUT"
rate() { timeout 12 rostopic hz -w 50 "$1" 2>/dev/null | grep "average rate" | tail -1 | awk '{print $3}'; }
cpu() { top -b -n2 -d2 | grep "^%Cpu" | tail -1 | awk '{print 100 - $8}'; }
measure() {
  local label="$1"
  local s o c l
  s=$(rate /scan); o=$(rate /odom); c=$(cpu); l=$(cut -d' ' -f1 /proc/loadavg)
  echo "$label | scan_hz=$s odom_hz=$o cpu_busy_pct=$c load=$l" | tee -a "$OUT"
}
measure "baseline (parked, full ROS stack)"

python3 - <<'PY' &
import os, sys, time
sys.path.insert(0, os.path.expanduser("~/dhwani_parser"))
from dhwani_parser import MissionParser
p = MissionParser(os.path.expanduser("~/dhwani_parser/models/tinybert4"))
t = time.time()
while time.time() - t < 40:
    p.parse("go through this mine and find any person alive and come back if the link drops")
PY
PID=$!
sleep 6
measure "parser back to back (worst case)"
kill $PID 2>/dev/null; wait $PID 2>/dev/null

python - <<'PY' &
from __future__ import print_function
import os, sys, time, numpy as np
sys.path.insert(0, os.path.expanduser("~/dhwani_brain"))
from dhwani_brain import Brain
g = np.zeros((400, 400), np.int8); g[:] = -1; g[150:250, 20:380] = 0; g[20:380, 180:220] = 0
b = Brain({"task": "SEARCH_SURVIVORS", "understood": True}, {"use_learned": False})
obs = {"grid": g, "res": 0.05, "origin": (-10.0, -10.0), "pose": (0.0, 0.0, 0.0), "home": (0.0, 0.0),
       "battery_pct": 80.0, "link_ok": True, "since_contact_s": 0.0, "elapsed_s": 60.0, "odo_m": 5.0,
       "persons": [], "blocked": [], "last_link_pose": (0.0, 0.0), "trail": [(0.0, 0.0)]}
t = time.time()
while time.time() - t < 40:
    b.decide(obs)
PY
PID=$!
sleep 6
measure "brain back to back (worst case)"
kill $PID 2>/dev/null; wait $PID 2>/dev/null
sleep 3
measure "baseline again"
