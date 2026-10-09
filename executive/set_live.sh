#!/bin/bash
# Switch the on-board executive between TEST MODE (decides, never moves) and LIVE (drives when ARMED).
#   ./set_live.sh on     LIVE: after an operator presses ARM on the dashboard, the rover drives itself
#   ./set_live.sh off    TEST MODE (default)
#   ./set_live.sh        show the current setting
CFG=~/dhwani_executive.json
python - "$1" <<'PY'
import json, os, sys
p = os.path.expanduser("~/dhwani_executive.json")
try:
    c = json.load(open(p))
except Exception:
    c = {}
a = sys.argv[1] if len(sys.argv) > 1 else ""
if a in ("on", "off"):
    c["live"] = (a == "on")
    json.dump(c, open(p, "w"), indent=1)
print("executive live = %s  (%s)" % (c.get("live", False), p))
PY
if [ "$1" = "on" ] || [ "$1" = "off" ]; then
  # restart the node now so it reads the new setting (the boot supervisor skips it if it is running)
  screen -S executive -X quit 2>/dev/null
  sleep 1
  screen -dmS executive bash -c "source /opt/ros/melodic/setup.bash; source /home/jetson/catkin_ws/devel/setup.bash; \
    export ROS_MASTER_URI=http://localhost:11311 ROS_IP=127.0.0.1; python /home/jetson/executive_node.py > /tmp/executive.log 2>&1"
  echo "executive restarted with the new setting (log: /tmp/executive.log)"
fi
