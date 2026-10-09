#!/bin/bash
# Start the autonomy layer once the ROS stack is actually up. Run by dhwani-autonomy.service.
#
# WHY IT WAITS RATHER THAN DEPENDS: jetracer.service returns as soon as its bring-up script has
# been *launched*, not when roscore is answering. systemd's After= therefore fires this far too
# early, the nodes fail to reach the master, and the unit dies. Probing the master port is the
# only honest readiness signal - the same reason jetracer-autostart.sh uses a TCP probe instead
# of pgrep (rosmaster is a python script, so its process comm is "python" and pgrep -x misses it).
#
# WHAT IT STARTS
#   mission_node.py    takes orders over LoRa or Wi-Fi, holds the dead-man's switch, fires events
#   junction_node.py   publishes map junctions as markers for Foxglove
#
# Neither commands motion on its own. mission_node only drives when it receives an explicit
# GOTO/TOUR/HOME, and it halts itself the moment operator permission stops arriving.
set +u
SRC='source /opt/ros/melodic/setup.bash; source /home/jetson/catkin_ws/devel/setup.bash; export ROS_MASTER_URI=http://localhost:11311; export ROS_IP=127.0.0.1;'
LOG=/tmp/dhwani-autonomy.log

master_up() { python -c "import socket,sys
s=socket.socket(); s.settimeout(2)
try:
    s.connect(('127.0.0.1',11311)); s.close(); sys.exit(0)
except Exception: sys.exit(1)" 2>/dev/null; }

echo "=== autonomy start $(date) ===" >> $LOG
for i in $(seq 1 60); do
  master_up && break
  sleep 2
done
if ! master_up; then
  echo "FATAL: roscore never came up" >> $LOG
  exit 1
fi
# the master answering is not the same as move_base having registered; give the stack a moment
sleep 8

# Purge dead node registrations left by a previous run. Every node here registers under a
# PID-stamped name, so a hard kill (or a power cut) leaves a ghost publisher behind - which is
# exactly what made a one-shot `rostopic echo /mission/state` read empty after a restart on
# 2026-09-24. rosnode cleanup prompts y/n; feed it y. Harmless when there is nothing to purge.
$SRC yes 2>/dev/null | rosnode cleanup >> $LOG 2>&1 || true

# oak_node runs under python3.6 - depthai has no cp27 wheel - and needs the ROS python path
# added by hand, since Melodic only sets it up for 2.7.
PY3='PYTHONPATH=/opt/ros/melodic/lib/python2.7/dist-packages export ROS_MASTER_URI=http://localhost:11311; export ROS_IP=127.0.0.1;'

oak_present() {   # 03e7 is Movidius/Intel: the OAK enumerates as this whether booted or not
  lsusb -d 03e7: >/dev/null 2>&1
}

start() {   # start() <screen-name> <script>
  if screen -ls 2>/dev/null | grep -q "[.]$1"; then
    echo "  [$1] already running" >> $LOG
    return
  fi
  screen -dmS "$1" bash -c "$SRC python /home/jetson/$2 > /tmp/$1.log 2>&1"
  echo "  [$1] started" >> $LOG
}

start_oak() {
  if screen -ls 2>/dev/null | grep -q "[.]oak"; then return; fi
  screen -dmS oak bash -c "source /opt/ros/melodic/setup.bash; $PY3 python3 /home/jetson/oak_node.py > /tmp/oak.log 2>&1"
  echo "  [oak] started" >> $LOG
}

start mission   mission_node.py
start junctions junction_node.py
# Only start the camera node if a camera is actually plugged in. Without one it would die on
# every launch and systemd would restart-loop it; worse, its own failure handler zeroes
# /cmd_vel, so a restart loop would fight the planner for control of the robot.
if oak_present; then
  start_oak
else
  echo "  [oak] no 03e7 device on the USB bus - skipped" >> $LOG
fi
echo "autonomy up at $(date)" >> $LOG

# Supervise. jetracer-autostart.sh only ever probes port 11311, which is why gmapping, move_base
# and the camera could all die on 2026-09-23 while the supervisor reported the stack healthy.
# Check the actual processes here and restart what has gone.
while sleep 20; do
  master_up || { echo "master gone at $(date) - exiting for restart" >> $LOG; exit 1; }
  start mission   mission_node.py
  start junctions junction_node.py
  oak_present && start_oak
done
