#!/bin/bash
# Switch the rover between building a map and driving a saved one.
#
# gmapping and amcl BOTH publish map->odom. Run both and the transform tree has two writers,
# the robot's pose jumps between their estimates, and move_base plans against whichever won
# the last update. So this script's real job is making sure exactly one of them is alive.
#
#   switch_mode.sh mapping              build a new map (what the rover has always done)
#   switch_mode.sh localise [map.yaml]  drive a saved map with AMCL
#   switch_mode.sh status               what is running now
# ROS's own setup.sh references unbound vars; jetracer-autostart.sh disables -u for the
# same reason. Keep it off or every source line aborts the script.
set +u
SRC='source /opt/ros/melodic/setup.bash; source /home/jetson/catkin_ws/devel/setup.bash; export ROS_MASTER_URI=http://localhost:11311; export ROS_IP=127.0.0.1;'
MAP="${2:-/home/jetson/maps/tour_20260923.yaml}"

kill_screen() { screen -ls 2>/dev/null | awk -v n="$1" '$0 ~ n {print $1}' | while read s; do screen -S "$s" -X quit; done; }
running()     { source /opt/ros/melodic/setup.bash; export ROS_MASTER_URI=http://127.0.0.1:11311; rosnode list 2>/dev/null | grep -qx "$1"; }

case "${1:-status}" in
  mapping)
    echo "-> mapping mode"
    kill_screen localise; screen -wipe >/dev/null 2>&1
    source /opt/ros/melodic/setup.bash; export ROS_MASTER_URI=http://127.0.0.1:11311
    rosnode kill /amcl /map_server >/dev/null 2>&1
    sleep 2
    if ! running /slam_gmapping; then
      screen -dmS gmapping bash -c "$SRC roslaunch ~/gmapping25.launch > /tmp/gmapping.log 2>&1"
      echo "   gmapping started (blank map)"
    else
      echo "   gmapping already running"
    fi
    ;;

  localise)
    [ -f "$MAP" ] || { echo "no such map: $MAP" >&2; exit 1; }
    echo "-> localisation mode on $MAP"
    # gmapping must die FIRST or there are two writers on map->odom
    kill_screen gmapping; screen -wipe >/dev/null 2>&1
    source /opt/ros/melodic/setup.bash; export ROS_MASTER_URI=http://127.0.0.1:11311
    rosnode kill /slam_gmapping >/dev/null 2>&1
    sleep 3
    screen -dmS localise bash -c "$SRC roslaunch ~/localise.launch map:=$MAP > /tmp/localise.log 2>&1"
    echo "   map_server + amcl starting..."
    sleep 12
    ;;

  status) ;;
  *) echo "usage: $0 {mapping|localise [map.yaml]|status}" >&2; exit 2 ;;
esac

source /opt/ros/melodic/setup.bash; export ROS_MASTER_URI=http://127.0.0.1:11311
echo "--- status ---"
for n in /slam_gmapping /map_server /amcl /move_base; do
  printf '  %-16s ' "$n"
  rosnode list 2>/dev/null | grep -qx "$n" && echo UP || echo down
done
n=$(rosnode list 2>/dev/null | grep -cE '^/(slam_gmapping|amcl)$')
[ "$n" -gt 1 ] && echo "  !! BOTH gmapping and amcl are up - map->odom has two writers"
exit 0
