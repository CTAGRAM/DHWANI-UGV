#!/usr/bin/env python
"""Named places on a saved map: stand somewhere, name it, drive back to it later.

This only means anything in localisation mode. While gmapping is running the map frame is
rebuilt from scratch on every start, so a coordinate saved yesterday points nowhere today -
which is exactly why "return to start" stopped working after the stack restarted on 2026-09-23.
With map_server + amcl on a saved map, a coordinate is durable.

  waypoint.py save hall_end       name wherever the robot is standing now
  waypoint.py list                show them, with distance from here
  waypoint.py go hall_end         drive there
  waypoint.py tour A B A          drive a sequence, reporting error at each stop

Reuses goto.Nav so every goal gets the same standable/reachable checks the manual tool applies -
the costmap treats unknown cells as free, so an unchecked goal into unmapped space gets a route
straight through territory the robot has never seen.
"""
import json, math, os, sys, time
import rospy
from goto import Nav

STORE = os.path.expanduser("~/waypoints.json")
ARRIVE_TOL = 0.15          # m, tightened for the final approach of each leg
ARRIVE_YAW = 0.25          # rad
LEG_TIMEOUT = 180


def load():
    try:
        return json.load(open(STORE))
    except Exception:
        return {}


def save_all(d):
    json.dump(d, open(STORE, "w"), indent=2, sort_keys=True)


def tighten(xy, yaw):
    try:
        import dynamic_reconfigure.client as dc
        for ns in ("/move_base/TebLocalPlannerROS", "/move_base/DWAPlannerROS"):
            try:
                dc.Client(ns, timeout=3.0).update_configuration(
                    {"xy_goal_tolerance": xy, "yaw_goal_tolerance": yaw})
                return True
            except Exception:
                pass
    except Exception:
        pass
    return False


def require_localised(nav):
    """Refuse to drive if gmapping is running - the map frame would not be the saved one."""
    import rosnode
    nodes = rosnode.get_node_names()
    if "/slam_gmapping" in nodes:
        print("REFUSED: gmapping is running, so /map is a NEW map, not the saved one.")
        print("         run:  ~/switch_mode.sh localise")
        return False
    if "/amcl" not in nodes:
        print("REFUSED: amcl is not running - there is no localisation against the saved map.")
        print("         run:  ~/switch_mode.sh localise")
        return False
    return True


def drive_to(nav, name, wp):
    x, y, yaw = nav.pose()
    gx, gy, gyaw = wp["x"], wp["y"], math.radians(wp["yaw_deg"])
    d = math.hypot(gx - x, gy - y)
    print("\n=== %s : (%.2f, %.2f) yaw %.0f, %.2f m away ===" % (name, gx, gy, wp["yaw_deg"], d))
    ok, why, _ = nav.corridor_clear(x, y, gx, gy)
    print("  straight line: %s" % ("clear" if ok else why + " - planner must route around"))
    if not nav.send(gx, gy, gyaw):
        return False
    if not nav.watch(gx, gy, timeout=LEG_TIMEOUT):
        nav.stop()
        return False
    time.sleep(1.0)
    fx, fy, fyaw = nav.pose()
    err = math.hypot(fx - gx, fy - gy)
    dyaw = math.degrees(math.atan2(math.sin(fyaw - gyaw), math.cos(fyaw - gyaw)))
    print("  ARRIVED  position error %.3f m   heading error %.1f deg" % (err, dyaw))
    return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd = sys.argv[1]
    wps = load()

    if cmd == "list":
        if not wps:
            print("no waypoints saved yet")
            return 0
        nav = Nav(); time.sleep(1.5)   # Nav.__init__ calls rospy.init_node itself
        try:
            x, y, _ = nav.pose()
        except Exception:
            x = y = None
        for n in sorted(wps):
            w = wps[n]
            if x is None:
                print("  %-14s (%7.2f, %7.2f) yaw %6.1f" % (n, w["x"], w["y"], w["yaw_deg"]))
            else:
                print("  %-14s (%7.2f, %7.2f) yaw %6.1f   %5.2f m away"
                      % (n, w["x"], w["y"], w["yaw_deg"], math.hypot(w["x"] - x, w["y"] - y)))
        return 0

    nav = Nav(); time.sleep(2.0)       # Nav.__init__ calls rospy.init_node itself

    if cmd == "save":
        name = sys.argv[2]
        x, y, yaw = nav.pose()
        wps[name] = {"x": round(x, 3), "y": round(y, 3), "yaw_deg": round(math.degrees(yaw), 1)}
        save_all(wps)
        print("saved %-12s (%.3f, %.3f) yaw %.1f" % (name, x, y, math.degrees(yaw)))
        return 0

    if cmd in ("go", "tour"):
        names = sys.argv[2:]
        missing = [n for n in names if n not in wps]
        if missing:
            print("unknown waypoint(s): %s" % ", ".join(missing))
            print("known: %s" % (", ".join(sorted(wps)) or "(none)"))
            return 2
        if not require_localised(nav):
            return 3
        tighten(ARRIVE_TOL, ARRIVE_YAW)
        t0 = time.time()
        done = 0
        for n in names:
            if not drive_to(nav, n, wps[n]):
                print("\nFAILED at %s - stopping" % n)
                nav.stop()
                break
            done += 1
        print("\n=== %d/%d legs in %.1f min ===" % (done, len(names), (time.time() - t0) / 60.0))
        nav.stop()
        return 0 if done == len(names) else 1

    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
