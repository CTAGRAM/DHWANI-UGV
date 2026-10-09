# DHWANI on-board interfaces

The mission parser, the decision layer (`dhwani_brain`), the executive and the dashboard talk
to each other only through the contracts below. They were fixed on 25 September 2026, before
the parts were built, so each part could be written and tested on its own.

## 1. Mission JSON (parser output, published as JSON on `/mission/plan`, std_msgs/String)
```json
{
  "task": "SEARCH_SURVIVORS | MAP_AREA | GOTO_WAYPOINT | RETURN_HOME | HALT | UNSUPPORTED",
  "report_person": true,
  "approach_person": true,
  "on_person_found": "REPORT_AND_CONTINUE | REPORT_AND_RETURN | STAY",
  "on_link_loss": "CONTINUE_THEN_RETURN | RETURN_NOW | CONTINUE | HOLD",
  "return_when_done": true,
  "first_branch": "ANY | LEFT | RIGHT | STRAIGHT",
  "speed": "NORMAL | SLOW",
  "waypoint": null,
  "max_duration_min": null,
  "max_distance_m": null,
  "min_battery_pct": null,
  "confidence": 0.97,
  "understood": true,
  "text": "original instruction"
}
```
Defaults when the instruction is silent, per task:
- SEARCH_SURVIVORS: report_person true, approach_person true, on_person_found REPORT_AND_CONTINUE,
  on_link_loss CONTINUE_THEN_RETURN, return_when_done true, first_branch ANY, speed NORMAL.
- MAP_AREA: report_person true, approach_person false, on_person_found REPORT_AND_CONTINUE,
  on_link_loss CONTINUE_THEN_RETURN, return_when_done true.
- GOTO_WAYPOINT: waypoint letter (A to Z) required; on_link_loss RETURN_NOW; return_when_done false.
- RETURN_HOME / HALT: everything else default, return_when_done false.
- UNSUPPORTED: understood false. Numbers null unless stated. Ranges: duration 1 to 240 min,
  distance 5 to 2000 m, battery 5 to 90 %.

## 2. Parser API (Python 3.6, numpy 1.13, runs in the dashboard gateway on the rover)
```python
from dhwani_parser import MissionParser
p = MissionParser(model_dir)          # loads npz + vocab once
m = p.parse("go through this area and find anyone alive")   # returns the dict above
```

## 3. Brain API (pure NumPy, must run under BOTH python 2.7 and 3.6 with numpy 1.13)
```python
from dhwani_brain import Brain
brain = Brain(mission_dict, params=None)
decision = brain.decide(obs)
```
`obs` keys:
- `grid`: numpy int8 HxW, -1 unknown, 0..100 occupancy (ROS OccupancyGrid data reshaped, row 0 = origin y)
- `res` (m/cell), `origin` (x0, y0) of cell (0,0) in the map frame
- `pose` (x, y, yaw) and `home` (x, y) in the map frame
- `battery_pct` (estimate), `link_ok` (bool), `since_contact_s`, `elapsed_s`, `odo_m`
- `persons`: list of {"x","y","confirmed","reported","hits"}
- `blocked`: list of (x, y, radius_m) regions to treat as occupied (failed plans, new rubble)
- `last_link_pose`: (x, y) or None; `trail`: list of recent (x, y)

`decision` keys: `kind` in EXPLORE | VERIFY | RETURN_HOME | RETURN_TO_LINK | HOLD | DONE,
`goal` (x, y, yaw) or None, `sensor` in MOVE | DWELL_SCAN, `reason` (short English),
`source` rules | learned, `confidence`, `candidates` (list of {kind, goal, score, features}).

## 4. ROS topics
| Topic | Type | Producer | Consumer |
|---|---|---|---|
| `/mission/plan` | String (mission JSON) | dashboard gateway, LoRa | executive |
| `/executive/state` | String (JSON: mode, phase, decision, reason, candidates) | executive | dashboard |
| `/executive/arm` | Bool | dashboard (PIN) | executive (stays DRY-RUN unless armed AND launched live) |
| `/dashboard/heartbeat` | String (JSON) | gateway, 2 Hz while a controller is connected | executive (link_ok) |
| `/oak/person` | PointStamped, frame base_footprint | oak node | mission_node, executive |
| `/oak/cliff` | Bool | oak node | executive safety |
| `/oak/scan` | LaserScan | oak node | dashboard drive guard; planner costmap (planned) |
| `/oak/status` | String (latched) | oak node | dashboard |
| UDP 127.0.0.1:5601 | raw JPEG datagrams (<60 KB, ~5 fps, 416x234 or smaller) | OAK-D pipeline process (not published yet) | dashboard gateway camera panel |

Existing, unchanged: `/mission/command` (STOP, HOME, GOTO A, TOUR A B, PING), `/mission/permit`,
`/mission/state`, `/events` (JSON), `/events/markers`, `/junctions/poses`, `/map`, `/scan`,
`/odom`, `/imu`, `/tf`, `/move_base/*`, `/csi_cam_0/*`.
