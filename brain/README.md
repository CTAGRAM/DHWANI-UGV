# DHWANI brain: the rover's on-board decision layer

> The decision layer was first written for search and rescue in underground tunnels, where GPS
> never works, so its simulator, tests and held-out evaluation use generated tunnel maps. The same
> rules run DHWANI's missions on the rover, on the map built by the rover's own SLAM.

`dhwani_brain` decides what the rover does next underground: which frontier to explore, whether
to go and look at a person sighting, when to turn back, where to regain radio contact, when to
hold still. It runs on the Jetson Nano in pure NumPy under **python 2.7 and 3.6 with numpy
1.13.3**. It is a library for the ROS executive: it never publishes anything, never touches a
motion topic, and starts no nodes.

```python
from dhwani_brain import Brain
brain = Brain(mission_dict, params=None)   # one Brain per mission
decision = brain.decide(obs)               # every ~10 s, on goal reached, and on events
```

**Status, 25 Sep 2026.** The rules brain is complete and is what should run. 57 tests pass on the Jetson Nano under python 2.7.17 and 3.6.9 (numpy 1.13.3) and on the Mac. On the Nano a full decision on a deep 60 m mine takes 163 ms median (168 ms p99), of which the robot and home wavefronts take 87-125 ms; the rules' choice itself is 0.03 ms and the learned network's 2.6 ms. On 1600 held-out simulated missions (400 unseen mines, two generator styles, with and without rockfalls and link loss) the rules score -9.28 +- 0.85, report 0.70 people per mission, come home in 78 % and are lost in 11 % (82 % of those were trapped: no way home existed in the true map). The learned policy (38 k parameters, BC + PPO) scores -9.37 +- 0.85; paired difference -0.09 +- 0.14 (drift -0.13 +- 0.19, lattice -0.05 +- 0.21). **It does not beat the rules, so it ships disabled (`use_learned` False, the default).** The specified score rewards ending searches early (section 6); read it together with people reported and lost rate.

Contents: [calling it from ROS](#1-calling-the-brain-from-the-ros-executive) ·
[contract](#2-the-contract) · [design](#3-design) · [mission semantics](#4-mission-semantics) ·
[learned policy](#5-learned-policy) · [simulator and score](#6-simulator-and-score) ·
[evaluation](#7-evaluation-held-out-mines) · [latency on the rover](#8-latency-on-the-rover) ·
[tests](#9-tests) · [files](#10-files) · [limits](#11-limits-and-known-issues) ·
[reproduce](#12-reproduce)

---

## 1. Calling the brain from the ROS executive

A minimal python2.7 executive loop. Everything the brain needs is already on the rover's topics.

```python
import json, math, time
import numpy as np
import rospy, tf
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
from rospy.numpy_msg import numpy_msg          # /map data arrives as a NumPy array, not a tuple
from dhwani_brain import Brain

class Executive(object):
    def __init__(self):
        self.brain, self.map, self.persons, self.blocked, self.trail = None, None, [], [], []
        self.t0 = self.last_hb = None
        self.last_link_pose = None
        self.tf = tf.TransformListener()
        rospy.Subscriber("/mission/plan", String, self.on_plan)
        rospy.Subscriber("/map", numpy_msg(OccupancyGrid), lambda m: setattr(self, "map", m))
        rospy.Subscriber("/dashboard/heartbeat", String, lambda m: setattr(self, "last_hb", time.time()))
        self.state = rospy.Publisher("/executive/state", String, queue_size=1, latch=True)

    def on_plan(self, msg):
        mission = json.loads(msg.data)
        wps = json.load(open("/home/jetson/waypoints.json"))           # for GOTO_WAYPOINT
        self.brain = Brain(mission, params={"waypoints": wps})
        self.t0, self.odo, self.persons, self.blocked, self.trail = time.time(), 0.0, [], [], []

    def obs(self):
        m = self.map
        (x, y, _), q = self.tf.lookupTransform("map", "base_footprint", rospy.Time(0))
        yaw = tf.transformations.euler_from_quaternion(q)[2]
        link_ok = self.last_hb is not None and time.time() - self.last_hb < 3.0   # or LoRa permit fresh
        if link_ok:
            self.last_link_pose = (x, y)
        return {
            "grid": m.data.reshape(m.info.height, m.info.width),     # int8, read-only is fine
            "res": m.info.resolution,
            "origin": (m.info.origin.position.x, m.info.origin.position.y),
            "pose": (x, y, yaw), "home": self.home,
            "battery_pct": self.battery_pct(), "link_ok": link_ok,
            "since_contact_s": 0.0 if link_ok else time.time() - (self.last_hb or self.t0),
            "elapsed_s": time.time() - self.t0, "odo_m": self.odo,
            "persons": self.persons,          # [{"x","y","confirmed","reported","hits"}] from /oak/person
            "blocked": self.blocked,          # [(x, y, r)]: move_base aborts, new rubble
            "last_link_pose": self.last_link_pose,
            "trail": self.trail[-60:],        # recent (x, y), one per metre
        }

    def step(self):
        d = self.brain.decide(self.obs())
        self.state.publish(String(json.dumps({"decision": d})))       # dashboard sees reason + candidates
        # act on d["kind"] / d["goal"] / d["sensor"] only when armed (see below)
```

Use `numpy_msg` (or convert once per map message, not per decision): measured on the Nano, turning
the 1984 x 1984 `data` tuple rospy normally delivers into an array takes 481 ms, while
`np.frombuffer`, which is what `numpy_msg` does, takes 4.4 ms.

**A concrete obs**, captured from simulated mission `drift` seed 4247 at t = 178 s (the grid is
the 725 x 319 int8 map the rover had built by then, at 0.1 m):

```python
obs = {"grid": grid, "res": 0.1, "origin": (0.0, 0.0),
       "pose": (27.58, 11.87, -2.03), "home": (2.19, 35.55),
       "battery_pct": 68.3, "link_ok": False, "since_contact_s": 73.0,
       "elapsed_s": 178.0, "odo_m": 53.0,
       "persons": [{"x": 27.93, "y": 9.6, "confirmed": False, "reported": False, "hits": 1}],
       "blocked": [], "last_link_pose": (27.1, 31.91),
       "trail": [(28.18, 14.66), (27.7, 13.64), (27.7, 12.44)]}
d = Brain({"task": "SEARCH_SURVIVORS", ...}).decide(obs)
# d["kind"] == "VERIFY", d["goal"] == (27.7, 11.1, -1.421), d["sensor"] == "DWELL_SCAN", d["tier"] == "mission"
# d["reason"] == "unconfirmed sighting at (27.9, 9.6) (1 hit); going 0.9 m to look and dwell 5 s"
```

**What to do with the decision** (the simulator's executive in `dhwani_brain/sim/runner.py` does
exactly this and is the reference):

| `kind` | executive action |
|---|---|
| `EXPLORE`, `VERIFY`, `RETURN_HOME`, `RETURN_TO_LINK` | send `goal` (x, y, yaw) to move_base, e.g. through `goto.Nav.send` (it refuses goals on unknown ground or in obstacles). `d["path"]` is the brain's own route, usable as via points. |
| `HOLD` | `goal` None: cancel goals and stay. `goal` set: drive there (a safe open spot, or next to a person), then stay. |
| `DONE` | cancel goals, report the mission finished. |
| `sensor == "DWELL_SCAN"` | on arrival, stop for `params["dwell_s"]` (5 s) so the camera can look. |

Ask again when the goal is reached (after the dwell), when move_base aborts (add
`(x, y, 0.4)` at the failure point to `blocked`), every 10 s while driving, and immediately on an
event: a new or confirmed sighting, link up or down, new rubble (cells that were free in the map
and are now occupied: add small `(x, y, 0.1)` discs on them). A decision is about 0.1-0.2 s on the
Nano (section 8), so none of this is time-critical. Keep the executive DRY-RUN unless armed, as
`/executive/arm` specifies; the brain has no notion of arming.

`link_ok` should be true when **either** transport is alive (dashboard heartbeat or LoRa
permission), otherwise a mission sent over LoRa would look like a mission without link.

## 2. The contract

`obs` and `decision` are exactly section 3 of `docs/INTERFACES.md`. Additions (all
additive; nothing in the interfaces doc was changed):

| key | where | meaning |
|---|---|---|
| `tier` | decision | `safety`, `mission` or `discretionary`: which rule layer decided (section 3.5) |
| `path` | decision | the brain's geodesic route to `goal`, one point per 0.4 m |
| `status` | decision | battery, floor, reserve, distance home, frontier count, latched return, grid size |
| `timing_ms` | decision | map / geodesic / frontiers / total |
| `rules_decision`, `learned` | decision | when the learned policy decided: what the rules would have done, its confidence |
| `target` | decision | `"WAYPOINT"` for GOTO_WAYPOINT (reported as kind EXPLORE: the doc has no GOTO kind) |
| `slot`, `feasible`, `why_not`, `d_robot_m`, `d_home_m`, `chosen`, `p_learned` | each candidate | per-candidate detail for the dashboard |
| `params["waypoints"]` | Brain params | waypoint letter -> {"x","y","yaw_deg"} (the rover's waypoints.json) |
| `params["start_yaw"]` | Brain params | heading at mission start, if the Brain is re-created mid-mission |

A HOLD may carry a goal (go there, then hold). All numbers in the decision are plain Python
floats, so `json.dumps(decision)` works as-is. If anything inside the brain raises, `decide`
returns a HOLD whose reason starts with "brain error" instead of raising.

## 3. Design

### 3.1 Planning grid (`mapview.py`)
The 5 cm gmapping grid (1984 x 1984 on this rover, mostly unknown) is cropped to the known
bounding box and pooled to 0.2 m cells: **occupied** if any fine cell is occupied (walls are never
averaged away), **free** if at least half the fine cells are known free, else unknown. Occupied
cells are inflated by 0.15 m (a disc, centre to cell edge) and the executive's `blocked` discs are
painted in with the same margin. `passable` = free and not inflated and not blocked. Within 1 m of
the robot, free cells inside the inflation band also count as passable: lidar keeps refining walls
and the band can close behind a robot that squeezed past a boulder (found in the simulator: the
robot looked sealed in a pocket it had just driven into). Pooling views the int8 grid as uint8 so
unknown (-1) becomes 255 and each class is one comparison; this cut pooling on the Nano from 32 to
11.5 ms for a 60 m map.

### 3.2 Geodesic distance (`geodesic.py`)
A vectorised NumPy wavefront: each round takes the open cells whose distance lies in [k, k+1)
and dilates that front one step through a precomputed 8-neighbour table (orthogonal 1, diagonal
sqrt 2, no corner cutting), keeping the smallest arrival per cell. Because the shortest step is 1,
nothing in the bucket can still improve (Dial/Dinitz), so the result **equals Dijkstra exactly**
(tested against a heap Dijkstra on random grids). Work is proportional to the front, not the grid.
The robot, home and last-link fields run in one loop over a stacked table, so the round count is
set by the farthest field; the camera-coverage field is the same code with many sources and a
3 m cut-off. The literal whole-grid dilation is kept (`distances_dense`) for the benchmark: it
gives identical numbers and is 43 to 86 times slower on the Nano (3.5 to 6.6 s for one field,
against 59 to 85 ms; section 8).

### 3.3 Candidates (`frontiers.py`, `features.py`)
* **Frontiers**: free cells 4-adjacent to unknown, clustered by 8-connectivity (union-find),
  clusters longer than 3 m split along their principal axis; goal = the reachable cell nearest the
  centroid (never unknown, never in an obstacle band); yaw faces the opening. Clusters under
  0.6 m are noise.
* **Camera search goals** (SEARCH_SURVIVORS only): lidar sees 8 m but the camera only 4 m, so
  frontier exploration alone maps a dead end from its mouth and never looks into it (the first
  simulator runs found 0 of 3 people that way). The brain keeps its own trail (from `pose` and
  `trail`), marks floor within 3 m geodesic of it as seen, and offers unseen patches of at least
  2 m2 as exploration goals.
* **Verify** standoff points (1.5 m from an unconfirmed sighting, line of sight, reachable),
  **RETURN_HOME**, **RETURN_TO_LINK**, **HOLD**.
* Every candidate gets battery / time / distance / link-silence **feasibility** and a 31-value
  feature vector; the state gets a 45-value global vector. All are listed with their
  normalisation in [`dhwani_brain/FEATURES.md`](dhwani_brain/FEATURES.md), generated from the code.

### 3.4 Energy model and reserve
Energy per metre = `drain_pct_per_m + drain_pct_per_s / speed`, times `path_factor` 1.15 for the
real path being longer than the geodesic. Once 20 m of odometry exist the brain also uses the
**measured** rate (battery used / odometry) if it is worse. Reserve from a point =
energy home x 1.3 + 4 %. The rover turns back when battery <= floor + reserve from 3 m further
on (one decision interval of driving); a frontier is feasible only if battery - energy to it -
dwell - reserve from it >= floor. `floor = max(5 %, min_battery_pct)`. Set `drain_pct_per_m` /
`drain_pct_per_s` from a battery log of the real pack; the defaults are the simulator's nominal
values.

### 3.5 Rules policy (`rules.py`)
Fixed order, every decision with a plain-English reason and a tier:

1. **safety**: robot off the map -> HOLD; no known route home -> look for one only within the
   reserve, else go to radio contact, else HOLD in the most open spot nearby, away from the
   rubble; battery at the return threshold -> RETURN_HOME; time / distance limit -> RETURN_HOME
   (or DONE when not returning).
2. **orders**: RETURN_HOME and HALT tasks; returns latched by earlier rules.
3. **link loss** per `on_link_loss`.
4. **persons** per `on_person_found`, `report_person`, `approach_person`.
5. **explore**: best feasible exploration goal by utility = gain / (distance + 2 m), gain = frontier
   width + 2 x unknown fraction around it, or 0.35 x unseen m2 for search goals; x4 for the
   first_branch sector until taken, x1.3 for the previous goal (hysteresis). DWELL_SCAN at open,
   chamber-like goals.
6. **nothing feasible left**: RETURN_HOME if `return_when_done`, else DONE; under
   CONTINUE_THEN_RETURN, if the rest is only out of reach because of the silence limit, check in
   at the last contact point first.

`safety` and `mission` decisions are final. Only `discretionary` EXPLORE decisions (which goal,
whether to dwell) may go to the learned policy.

## 4. Mission semantics

Where the parser's fields leave room for interpretation, this is what the brain does:

| field | behaviour |
|---|---|
| `min_battery_pct` | a floor the battery must not cross anywhere in the mission, **including the drive home** (the rover turns back early enough to arrive above it) |
| `max_duration_min` | total mission time including the return (with 20 % margin on the time home) |
| `max_distance_m` | an **odometer** budget including the return (it is what `odo_m` measures) |
| `return_when_done` false | limits and "nothing left" end with DONE where the rover stands |
| `on_link_loss` CONTINUE | ignore link loss (reports still trigger a trip to contact, see persons) |
| CONTINUE_THEN_RETURN | keep exploring out of contact; after `link_grace_s` (600 s) of silence go back to the last contact point, then carry on; goals that cannot be reached and brought back within the silence limit are skipped |
| RETURN_NOW | return home, latched: it keeps going home even if the link comes back |
| HOLD | stop and wait for the link; the battery reserve still forces a return |
| `on_person_found` REPORT_AND_CONTINUE | if the person is not reported (no link), go back to the last contact point, then continue |
| REPORT_AND_RETURN | latched return home; the report goes out as soon as the link is back on the way |
| STAY | report first if needed, then hold next to the person |
| `approach_person` | go to a 1.5 m standoff with line of sight and DWELL_SCAN; after 2 looks without confirmation the sighting is written off |
| `first_branch` | relative to the heading at mission start: STRAIGHT within 30 deg, LEFT 30-150 deg, RIGHT the mirror; enforced (tier mission) at the first real choice (two or more frontiers), then released |
| MAP_AREA | lidar frontiers only, no camera search goals, no approaching |
| GOTO_WAYPOINT | kind EXPLORE toward `params["waypoints"][letter]`, DONE on arrival (or return) |
| UNSUPPORTED / not understood | HOLD |

Ending the search is decided by the rules only (section 5.4).

## 5. Learned policy

### 5.1 Network (`learned.py`, weights in `weights/policy.npz`)
DeepSets over the candidate set, NumPy inference only:

```
per candidate   e_i = phi(c_i)              31 -> 64 -> 64, ReLU, shared across candidates
pooled          [mean_i e_i, max_i e_i]     over present candidates (order does not matter)
context         h = psi([g, pooled])        45 + 128 -> 64 -> 64, ReLU
choose          logit_i = rho([e_i, h])     128 -> 64 -> 1, masked softmax over ALLOWED candidates
sensor          dwell_i = tau([e_i, h])     128 -> 64 -> 1, sigmoid; DWELL_SCAN if > 0.5 for the pick
confidence      Platt scaling of the pick's probability (5.5)
```

38,146 parameters at inference (a 4,225-parameter value head exists only in training), float32,
inputs standardised with training means and standard deviations stored in the .npz. The .npz
holds plain arrays plus a JSON blob as uint8, so it loads without pickle on numpy 1.13; outputs
on the rover match the Mac to 8e-9. No GRU memory: the features already carry the history that
matters (time since contact, odometry, previous goal, trail coverage), and a recurrent state is
one more thing that can silently desynchronise when the executive restarts.

### 5.2 What it may decide, and what it may not
Only when the rules' decision is a **discretionary EXPLORE**: which of the up to six feasible
exploration goals (lidar frontiers and camera search goals, already filtered for battery, time,
distance and link-silence feasibility) to take, and whether to DWELL_SCAN there. It never sees an
infeasible or unreachable goal, cannot return, hold, verify, report, end the search, or override
any safety or mission decision. Its pick is re-checked (allowed, feasible, reachable) and used
only if its calibrated confidence >= `learned_threshold`; otherwise the rules' pick stands.
`source` in the decision says which policy decided, `rules_decision` what the rules would have
done, and the reason says so in words.

### 5.3 Training: behaviour cloning of the rules, then PPO on the score
The objective is the episode score under partial observability. An expert that is genuinely
stronger than the rules would have to be privileged (know the true map or where people are) or
search (minutes per decision); imitating a privileged expert teaches the student that expert's
average action, not the best action for what the student can see. PPO optimises the score
directly with the student's own information, so PPO is the training signal. BC on the rules is
only the warm start, so the first PPO policy is as good as the baseline instead of random.

Setup: behaviour cloning on 36,839 discretionary decisions from 400 training mines (drift, seeds
1,000,000+, all four conditions); PPO with clip 0.2, GAE lambda 0.95, gamma 1, reward = score
accrued between consecutive network decisions (x 0.1), Adam, minibatch 1,024, 160 episodes
(about 15,000 decisions) per iteration, 10 NumPy rollout processes; validation every 5 iterations on
50 validation mines x 4 conditions (seeds 2,000,000+), with the rules on the same 200 episodes for a
paired difference; the best validation checkpoint is kept. Four runs, all logged in `train/`:

| run | change | what happened (validation, paired difference to rules, 95 % CI) |
|---|---|---|
| 1 | network could also end the search | sampled episodes beat the rules on score by searching less (0.25-0.32 people found against 0.70); stopped, see 5.4 and `train/run1_aborted.txt` |
| 2 | lr 3e-4, entropy 0.01, 4 epochs | greedy policy did not change: validation at iterations 5 and 10 identical to the digit (+0.12 +- 0.82); entropy rose 0.34 -> 0.61; stopped |
| 3 | lr 1e-3, entropy 0.001, 8 epochs | critic warm-up through the shared trunk scrambled the policy (KL 0.13 in one iteration, entropy 0.7 -> 1.1): -2.09 +- 1.55 at iteration 5; stopped |
| 4 | run 3 + critic on a detached context + KL early stop at 0.02 | 34 PPO iterations (5440 episodes, 511946 decisions, 46 min). Paired validation difference at iterations 5: +0.38 +- 0.72, 10: +0.12 +- 0.82, 15: +0.09 +- 0.84, 20: +0.24 +- 0.84, 25: -0.10 +- 0.96, 30: +0.10 +- 0.93; entropy stayed 0.29-0.58. Kept: iteration 5 (the best validation score, within noise of the others). **This is the shipped network.** |


### 5.4 Why it is not allowed to end the search
The first PPO run could also pick "end the search". Its sampled episodes immediately scored
better than the rules (-4.2 to -6.5 against -10.8 on training mines) by finding fewer people
(0.25-0.32 against 0.70 per mission) in 506-670 s instead of 1,345 s (`train/run1_aborted.txt`). Under the specified score, 0.01 point per
second outweighs the expected value of searching, and a rover that goes home at t = 0 scores 0.
A network trained on that score learns to abandon searches. That run was stopped: ending the
search is mission semantics and stays with the rules, and the network only orders the
exploration and chooses dwell.

### 5.5 Calibrated confidence
**Definition.** conf = P(the network's action here scores at least as well as the rules' action | p),
with p = p(goal) x p(dwell choice) from the network. **Measurement**, not assumption: at random
discretionary decisions of greedy learned rollouts the whole simulation is copied and played out
twice under identical random numbers, once with each action, the network deciding afterwards in
both; the label is 1 if the network's branch scored at least as well. `conf = sigmoid(a logit(p) + b)`
(Platt scaling) is fitted on half the probes and its reliability checked on the other half.

Measured on validation mines 2,000,500-2,000,559 (all four conditions, 8 probe points per mission;
`train/finalize.py`, log `train/run4/calibration.json`): at 486 probed discretionary decisions the
network's action (goal and dwell) was the same as the rules' action 483 times. At the 3 where it
differed, the network's branch scored at least as well in 2. Platt scaling fitted on half the
probes gives a = 8.17, b = 2.24; on the other half the expected calibration error is 1.4e-05 (raw
probabilities: 0.112). That near-perfect number only says that both the confidence and the outcome
are about 1 almost everywhere: with 3 disagreements there is **no data to calibrate the
low-confidence range**, so in practice the gate (`learned_threshold` 0.5) almost never blocks
the network.

### 5.6 Verdict
Decision rule, fixed before the held-out run: enable the network by default only if the paired score difference over all held-out episodes has a 95 % CI above zero and its lost rate is not higher. Result: -0.09 +- 0.14 points per mission (1600 paired missions; learned better in 151, worse in 89, identical in 1360), lost rate 11.1 % against 10.9 %, people reported 0.70 against 0.70 per mission. **It does not beat the rules. It ships disabled** (`use_learned` False, the default); enabling it needs `params={'use_learned': True}` and, per `policy_runner.py`'s practice, a deliberate decision on the real rover.

Why it cannot do better here: the network only reorders the rules' own feasible exploration goals and chooses dwell, and the shipped network agrees with the rules at 483 of 486 probed validation decisions. PPO found no change of that ordering that the score could tell apart from noise within 511,946 decisions; the per-mission score has a standard deviation of 17 points, so a real gain of a fraction of a point needs far more episodes than one night gives. The rules are the brain; the learned path is wired, tested, calibrated and measured on the rover, ready for a better-posed objective (see section 11).

## 6. Simulator and score

`dhwani_brain/sim/` (pure NumPy; runs on the rover too, but is meant for the Mac):

* **Mines** (`mapgen.py`), 0.1 m cells, 30-80 m extents, corridors 1.2-2.5 m wide with +-0.1 m
  wall roughness. **drift** (training): a portal on the west edge, a random tree of 5-14 m
  segments bending up to 35 deg, junctions of degree 2-4 at 50-110 deg, dead ends, 0-3 loops,
  chambers at 30-60 % of dead ends plus side chambers, boulders in chambers. **lattice** (held out,
  never used in training): room-and-pillar, a rotated grid of corridors every 7-11 m with collapsed
  segments and pillar-free rooms.
* **Lidar**: 360 rays, 8 m, into an OccupancyGrid-like int8 map. **Motion**: 0.3 m/s (0.18 SLOW)
  along the path, 1 s steps. **Battery**: 0.04 %/m + 0.012 %/s nominal, true rates x U(0.85, 1.25)
  per episode (the brain only knows the nominal ones and what it measures), start 55-100 %.
* **Link**: loss dB = alpha x path metres + beta x corners from home (corners = direction changes
  over 35 deg along the shortest path, measured over 2 m); alpha 0.15-0.4 dB/m, beta 8-16 dB,
  budget 25-45 dB; restored when back in range.
* **Persons**: 0/1/2/3 with probability 0.15/0.35/0.30/0.20; placed in a chamber with probability
  0.45, the last 4 m of a dead end 0.35, a corridor 0.20; 6 m or more from home. **Camera**:
  4 m, line of sight, +-50 deg; hit probability per second 0.30 moving, 0.80 dwelling, scaled by
  (1 - 0.5 (d/4)^2); 3 hits confirm; false positives 0.002 per moving second, which rarely confirm.
* **Rockfall** (some episodes): once odometry passes 40-150 m, a 1.6 m disc of rubble fills the
  rover's own trail 8 m or more behind it; with probability 0.7 at a spot that leaves another way
  home, otherwise anywhere (it may trap the rover).
* **Executive** (`runner.py`): the loop described in section 1.

**Score** (task statement): `10 x reported + 4 x found_not_reported_but_returned - 30 x lost - 0.01 x seconds`
with *found* = a real person confirmed, *reported* = found and the link was up at or after the
confirmation **or the rover got home**, *returned* = ended within 1 m of home, *lost* = battery
empty away from home, trapped (no route home in the true map), holding without a route for
10 min, or still out at the 3 h cap. Two consequences of the definitions as given:

* Because reaching home counts as reporting, `found_not_reported_but_returned` is always 0. It is
  computed anyway; `reported_radio` (reports that went out by radio before the rover was home) is
  logged for anyone who wants the stricter reading.
* **The time term dominates.** A rover that declares the mission over at t = 0 scores exactly 0,
  while the rules policy, searching until done, averages -9.28 on the held-out mines: its missions
  last 1,303 s on average (13 points of time) and report 0.70 people (7 points). The score therefore
  rewards abandoning searches. That is why the learned policy is not allowed to end the search (section 5.4), and why
  persons reported, return rate and lost rate are reported next to the score.

## 7. Evaluation: held-out mines

Held-out mines, never used for training, validation, calibration or any choice of parameters: 200 unseen **drift** mines (seeds 9,000,000+) and 200 **lattice** mines (seeds 9,500,000+, a generator style never used in training). Each mine is run in four conditions (nominal, rockfall, link loss, both) with the same mission, the same world random numbers and both policies, so differences are paired. Mission mix: `sim/runner.py` `sample_mission` (80 % SEARCH_SURVIVORS, 20 % MAP_AREA, all on_link_loss / on_person_found / first_branch options, some time, distance and battery limits). 95 % CIs: mean +- 1.96 standard errors; rates: Wilson intervals. Device: Mac M4 (the simulator), 3200 episodes, 0 errors. Every episode is in `eval/results_heldout.json`.

| mines | condition | policy | n | score (95% CI) | persons found | persons reported (share of all persons) | returned (95% CI) | lost (95% CI) | mission time s |
|---|---|---|---|---|---|---|---|---|---|
| drift | nominal | rules | 200 | -4.52 +- 1.49 | 0.77 +- 0.10 | 0.77 +- 0.10 (46%) | 86% (80%-90%) | 0% (0%-3%) | 1202 |
| drift | nominal | learned | 200 | -4.62 +- 1.48 | 0.76 +- 0.10 | 0.76 +- 0.10 (45%) | 86% (80%-90%) | 0% (0%-3%) | 1202 |
| drift | rockfall | rules | 200 | -15.68 +- 3.06 | 0.77 +- 0.10 | 0.77 +- 0.10 (46%) | 56% (49%-62%) | 34% (27%-40%) | 1328 |
| drift | rockfall | learned | 200 | -15.57 +- 3.09 | 0.77 +- 0.10 | 0.77 +- 0.10 (46%) | 56% (49%-62%) | 34% (27%-40%) | 1322 |
| drift | linkloss | rules | 200 | -7.26 +- 2.13 | 0.69 +- 0.10 | 0.69 +- 0.10 (41%) | 87% (82%-91%) | 3% (1%-6%) | 1321 |
| drift | linkloss | learned | 200 | -7.31 +- 2.13 | 0.68 +- 0.10 | 0.68 +- 0.10 (41%) | 87% (82%-91%) | 3% (1%-6%) | 1321 |
| drift | both | rules | 200 | -17.27 +- 3.32 | 0.62 +- 0.10 | 0.60 +- 0.10 (36%) | 62% (56%-69%) | 28% (23%-35%) | 1477 |
| drift | both | learned | 200 | -17.76 +- 3.35 | 0.61 +- 0.10 | 0.59 +- 0.10 (35%) | 62% (55%-68%) | 30% (24%-36%) | 1481 |
| drift | ALL | rules | 800 | -11.18 +- 1.35 | 0.71 +- 0.05 | 0.70 +- 0.05 (42%) | 73% (69%-76%) | 16% (14%-19%) | 1332 |
| drift | ALL | learned | 800 | -11.32 +- 1.36 | 0.70 +- 0.05 | 0.70 +- 0.05 (42%) | 72% (69%-75%) | 17% (14%-19%) | 1332 |
| lattice | nominal | rules | 200 | -4.91 +- 1.64 | 0.73 +- 0.11 | 0.73 +- 0.11 (49%) | 86% (80%-90%) | 2% (1%-4%) | 1176 |
| lattice | nominal | learned | 200 | -4.86 +- 1.65 | 0.73 +- 0.11 | 0.73 +- 0.11 (49%) | 86% (81%-91%) | 2% (1%-4%) | 1176 |
| lattice | rockfall | rules | 200 | -7.97 +- 2.33 | 0.80 +- 0.11 | 0.80 +- 0.11 (54%) | 78% (72%-83%) | 11% (7%-16%) | 1262 |
| lattice | rockfall | learned | 200 | -7.94 +- 2.29 | 0.80 +- 0.11 | 0.80 +- 0.11 (54%) | 78% (71%-83%) | 11% (7%-16%) | 1259 |
| lattice | linkloss | rules | 200 | -6.48 +- 1.72 | 0.68 +- 0.10 | 0.68 +- 0.10 (45%) | 89% (84%-93%) | 0% (0%-3%) | 1308 |
| lattice | linkloss | learned | 200 | -6.47 +- 1.73 | 0.68 +- 0.10 | 0.68 +- 0.10 (45%) | 89% (84%-93%) | 0% (0%-3%) | 1307 |
| lattice | both | rules | 200 | -10.12 +- 2.29 | 0.61 +- 0.10 | 0.60 +- 0.10 (41%) | 81% (75%-86%) | 9% (6%-14%) | 1347 |
| lattice | both | learned | 200 | -10.42 +- 2.31 | 0.60 +- 0.10 | 0.60 +- 0.10 (40%) | 81% (75%-86%) | 10% (6%-14%) | 1357 |
| lattice | ALL | rules | 800 | -7.37 +- 1.02 | 0.70 +- 0.05 | 0.70 +- 0.05 (47%) | 83% (81%-86%) | 6% (4%-7%) | 1273 |
| lattice | ALL | learned | 800 | -7.42 +- 1.02 | 0.70 +- 0.05 | 0.70 +- 0.05 (47%) | 84% (81%-86%) | 6% (4%-7%) | 1275 |
| ALL | ALL | rules | 1600 | -9.28 +- 0.85 | 0.71 +- 0.04 | 0.70 +- 0.04 (45%) | 78% (76%-80%) | 11% (10%-13%) | 1303 |
| ALL | ALL | learned | 1600 | -9.37 +- 0.85 | 0.70 +- 0.04 | 0.70 +- 0.04 (44%) | 78% (76%-80%) | 11% (10%-13%) | 1303 |

Paired difference learned - rules (same mine, mission, world seed):

| mines | condition | n | score diff (95% CI) | learned better / worse / same |
|---|---|---|---|---|
| drift | both | 200 | -0.50 +- 0.52 | 5 / 11 / 184 |
| drift | linkloss | 200 | -0.06 +- 0.10 | 6 / 6 / 188 |
| drift | nominal | 200 | -0.10 +- 0.14 | 16 / 4 / 180 |
| drift | rockfall | 200 | +0.11 +- 0.51 | 14 / 4 / 182 |
| drift | ALL | 800 | -0.13 +- 0.19 | 41 / 25 / 734 |
| lattice | both | 200 | -0.30 +- 0.56 | 21 / 16 / 163 |
| lattice | linkloss | 200 | +0.01 +- 0.20 | 25 / 13 / 162 |
| lattice | nominal | 200 | +0.05 +- 0.19 | 36 / 13 / 151 |
| lattice | rockfall | 200 | +0.03 +- 0.55 | 28 / 22 / 150 |
| lattice | ALL | 800 | -0.05 +- 0.21 | 110 / 64 / 626 |
| ALL | ALL | 1600 | -0.09 +- 0.14 | 151 / 89 / 1360 |

Learned runs: 219048 decisions, 161873 discretionary (74%), 148463 taken from the network (68% of all).

errors: 0


## 8. Latency on the rover

### 8.1 Geodesic wavefront (`bench/bench_wavefront.py`)

Jetson Nano 4 GB, python 2.7.17, numpy 1.13.3, ROS stack and another session's camera work running
(load average 3.76 and 5.18). Synthetic 60 m x 60 m mines at the rover's 0.05 m resolution (1200 x 1200
int8), mostly explored, robot at the known cell farthest from home. Two runs (5 and 15 repeats,
medians); where they differ both are given. Raw output: `bench/rover_wavefront.json`.

| map | planning grid / graph nodes | longest geodesic | pool + inflate ms | neighbour table ms | 1 field ms | robot + home ms | robot + home + link ms | dense dilation, 1 field |
|---|---|---|---|---|---|---|---|---|
| drift60 | 290 x 213 / 9223 | 112 m (robot 98 m from home) | 19-25 | 8-10 | 77 | 87-125 | 96-97 | 6.2-6.6 s |
| lattice60 | 233 x 213 / 11220 | 78 m (robot 59 m from home) | 16-20 | 8-12 | 59-85 | 72-103 | 83-121 | 3.5-3.7 s |

The two-field wavefront (robot + home) is 87-125 ms on the deeper drift map and 72-103 ms on the
lattice, inside the 150 ms target; with pooling and the neighbour table the geodesic stage totals
114-161 ms and 96-135 ms. Both fields share one loop: within a run the second field added 10.5,
13 and 18 ms in three of the four measurements (49 ms in the noisiest), against 59-85 ms for a
field on its own. The literal dense dilation computes the same numbers and is 43 to 86 times slower.

### 8.2 Decision latency of each policy (`bench/bench_latency.py`)

Same Nano, python 2.7.17, numpy 1.13.3, load average 5.66, shipped weights (38146 parameters).
Raw output: `bench/rover_latency.json`.

| what is timed | calls | median ms | p99 ms |
|---|---|---|---|
| rules policy, choice only (context built once, 6 frontier candidates) | 1000 | 0.030 | 0.065 |
| learned policy, choice only: 11 candidate + global feature vectors and the forward pass | 1000 | 2.605 | 3.329 |
| learned forward pass alone | 1000 | 1.024 | 1.325 |
| full `Brain.decide`, rules, 60 m drift mine at 0.05 m (1200 x 1200) | 25 | 163.3 | 167.7 |
| full `Brain.decide`, learned (same map) | 25 | 165.2 | 166.4 |

Where a full decision's time goes (median, rules): pooling and inflation 18.9 ms, robot and home
wavefronts 96.3 ms, frontiers 7.0 ms, camera coverage and search goals 21.8 ms, the rest
(features, the policy, the path) about 19 ms. Both policies share everything except the choice, so
the network adds about 2.6 ms per decision; a decision every few seconds costs a few percent of
one core. This map is a deliberately deep case (robot 98 m from home); decisions on the real
23 Sep hall map take about 7 ms on the Mac (`bench/real_map_check.py`).


## 9. Tests

`dhwani_brain/tests/` (unittest, no pytest needed), 57 tests: wavefront exactness against a heap
Dijkstra, multi-source and cut-off; frontier clustering, snapping, speckle rejection and camera
search goals; blocked handling (a loop closed behind the rover is taken the long way round and no
goal or path point is ever inside the rubble; a dead end holds; exploring for a route only within
the reserve; mission distance budget ignored when the way home is gone; rubble over home); every
`on_link_loss` option, the silence limit and the person-report interaction; reserve threshold,
min_battery_pct, measured drain, time and distance limits; first_branch LEFT / RIGHT / STRAIGHT,
relative to the start heading, released after the first branch; the decision contract,
JSON-safety, tasks, GOTO_WAYPOINT, bad input, unicode mission fields, persons (verify, give up,
report and return, stay), and the learned-policy gate.

```
cd ~/dhwani_brain && python -m unittest discover -s dhwani_brain/tests -t .     # rover, python2.7
python3 -m unittest discover -s dhwani_brain/tests -t .                          # rover python3.6 / Mac
```

Measured on 25 Sep 2026 (the three learned-gate tests need `weights/policy.npz`; they were
skipped in the first runs, before the weights existed, and ran in the final ones, section 12):

| where | python | numpy | result |
|---|---|---|---|
| Mac (M4) | 3.14.6 | 2.5.0 | 57 tests OK |
| rover (Jetson Nano) | 2.7.17 | 1.13.3 | 57 tests OK, 5.7 s |
| rover (Jetson Nano) | 3.6.9 | 1.13.3 | 57 tests OK, 5.8 s |

Final run with the shipped weights, where the three learned-gate tests also run (`bench/rover_tests.txt`):

| where | python | numpy | result |
|---|---|---|---|
| Mac (M4) | 3.14.6 | 2.5.0 | 57 tests OK, none skipped |
| rover (Jetson Nano) | 2.7.17 | 1.13.3 | 57 tests OK, none skipped, 3.6 s |
| rover (Jetson Nano) | 3.6.9 | 1.13.3 | 57 tests OK, none skipped, 5.1 s |

## 10. Files

```
brain/
  README.md                      this file
  dhwani_brain/                  the package (copy this directory to the rover; nothing else is needed)
    __init__.py                  Brain, DEFAULT_PARAMS, normalize_mission
    brain.py                     Brain: rules always, learned only on discretionary EXPLORE, output formatting
    params.py                    defaults, mission normalisation (interfaces doc section 1 defaults)
    mapview.py                   crop, pool to 0.2 m, inflate, blocked regions, escape zone
    geodesic.py                  vectorised wavefront (exact octile), paths, dense reference
    frontiers.py                 frontier clusters and camera search goals
    features.py                  context: fields, candidates, feasibility, utilities, feature vectors
    rules.py                     the rules policy
    learned.py                   NumPy inference of the network (.npz, no pickle)
    compat.py                    python 2.7 / 3.6 shims
    FEATURES.md                  every feature and its normalisation (generated from features.py)
    weights/policy.npz           trained network + calibration (38 k parameters, float32)
    sim/mapgen.py                drift and lattice mine generators
    sim/world.py                 lidar, motion, battery, link, camera, persons, rockfall
    sim/runner.py                episode loop = reference executive; metrics and score
    tests/                       57 unittest tests + synthetic maps
  train/rollout.py               NumPy-only rollout workers (BC data, PPO, validation, calibration probes)
  train/train.py                 BC warm start + PPO (torch, Mac only)
  train/finalize.py              confidence calibration, packaging of weights/policy.npz
  train/run1_aborted.txt         console lines of the run whose network could end searches (5.4)
  train/run2/ run3/ run4/        logs of the other runs (stdout.log, train_log.json); run2/bc.npz is the
                                 BC warm start, run4/best.npz the shipped network before calibration,
                                 run4/calibration.json the calibration probes summary
  eval/evaluate.py               batch evaluation, CIs, paired differences
  eval/report.py                 results JSON(s) -> markdown tables (merges files)
  eval/fill_readme.py            writes the evaluation section of this README from the JSON
  eval/results_heldout*.json     the held-out evaluation, every episode (rules, learned, merged)
  eval/results_heldout.md        the tables of section 7
  bench/make_bench_maps.py       60 m x 60 m benchmark maps at 0.05 m (bench_maps.npz)
  bench/bench_wavefront.py       wavefront timing (rover)
  bench/bench_latency.py         policy and full-decision latency (rover)
  bench/real_map_check.py        the brain on the rover's real 23 Sep map
  bench/rover_*.json, rover_tests.txt   raw rover measurements and test output quoted in sections 8-9
  examples/run_mission.py        one simulated mission with the rules policy -> PNG
  examples/mission.png           its output for the morning report (log: examples/mission_log.txt)
```

## 11. Limits and known issues

* **Everything learned or evaluated here happened in the simulator.** Its link model, camera
  model and person prior are stated assumptions (section 6), not measurements from a mine. The
  rules are the part to trust on the real rover first; the learned policy should stay off until
  it has been compared against the rules on real runs.
* **Energy defaults are the simulator's** (0.04 %/m, 0.012 %/s). Log the real pack once and set
  `drain_pct_per_m` / `drain_pct_per_s`; the online estimate only makes the reserve more
  conservative, never less.
* **Camera coverage is approximate**: floor within 3 m (geodesic) of the trail counts as seen,
  regardless of where the camera pointed. A person just outside the field of view of a passing
  rover can be missed; DWELL_SCAN at open areas partly compensates.
* **Rockfall losses are mostly unavoidable in tree-shaped mines**: when the only corridor home
  closes, the brain holds in a safe spot with the data (as specified), which still counts as lost.
* **The wavefront cost grows with the longest geodesic in the map** (one round per 0.2 m). A
  150 m-deep map is about 750 rounds, roughly 0.1 s per field on the Nano; beyond ~200 m consider
  `cell_m` 0.25.
* **gmapping loop-closure jumps** move the map under the brain's trail memory; the camera
  coverage then lags until the rover passes again. Nothing unsafe follows from it.
* **The score as specified rewards quitting** (section 6); treat score differences together with
  persons reported and lost rate, never alone.
* **The learned policy adds nothing yet** (section 5.6): same score, people, returns and losses as
  the rules within +-0.14 points on 1,600 paired missions, because it barely disagrees with them.
  What would give it something to learn: a score whose time term does not swamp people found (for
  example 0.001 per second, or a fixed mission budget), far more episodes (the per-mission score
  spread is 17 points), or a stronger teacher than the rules, such as a rollout-based one-step
  lookahead in the simulator (the episode snapshots in `sim/runner.py` already make that possible).
* **Not implemented**: the optional GRU memory (the features already carry the history that
  matters, and a recurrent state is one more thing that can silently desynchronise when the
  executive restarts); move_base plan-failure simulation beyond "goal unreachable".

## 12. Reproduce

```bash
cd brain
PY=python3

python3 -m unittest discover -s dhwani_brain/tests -t .           # tests (Mac)
python3 examples/run_mission.py --out examples/mission.png          # one mission, rules, PNG

# training as it was run (runs 2 and 4; run 2 is only used for its BC warm start)
$PY train/train.py --out train/run2 --bc-episodes 400 --iters 80 --episodes-per-iter 160 --val-mines 50 --minutes 85
$PY train/train.py --out train/run4 --init train/run2/bc.npz --lr 1e-3 --ent 0.001 --epochs 8 --target-kl 0.02 \
    --episodes-per-iter 160 --val-mines 50 --minutes 45 --seed0 1020000
$PY train/finalize.py train/run4/best.npz --out dhwani_brain/weights/policy.npz --mines 60 --probes 8 \
    --log train/run4/calibration.json

# held-out evaluation (the rules half ran at nice 19 while the network trained)
python3 eval/evaluate.py --styles drift lattice --seeds 9000000:9000200 --lattice-seeds 9500000:9500200 \
    --conds all --policies rules --procs 10 --out eval/results_heldout_rules.json
python3 eval/evaluate.py --styles drift lattice --seeds 9000000:9000200 --lattice-seeds 9500000:9500200 \
    --conds all --policies learned --weights dhwani_brain/weights/policy.npz --procs 10 --out eval/results_heldout_learned.json
python3 eval/report.py eval/results_heldout_rules.json eval/results_heldout_learned.json \
    --merged eval/results_heldout.json > eval/results_heldout.md

# on the rover (only inside ~/dhwani_brain; short runs)
ROVER="ssh jetson@<rover-ip>"
tar czf - dhwani_brain bench | $ROVER 'mkdir -p ~/dhwani_brain && tar xzf - -C ~/dhwani_brain'
$ROVER 'cd ~/dhwani_brain && python -m unittest discover -s dhwani_brain/tests -t .'
$ROVER 'cd ~/dhwani_brain && python bench/bench_wavefront.py 5 && python bench/bench_latency.py'
```
