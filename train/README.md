# DHWANI — Newton training pipeline

A terrain-reactive **driving reflex** for the JetRacer testbed: trained in NVIDIA Newton on a
Kaggle GPU, exported to a plain numpy `.npz`, and run on the Jetson Nano by
[`nodes/policy_runner.py`](../nodes/policy_runner.py) with no CUDA, no TensorRT, no ONNX.

It is deliberately **not** the mapper, the planner, or the mission logic — those stay in ROS
(gmapping, move_base/TEB, `mission_node.py`). This is the layer *underneath* move_base that
move_base does not have: modulate throttle and steering on rough ground so the rover keeps its
wheels down and keeps making progress toward the current goal.

## What is simulated

One rover = a 7-body Featherstone articulation, driven by the **MuJoCo (warp) solver**:

| part | joint | control |
|---|---|---|
| chassis (0.30×0.12×0.06 m box, battery-low CoM) | free | — |
| 2 front steering knuckles | revolute Z, ±0.38 rad | position |
| 2 front wheels | revolute Y | free-rolling |
| 2 rear wheels | revolute Y | velocity (torque-limited) |

Real JetRacer numbers throughout: wheelbase 0.20 m, track 0.19 m, wheel radius 32.5 mm, ~1.6 kg,
0.5 m/s top speed, min turning radius 0.5 m. Terrain is one 12×12 m heightfield whose roughness
grows along X (flat → undulating → bumps → rubble).

## The observation — only what the real robot measures

14 numbers, in this exact order (the export encodes it in `obs_spec`):

```
gravity_body(3)  linvel_body(3)  angvel_body(3)  goal_dir_body(2)  goal_dist/5m(1)  prev_action(2)
```

IMU gives gravity + angular rate, odom/EKF gives body velocity, map pose + waypoint gives the
goal. **No wheel encoders** (the JetRacer has none) and **no terrain samples** — so the policy is
a reactive stability/traction reflex, not a terrain planner. Output: `throttle, steering` in
[-1, 1], the two numbers the base takes.

## Why the deployment is numpy, not TensorRT

This Nano's JetRacer image has no CUDA and no TensorRT. An MLP policy is a few matrix multiplies;
numpy on the four A57 cores runs the largest one at ~945 Hz — ten times the 50 Hz control loop.
So we train on a GPU and ship the weights. The `.npz` contract (`W0,b0,…,obs_mean,obs_std,
act_scale,activation`) is exactly what `policy_runner.py --self-test` validates on the robot.

## Run it

```bash
# on any GPU box (Kaggle T4 used here). --smoke = 4 rovers, 2 iters, CPU is fine.
python newton_rover_train.py --envs 1024 --iters 600 --out out
# outputs: policy.npz (best checkpoint), terrain.png, curves.png, tracks.png, summary.json
```

Then on the rover, deliberately (never at boot — a sim-trained policy meeting real terrain is
switched on by hand):

```bash
scp out/policy.npz jetson@<rover>:~/models/policy.npz
# on the rover:
python nodes/policy_runner.py            # loads ~/models/policy.npz, idle
rostopic pub /policy/goal geometry_msgs/PointStamped '{header:{frame_id: map}, point:{x: 2,y: 0}}'
rostopic pub /policy/enable std_msgs/Bool 'data: true'   # now it publishes /policy/cmd_vel
```

`policy_runner.py` publishes to **`/policy/cmd_vel`**, never straight to the wheels — wiring it to
`/cmd_vel` is a deliberate remap, so the policy can be watched before it is trusted.

## Measured (see `evidence/`)

- Physics on a Kaggle T4: 1024 rovers at ~70k env-steps/s, ~1400× real time; 300 rover-hours of
  experience in about 5 wall-minutes.
- Pre-training self-checks (on GPU): rests on its wheels (0.005 m residual), reaches 0.49 m/s at
  full throttle, turns at the rate its geometry predicts.
- Export: numpy inference matches torch to < 1e-6.
- Deployment: `policy_runner.py` runs on the Nano at 50 Hz.

## Files

- `newton_rover_train.py` — the whole trainer (sim + PPO + export + plots + self-checks).
- `build_dashboard.py` — renders `evidence/` + a run's `summary.json` into one HTML dashboard.
- `evidence/` — measured JSON and logs every claim above traces to.
