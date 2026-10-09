# Setting up DHWANI on a JetRacer

This is how the code in this repository is laid out on our rover, a Waveshare JetRacer Pro ROS AI
kit running the kit's Ubuntu 18.04 and ROS Melodic image, with the user `jetson`. The paths below
are the ones the scripts expect. If yours differ, change them in the scripts.

Nothing in this guide makes the rover move. Driving needs the dashboard PIN, missions need an
operator to press ARM, and the executive stays in test mode until you switch it to live.

## 1. Where each part goes

| On the rover | From this repository | What it is |
|---|---|---|
| `~/mission_node.py`, `~/junction_node.py`, `~/oak_node.py`, `~/policy_runner.py` | `nodes/` | ROS nodes |
| `~/executive_node.py`, `~/set_live.sh` | `executive/` | mission executor and its live switch |
| `~/dhwani_brain/` | `brain/dhwani_brain/`, `brain/bench/` | decision layer, as `~/dhwani_brain/dhwani_brain/` |
| `~/dhwani_parser/` | `parser/dhwani_parser.py`, `parser/rover_check.py`, `parser/models/` | mission parser |
| `~/dhwani_remote/` | `remote/` | dashboard gateway and web page |
| `~/dhwani_oled/` | `oled/` | OLED status display |
| `~/switch_mode.sh`, `~/localise.launch`, `~/waypoint.py` | `tools/` | mapping and localisation |
| `/usr/local/bin/dhwani-autonomy.sh` | `service/dhwani-autonomy.sh` | boot supervisor |
| `/etc/systemd/system/dhwani-autonomy.service` | `service/dhwani-autonomy.service` | systemd unit |

From a laptop on the same network (replace `<rover-ip>`; over the USB cable it is `192.168.55.1`):

```bash
ROVER=jetson@<rover-ip>
scp nodes/*.py executive/executive_node.py executive/set_live.sh tools/switch_mode.sh tools/localise.launch tools/waypoint.py $ROVER:~/
ssh $ROVER 'mkdir -p ~/dhwani_brain ~/dhwani_parser ~/dhwani_remote ~/dhwani_oled'
scp -r brain/dhwani_brain brain/bench $ROVER:~/dhwani_brain/
scp -r parser/dhwani_parser.py parser/rover_check.py parser/models $ROVER:~/dhwani_parser/
scp -r remote/* $ROVER:~/dhwani_remote/
scp oled/* $ROVER:~/dhwani_oled/
```

## 2. Parser weights

Download `tinybert4-weights.npz` and `berttiny-weights.npz` from the
[v1.0 release](https://github.com/CTAGRAM/DHWANI-UGV/releases/tag/v1.0) and copy each one to
its model folder on the rover as `weights.npz`:

```bash
scp tinybert4-weights.npz $ROVER:~/dhwani_parser/models/tinybert4/weights.npz
scp berttiny-weights.npz  $ROVER:~/dhwani_parser/models/berttiny/weights.npz
ssh $ROVER 'cd ~/dhwani_parser && python3 rover_check.py'
```

Until the weights are there, the dashboard uses a simple keyword parser and marks each parsed
mission with `"parser": "keyword-fallback"`. It looks for the trained model again every minute.

## 3. Dashboard dependencies

The gateway runs under the rover's Python 3.6 and takes its extra packages from
`~/dhwani_remote/pylib`, so the system Python is left alone:

```bash
ssh $ROVER 'cd ~/dhwani_remote && python3 -m pip install --target pylib "numpy==1.19.5" "websockets<10" segno'
```

NumPy 1.19.5 brings its own OpenBLAS, which makes the TinyBERT parser about twice as fast as the
system NumPy. The gateway sets `OPENBLAS_CORETYPE=ARMV8` and `OPENBLAS_NUM_THREADS=1` itself; without
the first the Nano crashes with "Illegal instruction", and four threads were three times slower
than one because they fight ROS for the CPU.

For the public link, put Cloudflare's `cloudflared` binary for linux-arm64 at
`~/dhwani_remote/bin/cloudflared`. The quick tunnel it opens needs no Cloudflare account. Without
it the dashboard still works on the local network.

## 4. Start the dashboard

```bash
ssh $ROVER
cd ~/dhwani_remote
export DHWANI_PIN=<choose your own>      # or leave it unset: a random PIN is made and kept in pin.txt
./start_remote.sh --dry-run              # test mode: every motion command goes to a *_dryrun topic
```

It prints a public link, the local addresses, the PIN and a QR code. Open the link on any phone or
laptop. STOP works for everyone; driving and missions need the PIN, and only one person can
control the rover at a time. When everything looks right, restart it without `--dry-run`.
`./status_remote.sh` reports what is running and `./stop_remote.sh` stops it.

On a Mac, `tools/dhwani_start.command` does all of this with a double-click: it finds the rover
(USB cable first, then Wi-Fi), starts the dashboard and opens it. Run it once with
`DHWANI_SSH_PASS` set and `--install-key` so later logins use an SSH key instead of a password.

## 5. Boot service

The supervisor waits until the ROS master really answers, starts the mission node, the junction
node and, if an OAK-D is on the USB bus, the camera bridge, and restarts any of them that die.

```bash
sudo cp ~/dhwani-autonomy.sh /usr/local/bin/ && sudo chmod +x /usr/local/bin/dhwani-autonomy.sh
sudo cp ~/dhwani-autonomy.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now dhwani-autonomy
```

(Copy the two files from `service/` to the rover first.) The OLED display starts from the `jetson`
user's crontab: `@reboot /home/jetson/dhwani_oled/start.sh`.

## 6. Mapping and localisation

gmapping and AMCL both publish the `map` to `odom` transform, so only one of them may run at a
time. `switch_mode.sh` makes sure of that:

```bash
~/switch_mode.sh mapping                       # build a new map
~/switch_mode.sh localise ~/maps/<map>.yaml    # drive a saved map with AMCL
~/switch_mode.sh status
```

`mapping` starts `~/gmapping25.launch`, our copy of the kit's gmapping launch file with the map
limited to plus or minus 25 m, which keeps each map message small enough for the dashboard. Point
the script at your own launch file if you use another. Save a map with
`rosrun map_server map_saver -f ~/maps/<name>`; the map from our 74 m tour is in `maps/`.

The kit's planner defaults (up to 1.0 m/s, 0.1 m inflation) are too fast for indoor runs. We set,
with `dynamic_reconfigure`: TEB `max_vel_x` 0.28, `max_vel_x_backwards` 0.12, `acc_lim_x` 0.5,
`min_obstacle_dist` 0.25, `inflation_dist` 0.35, `weight_kinematics_forward_drive` 50, and a global
costmap `inflation_radius` of 0.25.

## 7. Missions and going live

The executive turns a parsed mission into planner goals. It starts in test mode, where it makes
every decision and shows it on the dashboard but sends nothing to the planner or the wheels:

```bash
~/set_live.sh off    # (re)start the executive in test mode
~/set_live.sh        # show the current setting
~/set_live.sh on     # live: drives itself only after an operator presses ARM on the dashboard
```

Go live only with someone standing next to the rover, and switch back with `~/set_live.sh off`
when you are done. On mirror-like floors, switch the camera's drop detection off: the stereo camera
sees reflections below the floor there and reports drops that are not real.

## 8. The camera

`nodes/oak_node.py` is the ROS side of the camera. It publishes `/oak/scan`, `/oak/cliff`,
`/oak/person` and `/oak/status` from the OAK-D pipeline process, which runs DepthAI v3 in a
separate Python 3.10 environment because DepthAI 3 does not run on the Nano's Ubuntu 18.04. That
pipeline is not published in this repository yet. Without a camera on the USB bus the boot
supervisor does not start the bridge, and the dashboard shows no camera.

## 9. Tests

```bash
cd brain && python3 -m unittest discover -s dhwani_brain/tests -t .     # decision layer, on a laptop
ssh $ROVER 'cd ~/dhwani_brain && python -m unittest discover -s dhwani_brain/tests -t .'   # on the rover
cd parser && python3 parity_test.py && python3 evaluate.py              # parser, needs the weights
python3 remote/tests/test_remote.py https://<link>/ <test-pin>          # dashboard, against --dry-run
```

`test_remote.py` checks on the rover what arrived, over ssh; set `DHWANI_ROVER_SSH` if your login
differs from `ssh jetson@10.42.0.1`. Use a throwaway PIN for tests, never the one you drive with.
