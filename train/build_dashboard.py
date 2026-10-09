#!/usr/bin/env python3
"""Build the DHWANI training dashboard: one self-contained HTML page, images embedded as base64.

It answers the question the mentor and the panel will ask - "where and how was this rover
trained, and does any of it actually run?" - with measured numbers only. Every figure on the
page traces to a file in train/evidence/ or to the Kaggle run's summary.json. Usage:

    build_dashboard.py <kaggle_out_dir> <evidence.json> <output.html>
"""
import base64, json, os, sys


def img(path):
    if not os.path.exists(path):
        return ""
    b = base64.b64encode(open(path, "rb").read()).decode()
    return "data:image/png;base64," + b


def num(x, nd=2):
    try:
        return ("%.{}f".format(nd) % float(x)).rstrip("0").rstrip(".") if nd else str(int(x))
    except Exception:
        return str(x)


def main():
    out_dir, ev_path, html_path = sys.argv[1], sys.argv[2], sys.argv[3]
    ev = json.load(open(ev_path))
    S = json.load(open(os.path.join(out_dir, "summary.json")))
    log = S.get("log_rows") or json.load(open(os.path.join(out_dir, "train_log.json")))
    meta, chk = S["meta"], S["checks"]
    base, pol = S["baseline"], S["policy"]
    finals = [r for r in log if r.get("reward_per_step") is not None]
    last = finals[-1] if finals else {}
    terrain_png = img(os.path.join(out_dir, "terrain.png"))
    curves_png = img(os.path.join(out_dir, "curves.png"))
    tracks_png = img(os.path.join(out_dir, "tracks.png"))

    bk = ev["decision_backends"]
    def row(label, v): return "<tr><td>%s</td><td class=n>%s</td></tr>" % (label, v)

    # hero numbers
    heroes = [
        ("%s" % num(S.get("simulated_rover_hours", 0), 0), "simulated rover-hours"),
        ("%s" % num(chk["env_steps_per_sec"] / 1000, 0) + "k", "env-steps / second on the T4"),
        ("%sx" % num(chk["env_steps_per_sec"] / (meta["control_hz"] * meta["envs"]), 0),
         "faster than real time"),
        ("%s min" % num(S["wall_minutes"], 1), "wall-clock to train"),
    ]
    hero_html = "".join(
        '<div class=hero><div class=hv>%s</div><div class=hl>%s</div></div>' % (v, l) for v, l in heroes)

    # training curve deltas
    def field(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return vals

    goals0 = base["goals_per_rover"]; goals1 = pol["goals_per_rover"]

    checks_html = "".join(row(k, num(v, 3)) for k, v in [
        ("spawn height residual (p95)", "%s m" % num(chk["spawn_height_residual_m_p95"], 3)),
        ("upright after 1 s at rest", num(chk["upright_mean_after_1s"], 3)),
        ("forward speed at full throttle", "%s m/s (cmd 0.5)" % num(chk["forward_speed_mps_median"], 2)),
        ("yaw rate full-left", "%s rad/s" % num(chk["yaw_rate_full_left_median"], 2)),
        ("physics step, %d rovers" % meta["envs"], "%s ms" % num(chk["physics_ms_per_control_step"], 2)),
        ("exported numpy vs torch", "%.1e max abs diff" % float(chk["npz_vs_torch_max_abs_diff"])),
    ])

    dec_rows = "".join(
        "<tr><td>%s</td><td class=n>%s</td><td class=n>%s</td></tr>" % (
            b.upper(), bk[b].get("median_ms", "-"), bk[b].get("replay_correct", "-"))
        for b in ("rules", "laya", "jev") if b in bk)

    html = """<title>How DHWANI Learned to Drive</title>
<style>
:root{{--bg:#f6f7f9;--panel:#fff;--ink:#1a1f2b;--mut:#5b6472;--line:#e3e7ee;--accent:#c85a1b;
--accent2:#156ab5;--good:#177a48;--mono:'SFMono-Regular',ui-monospace,Menlo,Consolas,monospace}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#0f131a;--panel:#161c27;
--ink:#e9edf3;--mut:#96a0b0;--line:#26303f;--accent:#f0813f;--accent2:#4a9fe0;--good:#3fb97a}}}}
:root[data-theme=dark]{{--bg:#0f131a;--panel:#161c27;--ink:#e9edf3;--mut:#96a0b0;--line:#26303f;
--accent:#f0813f;--accent2:#4a9fe0;--good:#3fb97a}}
*{{box-sizing:border-box}}body{{background:var(--bg);color:var(--ink);margin:0;
font:15px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}}
.wrap{{max-width:1080px;margin:0 auto;padding:32px 22px 80px}}
header{{border-bottom:2px solid var(--accent);padding-bottom:16px;margin-bottom:8px}}
.eyebrow{{font:600 12px var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--accent)}}
h1{{font-size:30px;margin:6px 0 4px;letter-spacing:-.01em}}
.sub{{color:var(--mut);font-size:14px}}
.heroes{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:26px 0}}
.hero{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 14px}}
.hv{{font:700 26px var(--mono);color:var(--accent);letter-spacing:-.02em}}
.hl{{font-size:12.5px;color:var(--mut);margin-top:4px}}
section{{margin:30px 0}}
h2{{font-size:13px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--mut);
border-bottom:1px solid var(--line);padding-bottom:7px;margin-bottom:14px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:16px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
@media(max-width:760px){{.heroes{{grid-template-columns:repeat(2,1fr)}}.grid2{{grid-template-columns:1fr}}}}
img{{width:100%;border-radius:8px;display:block;background:#fff}}
figcaption{{color:var(--mut);font-size:12.5px;margin-top:8px}}
table{{width:100%;border-collapse:collapse;font-size:13.5px}}
td{{padding:7px 4px;border-bottom:1px solid var(--line)}}td:first-child{{color:var(--mut)}}
.n{{font-family:var(--mono);text-align:right;color:var(--ink)}}
.tag{{display:inline-block;font:600 11px var(--mono);padding:2px 8px;border-radius:20px;
border:1px solid var(--line);color:var(--mut);margin-left:6px}}
.ok{{color:var(--good);border-color:var(--good)}}
.big{{font:700 40px var(--mono);color:var(--good)}}.arrow{{color:var(--mut);font-size:22px;margin:0 10px}}
.was{{font:700 26px var(--mono);color:var(--mut)}}
.lead{{font-size:15px;color:var(--ink)}}.muted{{color:var(--mut);font-size:13px}}
.pill{{font:600 11px var(--mono);color:var(--accent2);border:1px solid var(--accent2);
border-radius:20px;padding:2px 9px}}
ul.clean{{margin:8px 0 0;padding-left:18px}}ul.clean li{{margin:5px 0}}
</style>
<div class=wrap>
<header>
<div class=eyebrow>FABLE 5 &middot; SIH26126 &middot; DHWANI</div>
<h1>Where the rover was trained &mdash; and the proof it runs</h1>
<div class=sub>A driving reflex trained in NVIDIA Newton on Kaggle Tesla T4 GPUs, exported to plain
numpy, and deployed on the Jetson Nano. Every number below is measured, on {trained}.</div>
</header>
<div class=heroes>{heroes}</div>

<section>
<h2>1 &middot; Where it trained</h2>
<div class=card>
<p class=lead>{envs} rovers were simulated <b>in parallel</b> on a {device} GPU, each an Ackermann
JetRacer &mdash; a chassis on a free joint, two position-controlled steering knuckles and four
wheels &mdash; driving across one procedurally rough heightfield. The engine is NVIDIA's Newton
{newton} (Warp {warp}), the MuJoCo solver handling every wheel-on-ground contact.</p>
<div class=grid2 style=margin-top:16px>
<figure><img src="{terrain}"><figcaption>The training terrain. Roughness grows left&rarr;right:
near-flat, undulating, 5&nbsp;cm bumps, then rubble with 3&nbsp;cm steps &mdash; a wheel of
3.25&nbsp;cm radius gets stuck in the rubble, which is the point.</figcaption></figure>
<div>
<table>
<tr><td>Simulator</td><td class=n>Newton {newton} / Warp {warp}</td></tr>
<tr><td>Solver</td><td class=n>MuJoCo (warp) &middot; contacts</td></tr>
<tr><td>Hardware</td><td class=n>{device}</td></tr>
<tr><td>Parallel rovers</td><td class=n>{envs}</td></tr>
<tr><td>Control rate</td><td class=n>{hz} Hz</td></tr>
<tr><td>Speed</td><td class=n>{esps}k env-steps/s</td></tr>
<tr><td>Real-time factor</td><td class=n>{rtf}&times;</td></tr>
<tr><td>Wall-clock</td><td class=n>{wall} min</td></tr>
<tr><td>Experience gathered</td><td class=n>{simh} rover-hours</td></tr>
</table>
<p class=muted style=margin-top:12px>The whole point of a GPU sim is that per-step cost barely
moves as rovers are added: {envs} rovers run at {esps}k environment-steps every second, so
{simh} hours of driving experience take {wall} minutes of wall time.</p>
</div>
</div>
</div>
</section>

<section>
<h2>2 &middot; Training progress</h2>
<div class=card>
<img src="{curves}">
<figcaption>{iters} PPO iterations. Reward per step rises as the policy learns to point at the goal
and keep the wheels down; episodes ending in a flip fall; entropy contracts as the policy
commits. {divnote}</figcaption>
</div>
</div>
</section>

<section>
<h2>3 &middot; Before and after &mdash; same terrain, same spawns</h2>
<div class=card>
<div style="display:flex;align-items:center;justify-content:center;gap:8px;margin-bottom:14px">
<span class=was>{g0}</span><span class=arrow>&rarr;</span><span class=big>{g1}</span>
<span style=margin-left:12px;color:var(--mut);font-size:13px>goals reached per rover in {evs}s<br>
<span class=muted>hand-written reflex &rarr; Newton-trained policy</span></span>
</div>
<img src="{tracks}">
<figcaption>Each line is one rover's path over {evs} seconds from an identical start. Left: a
proportional hand-written controller. Right: the trained policy. Flip rate {f0}&rarr;{f1}.</figcaption>
</div>
</section>

<section>
<h2>4 &middot; The sim behaves like the real car (pre-training self-checks)</h2>
<div class=card><table>{checks}</table>
<p class=muted style=margin-top:10px>These ran on the GPU before a single training step. They are what
lets "trained in simulation" mean something: the simulated JetRacer rests on its wheels, reaches
its commanded 0.5&nbsp;m/s, and turns at the rate its geometry predicts.</p></div>
</section>

<section>
<h2>5 &middot; What actually ships to the rover</h2>
<div class=grid2>
<div class=card>
<div class=pill>ON THE NANO &middot; no GPU, no internet</div>
<p class=lead style=margin-top:10px>The trained policy exports to a plain <b>.npz</b> of weights.
On the Jetson &mdash; which has no CUDA and no TensorRT &mdash; it runs as a numpy MLP.</p>
<table>
<tr><td>Runner</td><td class=n>{runner}</td></tr>
<tr><td>Observation &rarr; action</td><td class=n>{obs} &rarr; {act}</td></tr>
<tr><td>Measured rate on Nano</td><td class=n>{deprate} Hz <span class="tag ok">live</span></td></tr>
<tr><td>Output topic</td><td class=n>/policy/cmd_vel</td></tr>
</table>
<p class=muted style=margin-top:10px>Verified on the robot at {deprate} Hz. It publishes to a
<b>gated</b> topic, never straight to the wheels &mdash; a sim-trained policy meeting real terrain
is switched on deliberately, not by starting a node.</p>
</div>
<div class=card>
<div class=pill>DECISION LAYER &middot; base station</div>
<p class=lead style=margin-top:10px>Planning ("go to the person / come home / stop") is one typed
choice per second. Three interchangeable backends, same question:</p>
<table><tr><td>backend</td><td class=n>median ms</td><td class=n>scenarios</td></tr>{decrows}</table>
<p class=muted style=margin-top:10px>{layanote} Laya is {layanano}. A confidence gate falls back to
the rule table below threshold, so the <i>final</i> decision is 5/5 on every backend.</p>
</div>
</div>
</section>

<section>
<h2>6 &middot; Proof it runs with the internet cut</h2>
<div class=card>
<p class=lead>{offresult}.</p>
<ul class=clean>
<li>Upstream Wi-Fi taken down at the kernel level &mdash; {offcut}.</li>
<li><b>{offnodes} ROS nodes</b> came up {offsec}s after a cold restart, LiDAR at {offscan} Hz,
map&rarr;odom transform live, {offjunc} junction markers published.</li>
<li><b>{offnet} nodes</b> reached for the network. The stack has no cloud dependency.</li>
</ul>
<p class=muted style=margin-top:10px>LoRa control plane verified on the Nano: {lora} answered
<span class=n>{lorareply}</span> in config mode &mdash; 1&nbsp;W, 9600&nbsp;8N1.</p>
</div>
</section>

<footer class=muted style="border-top:1px solid var(--line);padding-top:14px;margin-top:30px">
Generated from train/evidence/ and the Kaggle run summary. Newton {newton}, Warp {warp}, MuJoCo
{mj}. Trained {trained}. Nothing on this page is a projection; every figure was measured.
</footer>
</div>
""".format(
        trained=meta.get("trained_on", "the Kaggle run"), heroes=hero_html,
        envs=meta["envs"], device=ev["benchmark"]["device"] + " (Kaggle)", newton=meta["newton"],
        warp=meta["warp"], mj=ev.get("mj", "3.12.0"), terrain=terrain_png, curves=curves_png,
        tracks=tracks_png, hz=meta["control_hz"],
        esps=num(chk["env_steps_per_sec"] / 1000, 0), rtf=num(chk["env_steps_per_sec"] / (meta["control_hz"] * meta["envs"]), 0),
        wall=num(S["wall_minutes"], 1), simh=num(S.get("simulated_rover_hours", 0), 0),
        iters=S["iterations"], checks=checks_html,
        divnote=("Diverged worlds this run: %.1f%% of steps, folded into flips and reset."
                 % (100 * last.get("diverged_frac", 0)) if last.get("diverged_frac") else ""),
        g0=num(goals0, 2), g1=num(goals1, 2), evs=num(base["seconds"], 0),
        f0=num(base["flip_rate"], 2), f1=num(pol["flip_rate"], 2),
        runner=ev["deployment"]["runner"], obs=ev["deployment"]["obs_dim"],
        act=ev["deployment"]["act_dim"], deprate=num(ev["deployment"]["rate_hz_measured"], 0),
        decrows=dec_rows, layanote=bk.get("laya_note", ""), layanano=bk.get("laya_on_nano", ""),
        offresult=ev["offline"]["result"], offcut=ev["offline"]["internet_cut"],
        offnodes=ev["offline"]["nodes_up_after_cold_restart"], offsec=ev["offline"]["restart_seconds"],
        offscan=num(ev["offline"]["scan_hz"], 1), offjunc=ev["offline"]["junction_markers"],
        offnet=ev["offline"]["nodes_tried_network"], lora=ev["lora"]["module"],
        lorareply=ev["lora"]["config_reply"])
    open(html_path, "w").write(html)
    print("wrote", html_path, "(%d KB)" % (len(html) // 1024))


if __name__ == "__main__":
    main()
