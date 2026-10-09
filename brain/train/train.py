# -*- coding: utf-8 -*-
"""Train the learned candidate chooser: behaviour-cloning warm start, then PPO on the episode score.

Run on a desktop machine with PyTorch installed:
    PY=python3
    $PY train/train.py --out train/run1

WHY THIS RECIPE. The objective is the episode score (reports, getting home, time), under partial
observability. An expert "stronger than the rules" would have to be privileged (know the true map
or where people are) or search (minutes per decision); imitating a privileged expert teaches the
student that expert's average action, not the best action for what the student can see. PPO
optimises the score directly with the student's own information. BC on the rules is only the
warm start: it makes the first PPO policy as good as the baseline instead of random, so PPO
spends its samples on improvements rather than on relearning the basics.

Seeds: training mines 1,000,000+ (drift), validation 2,000,000+ (drift) for checkpoint choice and
calibration. The held-out sets (9,000,000+ drift, 9,500,000+ lattice) are never touched here.
"""
from __future__ import division, print_function, absolute_import

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")

TRAIN0, VAL0 = 1000000, 2000000
CONDS = ["nominal", "rockfall", "linkloss", "both"]
LAYERS = ("phi1", "phi2", "psi1", "psi2", "rho1", "rho2", "tau1", "tau2", "v1", "v2")


def build_net(torch, nc, ng, h):
    nn = torch.nn
    F = torch.nn.functional

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.phi1, self.phi2 = nn.Linear(nc, h), nn.Linear(h, h)
            self.psi1, self.psi2 = nn.Linear(ng + 2 * h, h), nn.Linear(h, h)
            self.rho1, self.rho2 = nn.Linear(2 * h, h), nn.Linear(h, 1)
            self.tau1, self.tau2 = nn.Linear(2 * h, h), nn.Linear(h, 1)
            self.v1, self.v2 = nn.Linear(h, h), nn.Linear(h, 1)
            for name in ("c_mean", "c_std", "g_mean", "g_std"):
                self.register_buffer(name, torch.zeros(nc if name[0] == "c" else ng))

        def forward(self, C, present, allowed, G):
            C = (C - self.c_mean) / self.c_std.clamp(min=1e-3)
            G = (G - self.g_mean) / self.g_std.clamp(min=1e-3)
            e = F.relu(self.phi2(F.relu(self.phi1(C))))
            pm = present.float().unsqueeze(-1)
            mean = (e * pm).sum(1) / pm.sum(1).clamp(min=1.0)
            mx = e.masked_fill(~present.unsqueeze(-1), -1e9).max(1).values
            mx = torch.where(present.any(1, keepdim=True), mx, torch.zeros_like(mx))
            hh = F.relu(self.psi2(F.relu(self.psi1(torch.cat([G, mean, mx], -1)))))
            eh = torch.cat([e, hh.unsqueeze(1).expand(-1, e.shape[1], -1)], -1)
            logit = self.rho2(F.relu(self.rho1(eh))).squeeze(-1).masked_fill(~allowed, -1e9)
            dwell = self.tau2(F.relu(self.tau1(eh))).squeeze(-1)
            # the critic reads a DETACHED context: value regression must not move the policy's
            # trunk (with a shared trunk, the critic warm-up alone scrambled the policy in run 3)
            v = self.v2(F.relu(self.v1(hh.detach()))).squeeze(-1)
            return logit, dwell, v
    return Net()


def export(net, path, meta):
    d = {}
    for name in LAYERS:
        lin = getattr(net, name)
        d[name + "_W"] = lin.weight.detach().cpu().numpy().T.astype(np.float32).copy()
        d[name + "_b"] = lin.bias.detach().cpu().numpy().astype(np.float32).copy()
    for name in ("c_mean", "c_std", "g_mean", "g_std"):
        d[name] = getattr(net, name).detach().cpu().numpy().astype(np.float32)
    d["calib_a"] = np.array(meta.get("calib_a", 1.0), dtype=np.float32)
    d["calib_b"] = np.array(meta.get("calib_b", 0.0), dtype=np.float32)
    d["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    np.savez(path, **d)


def cat_batches(parts):
    parts = [p for p in parts if p is not None]
    out = {}
    for k in parts[0]:
        out[k] = np.concatenate([p[k] for p in parts])
    return out


def to_t(torch, b, idx=None):
    sel = (lambda a: a[idx]) if idx is not None else (lambda a: a)
    return (torch.as_tensor(sel(b["C"]), dtype=torch.float32), torch.as_tensor(sel(b["present"])),
            torch.as_tensor(sel(b["allowed"])), torch.as_tensor(sel(b["G"]), dtype=torch.float32))


def summarize(ms):
    ms = [m for m in ms if m is not None]
    if not ms:
        return {}
    return {"n": len(ms), "score": float(np.mean([m["score"] for m in ms])),
            "reported": float(np.mean([m["reported"] for m in ms])),
            "found": float(np.mean([m["found"] for m in ms])),
            "return": float(np.mean([m["returned"] for m in ms])),
            "lost": float(np.mean([m["lost"] for m in ms])),
            "time": float(np.mean([m["time_s"] for m in ms]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "run"))
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--bc-episodes", type=int, default=400)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--episodes-per-iter", type=int, default=160)
    ap.add_argument("--val-mines", type=int, default=60)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--minutes", type=float, default=150.0, help="stop PPO after this wall time")
    ap.add_argument("--epochs", type=int, default=4, help="PPO epochs per batch")
    ap.add_argument("--target-kl", type=float, default=0.02, help="stop the epochs early past this KL")
    ap.add_argument("--init", default=None, help="start from this .npz (skip BC) - e.g. a previous bc.npz")
    ap.add_argument("--seed0", type=int, default=TRAIN0, help="first training mine seed (training range only)")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(8)
    from rollout import bc_job, ppo_job, eval_job
    from dhwani_brain.features import NC, NG, K_SLOTS
    os.makedirs(a.out, exist_ok=True)
    log = {"args": vars(a), "iters": [], "val": []}
    t0 = time.time()
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(a.procs)
    rng = np.random.RandomState(0)
    assert TRAIN0 <= a.seed0 < VAL0, "training seeds must stay in the training range"
    next_seed = [a.seed0]

    def train_seeds(n):
        s = list(range(next_seed[0], next_seed[0] + n))
        next_seed[0] += n
        return s

    def save_log():
        with open(os.path.join(a.out, "train_log.json"), "w") as f:
            json.dump(log, f, indent=1)

    # ---------------- 1. behaviour cloning of the rules (warm start) ----------------------------
    if a.init:
        net = build_net(torch, NC, NG, a.hidden)
        z = np.load(a.init)
        with torch.no_grad():
            for name in LAYERS:
                lin = getattr(net, name)
                lin.weight.copy_(torch.as_tensor(z[name + "_W"].T.copy()))
                lin.bias.copy_(torch.as_tensor(z[name + "_b"]))
            for name in ("c_mean", "c_std", "g_mean", "g_std"):
                getattr(net, name).copy_(torch.as_tensor(z[name]))
        print("initialised from %s (BC skipped)" % a.init, flush=True)
        log["init"] = a.init
        wpath = os.path.join(a.out, "current.npz")
        export(net, wpath, {"stage": "init", "from": a.init})
    else:
        jobs = [("drift", s, CONDS[i % 4]) for i, s in enumerate(train_seeds(a.bc_episodes))]
        res = pool.map(bc_job, jobs, chunksize=2)
        B = cat_batches([r[0] for r in res])
        rules_train = summarize([r[1] for r in res])
        n = len(B["k"])
        print("BC data: %d decisions from %d episodes (%.0f s); rules on these: %s" % (
            n, len(jobs), time.time() - t0, json.dumps(rules_train)), flush=True)
        net = build_net(torch, NC, NG, a.hidden)
        Cf = B["C"][B["present"]]
        net.c_mean.copy_(torch.as_tensor(Cf.mean(0)))
        net.c_std.copy_(torch.as_tensor(Cf.std(0) + 1e-3))
        net.g_mean.copy_(torch.as_tensor(B["G"].mean(0)))
        net.g_std.copy_(torch.as_tensor(B["G"].std(0) + 1e-3))
        n_params = sum(p.numel() for p in net.parameters())
        print("network parameters: %d (%d used at inference)" % (
            n_params, n_params - sum(getattr(net, l).weight.numel() + getattr(net, l).bias.numel() for l in ("v1", "v2"))))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        C, P, A, G = to_t(torch, B)
        K = torch.as_tensor(B["k"], dtype=torch.long)
        DW = torch.as_tensor(B["dw"], dtype=torch.float32)
        EX = torch.as_tensor(B["explore"])
        perm_all = np.arange(n)
        for epoch in range(12):
            rng.shuffle(perm_all)
            tot, acc = 0.0, 0.0
            for i in range(0, n, 512):
                idx = torch.as_tensor(perm_all[i:i + 512])
                logit, dwl, _ = net(C[idx], P[idx], A[idx], G[idx])
                logp = torch.log_softmax(logit, -1)
                nll = -logp.gather(1, K[idx, None]).squeeze(1)
                # label smoothing over allowed slots keeps some entropy for PPO to work with
                al = A[idx].float()
                smooth = -(logp * al).sum(1) / al.sum(1).clamp(min=1)
                loss = (0.9 * nll + 0.1 * smooth).mean()
                ex = EX[idx]
                if ex.any():
                    dsel = dwl[ex].gather(1, K[idx][ex][:, None]).squeeze(1)
                    loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(dsel, DW[idx][ex])
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
                acc += float((logit.argmax(1) == K[idx]).float().sum())
            print("  BC epoch %d loss %.4f agree %.3f" % (epoch, tot / n, acc / n), flush=True)
        wpath = os.path.join(a.out, "current.npz")
        export(net, wpath, {"stage": "bc"})
        export(net, os.path.join(a.out, "bc.npz"), {"stage": "bc"})
        # parity: the NumPy inference the rover runs must match torch exactly
        from dhwani_brain.learned import LearnedPolicy
        lp_np = LearnedPolicy(wpath)
        with torch.no_grad():
            tl, td, _ = net(C[:64], P[:64], A[:64], G[:64])
            tp = torch.softmax(tl, -1).numpy()
            tdw = torch.sigmoid(td).numpy()
        err = 0.0
        for i in range(min(64, n)):
            pn, dn = lp_np.forward(B["C"][i], B["present"][i], B["allowed"][i], B["G"][i])
            err = max(err, float(np.abs(pn - tp[i]).max()), float(np.abs(dn - tdw[i]).max()))
        print("numpy/torch parity: max abs diff %.2e" % err, flush=True)
        log["parity_max_abs_diff"] = err

    # ---------------- validation helpers ------------------------------------------------------
    val_jobs = [("drift", VAL0 + i, CONDS[j]) for i in range(a.val_mines) for j in range(4)]
    rules_val = pool.map(eval_job, [(None,) + v + (0.0,) for v in val_jobs], chunksize=2)
    rules_val_s = summarize(rules_val)
    print("rules on validation: %s" % json.dumps(rules_val_s), flush=True)
    log["rules_val"] = rules_val_s
    rv = dict(((m["seed"], m["cond"]), m["score"]) for m in rules_val)

    def validate(path, tag):
        ms = pool.map(eval_job, [(path,) + v + (0.0,) for v in val_jobs], chunksize=2)
        s = summarize(ms)
        diffs = [m["score"] - rv[(m["seed"], m["cond"])] for m in ms]
        s["paired_diff"] = float(np.mean(diffs))
        s["paired_ci"] = float(1.96 * np.std(diffs, ddof=1) / math.sqrt(len(diffs)))
        s["tag"] = tag
        s["wall_min"] = (time.time() - t0) / 60.0
        log["val"].append(s)
        print("VAL %-10s %s" % (tag, json.dumps(s)), flush=True)
        save_log()
        return s

    best = validate(wpath, "bc")
    best_path = os.path.join(a.out, "best.npz")
    export(net, best_path, {"stage": "bc", "val": best})

    # ---------------- 2. PPO ------------------------------------------------------------------
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    clip, vcoef, lam = 0.2, 0.5, 0.95
    for it in range(1, a.iters + 1):
        if (time.time() - t0) / 60.0 > a.minutes:
            print("time budget reached", flush=True)
            break
        seeds = train_seeds(a.episodes_per_iter)
        jobs = [(wpath, "drift", s, CONDS[rng.randint(4)], int(rng.randint(1 << 30))) for s in seeds]
        res = pool.map(ppo_job, jobs, chunksize=2)
        parts, ms = [], []
        for b, m in res:
            ms.append(m)
            if b is None:
                continue
            r = b["r"].astype(np.float64) * 0.1               # points -> training scale
            b["ret_raw"] = r
            parts.append(b)
        B = cat_batches([dict((k, v) for k, v in p.items()) for p in parts])
        # values for GAE (current net), per episode
        with torch.no_grad():
            _, _, V = net(*to_t(torch, B))
        V = V.numpy().astype(np.float64)
        adv = np.zeros(len(V))
        ret = np.zeros(len(V))
        i0 = 0
        for p in parts:
            L = len(p["r"])
            v = V[i0:i0 + L]
            r = p["ret_raw"]
            g = 0.0
            for t in range(L - 1, -1, -1):
                nv = v[t + 1] if t + 1 < L else 0.0
                delta = r[t] + nv - v[t]
                g = delta + lam * g
                adv[i0 + t] = g
            ret[i0:i0 + L] = adv[i0:i0 + L] + v
            i0 += L
        advn = (adv - adv.mean()) / (adv.std() + 1e-8)
        C, P, A, G = to_t(torch, B)
        K = torch.as_tensor(B["k"], dtype=torch.long)
        DW = torch.as_tensor(B["dw"].astype(np.float32))
        EX = torch.as_tensor(B["explore"])
        OLD = torch.as_tensor(B["logp"], dtype=torch.float32)
        ADV = torch.as_tensor(advn, dtype=torch.float32)
        RET = torch.as_tensor(ret, dtype=torch.float32)
        n = len(K)
        idx_all = np.arange(n)
        warm = it <= 2                                        # fit the critic before moving the actor
        stats = []
        for epoch in range(a.epochs):
            rng.shuffle(idx_all)
            ep_kl = []
            for i in range(0, n, 1024):
                idx = torch.as_tensor(idx_all[i:i + 1024])
                logit, dwl, v = net(C[idx], P[idx], A[idx], G[idx])
                logp_all = torch.log_softmax(logit, -1)
                lp = logp_all.gather(1, K[idx, None]).squeeze(1)
                dl = dwl.gather(1, K[idx, None]).squeeze(1)
                lpd = torch.where(DW[idx] > 0.5, torch.nn.functional.logsigmoid(dl),
                                  torch.nn.functional.logsigmoid(-dl))
                lp = lp + torch.where(EX[idx], lpd, torch.zeros_like(lpd))
                ratio = torch.exp(lp - OLD[idx])
                s1 = ratio * ADV[idx]
                s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * ADV[idx]
                pl = -torch.min(s1, s2).mean()
                al = A[idx]
                pr = torch.softmax(logit, -1)
                ent = -(pr * logp_all).masked_fill(~al, 0.0).sum(1).mean()
                vl = ((v - RET[idx]) ** 2).mean()
                loss = vcoef * vl if warm else pl + vcoef * vl - a.ent * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    kl = float((OLD[idx] - lp).mean())
                stats.append((float(pl), float(vl), float(ent), kl))
                ep_kl.append(kl)
            if not warm and np.mean(ep_kl) > a.target_kl:
                break
        export(net, wpath, {"stage": "ppo", "iter": it})
        st = np.mean(stats, axis=0)
        s = summarize(ms)
        s.update({"iter": it, "decisions": int(n), "pi_loss": float(st[0]), "v_loss": float(st[1]),
                  "entropy": float(st[2]), "kl": float(st[3]), "wall_min": (time.time() - t0) / 60.0})
        log["iters"].append(s)
        print("IT %3d %s" % (it, json.dumps(s)), flush=True)
        save_log()
        if it % a.val_every == 0:
            vs = validate(wpath, "it%d" % it)
            if vs["score"] > best["score"]:
                best = vs
                export(net, best_path, {"stage": "ppo", "iter": it, "val": vs})
                print("  new best checkpoint (iter %d)" % it, flush=True)
    log["best"] = best
    save_log()
    pool.close()
    print("done in %.1f min; best %s" % ((time.time() - t0) / 60.0, json.dumps(best)))


if __name__ == "__main__":
    main()
