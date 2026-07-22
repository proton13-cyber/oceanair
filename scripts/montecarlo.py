"""Monte Carlo stability comparison of fleet configurations.

Runs many random-spawn episodes per config and reports survival probability and the
catch distribution — so you can contrast, e.g., 8 small tankers vs 4 big ones at the
SAME total fuel and see which fleet is more *stable* (survives more often, steadier
catches), not just which has the higher average.

    # explicit configs, "n_barges:tank_lb"
    python -m scripts.montecarlo --config fishing.toml --runs 100 \
        --configs 4:200000 6:133000 8:100000

    # same total fuel, auto-split across fleet sizes
    python -m scripts.montecarlo --config fishing.toml --runs 100 \
        --equal-total 800000 --barges 4 6 8

Each run uses a fresh random seed, so fish spawns/positions differ every episode.
"""
from __future__ import annotations

import argparse

import numpy as np

from fishing_rl.config import Config
from fishing_rl.config_io import load_config
from fishing_rl.env import FishingEnv
from fishing_rl.heuristic import HeuristicPolicy


def evaluate(base_config, n_barges, tank, runs, stage):
    cfg = load_config(base_config) if base_config else Config()
    cfg.n_barges = n_barges
    cfg.barge.tank = float(tank)
    env = FishingEnv(cfg)
    env.set_stage(stage)
    pol = HeuristicPolicy(cfg)
    catches, deaths, lengths = [], [], []
    sea, dock_t, dock_b = [], [], []
    for s in range(runs):
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(pol(env))
            done = any(term.values()) or any(trunc.values())
        e = info[env.agents[0]]["episode"]
        catches.append(e["catches"])
        deaths.append(1 if e["deaths"] > 0 else 0)
        lengths.append(e["length"])          # steps = minutes (1 step = 1 min)
        sea.append(e["sea_refuels"])
        dock_t.append(e["barge_dock_refuels"])
        dock_b.append(e["boat_dock_refuels"])
    c = np.array(catches, dtype=float)
    d = np.array(deaths, dtype=float)
    mean = c.mean()
    # elapsed survival time until first boat lost (only meaningful for failed voyages)
    loss_times = [ln for ln, dd in zip(lengths, deaths) if dd]  # minutes
    return {
        "sea_ref": float(np.mean(sea)), "dock_t_ref": float(np.mean(dock_t)),
        "dock_b_ref": float(np.mean(dock_b)),
        "n": n_barges, "tank": int(tank), "total": int(n_barges * tank),
        "survival": 1.0 - d.mean(),
        "mean": mean, "std": c.std(),
        "min": c.min(), "p10": np.percentile(c, 10), "median": np.median(c),
        "cv": (c.std() / mean) if mean > 0 else float("inf"),
        "loss_med_min": float(np.median(loss_times)) if loss_times else None,
        "loss_min_min": float(np.min(loss_times)) if loss_times else None,
        "catches": catches, "deaths": deaths, "lengths": lengths,   # raw per-episode
    }


def parse_configs(args):
    if args.equal_total:
        return [(n, args.equal_total // n) for n in args.barges]
    out = []
    for spec in args.configs:
        n, tank = spec.split(":")
        out.append((int(n), int(tank)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="TOML base config")
    ap.add_argument("--runs", type=int, default=60, help="random episodes per config")
    ap.add_argument("--stage", type=float, default=1.0)
    ap.add_argument("--configs", nargs="+", default=None,
                    help="explicit configs as n_barges:tank_lb (e.g. 4:200000 8:100000)")
    ap.add_argument("--equal-total", type=int, default=None,
                    help="fixed total fuel; splits it across --barges")
    ap.add_argument("--barges", type=int, nargs="+", default=[4, 6, 8],
                    help="fleet sizes for --equal-total")
    ap.add_argument("--json", type=str, default=None,
                    help="also dump full results (incl. raw per-episode data) to this file")
    args = ap.parse_args()
    if not args.configs and not args.equal_total:
        ap.error("give either --configs or --equal-total")

    configs = parse_configs(args)
    print(f"Monte Carlo | base={args.config or 'defaults'} | {args.runs} runs/config "
          f"| stage {args.stage} | controller=heuristic\n")

    rows = [evaluate(args.config, n, tank, args.runs, args.stage) for n, tank in configs]

    def hrs(m):
        return f"{m/60:>5.1f}h" if m is not None else "   —  "

    hdr = (f"{'fleet':>13} | {'survival':>8} | {'catches (mean±std)':>19} | "
           f"{'CV':>5} | {'time-to-loss med/min':>20} | {'refuels sea/tkr/boat':>20}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['n']:>2} x {r['tank']:>7} | {r['survival']*100:>6.0f}% | "
              f"{r['mean']:>7.0f} ± {r['std']:<7.0f} | {r['cv']:>5.2f} | "
              f"{hrs(r['loss_med_min'])} / {hrs(r['loss_min_min'])} | "
              f"{r['sea_ref']:>6.0f} /{r['dock_t_ref']:>4.0f} /{r['dock_b_ref']:>4.0f}")

    # "most stable" = highest survival, then steadiest catches (lowest CV)
    best = sorted(rows, key=lambda r: (-r["survival"], r["cv"]))[0]
    print(f"\nMost stable: {best['n']} x {best['tank']:,} lb  "
          f"({best['survival']*100:.0f}% survival, CV {best['cv']:.2f}, "
          f"~{best['mean']:.0f} catches).  Lower CV = steadier run-to-run.")

    if args.json:
        import json
        with open(args.json, "w") as fh:
            json.dump({"runs": args.runs, "stage": args.stage, "configs": rows}, fh)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
