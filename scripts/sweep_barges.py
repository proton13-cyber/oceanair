"""Barge fleet-sizing sweep — an operations-research feasibility study.

Question: given the boats' fuel burn, what (n_barges x barge_tank) combinations can
sustain the fleet (no deaths) at the hardest deep-sea stage, and what's the
few-big-barges vs many-small-barges tradeoff frontier?

Uses the scripted heuristic as the logistics controller (fast, no training, works for
any barge count), so we can sweep the whole grid in seconds. Reports, per cell, the
death-free rate over N seeds and mean catches (throughput).

    python -m scripts.sweep_barges
"""
from __future__ import annotations

import argparse
import numpy as np

from fishing_rl.config import Config
from fishing_rl.config_io import load_config
from fishing_rl.env import FishingEnv
from fishing_rl.heuristic import HeuristicPolicy


def evaluate(n_barges, tank, seeds, stage, max_steps, base_config=None):
    cfg = load_config(base_config) if base_config else Config()
    cfg.n_barges = n_barges
    cfg.barge.tank = float(tank)
    cfg.max_steps = max_steps
    env = FishingEnv(cfg)
    env.set_stage(stage)
    pol = HeuristicPolicy(cfg)
    deaths, catches, lengths = [], [], []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(pol(env))
            done = any(term.values()) or any(trunc.values())
        e = info[env.agents[0]]["episode"]
        deaths.append(1 if e["deaths"] > 0 else 0)
        catches.append(e["catches"])
        lengths.append(e["length"])
    survive = 1.0 - np.mean(deaths)          # fraction of seeds with zero deaths
    return survive, float(np.mean(catches)), float(np.mean(lengths))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--barges", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--tanks", type=int, nargs="+",
                    default=[200, 300, 400, 500, 600, 800, 1000, 1200])
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--stage", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=2500)
    ap.add_argument("--config", type=str, default=None,
                    help="TOML base config; the sweep varies n_barges x tank on top of it")
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    print(f"Fleet-sizing sweep | stage {args.stage} | {args.seeds} seeds | "
          f"{args.max_steps} steps | boats=4 (fixed burn)")
    print("Controller: scripted heuristic.  Cell = survive%% (mean catches)\n")

    # header
    hdr = "n_barges \\ tank | " + " | ".join(f"{t:>5d}" for t in args.tanks)
    print(hdr)
    print("-" * len(hdr))
    frontier = {}
    for n in args.barges:
        cells = []
        min_ok = None
        for t in args.tanks:
            surv, catch, length = evaluate(n, t, seeds, args.stage, args.max_steps,
                                           base_config=args.config)
            flag = "*" if surv >= 1.0 else " "   # * = fully sustainable
            cells.append(f"{int(surv*100):3d}{flag}({catch:4.0f})")
            if surv >= 1.0 and min_ok is None:
                min_ok = t
        frontier[n] = min_ok
        print(f"{n:>14d} | " + " | ".join(cells))

    print("\nFrontier — minimum tank for 100%% sustainable (0 deaths, all seeds):")
    for n in args.barges:
        mt = frontier[n]
        if mt is None:
            print(f"  {n} barges: NONE in range (need bigger tanks or more barges)")
        else:
            print(f"  {n} barges: tank >= {mt:5d}   (total fuel capacity {n*mt:6d})")


if __name__ == "__main__":
    main()
