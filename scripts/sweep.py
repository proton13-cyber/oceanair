"""General parameter sweep for sensitivities — vary any one or two parameters.

    # 2D: barge tank (x) vs fleet size (y)
    python -m scripts.sweep --config fishing.toml \
        --x barge.tank  --x-vals 100000 150000 200000 250000 \
        --y n_barges    --y-vals 4 6 8  --seeds 8

    # 1D: how fish speed drives survival
    python -m scripts.sweep --x fish_speed_frac --x-vals 0.2 0.3 0.4 0.5 0.6

Any dotted parameter path works, e.g.:
    n_barges  barge.tank  boat.idle_burn  boat.turn_agility  harpoon_ammo
    fish_speed_frac  max_fish  grounds_center_x  harpoon_ammo

Each cell = survive% (mean catches) over the seeds at the given curriculum stage,
using the scripted heuristic controller.
"""
from __future__ import annotations

import argparse

import numpy as np

from fishing_rl.config import Config
from fishing_rl.config_io import load_config
from fishing_rl.env import FishingEnv
from fishing_rl.heuristic import HeuristicPolicy


def set_param(cfg, path, value):
    """Set a dotted parameter path (e.g. 'barge.tank'), coercing to the field's type."""
    obj = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        if not hasattr(obj, p):
            raise KeyError(f"unknown parameter path '{path}' (no '{p}')")
        obj = getattr(obj, p)
    leaf = parts[-1]
    if not hasattr(obj, leaf):
        raise KeyError(f"unknown parameter '{path}'")
    cur = getattr(obj, leaf)
    if isinstance(cur, bool):
        value = bool(value)
    elif isinstance(cur, int):
        value = int(round(value))
    elif isinstance(cur, float):
        value = float(value)
    setattr(obj, leaf, value)


def evaluate(base_config, overrides, seeds, stage):
    cfg = load_config(base_config) if base_config else Config()
    for path, val in overrides:
        set_param(cfg, path, val)
    env = FishingEnv(cfg)
    env.set_stage(stage)
    pol = HeuristicPolicy(cfg)
    deaths, catches = [], []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(pol(env))
            done = any(term.values()) or any(trunc.values())
        e = info[env.agents[0]]["episode"]
        deaths.append(1 if e["deaths"] > 0 else 0)
        catches.append(e["catches"])
    return 1.0 - float(np.mean(deaths)), float(np.mean(catches))


def _g(v):
    return f"{v:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="TOML base config")
    ap.add_argument("--x", required=True, help="parameter path to sweep (columns)")
    ap.add_argument("--x-vals", type=float, nargs="+", required=True)
    ap.add_argument("--y", default=None, help="optional 2nd parameter path (rows)")
    ap.add_argument("--y-vals", type=float, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--stage", type=float, default=1.0)
    args = ap.parse_args()

    if args.y and not args.y_vals:
        ap.error("--y requires --y-vals")
    seeds = list(range(args.seeds))
    y_vals = args.y_vals if args.y else [None]

    print(f"Sweep | base={args.config or 'defaults'} | stage {args.stage} | "
          f"{args.seeds} seeds | controller=heuristic")
    print(f"x = {args.x}" + (f"   y = {args.y}" if args.y else ""))
    print("cell = survive% (mean catches);  * = 100% sustained\n")

    ylab = (args.y or "").split(".")[-1]
    hdr = f"{ylab:>12} \\ {args.x.split('.')[-1]:<12} | " \
          + " | ".join(f"{_g(x):>10}" for x in args.x_vals)
    print(hdr)
    print("-" * len(hdr))
    for y in y_vals:
        cells = []
        for x in args.x_vals:
            overrides = [(args.x, x)] + ([(args.y, y)] if args.y else [])
            surv, catch = evaluate(args.config, overrides, seeds, args.stage)
            flag = "*" if surv >= 1.0 else " "
            cells.append(f"{int(surv * 100):3d}{flag}({catch:4.0f})")
        lab = _g(y) if y is not None else ""
        print(f"{lab:>12} | " + " | ".join(f"{c:>10}" for c in cells))


if __name__ == "__main__":
    main()
