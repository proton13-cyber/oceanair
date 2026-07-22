"""Record one voyage for a single study cell, using the SAME config the study builds.

Loads the study's base_config + all its pinned (non-swept) params, then pins the
swept axes to the cell you name on the CLI, so the video matches a study row exactly.

    python -m scripts.record_cell study.json --tankers 6 --tank 125000 \
        --out runs/6x125k.mp4 --seed 3
"""
from __future__ import annotations

import argparse
import json

from fishing_rl.config import Config
from fishing_rl.config_io import load_config, set_param
from fishing_rl.env import FishingEnv
from fishing_rl.heuristic import HeuristicPolicy
from fishing_rl.render import Renderer

CONTROL = {"runs", "stage", "base_config", "report", "out", "json"}


def build_config(study_path, overrides):
    with open(study_path, "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    base = spec.get("base_config")
    cfg = load_config(base) if base else Config()
    for key, val in spec.items():
        if key.startswith("_") or key in CONTROL:
            continue
        if isinstance(val, (list, dict)):        # swept axis — pinned via overrides
            continue
        set_param(cfg, key, val)
    for key, val in overrides.items():           # the specific cell
        set_param(cfg, key, val)
    return cfg, float(spec.get("stage", 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study", help="study.json used for the sweep")
    ap.add_argument("--tankers", type=int, required=True)
    ap.add_argument("--tank", type=float, required=True)
    ap.add_argument("--out", required=True, help="output mp4 path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    cfg, stage = build_config(args.study, {"tankers": args.tankers, "tank": args.tank})
    env = FishingEnv(cfg)
    env.set_stage(stage)
    policy = HeuristicPolicy(cfg)
    renderer = Renderer(env, window=args.window, fps=args.fps, record_path=args.out)

    env.reset(seed=args.seed)
    done = False
    while not done:
        if not renderer.draw():
            break
        _, _, term, trunc, infos = env.step(policy(env))
        done = any(term.values()) or any(trunc.values())
    info = infos[env.agents[0]].get("episode", {})
    print(f"cell {args.tankers}x{int(args.tank)}: {info}")
    renderer.close()


if __name__ == "__main__":
    main()
