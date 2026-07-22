"""Watch the scripted heuristic policy — validates the sim before any training.

    python -m scripts.play_heuristic
"""
from __future__ import annotations

import argparse

from fishing_rl.config import Config
from fishing_rl.config_io import load_config
from fishing_rl.env import FishingEnv
from fishing_rl.heuristic import HeuristicPolicy
from fishing_rl.render import Renderer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None,
                    help="TOML parameter file (see fishing.toml); CLI flags override it")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--stage", type=float, default=1.0,
                    help="curriculum stage 0=easy .. 1=hard (matches the sweeps' default)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--window", type=int, default=1000, help="window size in pixels")
    ap.add_argument("--barges", type=int, default=None, help="override n_barges")
    ap.add_argument("--tank", type=float, default=None, help="override barge tank size")
    ap.add_argument("--max-steps", type=int, default=None, help="override episode length")
    ap.add_argument("--seed", type=int, default=None,
                    help="fixed seed to replay a specific scenario")
    ap.add_argument("--record", type=str, default=None,
                    help="save the replay to this video file, e.g. runs/demo.mp4")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else Config()
    if args.barges is not None:
        cfg.n_barges = args.barges
    if args.tank is not None:
        cfg.barge.tank = args.tank
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    env = FishingEnv(cfg)
    env.set_stage(args.stage)
    policy = HeuristicPolicy(cfg)
    renderer = Renderer(env, window=args.window, fps=args.fps, record_path=args.record)

    for ep in range(args.episodes):
        seed = None if args.seed is None else args.seed + ep
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            if not renderer.draw():
                renderer.close()
                return
            actions = policy(env)
            obs, rew, term, trunc, infos = env.step(actions)
            done = any(term.values()) or any(trunc.values())
        info = infos[env.agents[0]].get("episode", {})
        print(f"episode {ep}: {info}")

    renderer.close()


if __name__ == "__main__":
    main()
