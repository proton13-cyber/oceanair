"""Run a sensitivity study defined entirely in a JSON file — no CLI flags.

    python -m scripts.study study.json

The JSON lists parameters. A single value pins one; a list (or a {"from","to","step"}
range) makes it a swept variable; anything you omit stays at its default. Every cell of
the resulting grid is evaluated as a Monte Carlo (many random voyages), so you get
survival probability and catch stability, not just a mean.

Example study.json::

    {
      "runs": 60,
      "base_config": "fishing.toml",
      "report": "study_report.html",

      "harpoons": 10,                                     // pinned (fixed)
      "boat.idle_burn": 15,                               // pinned

      "tankers": [4, 6, 8],                               // swept (list)
      "tank": {"from": 150000, "to": 250000, "step": 50000}  // swept (range)
    }

Control keys: runs, stage, base_config, report, out, json. Friendly aliases (tankers,
harpoons, tank, fish, ...) map to the real parameter names; dotted paths like
"boat.idle_burn" or "grounds_center_x" (theater depth) also work. Up to 3 swept parameters.
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

from fishing_rl.config import Config
from fishing_rl.config_io import load_config, resolve_path, set_param
from fishing_rl.env import FishingEnv
from fishing_rl.heuristic import HeuristicPolicy

CONTROL = {"runs", "stage", "base_config", "report", "out", "json"}


def expand(value):
    """Return (is_swept, values). Scalar -> fixed; list/range -> swept."""
    if isinstance(value, list):
        return True, value
    if isinstance(value, dict) and "from" in value:
        step = value.get("step", 1)
        vals, v, stop = [], value["from"], value["to"]
        while v <= stop + 1e-9:
            vals.append(round(v, 6) if isinstance(step, float) else int(v))
            v += step
        return True, vals
    return False, value


def run_cell(base_config, fixed, combo, runs, stage):
    cfg = load_config(base_config) if base_config else Config()
    for path, val in fixed.items():
        set_param(cfg, path, val)
    for path, val in combo.items():
        set_param(cfg, path, val)
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
        lengths.append(e["length"])          # steps = minutes
        sea.append(e["sea_refuels"])
        dock_t.append(e["barge_dock_refuels"])
        dock_b.append(e["boat_dock_refuels"])
    c = np.array(catches, dtype=float)
    d = np.array(deaths, dtype=float)
    mean = c.mean()
    loss_times = [ln for ln, dd in zip(lengths, deaths) if dd]
    return {
        "survival": 1.0 - d.mean(), "mean": mean, "std": c.std(),
        "min": float(c.min()), "p10": float(np.percentile(c, 10)),
        "median": float(np.median(c)), "cv": (c.std() / mean) if mean > 0 else 0.0,
        "loss_med_min": float(np.median(loss_times)) if loss_times else None,
        "loss_min_min": float(np.min(loss_times)) if loss_times else None,
        "sea_ref": float(np.mean(sea)), "dock_t_ref": float(np.mean(dock_t)),
        "dock_b_ref": float(np.mean(dock_b)),
        "catches": catches, "deaths": deaths, "lengths": lengths,
        "total": int(cfg.n_barges * cfg.barge.tank),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", help="study JSON file")
    args = ap.parse_args()
    with open(args.spec, "r", encoding="utf-8") as fh:
        spec = json.load(fh)

    runs = spec.get("runs", 40)
    stage = spec.get("stage", 1.0)
    base_config = spec.get("base_config")

    fixed, swept = {}, {}
    for name, value in spec.items():
        if name in CONTROL or name.startswith("_"):
            continue
        is_swept, vals = expand(value)
        (swept if is_swept else fixed)[name] = vals
    if not swept:
        raise SystemExit("no swept variables — give at least one list or range value")
    if len(swept) > 3:
        raise SystemExit(f"too many swept variables ({len(swept)}); keep it to <= 3")

    names = list(swept)
    grid = list(itertools.product(*(swept[n] for n in names)))
    print(f"Study | base={base_config or 'defaults'} | {runs} runs/cell | "
          f"{len(grid)} cells | stage {stage}")
    print(f"fixed: {fixed or '(none)'}")
    print(f"swept: {swept}\n")

    # only surface "total tanker fuel" framing when the study actually varies the fleet
    fleet_study = any(resolve_path(n) in ("n_barges", "barge.tank") for n in names)

    def hrs(m):
        return f"{m/60:>5.1f}h" if m is not None else "  —  "

    rows = []
    hdr = (" | ".join(f"{n:>14}" for n in names)
           + " | survival |  catches(mean±sd) |   CV | time-to-loss med/min")
    print(hdr); print("-" * len(hdr))
    for values in grid:
        combo = dict(zip(names, values))
        res = run_cell(base_config, fixed, combo, runs, stage)
        label = ", ".join(f"{n}={combo[n]:g}" if isinstance(combo[n], float)
                          else f"{n}={combo[n]}" for n in names)
        cells = " | ".join(f"{combo[n]:>14g}" for n in names)
        print(f"{cells} | {res['survival']*100:>6.0f}%  | "
              f"{res['mean']:>6.0f} ± {res['std']:<5.0f} | {res['cv']:>5.2f} | "
              f"{hrs(res['loss_med_min'])} / {hrs(res['loss_min_min'])}")
        row = {"label": label, **res}
        if not fleet_study:
            row.pop("total", None)
        rows.append(row)

    best = sorted(rows, key=lambda r: (-r["survival"], r["cv"]))[0]
    print(f"\nMost stable: {best['label']}  "
          f"({best['survival']*100:.0f}% survival, CV {best['cv']:.2f}, ~{best['mean']:.0f} catches)")

    out_json = spec.get("json")
    result = {"runs": runs, "stage": stage, "title": spec.get("title"),
              "configs": rows}
    if out_json:
        with open(out_json, "w") as fh:
            json.dump(result, fh)
        print(f"wrote {out_json}")
    if spec.get("report"):
        import tempfile
        import os
        from scripts.mc_report import build
        tmp = os.path.join(tempfile.gettempdir(), "_study_mc.json")
        with open(tmp, "w") as fh:
            json.dump(result, fh)
        build(tmp, spec["report"])
        print(f"wrote {spec['report']}  (open in a browser)")


if __name__ == "__main__":
    main()
