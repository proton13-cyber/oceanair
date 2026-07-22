# Fishing RL — cooperative multi-agent logistics game

A continuous 2D multi-agent reinforcement learning environment where **fishing boats**
(fast, short range) must chase down stochastically spawning **fish**, supported by
**barges** (slow, long range) that ferry and transfer fuel from **port**. Barges scare
fish if they get too close, so the fleet must learn a **relay / cycling strategy**.

If **any** fishing boat runs out of fuel, the episode ends with a large penalty — the
boats cannot be replaced.

This is a **cooperative, heterogeneous MARL** problem (foraging + logistics).

## Layout

```
fishing_rl/
  config.py      # all tunable parameters (world, dynamics, rewards, obs sizes)
  env.py         # FishingEnv: PettingZoo-style ParallelEnv, pure NumPy dynamics
  render.py      # pygame renderer
  heuristic.py   # scripted baseline policy (chase fish / refuel / stage barges)
  models.py      # PyTorch actor-critic (Gaussian policy)
  train.py       # PPO with parameter-sharing-by-role (boat policy + barge policy)
scripts/
  play_heuristic.py   # watch the heuristic policy in the renderer
  train.py            # run training, periodically render a greedy rollout
```

## Setup

```powershell
cd path\to\fishing-rl
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For CUDA, install the matching PyTorch build from https://pytorch.org (the
`requirements.txt` pin is CPU-safe; swap in the CUDA wheel for GPU training).

## Run

Watch the heuristic (no training needed — validates the sim):

```powershell
python -m scripts.play_heuristic
```

Train:

```powershell
python -m scripts.train --timesteps 2000000 --render-every 50
```

## Design notes / where to iterate

- **Reward shaping** lives in `config.py` (`Rewards`) and `env.py`. The relay behavior
  should emerge; start with shaping on (`shaping_coef > 0`) and anneal it down.
- **Curriculum**: `Config.curriculum` scales fish spawn distance and fuel generosity.
  Start easy, push fish outward — cold-start with sparse catch + catastrophic death is
  very hard to explore otherwise.
- **Refueling is automatic on proximity** in this scaffold (positioning is the decision).
  Making it an explicit action is a documented next step.
- **Centralized critic**: this scaffold uses IPPO-style parameter sharing (per-role
  shared actor + per-agent value). Upgrading the critic to take a global state is the
  natural MAPPO step and usually improves credit assignment.
- **Throughput**: single-env Python loop for clarity. Vectorizing the env (batch the
  NumPy dynamics across N worlds) is the biggest training-speed win.
