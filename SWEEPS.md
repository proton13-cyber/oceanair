# Running Sweeps & Sensitivities

All parameters live in a TOML file (`fishing.toml`), and any script can load it with
`--config`. The general sweep tool (`scripts.sweep`) varies **any one or two
parameters** over any values and reports survival + catches on a fast heuristic
controller — so you can run a lot of sensitivities without touching code.

---

## 1. Quick start

```powershell
# activate the venv first
.\.venv\Scripts\Activate.ps1

# (re)generate a full parameter file with every default filled in
python -c "from fishing_rl.config_io import write_template; write_template('fishing.toml')"

# 1D sweep: how fish speed drives survival
python -m scripts.sweep --config fishing.toml --x fish_speed_frac --x-vals 0.3 0.4 0.5 0.6

# 2D sweep: tanker size (columns) vs fleet size (rows)
python -m scripts.sweep --config fishing.toml `
    --x barge.tank --x-vals 100000 150000 200000 250000 `
    --y n_barges   --y-vals 4 6 8 --seeds 8
```

Each cell prints **`survive% (mean catches)`**; a `*` marks 100% sustained (no boat
lost across all seeds).

---

## 1b. JSON studies — the no-CLI path

Prefer editing a file over remembering flags? Define a whole study in **`study.json`**
and run it with one command. A single value pins a parameter; a `[list]` or a
`{"from","to","step"}` range makes it a swept variable; delete a line to keep the
default. Every grid cell is evaluated as a Monte Carlo (many random voyages), and it
prints a table and (optionally) writes the HTML report.

```powershell
# (re)generate a study.json pre-filled with EVERY tunable at its current value
python -c "from fishing_rl.config_io import write_study_template; write_study_template('study.json')"

# run it
python -m scripts.study study.json
```

`study.json` groups all controls — boats, tankers, world size, the fish/"threats" that
pop up, harpoon & dock, refuel/transfer, spawn band. Friendly aliases (`tankers`,
`harpoons`, `tank`, `fish`, …) and dotted paths (`boat.max_speed`,
`curriculum.hard_max_x`) both work. Keys starting with `_` are comments. Example:

```json
{
  "runs": 40,
  "base_config": "fishing.toml",
  "report": "runs/study_report.html",

  "harpoons": 8,                 // pinned
  "boat.idle_burn": 15,          // pinned

  "tankers": [4, 6, 8],                                  // swept
  "tank": {"from": 150000, "to": 250000, "step": 50000}  // swept range
}
```

Up to 3 swept variables (full-factorial grid). `report`/`json` keys are optional
outputs. This is the friendliest way to drive everything below.

---

## 2. The parameter file (`fishing.toml`)

Edit any value; delete a line to keep its default. Unknown keys are rejected with the
list of valid names, so typos surface immediately. Five sections:

| Section        | What it controls |
|----------------|------------------|
| `[env]`        | world size, fleet counts, fish/schools, harpoon, dock, radii, timing |
| `[boat]`       | jet dynamics (speed, tank, burn, turn) |
| `[barge]`      | tanker dynamics |
| `[rewards]`    | reward shaping (only used when training the RL barge policy) |
| `[curriculum]` | difficulty ramp and fish spawn band |

Load it into any script:

```powershell
python -m scripts.play_heuristic --config fishing.toml --fps 20 --seed 2   # watch it
python -m scripts.sweep_barges   --config fishing.toml                     # fleet grid
python -m scripts.train          --config fishing.toml                     # RL training
```

CLI flags override the file (e.g. `--barges 6` beats `n_barges` in the TOML).

---

## 3. The general sweep — `scripts.sweep`

```
python -m scripts.sweep [--config FILE] --x PATH --x-vals V... [--y PATH --y-vals V...]
                        [--seeds N] [--stage S]
```

| Flag        | Meaning | Default |
|-------------|---------|---------|
| `--config`  | base TOML; the sweep varies params on top of it | code defaults |
| `--x`       | parameter to sweep across **columns** (dotted path) | required |
| `--x-vals`  | the values for `--x` | required |
| `--y`       | optional 2nd parameter, across **rows** | none (1D) |
| `--y-vals`  | the values for `--y` | — |
| `--seeds`   | episodes per cell (higher = less noise) | 8 |
| `--stage`   | curriculum stage, 0 = easy … 1 = hard | 1.0 |

**Parameter paths.** Anything in `[env]` is a bare name; nested sections use a dot:

```
n_barges            barge.tank            rewards.catch
n_boats             barge.max_speed       curriculum.hard_max_x
harpoon_ammo        boat.idle_burn        curriculum.hard_min_x
fish_speed_frac     boat.turn_agility     curriculum.easy_tank_mult
max_fish            boat.tank
dock_service_steps  boat.max_speed
```

---

## 4. Parameter reference (sweepable paths)

**Fleet & world (`[env]`)**
`n_boats`, `n_barges`, `world_size`, `max_steps`, `min_speed_frac`, `accel_frac`, `seed`

**Fish / schools**
`max_fish`, `max_schools`, `school_size`, `school_spawn_prob`, `fish_speed_frac`,
`fish_wander`, `school_spread`, `fish_y_min`, `fish_y_max`

**Harpoon & dock**
`harpoon_range`, `harpoon_cooldown`, `harpoon_ammo`, `dock_service_steps`, `refuel_full_frac`

**Radii & transfer**
`catch_radius`, `scare_radius`, `barge_avoid_radius`, `refuel_radius`, `port_radius`,
`transfer_rate`, `port_refill_rate`

**Jet (`boat.`) and tanker (`barge.`)**
`max_speed`, `max_turn_rate`, `tank`, `idle_burn`, `move_burn`, `turn_agility`

**Curriculum (`curriculum.`)**
`hard_min_x`, `hard_max_x`, `easy_min_x`, `easy_max_x`, `easy_tank_mult`, `hard_tank_mult`

**Rewards (`rewards.`, RL only)**
`catch`, `team_catch_share`, `death_penalty`, `scare_penalty`, `step_penalty`,
`shaping_coef`, `refuel_reward_per_unit`, `boat_refuel_reward_per_unit`,
`barge_stage_reward`, `low_fuel_threshold`, `low_fuel_penalty`

---

## 5. Example sensitivities

```powershell
# Fleet sizing: how much tanker fuel each fleet size needs
python -m scripts.sweep --config fishing.toml `
    --x barge.tank --x-vals 100000 150000 200000 250000 300000 `
    --y n_barges   --y-vals 3 4 5 6 7 8 --seeds 8

# Boat thirst: idle burn vs harpoon magazine
python -m scripts.sweep --config fishing.toml `
    --x boat.idle_burn --x-vals 10 15 20 25 `
    --y harpoon_ammo   --y-vals 4 8 12 --seeds 8

# How far can the grounds be? (dock-run cost vs fleet size)
python -m scripts.sweep --config fishing.toml `
    --x curriculum.hard_max_x --x-vals 0.35 0.45 0.55 0.65 `
    --y n_barges              --y-vals 6 8 --seeds 8

# Fish behavior: speed vs how many fish are present
python -m scripts.sweep --config fishing.toml `
    --x fish_speed_frac --x-vals 0.2 0.3 0.4 0.5 `
    --y max_fish        --y-vals 6 9 12 --seeds 8

# Dock service time (deck dwell) sensitivity — 1D
python -m scripts.sweep --config fishing.toml `
    --x dock_service_steps --x-vals 0 15 25 40 --seeds 12

# Tanker speed / agility (KC-135 stand-in)
python -m scripts.sweep --config fishing.toml `
    --x barge.max_speed --x-vals 4 5 6 7 `
    --y n_barges         --y-vals 6 8 --seeds 8
```

---

## 6. Monte Carlo — comparing fleet stability

A sweep reports the *average* over a few seeds. To compare configurations on
**stability** — how *reliably* they survive and how *steady* the catch is across many
random spawns — use `scripts.montecarlo`. It runs many random episodes per config and
reports the distribution.

```powershell
# Same total fuel, split across fleet sizes: is 8 small tankers better than 4 big ones?
python -m scripts.montecarlo --config fishing.toml --equal-total 1200000 --barges 4 6 8 --runs 100

# Explicit head-to-head (n_barges:tank_lb)
python -m scripts.montecarlo --config fishing.toml --configs 4:300000 8:150000 --runs 100

# Dump raw per-episode data (for plotting / the artifact)
python -m scripts.montecarlo --config fishing.toml --equal-total 1200000 --barges 4 6 8 --runs 100 --json mc.json
```

| Flag           | Meaning | Default |
|----------------|---------|---------|
| `--configs`    | explicit list, `n_barges:tank_lb` | — |
| `--equal-total`| fixed total fuel; splits across `--barges` | — |
| `--barges`     | fleet sizes for `--equal-total` | 4 6 8 |
| `--runs`       | random episodes per config (↑ = tighter estimate) | 60 |
| `--config`, `--stage`, `--json` | as elsewhere | — |

**Reading it:** each row shows **survival %**, **catches mean ± std**, the **min** and
**10th-percentile** run (worst-case tail), and **CV** (coefficient of variation =
std/mean). *Lower CV = steadier run-to-run.* The tool names the most stable config
(highest survival, then lowest CV). Two fleets with the same average can differ a lot
in reliability — that's what this surfaces.

**Visual report.** Turn any `--json` dump into a standalone HTML chart (survival bars +
catch-distribution strip + table + takeaways), generated entirely from the data:

```powershell
python -m scripts.montecarlo --config fishing.toml --equal-total 1200000 --barges 4 6 8 --runs 100 --json mc.json
python -m scripts.mc_report mc.json --out mc_report.html
```

`mc_report.html` embeds its data and opens in any browser — works for any set of
configs and run count. (`scripts/mc_report_template.html` is the reusable template.)

---

## 7. Tips

- **Noise.** Survival is measured over `--seeds` random episodes; at 4 seeds the grid
  jitters. Use `--seeds 8`–`16` for a trustworthy frontier, fewer for a quick look.
- **Speed.** Sweeps use the scripted heuristic (no training), so each episode is
  milliseconds. A big grid is a couple of minutes; run it in the background if needed.
- **Reading a cell.** `100*(194)` = every seed survived, ~194 fish landed.
  `25 ( 90)` = only 1 in 4 seeds survived — under-provisioned.
- **Stage.** `--stage 1.0` (default) is the hardest spawn band (deepest fish); lower it
  to probe easier conditions.
- **Isolate one change.** Start from `fishing.toml` (a known-good config) and sweep a
  single parameter so the effect is clean.
- **Fleet grid shortcut.** `scripts.sweep_barges --config fishing.toml` is a dedicated
  n_barges × tank sweep with a frontier summary, if that's all you need.
