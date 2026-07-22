"""Load / dump the whole parameter set as a TOML file, so sensitivity runs never
need code edits.

    from fishing_rl.config_io import load_config, write_template
    cfg = load_config("fishing.toml")     # any omitted key keeps its default
    write_template("fishing.toml")        # emit a fully-populated, editable template

File layout (four sections)::

    [env]         world / fleet counts / fish / harpoon / radii / timing
    [boat]        jet dynamics (Dynamics fields)
    [barge]       tanker dynamics (Dynamics fields)
    [curriculum]  difficulty ramp + spawn band

Unknown sections or keys raise an error, so typos surface immediately instead of
silently doing nothing.
"""
from __future__ import annotations

from dataclasses import fields

from .config import Config

_SUBOBJECTS = ("boat", "barge", "curriculum")


def _parse_val(s):
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        return [_parse_val(x) for x in inner.split(",")] if inner else []
    if s in ("true", "false"):
        return s == "true"
    if len(s) >= 2 and s[0] == s[-1] == '"':
        return s[1:-1]
    try:
        return int(s) if s.lstrip("-").isdigit() else float(s)
    except ValueError:
        return s


def _mini_toml(text):
    """Minimal TOML reader for this project's flat [section] key = value files.

    Used as a fallback when the stdlib `tomllib` (Python 3.11+) is unavailable, so
    the config files work on any Python version without extra dependencies.
    """
    doc, cur = {}, {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            cur = doc.setdefault(line[1:-1].strip(), {})
        elif "=" in line:
            key, val = line.split("=", 1)
            cur[key.strip()] = _parse_val(val)
    return doc


def _parse_toml(text):
    try:
        import tomllib
        return tomllib.loads(text)
    except ModuleNotFoundError:
        try:
            import tomli
            return tomli.loads(text)
        except ModuleNotFoundError:
            return _mini_toml(text)


def _apply(obj, data, section, exclude=()):
    valid = {f.name for f in fields(obj)} - set(exclude)
    for key, val in data.items():
        if key not in valid:
            raise KeyError(
                f"[{section}] unknown parameter '{key}'.\n  valid: {sorted(valid)}")
        cur = getattr(obj, key)
        if isinstance(cur, tuple) and isinstance(val, list):
            val = tuple(val)              # TOML arrays -> tuples (e.g. port_frac)
        setattr(obj, key, val)


def load_config(path) -> Config:
    """Build a Config from a TOML file; unspecified values keep their defaults."""
    with open(path, "r", encoding="utf-8") as fh:
        doc = _parse_toml(fh.read())
    valid_sections = {"env", *_SUBOBJECTS}
    unknown = set(doc) - valid_sections
    if unknown:
        raise KeyError(f"unknown section(s) {sorted(unknown)}; "
                       f"valid: {sorted(valid_sections)}")
    cfg = Config()
    if "env" in doc:
        _apply(cfg, doc["env"], "env", exclude=_SUBOBJECTS)
    for name in _SUBOBJECTS:
        if name in doc:
            _apply(getattr(cfg, name), doc[name], name)
    return cfg


# Friendly names -> real dotted parameter paths (so a study JSON can say "tankers"
# or "harpoons" instead of "n_barges" / "harpoon_ammo").
ALIASES = {
    "tankers": "n_barges", "n_tankers": "n_barges", "barges": "n_barges",
    "boats": "n_boats", "n_boats_": "n_boats",
    "harpoons": "harpoon_ammo", "harpoon": "harpoon_ammo", "ammo": "harpoon_ammo",
    "tank": "barge.tank", "barge_tank": "barge.tank", "boat_tank": "boat.tank",
    "fish": "max_fish", "fish_speed": "fish_speed_frac",
    "dock_time": "dock_service_steps",
    "width": "world_width", "height": "world_height",
}


def resolve_path(name):
    """Map a friendly alias to the real dotted parameter path (identity otherwise)."""
    return ALIASES.get(name, name)


def set_param(cfg, path, value):
    """Set a dotted parameter path on a Config, coercing to the field's type."""
    obj = cfg
    parts = resolve_path(path).split(".")
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


def _toml_val(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (tuple, list)):
        return "[" + ", ".join(_toml_val(x) for x in v) + "]"
    return f'"{v}"'


def write_template(path):
    """Write a TOML file pre-filled with every parameter at its current default."""
    cfg = Config()
    out = [
        "# Fishing-RL parameters. Edit any value; delete a line to keep the default.",
        "# Run with:  python -m scripts.play_heuristic --config fishing.toml",
        "#            python -m scripts.sweep_barges  --config fishing.toml",
        "",
    ]

    def section(name, obj, exclude=()):
        out.append(f"[{name}]")
        for f in fields(obj):
            if f.name in exclude:
                continue
            out.append(f"{f.name} = {_toml_val(getattr(obj, f.name))}")
        out.append("")

    section("env", cfg, exclude=_SUBOBJECTS)
    section("boat", cfg.boat)
    section("barge", cfg.barge)
    section("curriculum", cfg.curriculum)
    with open(path, "w") as fh:
        fh.write("\n".join(out))
    return path


def write_study_template(path):
    """Write a comprehensive study.json exposing every tunable at its current value.

    A single value pins a parameter; a [list] or {"from","to","step"} range sweeps it;
    delete a line to keep the default. `_`-prefixed keys are comments (ignored).
    """
    import json
    c = Config()
    b, g, cur = c.boat, c.barge, c.curriculum
    doc = {}

    def sect(note):
        doc[f"_{len(doc)}"] = note      # unique comment key

    doc["_help"] = ("study spec: one value pins a param, a [list] or {from,to,step} "
                    "range sweeps it, delete a line to keep the default. Friendly "
                    "aliases and dotted paths (boat.max_speed) both work.")
    doc["runs"] = 40
    doc["stage"] = 1.0
    doc["base_config"] = "fishing.toml"
    doc["report"] = "runs/study_report.html"
    doc["json"] = "runs/study_data.json"

    sect("=== EXAMPLE SWEEPS (lists = varied; edit or replace) ===")
    doc["tankers"] = [4, 6, 8]                       # alias -> n_barges
    doc["tank"] = {"from": 150000, "to": 250000, "step": 50000}  # alias -> barge.tank

    sect("=== FISHING BOATS (jets) ===")
    doc["n_boats"] = c.n_boats
    doc["boat.max_speed"] = b.max_speed              # nmi/step at full throttle
    doc["boat.max_turn_rate"] = b.max_turn_rate      # rad/step hard cap
    doc["boat.turn_agility"] = b.turn_agility        # lower = wider arcs at speed
    doc["boat.tank"] = b.tank                        # lb; 20000 = 1000 nmi range
    doc["boat.idle_burn"] = b.idle_burn              # lb/step at loiter
    doc["boat.move_burn"] = b.move_burn              # lb/step per unit speed

    sect("=== TANKERS (barges) ===")
    doc["barge.max_speed"] = g.max_speed
    doc["barge.max_turn_rate"] = g.max_turn_rate
    doc["barge.turn_agility"] = g.turn_agility
    doc["barge.idle_burn"] = g.idle_burn
    doc["barge.move_burn"] = g.move_burn
    doc["_barge_tank"] = "barge.tank shown above as the 'tank' sweep"

    sect("=== WORLD / TIMING ===")
    doc["world_width"] = c.world_width               # nmi east-west
    doc["world_height"] = c.world_height             # nmi north-south (!= width for a non-square ocean)
    doc["max_steps"] = c.max_steps                   # voyage length (steps ~ minutes)
    doc["min_speed_frac"] = c.min_speed_frac         # nothing can sit still below this
    doc["accel_frac"] = c.accel_frac                 # momentum (max speed change/step)

    sect("=== FISH / TARGETS THAT RANDOMLY POP UP ===")
    doc["max_schools"] = c.max_schools               # how many schools can appear
    doc["max_fish"] = c.max_fish                     # total fish cap on the map
    doc["school_size"] = c.school_size               # fish per school
    doc["school_spawn_prob"] = c.school_spawn_prob   # per-step chance a new school appears
    doc["fish_speed_frac"] = c.fish_speed_frac       # fish speed as fraction of boat speed
    doc["fish_wander"] = c.fish_wander               # heading jitter (radians/step)
    doc["school_spread"] = c.school_spread           # nmi cluster radius of a school
    doc["fish_barge_clearance_mult"] = c.fish_barge_clearance_mult  # * scare_radius keep-out at spawn
    doc["fish_y_min"] = c.fish_y_min                 # spawn band (fraction of world height)
    doc["fish_y_max"] = c.fish_y_max

    sect("=== HARPOON & DOCK ===")
    doc["harpoons"] = c.harpoon_ammo                 # alias -> harpoon_ammo (magazine)
    doc["harpoon_range"] = c.harpoon_range           # nmi ranged catch
    doc["harpoon_cooldown"] = c.harpoon_cooldown     # steps between shots
    doc["dock_service_steps"] = c.dock_service_steps # deck time to rearm + refuel

    sect("=== REFUEL / TRANSFER / RADII ===")
    doc["transfer_rate"] = c.transfer_rate           # lb/step tanker->boat (refuel speed)
    doc["port_refill_rate"] = c.port_refill_rate     # lb/step dock->tanker
    doc["refuel_full_frac"] = c.refuel_full_frac     # boat tops up to this before leaving
    doc["refuel_radius"] = c.refuel_radius           # nmi boat<->tanker transfer range
    doc["port_radius"] = c.port_radius               # nmi dock service range
    doc["catch_radius"] = c.catch_radius             # nmi close-net catch
    doc["scare_radius"] = c.scare_radius             # nmi a tanker scares fish
    doc["barge_avoid_radius"] = c.barge_avoid_radius # nmi tankers keep off fish

    sect("=== THEATER DEPTH (single knob pushes the whole operating area out) ===")
    doc["grounds_center_x"] = c.grounds_center_x     # center of fish grounds (fraction of width); raise to push everything DEEPER
    doc["grounds_half_span"] = c.grounds_half_span   # half the fish band width (band = center +/- this)

    sect("=== TANKER STAGING / DEFENSE ===")
    doc["barge_stage_standoff"] = c.barge_stage_standoff  # tanker line this far port-side of grounds center
    doc["barge_stage_y"] = c.barge_stage_y           # tanker loiter center (fraction of height)
    doc["barge_loiter_radius"] = c.barge_loiter_radius    # nmi width of the tanker loiter loop
    doc["boat_defend_radius"] = c.boat_defend_radius # nmi: fish within this of a tanker -> nearest boat intercepts

    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)
    return path
