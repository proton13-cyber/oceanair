"""FishingEnv — the continuous 2D fishing/tanker simulation (pure NumPy).

    _, infos                 = env.reset(seed=...)
    _, _, term, trunc, infos = env.step(actions)   # actions: dict of [throttle, turn]

Actions come from the scripted controller (`heuristic.HeuristicPolicy`). Refueling and
harpoon reload happen automatically on proximity; any boat hitting zero fuel ends the
episode. `infos[agent]["episode"]` carries {catches, length, deaths} on the final step.
"""
from __future__ import annotations

import math
import numpy as np

from .config import Config


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class _Fish:
    __slots__ = ("pos", "vel")

    def __init__(self, pos, vel):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.vel = np.asarray(vel, dtype=np.float64)


class _Entity:
    __slots__ = ("pos", "heading", "fuel", "alive", "role", "speed",
                 "refueling", "harpoon_cd", "harpoons_left", "reloading", "dock_timer")

    def __init__(self, pos, heading, fuel, role):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.heading = float(heading)
        self.fuel = float(fuel)
        self.alive = True
        self.role = role
        self.speed = 0.0
        self.refueling = False   # boat latched into a refuel stop (float alongside)
        self.harpoon_cd = 0      # steps until this boat can harpoon again
        self.harpoons_left = 0   # harpoons in the magazine (boats; set in reset)
        self.reloading = False   # boat latched into a dock trip to restock harpoons
        self.dock_timer = 0      # steps spent "on the deck" servicing at the dock


class FishingEnv:
    metadata = {"render_modes": ["human", "rgb_array"], "name": "fishing_v0"}

    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self.possible_agents = list(self.cfg.agent_names)
        self.agents: list[str] = []
        self.rng = np.random.default_rng(self.cfg.seed)

        # populated in reset()
        self.ent: dict[str, _Entity] = {}
        self.fish: list[_Fish] = []
        self.t = 0
        self.stage = 0.0  # curriculum stage in [0, 1]
        self._ep_catches = 0
        self._ep_deaths = 0
        self._lost_barges = 0        # tankers that ran dry at sea and were removed
        self._sea_refuels = self._barge_dock_refuels = self._boat_dock_refuels = 0
        self._receiving = set()      # boats taking fuel from a barge this step
        self._refuel_barges = set()  # barges giving fuel this step
        self._harpoon_shots = []  # (boat_pos, fish_pos) harpooned THIS step (for rendering)

    # ---- curriculum ---------------------------------------------------------
    def set_stage(self, stage: float):
        self.stage = float(np.clip(stage, 0.0, 1.0))

    def _tank_mult(self):
        c = self.cfg.curriculum
        if not c.enabled:
            return 1.0
        return c.easy_tank_mult + self.stage * (c.hard_tank_mult - c.easy_tank_mult)

    def _spawn_band(self):
        c = self.cfg
        # hard (stage 1.0) band comes from the single theater-depth control
        hmin = c.grounds_center_x - c.grounds_half_span
        hmax = c.grounds_center_x + c.grounds_half_span
        cur = c.curriculum
        if not cur.enabled:
            return hmin, hmax
        lo = cur.easy_min_x + self.stage * (hmin - cur.easy_min_x)
        hi = cur.easy_max_x + self.stage * (hmax - cur.easy_max_x)
        return lo, hi

    # ---- reset --------------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        cfg = self.cfg
        self.agents = list(self.possible_agents)
        self.t = 0
        self._ep_catches = 0
        self._ep_deaths = 0
        self._lost_barges = 0        # tankers that ran dry at sea and were removed
        self._sea_refuels = 0        # boat topped up by a barge out at sea (rendezvous count)
        self._barge_dock_refuels = 0  # tanker completed a dock refuel
        self._boat_dock_refuels = 0   # boat completed a dock reload + refuel
        self._receiving = set()       # boats receiving fuel at sea last step
        self._refuel_barges = set()
        tank_mult = self._tank_mult()

        px, py = cfg.port_pos
        self.ent = {}
        for i in range(cfg.n_boats):
            jitter = self.rng.uniform(-20, 20, size=2)
            e = _Entity(pos=[px + jitter[0], py + jitter[1]], heading=0.0,
                        fuel=cfg.boat.tank * tank_mult, role="boat")
            e.harpoons_left = cfg.harpoon_ammo
            self.ent[f"boat_{i}"] = e
        for i in range(cfg.n_barges):
            jitter = self.rng.uniform(-20, 20, size=2)
            self.ent[f"barge_{i}"] = _Entity(
                pos=[px + jitter[0], py + jitter[1]], heading=0.0,
                fuel=cfg.barge.tank, role="barge")

        self.fish = []
        for _ in range(max(0, cfg.initial_schools)):   # small seed; the rest accumulate over time
            self._spawn_school()

        infos = {a: {} for a in self.agents}
        return {}, infos

    def _spawn_school(self):
        """Add a school at a random spot in the spawn band, clear of every barge."""
        cfg = self.cfg
        Wx, Wy = cfg.world_width, cfg.world_height
        lo, hi = self._spawn_band()
        clearance = cfg.fish_barge_clearance_mult * cfg.scare_radius
        center = None
        for _ in range(12):                       # retry until we find water clear of barges
            c = np.array([self.rng.uniform(lo, hi) * Wx,
                          self.rng.uniform(cfg.fish_y_min, cfg.fish_y_max) * Wy])
            if all(np.linalg.norm(self.ent[f"barge_{i}"].pos - c) >= clearance
                   for i in range(cfg.n_barges)):
                center = c
                break
        if center is None:
            return                                # crowded by barges this tick — skip
        speed = cfg.fish_speed_frac * cfg.boat.max_speed
        ang = self.rng.uniform(-math.pi, math.pi)
        school_vel = np.array([math.cos(ang), math.sin(ang)]) * speed
        for _ in range(cfg.school_size):
            if len(self.fish) >= cfg.max_fish:
                break
            offs = self.rng.uniform(-cfg.school_spread, cfg.school_spread, size=2)
            self.fish.append(_Fish(pos=center + offs, vel=school_vel.copy()))

    def _move_fish(self):
        """Fish swim: wander their heading, advance, reflect off the world edges."""
        cfg = self.cfg
        bound = (cfg.world_width, cfg.world_height)
        speed = cfg.fish_speed_frac * cfg.boat.max_speed
        for f in self.fish:
            ang = self.rng.uniform(-cfg.fish_wander, cfg.fish_wander)
            c, s = math.cos(ang), math.sin(ang)
            vx, vy = f.vel
            f.vel[0], f.vel[1] = vx * c - vy * s, vx * s + vy * c
            n = math.hypot(f.vel[0], f.vel[1]) + 1e-9
            f.vel *= speed / n
            f.pos += f.vel * cfg.dt
            for k in (0, 1):
                if f.pos[k] < 0:
                    f.pos[k] = -f.pos[k]; f.vel[k] = -f.vel[k]
                elif f.pos[k] > bound[k]:
                    f.pos[k] = 2 * bound[k] - f.pos[k]; f.vel[k] = -f.vel[k]

    def _near_barge(self, e, radius):
        for i in range(self.cfg.n_barges):
            b = self.ent[f"barge_{i}"]
            if b.alive and np.linalg.norm(b.pos - e.pos) <= radius:
                return True
        return False

    # ---- step ---------------------------------------------------------------
    def step(self, actions: dict[str, np.ndarray]):
        cfg = self.cfg

        self._harpoon_shots = []   # cleared each step; filled by ranged catches below
        # harpoon reload: tick down each boat's cooldown
        for j in range(cfg.n_boats):
            bt = self.ent[f"boat_{j}"]
            if bt.harpoon_cd > 0:
                bt.harpoon_cd -= 1

        # 1) movement + fuel burn
        for name in self.agents:
            e = self.ent[name]
            if not e.alive:
                continue
            dyn = cfg.boat if e.role == "boat" else cfg.barge
            a = np.asarray(actions[name], dtype=np.float64).reshape(-1)
            throttle = float(np.clip((a[0] + 1.0) * 0.5, 0.0, 1.0))
            turn = float(np.clip(a[1], -1.0, 1.0))

            # turn agility: fast -> wide arc, slow -> tight (angular cap =
            # turn_agility / speed, never above max_turn_rate)
            eff_turn = min(dyn.max_turn_rate, dyn.turn_agility / max(e.speed, 1e-3))
            e.heading = _wrap_angle(e.heading + turn * eff_turn * cfg.dt)
            # a boat moored alongside a barge, or sitting on the deck at the dock, may
            # be still; everyone else must keep moving (>= min_speed_frac of max)
            moored = (e.role == "boat" and e.refueling
                      and self._near_barge(e, cfg.refuel_radius))
            # anything sitting at the dock may be still (boats on the deck, and
            # reserve tankers parked on the tarmac between wave deployments)
            grounded = np.linalg.norm(e.pos - np.array(cfg.port_pos)) <= cfg.port_radius
            floor = 0.0 if (moored or grounded) else cfg.min_speed_frac * dyn.max_speed
            target_speed = max(floor, throttle * dyn.max_speed)
            if e.fuel <= 0.0:          # (A) out of fuel -> dead in the water
                target_speed = 0.0
            # momentum: speed ramps toward target, capped by max acceleration
            max_dv = cfg.accel_frac * dyn.max_speed
            e.speed += float(np.clip(target_speed - e.speed, -max_dv, max_dv))
            vel = np.array([math.cos(e.heading), math.sin(e.heading)]) * e.speed
            e.pos = np.clip(e.pos + vel * cfg.dt, 0.0,
                            (cfg.world_width, cfg.world_height))

            burn = (dyn.idle_burn + dyn.move_burn * e.speed) * cfg.dt
            e.fuel = max(0.0, e.fuel - burn)

        # 2) death check (any boat dry -> catastrophic terminal)
        dead = False
        for i in range(cfg.n_boats):
            e = self.ent[f"boat_{i}"]
            if e.alive and e.fuel <= 0.0:
                e.alive = False
                dead = True
        if dead:
            self._ep_deaths += 1

        # 2b) a tanker that runs dry away from port is a lost asset -> remove it
        # (prevents a stranded, empty barge from sitting there dribbling fuel).
        px, py = cfg.port_pos
        for i in range(cfg.n_barges):
            b = self.ent[f"barge_{i}"]
            if (b.alive and b.fuel <= 0.0
                    and np.hypot(b.pos[0] - px, b.pos[1] - py) > cfg.port_radius):
                b.alive = False
                self._lost_barges += 1

        # 3) refueling (barge->boat, port->barge) — automatic on proximity
        px, py = cfg.port_pos
        port = np.array([px, py])
        # boats service at the dock: refuel continuously, but rearming harpoons takes
        # a ground dwell (dock_service_steps) — they sit "on the deck" before launching
        for j in range(cfg.n_boats):
            bt = self.ent[f"boat_{j}"]
            if bt.alive and np.linalg.norm(bt.pos - port) <= cfg.port_radius:
                bt.dock_timer += 1
                bt.fuel = min(cfg.boat.tank, bt.fuel + cfg.port_refill_rate * cfg.dt)
                if bt.dock_timer == cfg.dock_service_steps:
                    bt.harpoons_left = cfg.harpoon_ammo
                    self._boat_dock_refuels += 1
            else:
                bt.dock_timer = 0
        receiving_now = set()
        giving_now = set()
        for i in range(cfg.n_barges):
            b = self.ent[f"barge_{i}"]
            if not b.alive:
                continue
            # barge refuels over a fixed dock dwell (fully tops up across the service)
            if np.linalg.norm(b.pos - port) <= cfg.port_radius:
                b.dock_timer += 1
                if b.dock_timer == cfg.barge_dock_service_steps:
                    self._barge_dock_refuels += 1
                b.fuel = min(cfg.barge.tank,
                             b.fuel + cfg.barge.tank / cfg.barge_dock_service_steps)
            else:
                b.dock_timer = 0
            # barge transfers to nearby boats (draws from its own tank)
            for j in range(cfg.n_boats):
                bt = self.ent[f"boat_{j}"]
                if not bt.alive:
                    continue
                # a boat only takes fuel when it has COMMITTED to a refuel stop (low
                # and latched) — no unrealistic tickle-topping of boats passing by
                if (bt.refueling and b.fuel > 0
                        and np.linalg.norm(b.pos - bt.pos) <= cfg.refuel_radius):
                    want = cfg.boat.tank - bt.fuel
                    amount = min(cfg.transfer_rate * cfg.dt, b.fuel, want)
                    if amount > 0:
                        bt.fuel += amount
                        b.fuel -= amount
                        receiving_now.add(f"boat_{j}")
                        giving_now.add(f"barge_{i}")
        # count each at-sea rendezvous once (boats that just started taking fuel)
        self._sea_refuels += len(receiving_now - self._receiving)
        self._receiving = receiving_now
        self._refuel_barges = giving_now

        # 3.5) fish swim
        self._move_fish()

        # 4) catching (boats). Barges no longer remove fish — instead the barge
        # steers away from any fish that gets close (see the heuristic's evasion
        # reflex). Fish only leave the sim when a boat catches them.
        surviving_fish = []
        for f in self.fish:
            caught_by = None
            best_d = float("inf")
            by_harpoon = False
            for j in range(cfg.n_boats):
                bt = self.ent[f"boat_{j}"]
                if not bt.alive:
                    continue
                d = np.linalg.norm(bt.pos - f.pos)
                # close net always available; harpoon reaches farther but needs a
                # reloaded gun, a harpoon in the magazine, AND not be mid-refuel
                can_harpoon = (bt.harpoon_cd == 0 and bt.harpoons_left > 0
                               and f"boat_{j}" not in self._receiving)
                rng = cfg.harpoon_range if can_harpoon else cfg.catch_radius
                if d <= rng and d < best_d:
                    best_d = d
                    caught_by = f"boat_{j}"
                    by_harpoon = d > cfg.catch_radius
            if caught_by is not None:
                if by_harpoon:
                    self.ent[caught_by].harpoon_cd = cfg.harpoon_cooldown
                    self.ent[caught_by].harpoons_left -= 1
                    self._harpoon_shots.append((self.ent[caught_by].pos.copy(),
                                                f.pos.copy()))
                self._ep_catches += 1
                continue  # caught by a boat -> removed

            surviving_fish.append(f)  # near a barge is fine now — the barge dodges
        self.fish = surviving_fish

        # 6) spawn a fresh school now and then (keeps the grounds stocked)
        if len(self.fish) < cfg.max_fish and self.rng.random() < cfg.school_spawn_prob:
            self._spawn_school()

        # 7) termination / truncation
        self.t += 1
        truncated = self.t >= cfg.max_steps
        terminated = dead
        term = {a: terminated for a in self.agents}
        trunc = {a: truncated for a in self.agents}

        infos = {a: {} for a in self.agents}
        if terminated or truncated:
            ep = {"catches": self._ep_catches, "length": self.t,
                  "deaths": self._ep_deaths,
                  "lost_barges": self._lost_barges,
                  "sea_refuels": self._sea_refuels,
                  "barge_dock_refuels": self._barge_dock_refuels,
                  "boat_dock_refuels": self._boat_dock_refuels}
            for a in self.agents:
                infos[a]["episode"] = ep

        return {}, {}, term, trunc, infos

    # ---- convenience --------------------------------------------------------
    def state_snapshot(self):
        """Lightweight dict for the renderer."""
        return {
            "port": self.cfg.port_pos,
            "boats": [(self.ent[f"boat_{i}"].pos.copy(),
                       self.ent[f"boat_{i}"].fuel / self.cfg.boat.tank,
                       self.ent[f"boat_{i}"].alive,
                       self.ent[f"boat_{i}"].heading,
                       self.ent[f"boat_{i}"].harpoons_left,
                       f"boat_{i}" in self._receiving)
                      for i in range(self.cfg.n_boats)],
            "barges": [(self.ent[f"barge_{i}"].pos.copy(),
                        self.ent[f"barge_{i}"].fuel / self.cfg.barge.tank,
                        self.ent[f"barge_{i}"].heading,
                        f"barge_{i}" in self._refuel_barges)
                       for i in range(self.cfg.n_barges)
                       if self.ent[f"barge_{i}"].alive],   # lost tankers vanish
            "fish": [f.pos.copy() for f in self.fish],
            "harpoon_shots": list(self._harpoon_shots),
            "t": self.t,
            "catches": self._ep_catches,
            "scare_radius": self.cfg.scare_radius,
        }
