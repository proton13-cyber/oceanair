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
    __slots__ = ("pos", "vel", "aa12_cd")

    def __init__(self, pos, vel):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.vel = np.asarray(vel, dtype=np.float64)
        self.aa12_cd = 0         # escort mode: steps until this fish can fire an AA-12 again


class _Shellfish:
    """A static reef target (escort mode). No velocity — dive boats must close to it."""
    __slots__ = ("pos",)

    def __init__(self, pos):
        self.pos = np.asarray(pos, dtype=np.float64)


class _Entity:
    __slots__ = ("pos", "heading", "fuel", "alive", "role", "speed",
                 "refueling", "harpoon_cd", "harpoons_left", "reloading", "dock_timer",
                 "maverick_cd", "maverick_ammo", "killed")

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
        self.maverick_cd = 0     # dive boat: steps until next Maverick shot
        self.maverick_ammo = 0   # dive boat: Mavericks in the rack (set in reset)
        self.killed = False      # shot down (escort attrition) — distinct from fuel-death


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
        self.shellfish: list[_Shellfish] = []   # escort mode: static reef targets
        self.t = 0
        self.stage = 0.0  # curriculum stage in [0, 1]
        self._ep_catches = 0
        self._ep_deaths = 0
        self._lost_barges = 0        # tankers that ran dry at sea and were removed
        self._sea_refuels = self._barge_dock_refuels = self._boat_dock_refuels = 0
        self._ep_shellfish = 0       # escort: shellfish harvested by dive boats
        self._dive_lost = 0          # escort: dive boats shot down (attrition)
        self._ep_fish_killed = 0     # escort: threat fish downed by escorts
        self._receiving = set()      # boats taking fuel from a barge this step
        self._refuel_barges = set()  # barges giving fuel this step
        self._harpoon_shots = []  # (shooter_pos, target_pos) AMRAAM shots THIS step (render)
        self._maverick_shots = []  # (dive_pos, reef_pos) Maverick strikes this step (render)
        self._aa12_shots = []      # (fish_pos, dive_pos) AA-12 shots this step (render)

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
        self._ep_shellfish = 0
        self._dive_lost = 0
        self._ep_fish_killed = 0
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
        if cfg.game_mode == "escort":       # dive boats launch from the dock too
            for i in range(cfg.n_dive_boats):
                jitter = self.rng.uniform(-20, 20, size=2)
                d = _Entity(pos=[px + jitter[0], py + jitter[1]], heading=0.0,
                            fuel=cfg.dive.tank * tank_mult, role="dive")
                d.maverick_ammo = cfg.maverick_ammo
                self.ent[f"dive_{i}"] = d

        self.fish = []
        for _ in range(max(0, cfg.initial_schools)):   # small seed; the rest accumulate over time
            self._spawn_school()

        self.shellfish = []
        if cfg.game_mode == "escort":       # seed the reef sites (min_active up front)
            for _ in range(cfg.shellfish_min_active):
                self._spawn_shellfish()

        infos = {a: {} for a in self.agents}
        return {}, infos

    def _spawn_shellfish(self):
        """Add one static reef in the shellfish band (its own depth, for mission planning)."""
        cfg = self.cfg
        if len(self.shellfish) >= cfg.shellfish_max_active:
            return
        lo = cfg.shellfish_center_x - cfg.shellfish_half_span
        hi = cfg.shellfish_center_x + cfg.shellfish_half_span
        pos = np.array([self.rng.uniform(lo, hi) * cfg.world_width,
                        self.rng.uniform(cfg.shellfish_y_min, cfg.shellfish_y_max) * cfg.world_height])
        self.shellfish.append(_Shellfish(pos=pos))

    def _spawn_school(self):
        """Add a school at a random spot in the spawn band, clear of every barge.
        In escort mode the A-4 threats enter from the right (east) side instead."""
        cfg = self.cfg
        Wx, Wy = cfg.world_width, cfg.world_height
        if cfg.game_mode == "escort":
            lo = cfg.fish_spawn_center_x - cfg.fish_spawn_half_span
            hi = cfg.fish_spawn_center_x + cfg.fish_spawn_half_span
        else:
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
        """Fish swim: wander their heading, advance, reflect off the world edges.
        In ESCORT mode they instead pursue the nearest dive boat (simple homing)."""
        cfg = self.cfg
        bound = (cfg.world_width, cfg.world_height)
        if cfg.game_mode == "escort":
            speed = cfg.fish_threat_speed_frac * cfg.boat.max_speed
            dives = [self.ent[f"dive_{i}"] for i in range(cfg.n_dive_boats)
                     if self.ent[f"dive_{i}"].alive]
            for f in self.fish:
                if dives:   # steer straight at the nearest live dive boat
                    tgt = min(dives, key=lambda d: np.linalg.norm(d.pos - f.pos)).pos
                    dvec = tgt - f.pos
                    f.vel = dvec / (np.linalg.norm(dvec) + 1e-9) * speed
                else:
                    f.vel *= speed / (math.hypot(f.vel[0], f.vel[1]) + 1e-9)
                ang = self.rng.uniform(-cfg.fish_wander, cfg.fish_wander)  # slight jitter
                c, s = math.cos(ang), math.sin(ang)
                vx, vy = f.vel
                f.vel[0], f.vel[1] = vx * c - vy * s, vx * s + vy * c
                f.pos += f.vel * cfg.dt
                for k in (0, 1):
                    if f.pos[k] < 0:
                        f.pos[k] = -f.pos[k]; f.vel[k] = -f.vel[k]
                    elif f.pos[k] > bound[k]:
                        f.pos[k] = 2 * bound[k] - f.pos[k]; f.vel[k] = -f.vel[k]
            return
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

    def _pk_hit(self, d, rng, pk_near, pk_far):
        """Probability-of-kill roll: Pk ramps from pk_far at max range to pk_near point-blank."""
        frac = 1.0 - min(d, rng) / max(rng, 1e-9)
        return self.rng.random() < pk_far + (pk_near - pk_far) * frac

    def _escort_combat(self):
        """Escort-mode combat: dive->shellfish (Maverick), F-15->fish (AMRAAM), fish->dive (AA-12)."""
        cfg = self.cfg
        # (a) dive boats strike the nearest reef in Maverick range (probabilistic kill)
        struck = 0
        for i in range(cfg.n_dive_boats):
            dv = self.ent[f"dive_{i}"]
            if (not dv.alive or dv.maverick_cd > 0 or dv.maverick_ammo <= 0
                    or f"dive_{i}" in self._receiving):
                continue
            best, bestd = None, cfg.maverick_range
            for r in self.shellfish:
                d = np.linalg.norm(dv.pos - r.pos)
                if d <= bestd:
                    bestd, best = d, r
            if best is not None:
                dv.maverick_cd = cfg.maverick_cooldown
                dv.maverick_ammo -= 1
                self._maverick_shots.append((dv.pos.copy(), best.pos.copy()))
                if self._pk_hit(bestd, cfg.maverick_range,
                                cfg.maverick_pk_near, cfg.maverick_pk_far):
                    self.shellfish.remove(best)
                    self._ep_shellfish += 1
                    struck += 1
        for _ in range(struck):             # struck reef -> chance a fresh one pops up
            if self.rng.random() < cfg.shellfish_respawn_prob:
                self._spawn_shellfish()
        while (len(self.shellfish) < cfg.shellfish_min_active      # keep min reefs up
               and len(self.shellfish) < cfg.shellfish_max_active):
            self._spawn_shellfish()

        # (b) escorts shoot down threat fish with AMRAAM (defense first, so a fish downed
        # this step can't fire; Pk misses let some leak through -> attrition emerges)
        killed = set()
        for j in range(cfg.n_boats):
            bt = self.ent[f"boat_{j}"]
            if (not bt.alive or bt.harpoon_cd > 0 or bt.harpoons_left <= 0
                    or f"boat_{j}" in self._receiving):
                continue
            best, bestd = None, cfg.harpoon_range
            for fi, f in enumerate(self.fish):
                if fi in killed:
                    continue
                d = np.linalg.norm(bt.pos - f.pos)
                if d <= bestd:
                    bestd, best = d, fi
            if best is not None:
                bt.harpoon_cd = cfg.harpoon_cooldown
                bt.harpoons_left -= 1
                self._harpoon_shots.append((bt.pos.copy(), self.fish[best].pos.copy()))
                if self._pk_hit(bestd, cfg.harpoon_range,
                                cfg.amraam_pk_near, cfg.amraam_pk_far):
                    killed.add(best)
                    self._ep_fish_killed += 1
        if killed:
            self.fish = [f for fi, f in enumerate(self.fish) if fi not in killed]

        # (c) surviving threat fish fire AA-12 at the nearest dive boat in range
        for f in self.fish:
            if f.aa12_cd > 0:
                continue
            best, bestd = None, cfg.aa12_range
            for i in range(cfg.n_dive_boats):
                dv = self.ent[f"dive_{i}"]
                if not dv.alive:
                    continue
                d = np.linalg.norm(f.pos - dv.pos)
                if d <= bestd:
                    bestd, best = d, dv
            if best is not None:
                f.aa12_cd = cfg.aa12_cooldown
                self._aa12_shots.append((f.pos.copy(), best.pos.copy()))
                if self._pk_hit(bestd, cfg.aa12_range,
                                cfg.aa12_pk_near, cfg.aa12_pk_far):
                    best.alive = False
                    best.killed = True
                    self._dive_lost += 1

    # ---- step ---------------------------------------------------------------
    def step(self, actions: dict[str, np.ndarray]):
        cfg = self.cfg

        self._harpoon_shots = []   # cleared each step; filled by ranged catches below
        self._maverick_shots = []
        self._aa12_shots = []
        # harpoon reload: tick down each boat's cooldown
        for j in range(cfg.n_boats):
            bt = self.ent[f"boat_{j}"]
            if bt.harpoon_cd > 0:
                bt.harpoon_cd -= 1
        if cfg.game_mode == "escort":   # tick Maverick (dive) and AA-12 (fish) cooldowns
            for i in range(cfg.n_dive_boats):
                dv = self.ent[f"dive_{i}"]
                if dv.maverick_cd > 0:
                    dv.maverick_cd -= 1
            for f in self.fish:
                if f.aa12_cd > 0:
                    f.aa12_cd -= 1

        # 1) movement + fuel burn
        for name in self.agents:
            e = self.ent[name]
            if not e.alive:
                continue
            dyn = {"boat": cfg.boat, "dive": cfg.dive}.get(e.role, cfg.barge)
            a = np.asarray(actions[name], dtype=np.float64).reshape(-1)
            throttle = float(np.clip((a[0] + 1.0) * 0.5, 0.0, 1.0))
            turn = float(np.clip(a[1], -1.0, 1.0))

            # turn agility: fast -> wide arc, slow -> tight (angular cap =
            # turn_agility / speed, never above max_turn_rate)
            eff_turn = min(dyn.max_turn_rate, dyn.turn_agility / max(e.speed, 1e-3))
            e.heading = _wrap_angle(e.heading + turn * eff_turn * cfg.dt)
            # a boat moored alongside a barge, or sitting on the deck at the dock, may
            # be still; everyone else must keep moving (>= min_speed_frac of max)
            moored = (e.role in ("boat", "dive") and e.refueling
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

        # 2) death check. FISHING: a boat running dry is catastrophic-terminal.
        # ESCORT: boat/dive fuel-loss is a logged, NON-terminal loss (episode runs full
        # length so we can measure attrition).
        escort = cfg.game_mode == "escort"
        dead = False
        for i in range(cfg.n_boats):
            e = self.ent[f"boat_{i}"]
            if e.alive and e.fuel <= 0.0:
                e.alive = False
                if escort:
                    self._ep_deaths += 1
                else:
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
        # dive boats that run dry away from port are lost too (escort, non-terminal)
        if escort:
            for i in range(cfg.n_dive_boats):
                dv = self.ent[f"dive_{i}"]
                if (dv.alive and dv.fuel <= 0.0
                        and np.hypot(dv.pos[0] - px, dv.pos[1] - py) > cfg.port_radius):
                    dv.alive = False
                    self._dive_lost += 1

        # 3) refueling (barge->boat, port->barge) — automatic on proximity
        px, py = cfg.port_pos
        port = np.array([px, py])
        # refuel/rearm clients: fishing boats, plus dive boats in escort mode
        clients = [f"boat_{j}" for j in range(cfg.n_boats)]
        if escort:
            clients += [f"dive_{i}" for i in range(cfg.n_dive_boats)]

        def _tank(role):
            return cfg.dive.tank if role == "dive" else cfg.boat.tank

        # craft service at the dock: refuel continuously, but rearming (harpoons /
        # Mavericks) takes a ground dwell (dock_service_steps) on the deck
        for name in clients:
            bt = self.ent[name]
            if bt.alive and np.linalg.norm(bt.pos - port) <= cfg.port_radius:
                bt.dock_timer += 1
                bt.fuel = min(_tank(bt.role), bt.fuel + cfg.port_refill_rate * cfg.dt)
                if bt.dock_timer == cfg.dock_service_steps:
                    if bt.role == "dive":
                        bt.maverick_ammo = cfg.maverick_ammo
                    else:
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
            # barge transfers to nearby clients (boats + dive boats; draws from its tank)
            for name in clients:
                bt = self.ent[name]
                if not bt.alive:
                    continue
                # a craft only takes fuel when it has COMMITTED to a refuel stop (low
                # and latched) — no unrealistic tickle-topping of craft passing by
                if (bt.refueling and b.fuel > 0
                        and np.linalg.norm(b.pos - bt.pos) <= cfg.refuel_radius):
                    want = _tank(bt.role) - bt.fuel
                    amount = min(cfg.transfer_rate * cfg.dt, b.fuel, want)
                    if amount > 0:
                        bt.fuel += amount
                        b.fuel -= amount
                        receiving_now.add(name)
                        giving_now.add(f"barge_{i}")
        # count each at-sea rendezvous once (boats that just started taking fuel)
        self._sea_refuels += len(receiving_now - self._receiving)
        self._receiving = receiving_now
        self._refuel_barges = giving_now

        # 3.5) fish swim
        self._move_fish()

        if not escort:
            # 4) FISHING: boats catch fish (hitscan, deterministic within range).
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
        else:
            self._escort_combat()   # 4) ESCORT: Maverick / AMRAAM / AA-12 (hitscan + Pk)

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
                  "boat_dock_refuels": self._boat_dock_refuels,
                  # escort-mode metrics (0 in fishing mode)
                  "game_mode": cfg.game_mode,
                  "shellfish_harvested": self._ep_shellfish,
                  "dive_boats_lost": self._dive_lost,
                  "dive_boats_start": cfg.n_dive_boats if cfg.game_mode == "escort" else 0,
                  "fish_killed": self._ep_fish_killed}
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
            # escort mode (empty/zero in fishing mode so fishing frames are unchanged)
            "game_mode": self.cfg.game_mode,
            "dive_boats": [(self.ent[f"dive_{i}"].pos.copy(),
                            self.ent[f"dive_{i}"].fuel / self.cfg.dive.tank,
                            self.ent[f"dive_{i}"].alive,
                            self.ent[f"dive_{i}"].heading,
                            self.ent[f"dive_{i}"].maverick_ammo,
                            f"dive_{i}" in self._receiving)
                           for i in range(self.cfg.n_dive_boats)
                           if self.ent[f"dive_{i}"].alive],
            "shellfish": [r.pos.copy() for r in self.shellfish],
            "maverick_shots": list(self._maverick_shots),
            "aa12_shots": list(self._aa12_shots),
            "shellfish_harvested": self._ep_shellfish,
            "dive_boats_lost": self._dive_lost,
        }
