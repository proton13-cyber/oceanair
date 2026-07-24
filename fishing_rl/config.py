"""Central configuration for the fishing simulation.

Everything tunable lives here so a study has a single surface to edit: world size,
vessel dynamics, fuel economy, harpoon/dock, fish spawning, curriculum spawn band.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Dynamics:
    max_speed: float          # nmi / step at full throttle
    max_turn_rate: float      # radians / step at full turn (hard cap)
    tank: float               # max fuel (lb)
    idle_burn: float          # lb / step regardless of speed
    move_burn: float          # lb / step per unit speed
    # Turn agility: angular rate is capped at turn_agility / speed, so a FAST craft
    # arcs wide while a SLOW one still turns tight (matches real aircraft: radius grows
    # with v^2). NOTE: the real F-15 radius is a pivot-in-place at this 1000-nmi scale;
    # this is STYLIZED for readable, graceful cruise turns.
    turn_agility: float = 1e9  # default huge = no limit (pivot freely)


@dataclass
class Curriculum:
    """Scales difficulty. Interpolate stage in [0,1] during training."""
    enabled: bool = True
    # fish spawn band as fraction of world width (world = 1000 nmi), easy -> hard.
    # Port is on the left shore (x=100 nmi). Grounds are bounded by the boats'
    # 1000 nmi range: at 350-550 nmi out a boat arrives with working margin, then
    # depends on forward barges (it can only refuel from barges, never port).
    easy_min_x: float = 0.30
    easy_max_x: float = 0.50
    # the HARD (stage 1.0) band is derived from grounds_center_x +/- grounds_half_span
    # on the Config below — that is the single "theater depth" control.
    # fuel generosity multiplier from easy->hard:
    easy_tank_mult: float = 1.5
    hard_tank_mult: float = 1.0


@dataclass
class Config:
    # ---- fleet ----
    game_mode: str = "fishing"  # "fishing" (classic) or "escort" (F-15s defend F-18 dive boats harvesting shellfish)
    n_boats: int = 4
    n_barges: int = 8
    n_dive_boats: int = 0       # escort mode only: F-18 strikers that harvest shellfish (0 keeps fishing mode inert)
    n_bingo_tankers: int = 0    # dedicated RECOVERY tankers (extra, beyond n_barges) that hold station near the dock to catch craft scrambling home on fumes. 0 = disabled
    bingo_tanker_dist: float = 0.15  # how far from the dock (fraction of world WIDTH) the bingo tankers loiter — small = right on the doorstep, larger = catch stragglers earlier

    # ---- world ----
    world_width: float = 1000.0       # nmi, east-west (x)
    world_height: float = 1000.0      # nmi, north-south (y) — set != width for a non-square ocean
    port_frac: tuple = (0.10, 0.50)   # port position as fraction of world (x, y)
    dt: float = 1.0
    max_steps: int = 3600      # ~60 hr voyage: long enough for several refuel / reload cycles
    min_speed_frac: float = 0.10  # boats & barges can NEVER sit still: always move >= this fraction of max_speed (when fueled)
    accel_frac: float = 0.15   # momentum: speed can change by at most this fraction of max_speed per step
    loiter_radius: float = 180.0  # nmi — when idle (no fish), boats patrol a ring this wide instead of piling on one point
    loiter_spin: float = 0.015    # rad/step the patrol ring rotates, so idle boats sweep wide arcs instead of spinning in place
    barge_reserve_mult: float = 2.0  # tanker JOKER fuel: it turns back to port once fuel drops to (straight-line-home burn x THIS). Higher = more padding = turns back earlier (fewer strandings, less time on station)
    barge_stage_standoff: float = 0.12  # tanker loiter line sits this fraction of width PORT-SIDE of the grounds center (so tankers stay just behind the fish band)
    barge_stage_y: float = 0.50  # tanker loiter-loop center as a fraction of world HEIGHT
    barge_waves: int = 1         # split tankers into this many cadence groups; each wave deploys forward only after the one before it has begun refueling boats, so they cycle OUT OF PHASE. 1 = all deploy together
    barge_wave_trigger: int = 1  # EVENT-based release: the next wave launches once this many at-sea refuels have happened (robust to stochastic timing vs a fixed clock). A not-yet-released wave PARKS at the dock, topped up, not burning fuel
    barge_loiter_radius: float = 110.0  # nmi — idle tankers fly a wide loiter loop this wide around the staging point (like the boats), spread by phase
    boat_defend_radius: float = 120.0  # nmi — a fish within this of a live tanker is a THREAT; the nearest boat is dispatched to intercept it before normal fishing
    boat_engage_margin: float = 40.0   # nmi a boat will close BEYOND harpoon_range to get a shot; a target farther than harpoon_range+this is not worth chasing (a fast fleeing mover would just drag the boat out of fuel range) — hold station instead
    harpoon_range: float = 86.0   # nmi — AIM-120D AMRAAM reach (~160 km); boats kill targets from standoff (vs the close-net catch_radius)...
    harpoon_speed: float = 44.0   # nmi/min ~Mach 4 AMRAAM (cosmetic: the projectile animation; the catch itself is instant within range)
    harpoon_cooldown: int = 8     # ...must reload this many steps between shots...
    harpoon_ammo: int = 8         # ...and carry only this many missiles before returning to the dock to restock
    amraam_pk_near: float = 0.95  # escort mode: AIM-120 kill probability point-blank... (fishing-mode catch stays deterministic)
    amraam_pk_far: float = 0.55   # ...falling to this at max harpoon_range
    dock_service_steps: int = 45       # min a boat sits on the deck to rearm + refuel (1 step = 1 min)
    barge_dock_service_steps: int = 25  # min a tanker sits at the dock to fully refuel
    refuel_full_frac: float = 0.90  # a refueling boat floats alongside the barge until this full (dwell, not instant)

    # ---- entity dynamics ----
    # Units: distance = nautical miles, 1 step ~ 1 minute, fuel = pounds.
    # Boat = F-14 analog (fast consumer): 20,000 lb tank = 1000 nmi range at cruise.
    boat: Dynamics = field(default_factory=lambda: Dynamics(
        max_speed=8.33, max_turn_rate=0.35, tank=20000.0,  # F-15C: 500 kts; 20k lb -> ~1250 nmi range
        idle_burn=67.0, move_burn=7.9,                     # loiter ~4000 lb/hr, cruise ~8000 lb/hr
        turn_agility=1.0))                                 # wide arc at cruise, tight when slow
    # Barge = KC-135 tanker analog (jet: fast, big fuel load, offloads to boats).
    barge: Dynamics = field(default_factory=lambda: Dynamics(
        max_speed=7.67, max_turn_rate=0.10, tank=200000.0,  # 460 kts; ~200k lb fuel
        idle_burn=120.0, move_burn=8.2,                    # cruise ~11000 lb/hr (183 lb/step)
        turn_agility=0.6))                                 # big jet -> wide turns
    # Dive boat = F/A-18E Super Hornet analog (escort-mode striker; AGM-65 Maverick).
    dive: Dynamics = field(default_factory=lambda: Dynamics(
        max_speed=8.0, max_turn_rate=0.30, tank=18000.0,   # ~480 kts; a touch less legs/agility than the F-15
        idle_burn=70.0, move_burn=8.1,
        turn_agility=0.9))

    # ---- fish ----
    # Fish travel in schools. Each episode has a random number of schools (< 10), and
    # they swim (slower than boats). Sparse on purpose so boats must range for them.
    max_fish: int = 9                 # total fish cap across all schools (sparse)
    max_schools: int = 6              # cap on how many schools can be on the map at once
    initial_schools: int = 1          # schools present at t=0 (keep low so fish ACCUMULATE over time instead of dumping a full map at the start)
    school_size: int = 3              # fish per school
    school_spawn_prob: float = 0.05   # per-step chance to add a fresh school (under cap)
    fish_speed_frac: float = 0.40     # fish swim at 40% of boat max speed
    fish_wander: float = 0.20         # random heading change per step (radians)
    school_spread: float = 25.0       # nmi jitter of fish around their school centre
    fish_barge_clearance_mult: float = 2.0  # schools spawn >= this * scare_radius from any barge
    # ---- theater depth: the single knob that pushes the whole operating area out ----
    grounds_center_x: float = 0.30  # center of the fishing grounds as a fraction of world WIDTH; raise it to push fish (and, via the standoff, the tankers and boat patrol) DEEPER into the ocean
    grounds_half_span: float = 0.10  # half the fish band's X-width (band = center +/- this); 0.10 -> a 0.20-wide band
    # fish spawn randomly within the grounds band (X) and this Y band
    fish_y_min: float = 0.35
    fish_y_max: float = 0.65

    # ---- interaction radii ----
    catch_radius: float = 15.0
    scare_radius: float = 60.0
    barge_avoid_radius: float = 70.0  # barges must keep clear of fish by this much; they evade if a fish gets closer
    refuel_radius: float = 25.0       # boat<->barge fuel transfer range
    port_radius: float = 40.0         # barge<->port refill range
    transfer_rate: float = 2000.0     # lb / step, barge -> boat (aerial-refuel offload rate)
    port_refill_rate: float = 4000.0  # lb / step, port -> barge (fast dockside pumping)

    # ---- escort mode: shellfish (dive-boat strike targets) ----
    # Reefs sit in their OWN depth band so you can mission-plan how deep the strike goes.
    shellfish_center_x: float = 0.45   # reef band center (fraction of world width)
    shellfish_half_span: float = 0.08  # half the reef band's X-width
    shellfish_y_min: float = 0.30
    shellfish_y_max: float = 0.70
    shellfish_min_active: int = 2      # always keep at least this many reefs up
    shellfish_max_active: int = 5      # never more than this many at once
    shellfish_respawn_prob: float = 0.50  # chance a struck reef is replaced by a fresh one

    # ---- escort mode: weapons (range triangle: AMRAAM 86 > AA-12 44 > Maverick 13) ----
    aa12_range: float = 44.0      # nmi — AA-12 Adder (R-77); fish fire at dive boats
    aa12_speed: float = 44.0      # nmi/min ~Mach 4 (cosmetic animation)
    aa12_cooldown: int = 10       # steps a fish waits between AA-12 shots
    aa12_pk_near: float = 0.60    # AA-12 kill probability point-blank...
    aa12_pk_far: float = 0.15     # ...falling to this at aa12_range
    maverick_range: float = 13.0  # nmi — AGM-65 Maverick; dive boats strike shellfish (must close in)
    maverick_speed: float = 10.0  # nmi/min ~Mach 0.9 (cosmetic animation)
    maverick_cooldown: int = 4    # steps between Maverick shots
    maverick_ammo: int = 12       # Mavericks per dive boat before a dock rearm
    maverick_pk_near: float = 0.95
    maverick_pk_far: float = 0.70
    fish_threat_speed_frac: float = 0.90  # escort mode: fish pursue dive boats at this fraction of boat speed (A-4)
    dive_bingo_mult: float = 2.0  # dive boats head to a tanker when fuel drops to (dist/speed * burn * THIS); raise it to bug out earlier with more reserve (fuel starvation is the main loss)
    # escort mode: A-4 threats enter from the RIGHT (east) side and drive in on the dive
    # boats, giving the escorts a defined threat axis to screen.
    fish_spawn_center_x: float = 0.85
    fish_spawn_half_span: float = 0.10
    escort_screen_standoff: float = 0.05  # escort mode: idle F-15s screen this fraction of WIDTH east of the dive boats. Tight (with the intercept-at-range logic) keeps the tankers' customers close, cutting tanker strandings ~40% at no defense cost; widen only if threats start leaking

    # ---- curriculum ----
    curriculum: Curriculum = field(default_factory=Curriculum)

    seed: int = 0

    # ---- derived ----
    @property
    def port_pos(self):
        return (self.port_frac[0] * self.world_width,
                self.port_frac[1] * self.world_height)

    @property
    def n_barges_total(self):
        # regular forward tankers PLUS any dedicated bingo/recovery tankers (extra)
        return self.n_barges + self.n_bingo_tankers

    def is_bingo(self, i):
        return i >= self.n_barges   # barge indices >= n_barges are bingo tankers

    @property
    def agent_names(self):
        names = ([f"boat_{i}" for i in range(self.n_boats)] +
                 [f"barge_{i}" for i in range(self.n_barges_total)])
        if self.game_mode == "escort":   # dive boats only exist in escort mode
            names += [f"dive_{i}" for i in range(self.n_dive_boats)]
        return names

    def role_of(self, name: str) -> str:
        if name.startswith("boat"):
            return "boat"
        if name.startswith("dive"):
            return "dive"
        return "barge"
