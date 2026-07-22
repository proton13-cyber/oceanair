"""A scripted baseline policy so the environment can be watched before any training.

Boats: chase the nearest fish; if low on fuel, head to the nearest barge; if nothing
to do, drift toward port.
Barges: stage between port and the fish cloud, hovering just OUTSIDE the scare radius;
return to port when low on fuel. This is intentionally simple — the point is to sanity
check the sim, not to be optimal.
"""
from __future__ import annotations

import math
import numpy as np

from .config import Config


BOAT_REFUEL_FRAC = 0.35   # boats head to a barge below this fraction of tank


def _steer_towards(pos, heading, target, dyn_max_turn):
    """Return [throttle, turn] in [-1,1] steering `pos` toward `target`."""
    d = target - pos
    dist = np.linalg.norm(d)
    if dist < 1e-6:
        return np.array([-1.0, 0.0])  # idle
    desired = math.atan2(d[1], d[0])
    err = (desired - heading + math.pi) % (2 * math.pi) - math.pi
    turn = float(np.clip(err / max(dyn_max_turn, 1e-6), -1.0, 1.0))
    # slow down when we need to turn hard, and when we're basically there
    throttle = 1.0 if abs(err) < 0.6 else 0.2
    if dist < 8.0:
        throttle = 0.0
    # map throttle [0,1] -> action space [-1,1]
    return np.array([throttle * 2 - 1, turn])


def boat_actions(env, cfg) -> dict:
    """Fully-scripted boat behavior (no learning).

    Low fuel  -> steer to the nearest barge to refuel (boats never go to the dock;
                 there is no unloading in this game).
    Otherwise -> chase fish with *finders-keepers*: a greedy one-fish-per-boat
    assignment (nearest global pair first) so two boats never target the same
    fish. Boats with no fish to claim loiter near the school (or hold position if
    the sea is empty), ready to grab the next fish — they do NOT return to port.
    """
    actions = {}
    fishing = []  # boats that want a fish this step
    port = np.array(cfg.port_pos)

    for i in range(cfg.n_boats):
        e = env.ent[f"boat_{i}"]
        if not e.alive:
            e.refueling = False
            e.reloading = False
            actions[f"boat_{i}"] = np.array([-1.0, 0.0])
            continue
        # reload latch: once out of harpoons, commit to a dock run to restock (the env
        # reloads on arrival). Barges refuel the boat automatically as it passes them.
        if e.reloading:
            if e.harpoons_left >= cfg.harpoon_ammo:
                e.reloading = False
        elif e.harpoons_left <= 0:
            e.reloading = True
        # refuel latch: once low, commit to a full refuel stop (float alongside the
        # barge) until ~full — no tickle-topping, so refueling has a real dwell cost
        if e.refueling:
            if e.fuel >= cfg.refuel_full_frac * cfg.boat.tank:
                e.refueling = False
        elif e.fuel < BOAT_REFUEL_FRAC * cfg.boat.tank:
            e.refueling = True

        # target priority: restock at dock > refuel at barge > go fishing
        if e.reloading:
            actions[f"boat_{i}"] = _steer_towards(
                e.pos, e.heading, port, cfg.boat.max_turn_rate)
        elif e.refueling:
            barges = [env.ent[f"barge_{k}"] for k in range(cfg.n_barges)
                      if env.ent[f"barge_{k}"].alive]
            # steer to the nearest live tanker; if the fleet is gone, run for the dock
            target = (min(barges, key=lambda b: np.linalg.norm(b.pos - e.pos)).pos
                      if barges else port)
            actions[f"boat_{i}"] = _steer_towards(
                e.pos, e.heading, np.asarray(target), cfg.boat.max_turn_rate)
        else:
            fishing.append(i)

    # DEFENSE FIRST, then finders-keepers. Fish that have drifted near a live tanker
    # are threats (they hurt the barges), so the nearest boat is dispatched to
    # intercept each one before any normal fishing. Remaining boats then claim their
    # nearest fish as usual (unique fish per boat, nearest pair first).
    claims = {}
    if env.fish and fishing:
        taken_boat, taken_fish = set(), set()

        # (a) threat fish: within defend_radius of a live tanker; most-threatening
        #     (closest to a tanker) handled first, each grabbing its nearest free boat
        threats = []
        for fj, f in enumerate(env.fish):
            dmin = min((np.linalg.norm(env.ent[f"barge_{k}"].pos - f.pos)
                        for k in range(cfg.n_barges) if env.ent[f"barge_{k}"].alive),
                       default=float("inf"))
            if dmin <= cfg.boat_defend_radius:
                threats.append((dmin, fj))
        for _, fj in sorted(threats, key=lambda x: x[0]):
            fp = env.fish[fj].pos
            cand = [(float(np.linalg.norm(env.ent[f"boat_{i}"].pos - fp)), i)
                    for i in fishing if i not in taken_boat]
            if cand:
                _, i = min(cand)
                claims[i] = fp
                taken_boat.add(i)
                taken_fish.add(fj)

        # (b) normal finders-keepers for the boats/fish not spoken for
        pairs = []
        for i in fishing:
            if i in taken_boat:
                continue
            ep = env.ent[f"boat_{i}"].pos
            for fj, f in enumerate(env.fish):
                if fj in taken_fish:
                    continue
                pairs.append((float(np.linalg.norm(f.pos - ep)), i, fj))
        pairs.sort(key=lambda x: x[0])
        for _, i, fj in pairs:
            if i in taken_boat or fj in taken_fish:
                continue
            claims[i] = env.fish[fj].pos
            taken_boat.add(i)
            taken_fish.add(fj)

    # chase claimed fish; boats with nothing to claim patrol a wide, slowly rotating
    # ring around the grounds — each boat gets its own phase slot so they SPREAD OUT
    # instead of dogpiling one point and spinning. The ring turns (loiter_spin) so the
    # target never sits still, keeping way on the boats -> wide arcs, no spin-in-place.
    centroid = (np.mean(np.stack([f.pos for f in env.fish]), axis=0)
                if env.fish else None)
    grounds = np.array([cfg.grounds_center_x * cfg.world_width, 0.5 * cfg.world_height])
    loiter_center = centroid if centroid is not None else grounds
    nb = max(1, cfg.n_boats)
    for i in fishing:
        e = env.ent[f"boat_{i}"]
        if i in claims:
            target = claims[i]
        else:
            ang = 2 * math.pi * i / nb + cfg.loiter_spin * env.t
            target = loiter_center + cfg.loiter_radius * np.array(
                [math.cos(ang), math.sin(ang)])
        act = _steer_towards(e.pos, e.heading, np.asarray(target),
                             cfg.boat.max_turn_rate)
        # fuel-saver: ease off only once inside harpoon range AND lined up on the fish
        # (schools swim, so a boat that fully stops loses them — keep enough way on to
        # stay in range). Coast at ~fish speed rather than full cruise.
        if i in claims and np.linalg.norm(np.asarray(target) - e.pos) <= cfg.harpoon_range:
            coast = cfg.fish_speed_frac * 2 - 1   # throttle ~ fish speed, mapped to [-1,1]
            act = np.array([min(act[0], coast), act[1]])
        actions[f"boat_{i}"] = act

    return actions


class HeuristicPolicy:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self, env) -> dict:
        cfg = self.cfg
        port = np.array(cfg.port_pos)

        # --- boats (scripted, finders-keepers) ---
        actions = boat_actions(env, cfg)

        # --- barges ---
        # Priority per barge: (1) refill at port if own fuel is low (staggered by index
        # so they don't all leave station at once); (2) if assigned a low boat, go
        # rendezvous with it; (3) otherwise stage forward, evading fish.
        # Assignment DISTRIBUTES: each low boat gets its OWN nearest available barge, so
        # adding barges actually increases coverage instead of all piling on one boat.
        # idle tankers fly a forward loiter loop (see priority 4 below); low tankers
        # still peel off to refuel and are assigned to low boats.

        going_port = {}
        # worst-case burn per step if the barge ran home at full cruise
        home_burn = cfg.barge.idle_burn + cfg.barge.move_burn * cfg.barge.max_speed
        held_back = {}   # tankers in a later wave, still parked at the dock
        for i in range(cfg.n_barges):
            b = env.ent[f"barge_{i}"]
            # echelon cadence: split the fleet into waves (contiguous blocks); a later
            # wave stays PARKED at the dock (topped up, not burning fuel) until the
            # wave before it has begun working — released once the cumulative at-sea
            # refuel count reaches wave*barge_wave_trigger. Event-based, not clock-
            # based, so it tracks the real (stochastic) operational tempo.
            wave = (i * max(1, cfg.barge_waves)) // max(1, cfg.n_barges)
            held_back[i] = env._sea_refuels < wave * cfg.barge_wave_trigger
            thr = 0.15 + 0.20 * (i / max(1, cfg.n_barges - 1))
            at_port = np.linalg.norm(b.pos - port) <= cfg.port_radius
            servicing = at_port and b.dock_timer < cfg.barge_dock_service_steps
            # distance-aware reserve: fuel needed to reach port from here, doubled
            # for safety. The 2x covers the slow turn-around and ramp-up (a barge
            # pointed the wrong way spends ~30 steps just turning) that a straight-
            # line estimate ignores. A barge turns back before it can strand — no
            # matter how big the world or how small its tank.
            dist_home = np.linalg.norm(b.pos - port)
            reserve = (dist_home / max(cfg.barge.max_speed, 1e-6)) * home_burn * 2.0
            going_port[i] = (servicing
                             or b.fuel < thr * cfg.barge.tank
                             or b.fuel < reserve)

        # neediest boats first; each grabs the nearest barge not already tasked/refilling
        refuel_boats = sorted(
            (env.ent[f"boat_{k}"] for k in range(cfg.n_boats)
             if env.ent[f"boat_{k}"].alive and env.ent[f"boat_{k}"].refueling),
            key=lambda bt: bt.fuel)
        assigned = {}   # barge index -> boat entity
        used = set()
        for bt in refuel_boats:
            cand = [(np.linalg.norm(env.ent[f"barge_{i}"].pos - bt.pos), i)
                    for i in range(cfg.n_barges)
                    if env.ent[f"barge_{i}"].alive and not going_port[i]
                    and not held_back[i] and i not in used]
            if cand:
                _, bi = min(cand)
                assigned[bi] = bt
                used.add(bi)

        for i in range(cfg.n_barges):
            b = env.ent[f"barge_{i}"]
            if not b.alive:
                continue   # lost tanker — removed from the sim

            # 0) reserve wave: sit on the tarmac (steer to port, park there topped up)
            # until this wave's deployment time. Conserves fuel for a fresh sortie.
            if held_back[i]:
                actions[f"barge_{i}"] = _steer_towards(
                    b.pos, b.heading, port, cfg.barge.max_turn_rate)
                continue

            # 1) own fuel low (or servicing at the dock) -> head to port. Self-
            # preservation outranks dodging: a stranded, empty barge is worse than
            # one that lets a fish brush past on its way out. Port is on the shore,
            # away from the grounds, so heading there leaves the fish behind anyway.
            if going_port[i]:
                actions[f"barge_{i}"] = _steer_towards(
                    b.pos, b.heading, port, cfg.barge.max_turn_rate)
                continue

            # 2) collision reflex: barges never delete fish, so a fish that gets
            # inside the scare radius MUST be dodged. Overrides servicing/staging
            # (but not the fuel run above) — give way, then resume.
            panic = [f for f in env.fish
                     if np.linalg.norm(f.pos - b.pos) <= cfg.scare_radius]
            if panic:
                repel = np.zeros(2)
                for f in panic:
                    d = b.pos - f.pos
                    dist = np.linalg.norm(d) + 1e-6
                    repel += d / (dist * dist)
                nrm = np.linalg.norm(repel)
                if nrm > 1e-9:
                    flee = b.pos + (repel / nrm) * 100.0
                    actions[f"barge_{i}"] = _steer_towards(
                        b.pos, b.heading, flee, cfg.barge.max_turn_rate)
                    continue

            # 3) assigned a low boat -> go rendezvous with it
            if i in assigned:
                actions[f"barge_{i}"] = _steer_towards(
                    b.pos, b.heading, assigned[i].pos, cfg.barge.max_turn_rate)
                continue

            # 4) nobody to service -> evade nearby fish, else stage forward
            near = [f for f in env.fish
                    if np.linalg.norm(f.pos - b.pos) < cfg.barge_avoid_radius]
            if near:
                repel = np.zeros(2)
                for f in near:
                    d = b.pos - f.pos
                    dist = np.linalg.norm(d) + 1e-6
                    repel += d / (dist * dist)
                nrm = np.linalg.norm(repel)
                if nrm > 1e-9:
                    flee = b.pos + (repel / nrm) * 100.0
                    actions[f"barge_{i}"] = _steer_towards(
                        b.pos, b.heading, flee, cfg.barge.max_turn_rate)
                    continue

            # forward loiter loop: idle tankers fly a wide, slowly rotating loop
            # around an explicit staging point out in the ocean (like the boats'
            # patrol ring). Each tanker gets its own phase slot so they spread out.
            # Raise grounds_center_x (theater depth) to forward-stage further out —
            # as long as tanks are big enough that the reserve still lets them home.
            depth_frac = cfg.grounds_center_x - cfg.barge_stage_standoff
            stage_center = np.array([depth_frac * cfg.world_width,
                                     cfg.barge_stage_y * cfg.world_height])
            ang = 2 * math.pi * i / max(1, cfg.n_barges) + cfg.loiter_spin * env.t
            target = stage_center + cfg.barge_loiter_radius * np.array(
                [math.cos(ang), math.sin(ang)])
            actions[f"barge_{i}"] = _steer_towards(
                b.pos, b.heading, np.asarray(target), cfg.barge.max_turn_rate)

        return actions
