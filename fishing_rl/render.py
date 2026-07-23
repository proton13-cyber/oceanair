"""Minimal pygame renderer for the fishing environment.

Blue = fishing boats (with fuel bar), orange = barges (with fuel bar + scare ring),
green = fish, gray square = port. Call `Renderer(env).draw()` each step; returns False
if the window was closed.
"""
from __future__ import annotations

import math

import numpy as np

# Top-down aircraft silhouettes in local coords (nose points +x), rotated to heading.
# Fighter jet (fishing boat): pointed nose, swept delta wings, tailplanes.
_JET = [(15, 0), (1, 2), (-9, 10), (-6, 2), (-13, 6), (-15, 0),
        (-13, -6), (-6, -2), (-9, -10), (1, -2)]
# KC-135 tanker (barge): long fuselage, straight high-aspect wings, tail.
_TANKER = [(22, 0), (7, 2), (3, 3), (1, 18), (-3, 18), (-2, 3), (-14, 3),
           (-15, 10), (-19, 10), (-18, 2), (-20, 0),
           (-18, -2), (-19, -10), (-15, -10), (-14, -3), (-2, -3),
           (-3, -18), (1, -18), (3, -3), (7, -2)]


class Renderer:
    def __init__(self, env, window=900, fps=30, record_path=None):
        import pygame
        self.pygame = pygame
        self.env = env
        self.window = window
        # uniform scale so the world keeps its true proportions; window matches aspect
        cfg = env.cfg
        self.scale = window / max(cfg.world_width, cfg.world_height)
        # round window dims up to even numbers — h264 video encoding requires it
        self.win_w = int(round(cfg.world_width * self.scale)) + 1 & ~1
        self.win_h = int(round(cfg.world_height * self.scale)) + 1 & ~1
        self.fps = fps
        pygame.init()
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption("Fishing RL")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)
        self._harpoons = []  # in-flight harpoon projectiles being animated
        # optional video recording (streams frames to a file via imageio)
        self._writer = None
        self._record_path = record_path
        if record_path:
            import imageio
            self._writer = imageio.get_writer(record_path, fps=fps,
                                              macro_block_size=None)

    def _px(self, pos):
        return int(pos[0] * self.scale), int(pos[1] * self.scale)

    def _draw_craft(self, pos, heading, pts, color, size):
        """Draw a top-down aircraft silhouette rotated to its heading."""
        cx, cy = self._px(pos)
        ct, st = math.cos(heading), math.sin(heading)
        poly = [(cx + (x * ct - y * st) * size, cy + (x * st + y * ct) * size)
                for (x, y) in pts]
        self.pygame.draw.polygon(self.screen, color, poly)
        self.pygame.draw.polygon(self.screen, (10, 15, 25), poly, 1)  # outline

    def _fuel_bar(self, pos, frac, color):
        pg = self.pygame
        x, y = self._px(pos)
        w, h = 26, 4
        frac = float(np.clip(frac, 0, 1))
        pg.draw.rect(self.screen, (60, 60, 60), (x - w // 2, y - 16, w, h))
        bar = (200, 60, 60) if frac < 0.2 else color
        pg.draw.rect(self.screen, bar, (x - w // 2, y - 16, int(w * frac), h))

    def draw(self) -> bool:
        pg = self.pygame
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return False

        s = self.env.state_snapshot()
        self.screen.fill((12, 22, 38))  # sea

        # nautical-mile grid: faint minor lines every 50 nmi, brighter labelled
        # majors every 200 nmi (finer than before so motion reads smoothly)
        cfg = self.env.cfg
        minor, major = 50.0, 200.0
        # vertical lines across the width
        x = 0.0
        while x <= cfg.world_width + 1e-6:
            g = int(x * self.scale)
            is_major = int(round(x)) % int(major) == 0
            col = (30, 47, 66) if is_major else (19, 31, 46)
            pg.draw.line(self.screen, col, (g, 0), (g, self.win_h), 1)
            if is_major:
                lab = self.font.render(f"{int(round(x))}", True, (60, 80, 102))
                self.screen.blit(lab, (g + 3, 3))
            x += minor
        # horizontal lines down the height
        y = 0.0
        while y <= cfg.world_height + 1e-6:
            g = int(y * self.scale)
            is_major = int(round(y)) % int(major) == 0
            col = (30, 47, 66) if is_major else (19, 31, 46)
            pg.draw.line(self.screen, col, (0, g), (self.win_w, g), 1)
            if is_major and g > 4:
                lab = self.font.render(f"{int(round(y))}", True, (60, 80, 102))
                self.screen.blit(lab, (3, g + 2))
            y += minor

        # port
        px, py = self._px(s["port"])
        pg.draw.rect(self.screen, (170, 170, 170), (px - 8, py - 8, 16, 16))

        # fish
        for f in s["fish"]:
            pg.draw.circle(self.screen, (80, 220, 120), self._px(f), 4)

        # barges = KC-135 tankers (draw scare ring first); red while offloading fuel
        for pos, fuel, heading, refueling in s["barges"]:
            c = self._px(pos)
            pg.draw.circle(self.screen, (90, 60, 20),
                           c, int(s["scare_radius"] * self.scale), 1)
            col = (235, 70, 70) if refueling else (240, 160, 40)
            self._draw_craft(pos, heading, _TANKER, col, 1.0)
            self._fuel_bar(pos, fuel, (240, 160, 40))

        # boats = fighter jets; red while taking on fuel
        for pos, fuel, alive, heading, ammo, refueling in s["boats"]:
            if not alive:
                color = (90, 90, 90)
            elif refueling:
                color = (235, 70, 70)
            else:
                color = (70, 150, 240)
            self._draw_craft(pos, heading, _JET, color, 1.0)
            if alive:
                self._fuel_bar(pos, fuel, (70, 150, 240))
                # harpoon magazine: a row of pips above the jet (dim when spent)
                cx, cy = self._px(pos)
                cap = self.env.cfg.harpoon_ammo
                x0 = cx - (cap - 1) * 3
                for k in range(cap):
                    col = (120, 235, 215) if k < ammo else (55, 74, 68)
                    pg.draw.circle(self.screen, col, (x0 + k * 6, cy - 23), 2)

        # harpoon shots: animate a projectile flying boat -> fish. The sim removes the
        # fish instantly, so we keep the doomed fish drawn until the harpoon arrives,
        # then flash on impact.
        hs_speed = self.env.cfg.harpoon_speed          # AMRAAM velocity (~Mach 4)
        for bp, fp in s.get("harpoon_shots", []):
            wd = float(np.hypot(fp[0] - bp[0], fp[1] - bp[1])) + 1e-6
            self._harpoons.append({"a": self._px(bp), "b": self._px(fp),
                                   "t": 0.0, "inc": hs_speed / wd})
        still_flying = []
        for h in self._harpoons:
            h["t"] += h["inc"]
            ax, ay = h["a"]; bx, by = h["b"]
            if h["t"] < 1.0:                      # in flight: fish still there, spear chasing
                t = h["t"]
                pg.draw.circle(self.screen, (80, 220, 120), (bx, by), 4)  # the target fish
                hx = int(ax + (bx - ax) * t); hy = int(ay + (by - ay) * t)
                pg.draw.line(self.screen, (90, 170, 230), (ax, ay), (hx, hy), 1)
                tx = int(ax + (bx - ax) * max(0.0, t - 0.18))
                ty = int(ay + (by - ay) * max(0.0, t - 0.18))
                pg.draw.line(self.screen, (220, 255, 255), (tx, ty), (hx, hy), 3)
                pg.draw.circle(self.screen, (255, 255, 255), (hx, hy), 3)
                still_flying.append(h)
            elif h["t"] < 1.4:                    # impact flash where the fish was struck
                pg.draw.circle(self.screen, (255, 240, 180), (bx, by), 8, 2)
                still_flying.append(h)
        self._harpoons = still_flying

        hud = f"t={s['t']}  catches={s['catches']}  fish={len(s['fish'])}"
        self.screen.blit(self.font.render(hud, True, (230, 230, 230)), (8, 8))

        pg.display.flip()
        if self._writer is not None:                 # capture this frame to the video
            frame = pg.surfarray.array3d(self.screen).transpose(1, 0, 2)
            self._writer.append_data(frame)
        self.clock.tick(self.fps)
        return True

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            print(f"saved video -> {self._record_path}")
        self.pygame.quit()
