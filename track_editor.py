import os, sys, math, json, copy, itertools
import pygame
import numpy as np
import cv2
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog

# ── Palette for dynamic rewards ────────────────────────────────────────────────
REWARD_PALETTE = itertools.cycle([
    (255,200,30),(255,130,40),(180,80,255),(60,220,180),
    (255,80,150),(80,200,255),(200,255,80),(255,120,120),
])

GRASS  = (45, 80, 45)
ROAD   = (20, 20, 20)
UI_BG  = (22, 22, 35)
ACCENT = (90,140,255)
HLIGHT = (255,220,60)
PANEL_W = 230

# ── Data classes ───────────────────────────────────────────────────────────────
class TrackObject:
    _counter = itertools.count(1)
    def __init__(self, kind, x, y, w, h, color, orientation="V", label=None):
        self.id   = next(TrackObject._counter)
        self.kind = kind
        self.x, self.y = x, y
        self.w, self.h = w, h
        self.color = color
        self.orientation = orientation  # "H" or "V"
        self.label = label or kind.replace("_"," ").title()
        self.angle = -90  # degrees; only used by start_point (default = up/north)

    def rect(self):
        if self.kind == "start_point":
            return pygame.Rect(self.x-12, self.y-12, 24, 24)
        return pygame.Rect(self.x, self.y, self.w, self.h)

    def to_tuple(self):
        return (self.x, self.y, self.w, self.h)

    def flip(self):
        self.w, self.h = self.h, self.w
        self.orientation = "H" if self.orientation=="V" else "V"


# ── Helpers ────────────────────────────────────────────────────────────────────
def pygame_to_gray(surface):
    arr = pygame.surfarray.array3d(surface)
    arr = np.transpose(arr,(1,0,2))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

def gen_road_points(surface, n=250):
    gray = pygame_to_gray(surface)
    # ROAD=(20,20,20) gray≈20, GRASS=(45,80,45) gray≈65 — threshold at 40
    _, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return []
    cnt = max(contours, key=cv2.contourArea)
    total = cv2.arcLength(cnt, True)
    if total == 0: return []
    step = total / n
    pts, cur = [], 0
    for i in range(len(cnt)):
        arc = cv2.arcLength(cnt[:i+1], False)
        if arc >= cur:
            pts.append(tuple(cnt[i][0]))
            cur += step
        if len(pts) >= n: break
    return pts


# ── Button widget ──────────────────────────────────────────────────────────────
class Btn:
    def __init__(self, rect, label, color=(40,42,60), active_color=(70,110,200), tag=None):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.color = color
        self.active_color = active_color
        self.tag = tag or label
        self.active = False
        self._hover = False

    def draw(self, surf, font):
        col = self.active_color if self.active else (self.color[0]+15,self.color[1]+15,self.color[2]+15) if self._hover else self.color
        pygame.draw.rect(surf, col, self.rect, border_radius=6)
        pygame.draw.rect(surf, ACCENT if self.active else (70,75,110), self.rect, 1, border_radius=6)
        t = font.render(self.label, True, (220,220,235))
        surf.blit(t, t.get_rect(center=self.rect.center))

    def hit(self, event):
        if event.type == pygame.MOUSEMOTION:
            self._hover = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button==1:
            if self.rect.collidepoint(event.pos): return True
        return False


class Slider:
    def __init__(self, rect, label, lo, hi, val):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.lo, self.hi = lo, hi
        self.value = val
        self._drag = False

    def draw(self, surf, font):
        t = font.render(f"{self.label}: {int(self.value)}", True, (200,200,215))
        surf.blit(t, (self.rect.x, self.rect.y-15))
        tr = pygame.Rect(self.rect.x, self.rect.y+3, self.rect.w, 6)
        pygame.draw.rect(surf, (55,58,90), tr, border_radius=3)
        fw = int(self.rect.w * (self.value-self.lo)/(self.hi-self.lo))
        pygame.draw.rect(surf, ACCENT, pygame.Rect(self.rect.x, self.rect.y+3, fw, 6), border_radius=3)
        hx = self.rect.x + fw
        pygame.draw.circle(surf, (180,210,255), (hx, self.rect.y+6), 8)

    def handle(self, event):
        hx = self.rect.x + int(self.rect.w*(self.value-self.lo)/(self.hi-self.lo))
        if event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
            if abs(event.pos[0]-hx)<12 and abs(event.pos[1]-(self.rect.y+6))<12:
                self._drag=True
        if event.type==pygame.MOUSEBUTTONUP: self._drag=False
        if event.type==pygame.MOUSEMOTION and self._drag:
            r = max(0,min(self.rect.w, event.pos[0]-self.rect.x))
            self.value = self.lo + (r/self.rect.w)*(self.hi-self.lo)


# ── Editor ─────────────────────────────────────────────────────────────────────
class Editor:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1210, 600), pygame.RESIZABLE)
        pygame.display.set_caption("Track Editor — DQN-CAR-2D")
        self.clock = pygame.time.Clock()
        self.font  = pygame.font.SysFont("Segoe UI", 13, bold=False)
        self.fontB = pygame.font.SysFont("Segoe UI", 13, bold=True)
        self.fontL = pygame.font.SysFont("Segoe UI", 17, bold=True)
        self.fontXS= pygame.font.SysFont("Segoe UI", 10)

        self.canvas = pygame.Surface((980, 600))
        self.canvas.fill(GRASS)

        # Objects
        self.objects: list[TrackObject] = []
        self.selected_id: int | None = None

        # Drawing state
        self.tool = "draw"   # "draw"|"erase"|"select"
        self.drawing = False
        self.last_pos = None
        self.drag_obj: TrackObject | None = None
        self.drag_off = (0,0)
        self.resize_obj: TrackObject | None = None
        self.resize_anchor = (0,0,0,0)  # (ox,oy,ow,oh)
        self.resize_mouse_start = (0,0)

        self.road_pts: list[tuple] = []
        self.show_pts = False
        self.track_name = "my_track"

        self._build_ui()

    def _build_ui(self):
        sw, sh = self.screen.get_size()
        px = sw - PANEL_W
        bw = PANEL_W - 16
        bx = px + 8

        def b(y, label, color=(40,42,60), acol=(70,110,200), tag=None):
            return Btn((bx, y, bw, 26), label, color, acol, tag)

        y = 8
        self.sl_brush = Slider((bx, y+16, bw, 20), "Brush", 5, 120, 50);  y+=50
        self.sl_pts   = Slider((bx, y+16, bw, 20), "Points", 50, 500, 250); y+=50

        # Tool buttons
        self.btn_draw   = b(y, "✏ Draw Road",  (35,50,35),(50,120,50),"draw");  y+=30
        self.btn_erase  = b(y, "⌫  Erase",     (50,35,35),(120,50,50),"erase"); y+=30
        self.btn_select = b(y, "↖ Select",     (35,35,55),(70,110,200),"select");y+=32
        self.tool_btns  = [self.btn_draw, self.btn_erase, self.btn_select]
        self._sync_tool()

        # Place buttons for fixed lines
        self.btn_sp  = b(y, "📍 Start Point", (55,35,65)); y+=28
        self.btn_sl  = b(y, "🟢 Start Line",  (30,65,30)); y+=28
        self.btn_ml  = b(y, "🔴 Mid Line",    (65,30,30)); y+=28
        self.btn_bl  = b(y, "🔵 Blue Line",   (25,35,75)); y+=30
        self.btn_nr  = b(y, "+ New Reward",   (30,70,55),(50,130,90)); y+=32

        # Selected-object actions
        self.btn_flip= b(y, "↔ Flip H/V",    (45,45,65)); y+=28
        self.btn_del = b(y, "🗑 Delete",       (80,30,30),(140,50,50)); y+=32

        # Bottom actions
        self.btn_load = b(y, "📂 Load Track",     (35,45,65),(60,90,140)); y+=28
        self.btn_gen  = b(y, "⚙ Generate Points",(30,65,40),(50,130,60)); y+=30
        self.btn_tog  = b(y, "👁 Show Points",   (35,35,55)); y+=28
        self.btn_save = b(y, "💾 Save Track",     (65,55,20),(130,110,30)); y+=28
        self.btn_clear= b(y, "🗑 Clear Canvas",   (70,25,25),(140,40,40))

        self.all_btns = [
            self.btn_draw, self.btn_erase, self.btn_select,
            self.btn_sp, self.btn_sl, self.btn_ml, self.btn_bl, self.btn_nr,
            self.btn_flip, self.btn_del,
            self.btn_load, self.btn_gen, self.btn_tog, self.btn_save, self.btn_clear,
        ]

    def _sync_tool(self):
        for b in self.tool_btns:
            b.active = (b.tag == self.tool)

    def _canvas_rect(self):
        sw, sh = self.screen.get_size()
        return pygame.Rect(0, 0, sw - PANEL_W, sh)

    def _canvas_pos(self, mp):
        return (mp[0], mp[1])

    def _on_canvas(self, mp):
        return self._canvas_rect().collidepoint(mp)

    def _resize_canvas(self):
        sw, sh = self.screen.get_size()
        cw, ch = sw - PANEL_W, sh
        if (cw, ch) != self.canvas.get_size():
            new = pygame.Surface((cw, ch))
            new.fill(GRASS)
            new.blit(self.canvas, (0,0))
            self.canvas = new

    def _stroke(self, p1, p2, r, color):
        dx,dy = p2[0]-p1[0], p2[1]-p1[1]
        dist = max(1, math.hypot(dx,dy))
        for i in range(int(dist/2)+1):
            t = i/max(1,int(dist/2))
            pygame.draw.circle(self.canvas, color,
                               (int(p1[0]+dx*t), int(p1[1]+dy*t)), r)

    def _find_obj_at(self, pos) -> TrackObject | None:
        for obj in reversed(self.objects):
            if obj.rect().collidepoint(pos):
                return obj
        return None

    def _selected(self) -> TrackObject | None:
        if self.selected_id is None: return None
        for obj in self.objects:
            if obj.id == self.selected_id: return obj
        return None

    def _add_obj(self, kind, x, y, w, h, color, label=None):
        obj = TrackObject(kind, x, y, w, h, color, "V", label)
        self.objects.append(obj)
        self.selected_id = obj.id
        return obj

    def _place_fixed(self, kind, color):
        sw, sh = self.screen.get_size()
        cw, ch = sw-PANEL_W, sh
        # Replace existing if same kind (except reward)
        self.objects = [o for o in self.objects if o.kind != kind]
        if kind == "start_point":
            self._add_obj(kind, cw//2, ch//2, 24, 24, color)
        else:
            self._add_obj(kind, cw//2-5, ch//2-40, 10, 80, color)

    def _generate_points(self):
        gray = pygame_to_gray(self.canvas)
        n = int(self.sl_pts.value)
        self.road_pts = gen_road_points(self.canvas, n)
        if not self.road_pts:
            print("[Editor] No road contour detected.")
        return self.road_pts

    def _save(self):
        root = tk.Tk(); root.withdraw()
        name = simpledialog.askstring("Track Name", "Name:", initialvalue=self.track_name)
        root.destroy()
        if not name: return
        self.track_name = name.strip().replace(" ","_")
        base = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(base,"images",f"{self.track_name}.png")
        pts_path = os.path.join(base,"road_points",f"road_points_{self.track_name}.txt")
        cfg_path = os.path.join(base,"road_points",f"lines_{self.track_name}.json")
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        os.makedirs(os.path.dirname(pts_path), exist_ok=True)
        # Save a game-compatible B/W image:
        # road (dark) → black(0) = driveable; grass (bright) → white(255) = wall
        arr = pygame.surfarray.array3d(self.canvas)   # (W,H,3) RGB
        # Convert to grayscale (W,H)
        gray_arr = (0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]).astype(np.uint8)
        # Use Otsu auto-threshold for robustness
        import cv2 as _cv
        _, bw_cv = _cv.threshold(gray_arr.T, 0, 255, _cv.THRESH_BINARY + _cv.THRESH_OTSU)
        # bw_cv is (H,W); convert back to (W,H,3) for surfarray
        bw_wh = bw_cv.T  # (W,H)
        bw_arr = np.stack([bw_wh, bw_wh, bw_wh], axis=2)  # (W,H,3)
        bw_surf = pygame.surfarray.make_surface(bw_arr)
        pygame.image.save(bw_surf, img_path)
        pts = self._generate_points()
        if pts:
            with open(pts_path,"w") as f:
                for px,py in pts: f.write(f"{px},{py}\n")
        cfg = {}
        for obj in self.objects:
            if obj.kind == "start_point":
                cfg["start_point"] = {"x": obj.x, "y": obj.y, "angle": obj.angle}
            else:
                cfg.setdefault("lines",[]).append({
                    "kind":obj.kind,"label":obj.label,
                    "rect":[obj.x,obj.y,obj.w,obj.h],
                    "orientation":obj.orientation,
                    "color":list(obj.color)
                })
        with open(cfg_path,"w") as f: json.dump(cfg, f, indent=2)

        sp = next((o for o in self.objects if o.kind=="start_point"), None)
        print("\n── Paste into train.py ──")
        print(f"TRACK_IMAGE      = '../images/{self.track_name}.png'")
        print(f"ROAD_POINTS_FILE = '../road_points/road_points_{self.track_name}.txt'")
        if sp:
            print(f"START_POS        = ({sp.x}, {sp.y})")
            print(f"# Start angle: {sp.angle}° (set car.angle = {sp.angle} in your Car init if needed)")
        print("─────────────────────────\n")

        root2=tk.Tk(); root2.withdraw()
        messagebox.showinfo("Saved!", f"Track '{self.track_name}' saved!\n\nImage:  {img_path}\nPoints: {pts_path}\nConfig: {cfg_path}")
        root2.destroy()

    def _load(self):
        """Load an existing track PNG + its lines JSON back into the editor."""
        root = tk.Tk(); root.withdraw()
        base = os.path.dirname(os.path.abspath(__file__))
        img_dir = os.path.join(base, "images")
        path = filedialog.askopenfilename(
            title="Open Track Image",
            initialdir=img_dir,
            filetypes=[("PNG images", "*.png")]
        )
        root.destroy()
        if not path: return

        stem = os.path.splitext(os.path.basename(path))[0]
        self.track_name = stem

        # Load PNG → display as green/dark in editor
        try:
            img = pygame.image.load(path).convert()
        except Exception as e:
            print(f"[Load] Could not load image: {e}"); return

        sw, sh = self.screen.get_size()
        cw = sw - PANEL_W
        new_canvas = pygame.Surface((cw, sh))
        new_canvas.fill(GRASS)
        # Scale image to fit canvas
        scaled = pygame.transform.scale(img, (cw, sh))
        # Convert B/W image to editor colors:
        # black (road) → ROAD color, white (wall) → GRASS color
        arr = pygame.surfarray.array3d(scaled)     # (W,H,3)
        gray_arr = (0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2])
        # Road = dark pixels (gray < 128)
        road_mask = gray_arr < 128                 # (W,H) bool
        # Build display array: road=dark, grass=green
        disp = np.empty(arr.shape, dtype=np.uint8)
        disp[:,:,0] = np.where(road_mask, ROAD[0], GRASS[0])
        disp[:,:,1] = np.where(road_mask, ROAD[1], GRASS[1])
        disp[:,:,2] = np.where(road_mask, ROAD[2], GRASS[2])
        self.canvas = pygame.surfarray.make_surface(disp)

        # Try to load matching JSON config
        cfg_path = os.path.join(base, "road_points", f"lines_{stem}.json")
        self.objects.clear()
        self.selected_id = None
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            sp = cfg.get("start_point")
            if sp:
                obj = TrackObject("start_point", sp["x"], sp["y"], 24, 24, (200,100,255))
                obj.angle = sp.get("angle", -90)
                self.objects.append(obj)
            for ld in cfg.get("lines", []):
                x,y,w,h = ld["rect"]
                col = tuple(ld["color"])
                obj = TrackObject(ld["kind"], x, y, w, h, col,
                                  ld.get("orientation","V"), ld.get("label"))
                self.objects.append(obj)
            print(f"[Load] Config loaded from {cfg_path}")
        else:
            print(f"[Load] No config found at {cfg_path}")
        print(f"[Load] Track '{stem}' loaded.")

    def run(self):
        running = True
        while running:
            sw, sh = self.screen.get_size()
            cw = sw - PANEL_W
            mp = pygame.mouse.get_pos()
            on_canvas = self._on_canvas(mp)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.VIDEORESIZE:
                    self._resize_canvas()
                    self._build_ui()

                # Sliders
                self.sl_brush.handle(event)
                self.sl_pts.handle(event)

                # Buttons
                if self.btn_draw.hit(event):   self.tool="draw";   self._sync_tool()
                if self.btn_erase.hit(event):  self.tool="erase";  self._sync_tool()
                if self.btn_select.hit(event): self.tool="select";  self._sync_tool()
                if self.btn_sp.hit(event):  self._place_fixed("start_point",(200,100,255)); self.tool="select"; self._sync_tool()
                if self.btn_sl.hit(event):  self._place_fixed("start_line",(50,220,50));    self.tool="select"; self._sync_tool()
                if self.btn_ml.hit(event):  self._place_fixed("mid_line",(220,50,50));      self.tool="select"; self._sync_tool()
                if self.btn_bl.hit(event):  self._place_fixed("blue_line",(50,100,255));    self.tool="select"; self._sync_tool()
                if self.btn_nr.hit(event):
                    col = next(REWARD_PALETTE)
                    n = sum(1 for o in self.objects if o.kind=="reward")+1
                    self._add_obj("reward", cw//2-5, sh//2-30, 10, 60, col, f"Reward {n}")
                    self.tool="select"; self._sync_tool()

                if self.btn_flip.hit(event):
                    sel = self._selected()
                    if sel and sel.kind != "start_point": sel.flip()

                if self.btn_del.hit(event):
                    self.objects = [o for o in self.objects if o.id != self.selected_id]
                    self.selected_id = None

                if self.btn_load.hit(event): self._load()
                if self.btn_gen.hit(event):
                    self._generate_points(); self.show_pts=True; self.btn_tog.active=True
                if self.btn_tog.hit(event):
                    self.show_pts = not self.show_pts; self.btn_tog.active = self.show_pts
                if self.btn_save.hit(event): self._save()
                if self.btn_clear.hit(event):
                    self.canvas.fill(GRASS); self.road_pts.clear()

                # Canvas mouse events
                if event.type == pygame.MOUSEBUTTONDOWN and event.button==1 and on_canvas:
                    if self.tool == "draw":
                        self.drawing=True; self.last_pos=mp
                    elif self.tool == "erase":
                        self.drawing=True; self.last_pos=mp
                    elif self.tool == "select":
                        obj = self._find_obj_at(mp)
                        if obj:
                            self.selected_id = obj.id
                            self.drag_obj = obj
                            if obj.kind == "start_point":
                                self.drag_off = (mp[0]-obj.x, mp[1]-obj.y)
                            else:
                                self.drag_off = (mp[0]-obj.x, mp[1]-obj.y)
                        else:
                            self.selected_id = None

                # Right-click to resize
                if event.type == pygame.MOUSEBUTTONDOWN and event.button==3 and on_canvas:
                    if self.tool == "select":
                        obj = self._find_obj_at(mp)
                        if obj and obj.kind != "start_point":
                            self.resize_obj = obj
                            self.resize_anchor = (obj.x, obj.y, obj.w, obj.h)
                            self.resize_mouse_start = mp

                if event.type == pygame.MOUSEMOTION:
                    if self.drawing and on_canvas and self.last_pos:
                        r = int(self.sl_brush.value)
                        col = ROAD if self.tool=="draw" else GRASS
                        self._stroke(self.last_pos, mp, r, col)
                        self.last_pos = mp

                    if self.drag_obj:
                        ox, oy = self.drag_off
                        if self.drag_obj.kind == "start_point":
                            self.drag_obj.x = mp[0]-ox
                            self.drag_obj.y = mp[1]-oy
                        else:
                            self.drag_obj.x = mp[0]-ox
                            self.drag_obj.y = mp[1]-oy

                    if self.resize_obj:
                        dx = mp[0]-self.resize_mouse_start[0]
                        dy = mp[1]-self.resize_mouse_start[1]
                        ax,ay,aw,ah = self.resize_anchor
                        self.resize_obj.w = max(4, aw+dx)
                        self.resize_obj.h = max(4, ah+dy)

                if event.type == pygame.MOUSEBUTTONUP:
                    if event.button==1:
                        self.drawing=False; self.last_pos=None; self.drag_obj=None
                    if event.button==3:
                        self.resize_obj=None

                # Keyboard shortcuts
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_DELETE:
                        self.objects=[o for o in self.objects if o.id!=self.selected_id]
                        self.selected_id=None
                    if event.key == pygame.K_f:
                        sel=self._selected()
                        if sel and sel.kind!="start_point": sel.flip()
                    if event.key == pygame.K_q:   # rotate start_point CCW
                        sel=self._selected()
                        if sel and sel.kind=="start_point": sel.angle = (sel.angle - 15) % 360
                    if event.key == pygame.K_e:   # rotate start_point CW
                        sel=self._selected()
                        if sel and sel.kind=="start_point": sel.angle = (sel.angle + 15) % 360
                        if not (sel and sel.kind=="start_point"): self.tool="erase"; self._sync_tool()
                    if event.key == pygame.K_d: self.tool="draw";   self._sync_tool()
                    if event.key == pygame.K_s and not (pygame.key.get_mods()&pygame.KMOD_CTRL):
                        self.tool="select"; self._sync_tool()

                # Scroll wheel to rotate selected start_point
                if event.type == pygame.MOUSEWHEEL:
                    sel=self._selected()
                    if sel and sel.kind=="start_point":
                        sel.angle = (sel.angle + event.y * 15) % 360

            # ── Render ─────────────────────────────────────────────────────────
            self.screen.fill(UI_BG)
            self.screen.blit(self.canvas, (0,0))

            # Brush cursor
            if on_canvas and self.tool in ("draw","erase"):
                r = int(self.sl_brush.value)
                col = ROAD if self.tool=="draw" else GRASS
                pygame.draw.circle(self.screen, col, mp, r, 2)

            # Draw objects
            sel = self._selected()
            for obj in self.objects:
                is_sel = (obj.id == self.selected_id)
                if obj.kind == "start_point":
                    sx, sy = obj.x, obj.y
                    ang_r = math.radians(obj.angle)
                    # Arrow direction
                    tip_x = sx + math.cos(ang_r) * 24
                    tip_y = sy + math.sin(ang_r) * 24
                    arr_l = sx + math.cos(ang_r) * 14
                    arr_r = sy + math.sin(ang_r) * 14
                    perp  = ang_r + math.pi/2
                    ax1 = tip_x - math.cos(ang_r)*8 + math.cos(perp)*5
                    ay1 = tip_y - math.sin(ang_r)*8 + math.sin(perp)*5
                    ax2 = tip_x - math.cos(ang_r)*8 - math.cos(perp)*5
                    ay2 = tip_y - math.sin(ang_r)*8 - math.sin(perp)*5
                    pygame.draw.circle(self.screen, obj.color, (sx, sy), 12)
                    pygame.draw.circle(self.screen, (255,255,255), (sx, sy), 12, 2)
                    pygame.draw.line(self.screen, (255,255,255), (sx,sy), (int(tip_x),int(tip_y)), 3)
                    pygame.draw.polygon(self.screen, (255,255,255),
                        [(int(tip_x),int(tip_y)),(int(ax1),int(ay1)),(int(ax2),int(ay2))])
                    t = self.fontXS.render(f"START {obj.angle}°", True, (255,255,255))
                    self.screen.blit(t, (sx+14, sy-8))
                    if is_sel:
                        pygame.draw.circle(self.screen, HLIGHT, (sx, sy), 16, 3)
                        hint = self.fontXS.render("Scroll/Q/E to rotate", True, HLIGHT)
                        self.screen.blit(hint, (sx+14, sy+6))
                else:
                    r = obj.rect()
                    s = pygame.Surface((r.w,r.h), pygame.SRCALPHA)
                    c = obj.color
                    s.fill((c[0],c[1],c[2],170))
                    self.screen.blit(s,(r.x,r.y))
                    pygame.draw.rect(self.screen, obj.color, r, 2)
                    lbl=self.fontXS.render(obj.label, True,(255,255,255))
                    self.screen.blit(lbl,(r.x+2,r.y-13))
                    if is_sel:
                        pygame.draw.rect(self.screen, HLIGHT, r.inflate(6,6), 3, border_radius=3)

            # Road points
            if self.show_pts:
                for px,py in self.road_pts:
                    pygame.draw.circle(self.screen,(255,60,60),(int(px),int(py)),3)

            # Canvas border
            pygame.draw.rect(self.screen, ACCENT, (0,0,cw,sh), 2)

            # ── Panel ──────────────────────────────────────────────────────────
            px = cw
            pygame.draw.rect(self.screen, UI_BG, (px,0,PANEL_W,sh))
            pygame.draw.line(self.screen, ACCENT,(px,0),(px,sh),2)

            # Reposition buttons
            bx=px+8; bw2=PANEL_W-16
            for btn in self.all_btns:
                btn.rect.x = bx
                btn.rect.w = bw2
            self.sl_brush.rect.x = bx; self.sl_brush.rect.w = bw2
            self.sl_pts.rect.x   = bx; self.sl_pts.rect.w   = bw2

            t=self.fontL.render("Track Editor", True,(180,210,255))
            self.screen.blit(t,(px+8,8))
            pygame.draw.line(self.screen,(50,55,80),(px+6,28),(px+PANEL_W-6,28))

            self.sl_brush.draw(self.screen, self.font)
            self.sl_pts.draw(self.screen, self.font)
            pygame.draw.line(self.screen,(50,55,80),(px+6,self.btn_draw.rect.y-4),(px+PANEL_W-6,self.btn_draw.rect.y-4))

            for btn in self.all_btns:
                btn.draw(self.screen, self.font)

            # Info for selected object
            sel=self._selected()
            if sel:
                iy = self.btn_del.rect.bottom+6
                info_lines=[
                    f"Selected: {sel.label}",
                    f"Pos: ({sel.x},{sel.y})",
                ]
                if sel.kind!="start_point":
                    info_lines+=[f"Size: {sel.w}x{sel.h}", f"Orient: {sel.orientation}"]
                for i,ln in enumerate(info_lines):
                    t=self.fontXS.render(ln,True,(160,170,200))
                    self.screen.blit(t,(px+8,iy+i*14))

            # Shortcuts hint
            hints=["D=Draw  E=Erase  S=Select","F=Flip  Del=Delete",
                   "LMB drag=move  RMB drag=resize"]
            hy=sh-len(hints)*14-6
            pygame.draw.line(self.screen,(50,55,80),(px+6,hy-4),(px+PANEL_W-6,hy-4))
            for i,h in enumerate(hints):
                t=self.fontXS.render(h,True,(110,115,145))
                self.screen.blit(t,(px+8,hy+i*14))

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


if __name__=="__main__":
    Editor().run()
