import math
import os
import json

import pygame


class TrackLines:
    """
    Track çizgileri ve ödül alanları.

    Kullanım:
      track_lines = TrackLines.from_track_image(TRACK_IMAGE)
      # veya doğrudan JSON ile:
      track_lines = TrackLines.from_json("road_points/lines_mytrack.json")
      # veya varsayılan değerlerle:
      track_lines = TrackLines()

    Reward line'larına dinamik olarak erişmek için:
      track_lines.reward_rects  -> list[pygame.Rect]  (tüm reward'lar)
    Geriye dönük uyumluluk için 4 sabit attr da var:
      track_lines.reward_line_1_rect, ...reward_line_4_rect
    """

    # ── Constructor ───────────────────────────────────────────────────────────
    def __init__(self):
        """Varsayılan değerlerle başlatır (fallback)."""
        self.start_line      = (240, 450, 10, 120)
        self.start_line_rect = pygame.Rect(*self.start_line)

        self.mid_line        = (500, 15, 10, 120)
        self.mid_line_rect   = pygame.Rect(*self.mid_line)

        self.blue_line       = (265, 450, 10, 120)
        self.blue_line_rect  = pygame.Rect(*self.blue_line)

        self.reward_rects: list[pygame.Rect] = [
            pygame.Rect(600, 500, 10, 40),
            pygame.Rect(900, 250, 30, 10),
            pygame.Rect(900, 400, 40, 10),
            pygame.Rect(200,  80, 10, 30),
        ]
        self._sync_reward_attrs()

    def _sync_reward_attrs(self):
        """reward_rects listesini reward_line_N_rect attr'larıyla senkronize eder."""
        for i in range(1, 5):
            attr = f"reward_line_{i}_rect"
            if i <= len(self.reward_rects):
                setattr(self, attr, self.reward_rects[i - 1])
            else:
                # Eksik reward'lar için ekranda olmayan bir rect
                setattr(self, attr, pygame.Rect(-9999, -9999, 1, 1))

        # Eski kod uyumluluğu: düz tuple erişimleri
        for i in range(1, 5):
            attr = f"reward_line_{i}"
            rect = getattr(self, f"reward_line_{i}_rect")
            setattr(self, attr, (rect.x, rect.y, rect.w, rect.h))

    # ── Alternate constructors ────────────────────────────────────────────────
    @classmethod
    def from_json(cls, json_path: str) -> "TrackLines":
        """JSON konfigürasyon dosyasından yükler."""
        tl = cls.__new__(cls)
        # Fallback'ler
        tl.start_line      = (240, 450, 10, 120)
        tl.start_line_rect = pygame.Rect(*tl.start_line)
        tl.mid_line        = (500, 15, 10, 120)
        tl.mid_line_rect   = pygame.Rect(*tl.mid_line)
        tl.blue_line       = (265, 450, 10, 120)
        tl.blue_line_rect  = pygame.Rect(*tl.blue_line)
        tl.reward_rects    = []

        if not os.path.exists(json_path):
            print(f"[TrackLines] JSON bulunamadı: {json_path} — varsayılanlar kullanılıyor.")
            tl.reward_rects = [
                pygame.Rect(600, 500, 10, 40),
                pygame.Rect(900, 250, 30, 10),
                pygame.Rect(900, 400, 40, 10),
                pygame.Rect(200,  80, 10, 30),
            ]
            tl._sync_reward_attrs()
            return tl

        with open(json_path, encoding="utf-8") as f:
            cfg = json.load(f)

        for line_data in cfg.get("lines", []):
            x, y, w, h = line_data["rect"]
            kind = line_data["kind"]
            rect = pygame.Rect(x, y, w, h)
            tup  = (x, y, w, h)

            if kind == "start_line":
                tl.start_line      = tup
                tl.start_line_rect = rect
            elif kind == "mid_line":
                tl.mid_line        = tup
                tl.mid_line_rect   = rect
            elif kind == "blue_line":
                tl.blue_line       = tup
                tl.blue_line_rect  = rect
            elif kind == "reward":
                tl.reward_rects.append(rect)

        tl._sync_reward_attrs()
        print(f"[TrackLines] {json_path} -> {len(tl.reward_rects)} reward, "
              f"start_line={tl.start_line}, mid={tl.mid_line}")
        return tl

    @classmethod
    def from_track_image(cls, track_image_path: str, caller_file: str = "") -> "TrackLines":
        """
        Sadece track image yolundan yükler.
        caller_file: __file__ → yol train.py'nin konumuna göre çözülür.
        Örn: TrackLines.from_track_image('../images/hard_track.png', __file__)
             → '../road_points/lines_hard_track.json' okunur.
        """
        abs_path  = _abs_image_path(track_image_path, caller_file)
        stem      = os.path.splitext(os.path.basename(abs_path))[0]
        rp_dir    = os.path.normpath(os.path.join(os.path.dirname(abs_path), "..", "road_points"))
        json_path = os.path.join(rp_dir, f"lines_{stem}.json")
        return cls.from_json(json_path)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _abs_image_path(track_image_path: str, caller_file: str = "") -> str:
    """
    Relative yolu, çağıran script'in konumuna göre mutlak yola çevirir.
    caller_file: __file__ (örn. 'Genetic Algorithm/train.py')
    """
    if os.path.isabs(track_image_path):
        return track_image_path
    base_dir = os.path.dirname(os.path.abspath(caller_file)) if caller_file else os.getcwd()
    return os.path.normpath(os.path.join(base_dir, track_image_path))


def road_points_path_from_image(track_image_path: str, caller_file: str = "") -> str:
    """
    Örn: road_points_path_from_image('../images/hard_track.png', __file__)
         → '<project>/road_points/road_points_hard_track.txt'
    """
    abs_path = _abs_image_path(track_image_path, caller_file)
    stem     = os.path.splitext(os.path.basename(abs_path))[0]
    rp_dir   = os.path.normpath(os.path.join(os.path.dirname(abs_path), "..", "road_points"))
    return os.path.join(rp_dir, f"road_points_{stem}.txt")


# ── Start position from JSON ──────────────────────────────────────────────────
def _load_start_point_cfg(track_image_path: str, caller_file: str = "") -> dict:
    """JSON'dan start_point bloğunu döner; bulunamazsa boş dict."""
    abs_path  = _abs_image_path(track_image_path, caller_file)
    stem      = os.path.splitext(os.path.basename(abs_path))[0]
    rp_dir    = os.path.normpath(os.path.join(os.path.dirname(abs_path), "..", "road_points"))
    json_path = os.path.join(rp_dir, f"lines_{stem}.json")
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            cfg = json.load(f)
        sp = cfg.get("start_point")
        if isinstance(sp, dict):
            return sp
    return {}


def start_pos_from_image(track_image_path: str, caller_file: str = "",
                          default=(280, 530)) -> tuple[int, int]:
    """Track image yolundan start_point koordinatını döner; JSON yoksa default."""
    sp = _load_start_point_cfg(track_image_path, caller_file)
    if sp:
        return (int(sp["x"]), int(sp["y"]))
    return default


def start_angle_from_image(track_image_path: str, caller_file: str = "",
                            default: float = 0.0) -> float:
    """Track image yolundan start_point açısını döner; JSON yoksa default."""
    sp = _load_start_point_cfg(track_image_path, caller_file)
    return float(sp.get("angle", default)) if sp else default


# ── Collision helper (unchanged) ──────────────────────────────────────────────
def handle_collision_with_lines(car, start_line_rect, mid_line_rect, blue_line_rect,
                                pass_startline, block_start=True):
    """Start line, mid line ve blue line ile çarpışma durumunu ele alır."""
    car_rect = car.car_image.get_rect(center=(car.x, car.y))

    if not pass_startline and block_start:
        if car_rect.colliderect(start_line_rect):
            car.x -= car.speed * math.cos(math.radians(car.angle))
            car.y -= car.speed * math.sin(math.radians(car.angle))

    if car_rect.colliderect(mid_line_rect):
        pass_startline = True

    if pass_startline:
        if car_rect.colliderect(blue_line_rect):
            pass_startline = False
            return pass_startline, True

    return pass_startline, False
