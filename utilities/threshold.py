import pygame
import numpy as np

try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def apply_threshold(image, threshold=200, wall_color=(255, 255, 255)):
    """
    Converts a track image into a wall-mask surface (SRCALPHA).

    Wall pixels  → (255, 255, 255, 255)  — white, opaque  (collision)
    Road pixels  → (0,   0,   0,   0)    — transparent    (driveable)

    Detection strategy (automatic, works for all track formats):
      • If cv2 is available: uses Otsu auto-threshold on the grayscale image.
        This correctly separates black road from white walls AND
        dark road from green/colored grass walls (track-editor format).
      • Fallback: original brightness check (r,g,b >= threshold).
    """
    width, height = image.get_size()
    threshold_surface = pygame.Surface((width, height), pygame.SRCALPHA)

    if _HAS_CV2:
        # ── Fast numpy/cv2 path ──────────────────────────────────────────────
        arr = pygame.surfarray.array3d(image)          # (W, H, 3) uint8 RGB
        # cv2 expects (H, W) so transpose
        arr_hw = np.transpose(arr, (1, 0, 2))          # (H, W, 3)
        gray = _cv2.cvtColor(arr_hw, _cv2.COLOR_RGB2GRAY)  # (H, W)

        # Otsu finds the best cut between road and wall automatically
        ret, bw = _cv2.threshold(gray, 0, 255,
                                  _cv2.THRESH_BINARY + _cv2.THRESH_OTSU)
        # bw: 0 = dark/road, 255 = bright/wall  (H, W)

        # Back to (W, H)
        wall_mask = bw.T > 127                         # (W, H) bool

        # Write RGB for wall pixels only (road stays 0,0,0)
        rgb_arr = pygame.surfarray.pixels3d(threshold_surface)
        rgb_arr[wall_mask] = list(wall_color)
        del rgb_arr  # unlock before touching alpha

        # Write alpha: wall=255 (opaque), road=0 (transparent)
        alpha_arr = pygame.surfarray.pixels_alpha(threshold_surface)
        alpha_arr[wall_mask] = 255
        alpha_arr[~wall_mask] = 0
        del alpha_arr

    else:
        # ── Fallback: original per-pixel loop ───────────────────────────────
        for x in range(width):
            for y in range(height):
                r, g, b, a = image.get_at((x, y))
                if r >= threshold and g >= threshold and b >= threshold:
                    threshold_surface.set_at((x, y), (*wall_color, 255))
                else:
                    threshold_surface.set_at((x, y), (0, 0, 0, 0))

    return threshold_surface
