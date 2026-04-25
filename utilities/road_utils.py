import numpy as np

# Numpy array cache to avoid re-converting on every call
_np_cache: dict = {}


def _to_np(road_points):
    key = id(road_points)
    if key not in _np_cache:
        _np_cache[key] = np.array(road_points, dtype=np.float32)
    return _np_cache[key]


def find_closest_point(car_pos, road_points):
    pts = _to_np(road_points)
    diff = pts - np.array(car_pos, dtype=np.float32)
    idx = int(np.argmin((diff * diff).sum(axis=1)))
    return road_points[idx]


def calculate_distance_from_start(road_points, closest_point):
    pts = _to_np(road_points)
    cp = np.array(closest_point, dtype=np.float32)
    diff = pts - cp
    idx = int(np.argmin((diff * diff).sum(axis=1)))
    return idx


def load_road_points(filepath):
    road_points = []
    with open(filepath, "r") as file:
        for line in file:
            x, y = map(int, line.strip().split(","))
            road_points.append((x, y))
    return road_points
