import math

def find_closest_point(car_pos, road_points):
    """
    Arabanın konumuna en yakın yol noktasını bulur.
    """
    closest_point = None
    min_distance = float('inf')
    for point in road_points:
        distance = math.sqrt((car_pos[0] - point[0]) ** 2 + (car_pos[1] - point[1]) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_point = point
    return closest_point


def calculate_distance_from_start(road_points, closest_point):
    """
    En yakın noktayı kullanarak başlangıçtan itibaren mesafeyi hesaplar.
    """
    distance = 0
    for i in range(len(road_points) - 1):
        # segment_distance = math.sqrt(
        #     (road_points[i+1][0] - road_points[i][0]) ** 2 +
        #     (road_points[i+1][1] - road_points[i][1]) ** 2
        # )
        distance += 1
        if road_points[i] == closest_point:
            break
    return distance


def load_road_points(filepath):
    """
    Yol noktalarını bir dosyadan yükler (örneğin, yolun bir haritası).
    """
    road_points = []
    with open(filepath, "r") as file:
        for line in file:
            x, y = map(int, line.strip().split(","))
            road_points.append((x, y))
    return road_points
