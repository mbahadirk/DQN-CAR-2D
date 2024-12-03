import math

def reorder_road_points(start_pos, road_points):
    # Başlangıç noktasına en yakın yolu bul
    def calculate_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    closest_index = min(range(len(road_points)), key=lambda i: calculate_distance(start_pos, road_points[i]))
    # En yakın noktayı bulduktan sonra, öncesindeki noktaları sona ekleyelim
    reordered_points = road_points[closest_index:] + road_points[:closest_index]
    return reordered_points
