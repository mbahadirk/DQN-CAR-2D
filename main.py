import pygame
from car import Car
from utilities.threshold import apply_threshold
from CFG import ROAD_IMAGE_PATH, CAR_IMAGE_PATH, WINDOW_SIZE, P_YELLOW, GRAY
from ray_list import create_rays
from utilities.road_utils import find_closest_point, calculate_distance_from_start, load_road_points
from utilities.reorder_road_points import reorder_road_points

from track_lines import TrackLines, handle_collision_with_lines
track_lines = TrackLines()

pygame.init()
window = pygame.display.set_mode(WINDOW_SIZE)
clock = pygame.time.Clock()
rays = create_rays(window)

# Yolu yükle ve eşik uygulayın
road_image = pygame.image.load(ROAD_IMAGE_PATH).convert_alpha()
threshold_image = apply_threshold(road_image)
threshold_mask = pygame.mask.from_surface(threshold_image)

# Yansıma maskelerini oluştur
mask_fx = pygame.mask.from_surface(pygame.transform.flip(threshold_image, True, False))
mask_fy = pygame.mask.from_surface(pygame.transform.flip(threshold_image, False, True))
mask_fx_fy = pygame.mask.from_surface(pygame.transform.flip(threshold_image, True, True))
flipped_masks = [[threshold_mask, mask_fy], [mask_fx, mask_fx_fy]]
beam_surface = pygame.Surface((WINDOW_SIZE[0], WINDOW_SIZE[1]), pygame.SRCALPHA)

start_pos = (60, 300)
car = Car(CAR_IMAGE_PATH, scale_factor=0.1,start_x=start_pos[0], start_y=start_pos[1], start_angle=90)
road_points = load_road_points("road_points.txt")
road_points = reorder_road_points(start_pos, road_points)


def reset_game():
    """Oyunun başlangıç durumuna dönmesini sağlar."""
    global car, distance_from_start, pass_startline,score  # Gerekli değişkenleri sıfırla
    car.x, car.y = start_pos[0], start_pos[1]
    car.angle = 90
    car.speed = 0  # Hız sıfırlama
    distance_from_start = 0  # Mesafeyi sıfırla
    score = 0
    pass_startline = False
    print("Oyun yeniden başlatıldı.")


run = True
distance_from_start = 0
pass_startline = False
score = 0

while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    keys = pygame.key.get_pressed()
    car.update(keys)

    window.fill((0, 0, 0))
    window.blit(threshold_image, threshold_image.get_rect(center=window.get_rect().center))

    car_pos = (car.x, car.y)
    for ray in rays:
        ray.draw_beam(car_pos, car.angle, flipped_masks, beam_surface, threshold_mask)

    car_mask = pygame.mask.from_surface(car.car_image)
    car_offset = (int(car.x - car.rect.width / 2), int(car.y - car.rect.height / 2))
    collision = threshold_mask.overlap(car_mask, car_offset)

    if collision:
        print("Collision detected with the road!")
        reset_game()

    # finding lap distance
    closest_point = find_closest_point(car_pos, road_points)
    distance_from_start = calculate_distance_from_start(road_points, closest_point) + score

    # draw the road points
    for point in road_points:
        pygame.draw.circle(window, P_YELLOW, point, 2)

    # draw distance
    font = pygame.font.SysFont(None, 24)
    text = font.render(f"Total Score: {int(distance_from_start)}", True, GRAY)
    window.blit(text, (10, 10))

    # collision with lines and a flag
    pass_startline, lap_flag = handle_collision_with_lines(car, track_lines.start_line_rect, track_lines.mid_line_rect, track_lines.blue_line_rect, pass_startline)

    if lap_flag:
        score += 3300

    # draw the start line
    pygame.draw.rect(window, (0, 255, 0), track_lines.start_line)
    pygame.draw.rect(window, (255, 0, 0), track_lines.mid_line)
    pygame.draw.rect(window, (0, 0, 255), track_lines.blue_line_rect)

    car.draw(window)
    pygame.display.update()

    clock.tick(60)

pygame.quit()
