import math

import pygame


class TrackLines:
    def __init__(self):
        self.start_line = (15, 270, 90, 10)
        self.start_line_rect = pygame.Rect(*self.start_line)

        self.mid_line = (860, 400, 90, 10)
        self.mid_line_rect = pygame.Rect(*self.mid_line)

        self.blue_line = (15, 310, 90, 10)
        self.blue_line_rect = pygame.Rect(*self.blue_line)


def handle_collision_with_lines(car, start_line_rect, mid_line_rect, blue_line_rect,pass_startline):
    """Start line, mid line ve blue line ile çarpışma durumunu ele alır."""
    car_rect = car.car_image.get_rect(center=(car.x, car.y))

    if not pass_startline:
        if car_rect.colliderect(start_line_rect):
            # Arabayı ters yöne geri it (hızın ve açının tersiyle)
            car.x -= car.speed * math.cos(math.radians(car.angle))
            car.y -= car.speed * math.sin(math.radians(car.angle))
            # print("Start line ile çarpışma! Geçiş engellendi.")

    if car_rect.colliderect(mid_line_rect):
        pass_startline = True
        # print("startLine is passable")
    if pass_startline:
        if car_rect.colliderect(blue_line_rect):
            # print("startline passed. now its not passable")
            pass_startline = False
            return pass_startline, True
    return pass_startline, False
