import math

import pygame


class TrackLines:
    def __init__(self):
        # its for game start with curve
        # self.start_line = (20, 270, 130, 10)
        # self.start_line_rect = pygame.Rect(*self.start_line)
        #
        # self.mid_line = (830, 400, 130, 10)
        # self.mid_line_rect = pygame.Rect(*self.mid_line)
        #
        # self.blue_line = (20, 300, 130, 10)
        # self.blue_line_rect = pygame.Rect(*self.blue_line)
        #
        # self.reward_line_1 = (200, 500, 10, 30)
        # self.reward_line_1_rect = pygame.Rect(*self.reward_line_1)
        #
        # self.reward_line_2 = (600, 500, 10, 30)
        # self.reward_line_2_rect = pygame.Rect(*self.reward_line_2)
        #
        # self.reward_line_3 = (900, 400, 40, 10)
        # self.reward_line_3_rect = pygame.Rect(*self.reward_line_3)
        #
        # self.reward_line_4 = (900, 250, 40, 10)
        # self.reward_line_4_rect = pygame.Rect(*self.reward_line_4)


        self.start_line = (240, 450, 10, 120)
        self.start_line_rect = pygame.Rect(*self.start_line)

        self.mid_line = (500, 15, 10, 120)
        self.mid_line_rect = pygame.Rect(*self.mid_line)

        self.blue_line = (265, 450, 10, 120)
        self.blue_line_rect = pygame.Rect(*self.blue_line)


        self.reward_line_1 = (600, 500, 10, 40)
        self.reward_line_1_rect = pygame.Rect(*self.reward_line_1)

        self.reward_line_2 = (900, 250, 30, 10)
        self.reward_line_2_rect = pygame.Rect(*self.reward_line_2)

        self.reward_line_3 = (900, 400, 40, 10)
        self.reward_line_3_rect = pygame.Rect(*self.reward_line_3)

        self.reward_line_4 = (200, 80, 10, 30)
        self.reward_line_4_rect = pygame.Rect(*self.reward_line_4)


        self.bottom_reward_line = (100, 500, 800, 10)
        self.bottom_reward_line_rect = pygame.Rect(*self.bottom_reward_line)

        self.top_reward_line = (100, 60, 800, 10)
        self.top_reward_line_rect = pygame.Rect(*self.top_reward_line)

        self.right_reward_line = (900, 80, 10, 430)
        self.right_reward_line_rect = pygame.Rect(*self.right_reward_line)

        self.left_reward_line = (80, 80, 10, 430)
        self.left_reward_line_rect = pygame.Rect(*self.left_reward_line)



def handle_collision_with_lines(car, start_line_rect, mid_line_rect, blue_line_rect,
                                pass_startline, block_start=True):
    """Start line, mid line ve blue line ile çarpışma durumunu ele alır.

    block_start=False ile çağrılırsa start_line fiziksel engel oluşturmaz,
    sadece tur tespiti yapılır. Eğitim ortamları için kullanılır.
    """
    car_rect = car.car_image.get_rect(center=(car.x, car.y))

    if not pass_startline and block_start:
        if car_rect.colliderect(start_line_rect):
            # Arabayı ters yöne geri it (hızın ve açının tersiyle)
            car.x -= car.speed * math.cos(math.radians(car.angle))
            car.y -= car.speed * math.sin(math.radians(car.angle))

    if car_rect.colliderect(mid_line_rect):
        pass_startline = True
        # print("startLine is passable")
    if pass_startline:
        if car_rect.colliderect(blue_line_rect):
            # print("startline passed. now its not passable")
            pass_startline = False
            return pass_startline, True

    return pass_startline, False


