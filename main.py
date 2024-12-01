import pygame
import math
import sys
from car import Car
from CFG import WIDTH, HEIGHT, WHITE, CAR_IMAGE_PATH,ROAD_IMAGE_PATH,GRAY
import pygame.surfarray


# Pygame'i başlat
pygame.init()

# Ekran boyutları
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gerçekçi Araba Sürüşü")

# Saat (FPS için)
clock = pygame.time.Clock()


road_img = pygame.image.load("images/track2.png").convert()
# grass_img = pygame.image.load("images/grass.jpg").convert()
# track = pygame.image.load("images/track.png").convert()

SPAWN_POINT = (150, 100, 90)



# main.py

from walls import getWalls

class Game:
    def __init__(self):
        self.car = Car(CAR_IMAGE_PATH, scale_factor=0.1, start_x=SPAWN_POINT[0], start_y=SPAWN_POINT[1], start_angle=SPAWN_POINT[2])

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            self.car.update(keys)

            screen.fill(GRAY)
            screen.blit(road_img, (0, 0))

            self.car.draw(screen)  # Araba ve ışınları çiz

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
        sys.exit()



if __name__ == "__main__":
    game = Game()
    game.run()


