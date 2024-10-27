import pygame
import math
import sys
from car import Car
from CFG import WIDTH, HEIGHT, WHITE, CAR_IMAGE_PATH,ROAD_IMAGE_PATH
import pygame.surfarray

# Pygame'i başlat
pygame.init()

# Ekran boyutları
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gerçekçi Araba Sürüşü")

# Saat (FPS için)
clock = pygame.time.Clock()


road_img = pygame.image.load(ROAD_IMAGE_PATH).convert()
grass_img = pygame.image.load("images/grass.jpg").convert()
track = pygame.image.load("images/track.png").convert()

SPAWN_POINT = (150, 100, 90)


class Game:
    def __init__(self):
        self.car = Car(CAR_IMAGE_PATH, scale_factor=0.1, start_x=SPAWN_POINT[0], start_y=SPAWN_POINT[1], start_angle=SPAWN_POINT[2])
        # Harita maskesi: Yolu gri renk ile temsil eden bir maske
        self.road_mask = pygame.mask.from_threshold(road_img, (255, 255, 255), (150, 150, 150))


    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            self.car.update(keys)

            # Yol içinde olup olmadığını kontrol et
            if self.is_on_road():
                print("Uyarı: Yoldan çıktın!")
                # self.car.reset()
            # screen.fill(WHITE)
            # screen.blit(road_img, (0, 0))  # Haritayı çiz


            self.car.draw(screen)

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
        sys.exit()

    def is_on_road(self):
        self.car.collide(self.road_mask)
        return False


if __name__ == "__main__":
    game = Game()
    game.run()
