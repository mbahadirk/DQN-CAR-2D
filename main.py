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


road_img = pygame.image.load(ROAD_IMAGE_PATH).convert()
# grass_img = pygame.image.load("images/grass.jpg").convert()
# track = pygame.image.load("images/track.png").convert()

SPAWN_POINT = (150, 100, 90)



class Game:
    def __init__(self):
        self.car = Car(CAR_IMAGE_PATH, scale_factor=0.1, start_x=SPAWN_POINT[0], start_y=SPAWN_POINT[1], start_angle=SPAWN_POINT[2])
        # Harita maskesi: Yolu gri renk ile temsil eden bir maske
        self.road_mask = pygame.mask.from_threshold(road_img, (255, 255, 255), (150, 150, 150))

    def check_collision(self, rayName):
        # Dinamik olarak rayName parametresine göre ışını al
        ray = getattr(self.car, rayName)

        # Işın yüzeyini maske olarak oluştur
        line_mask = pygame.mask.from_surface(ray.rotated_surface)

        # Çarpışma kontrolü
        offset = (int(ray.rotated_rect.left), int(ray.rotated_rect.top))
        overlap_point = self.road_mask.overlap(line_mask, offset)
        ray.point = (0, 0)  # Çarpışma olmadığında başlangıç noktasını sıfırla
        if overlap_point:
            print("Çarpışma algılandı!", overlap_point)  # Çarpışma konumunu yazdır
            ray.point = overlap_point  # Çarpışma noktasını güncelle
            return True
        return False



    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            self.car.update(keys)


            screen.fill(GRAY)
            screen.blit(road_img, (0, 0))  # Haritayı çiz
            # pygame.draw.rect(screen, (0, 0, 255), (250, 250, 50, 50))

            for ray_name, ray_info in self.car.rays.items():
                self.check_collision(f"{ray_name}Ray")

            self.check_collision("topLeftRay")
            self.check_collision("topRightRay")
            self.check_collision("backRightRay")
            self.check_collision("backLeftRay")

            self.car.draw(screen)

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
        sys.exit()



if __name__ == "__main__":
    game = Game()
    game.run()
