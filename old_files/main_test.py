import math
import pygame

from CFG import ROAD_IMAGE_PATH
from car import Car

pygame.init()
window = pygame.display.set_mode((1000, 600))
clock = pygame.time.Clock()


class Ray:
    def __init__(self, angle, surface, name):
        self.angle = angle
        self.surface = surface
        self.name = name

    def draw_beam(self, pos, car_angle):
        # Arabaya göre dönüş açısını ekleyin
        adjusted_angle = self.angle + car_angle  # Arabaya bağlı olarak açıya etki et

        c = math.cos(math.radians(adjusted_angle))
        s = math.sin(math.radians(adjusted_angle))

        flip_x = c < 0
        flip_y = s < 0
        filpped_mask = flipped_masks[flip_x][flip_y]

        # beam son noktası
        x_dest = self.surface.get_width() * abs(c)
        y_dest = self.surface.get_height() * abs(s)

        beam_surface.fill((0, 0, 0, 0))

        # Tek bir ışın çiz
        pygame.draw.line(beam_surface, (0, 0, 255), (0, 0), (x_dest, y_dest))
        beam_mask = pygame.mask.from_surface(beam_surface)

        # "global mask" ve mevcut beam maskesi arasında overlap kontrolü
        offset_x = self.surface.get_width() - 1 - pos[0] if flip_x else pos[0]
        offset_y = self.surface.get_height() - 1 - pos[1] if flip_y else pos[1]
        hit = filpped_mask.overlap(beam_mask, (offset_x, offset_y))
        if hit is not None and (hit[0] != pos[0] or hit[1] != pos[1]):
            hx = self.surface.get_width() - 1 - hit[0] if flip_x else hit[0]
            hy = self.surface.get_height() - 1 - hit[1] if flip_y else hit[1]
            hit_pos = (hx, hy)

            pygame.draw.line(self.surface, (0, 0, 255), pos, hit_pos)
            pygame.draw.circle(self.surface, (0, 255, 0), hit_pos, 3)
            self.distance = math.sqrt((hit_pos[0] - pos[0]) ** 2 + (hit_pos[1] - pos[1]) ** 2)




def apply_threshold(image, threshold=200):
    """Uygulamak için eşik değeri olan beyaz maske oluşturma."""
    width, height = image.get_size()
    threshold_surface = pygame.Surface((width, height), pygame.SRCALPHA)

    # İmajı piksellere dönüştür ve threshold işlemi yap
    for x in range(width):
        for y in range(height):
            r, g, b, a = image.get_at((x, y))
            # Beyaz olan pikseller için eşik kontrolü
            if r >= threshold and g >= threshold and b >= threshold:
                threshold_surface.set_at((x, y), (255, 255, 255, 255))  # Beyaz
            else:
                threshold_surface.set_at((x, y), (0, 0, 0, 0))  # Şeffaf

    return threshold_surface




# Yeni imajı yükle
road_image = pygame.image.load(ROAD_IMAGE_PATH).convert_alpha()  # İmajı yükle ve alpha kanalı koru

# Threshold ile maske oluştur
threshold_image = apply_threshold(road_image, threshold=200)
threshold_mask = pygame.mask.from_surface(threshold_image)

# Yansıma maskeleri oluştur
mask_fx = pygame.mask.from_surface(pygame.transform.flip(threshold_image, True, False))
mask_fy = pygame.mask.from_surface(pygame.transform.flip(threshold_image, False, True))
mask_fx_fy = pygame.mask.from_surface(pygame.transform.flip(threshold_image, True, True))
flipped_masks = [[threshold_mask, mask_fy], [mask_fx, mask_fx_fy]]
beam_surface = pygame.Surface(window.get_rect().center, pygame.SRCALPHA)


car = Car('../images/car.png', scale_factor=0.1)

rays = []
frontMidRay = Ray(0, window, name='frontMidRay')
rays.append(frontMidRay)

frontRightRay = Ray(30, window, name='frontRightRay')
rays.append(frontRightRay)

frontLeftRay = Ray(-30, window, name='frontLeftRay')
rays.append(frontLeftRay)

midLeftRay = Ray(-90, window, name='midLeftRay')
rays.append(midLeftRay)

midRightRay = Ray(90, window, name='midRightRay')
rays.append(midRightRay)

backMidRay = Ray(180, window, name='backMidRay')
rays.append(backMidRay)

backLeftRay = Ray(-150, window, name='backLeftRay')
rays.append(backLeftRay)

backRightRay = Ray(150, window, name='backRightRay')
rays.append(backRightRay)



run = True
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
        ray.draw_beam(car_pos, car.angle)
        # print(f"{ray.name} distance : {ray.distance}")


    # check collision between car's rectangle and the threshold mask
    car_mask = pygame.mask.from_surface(car.car_image)
    car_offset = (int(car.x - car.rect.width / 2), int(car.y - car.rect.height / 2))
    collision = threshold_mask.overlap(car_mask, car_offset)

    if collision:
        print("Collision detected with the road!")

    car.draw(window)

    pygame.display.update()
    clock.tick(60)

pygame.quit()
exit()
