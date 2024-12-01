import pygame
import math
from ray import Ray
from walls import getWalls

class Car:
    def __init__(self, image_path, scale_factor=0.1, start_x=0, start_y=0, start_angle=0):
        self.image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(self.image, (int(self.image.get_width() * scale_factor),
                                                         int(self.image.get_height() * scale_factor)))
        self.rotated_image = self.image
        self.rect = self.image.get_rect(center=(start_x, start_y))
        self.angle = start_angle
        self.speed = 0

        # Ray (ışın) tanımlamaları: sağ, sol, ön ve arka
        self.rays = [
            Ray(self.rect.centerx, self.rect.centery, self.angle - 45),  # Sol ön çapraz
            Ray(self.rect.centerx, self.rect.centery, self.angle + 45),  # Sağ ön çapraz
            Ray(self.rect.centerx, self.rect.centery, self.angle + 135),  # Sağ arka çapraz
            Ray(self.rect.centerx, self.rect.centery, self.angle - 135)  # Sol arka çapraz
        ]

    def update(self, keys):
        """
        Arabanın hareketini ve ışınların güncellenmesini sağla
        """
        # Hareket hızını belirle
        speed = 5

        # Yön tuşlarına göre hareket
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.rect.y -= speed  # Yukarı hareket
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.rect.y += speed  # Aşağı hareket
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.rect.x -= speed  # Sola hareket
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.rect.x += speed  # Sağa hareket

        # Arabanın açısını güncelle (isteğe bağlı, döndürme için)
        if keys[pygame.K_LEFT]:
            self.angle -= 5
        elif keys[pygame.K_RIGHT]:
            self.angle += 5
        self.rotated_image = pygame.transform.rotate(self.image, -self.angle)

        # Işınların güncellenmesi (sabit kalacak)
        for ray in self.rays:
            ray.start_x = self.rect.centerx
            ray.start_y = self.rect.centery
            ray.angle = self.angle + ray.angle  # Arabanın açısına göre ayarlanabilir
            ray.cast(getWalls())  # Çarpışma tespiti
            """
            Arabanın hareketini ve ışınların güncellenmesini sağla
            """
            if keys[pygame.K_UP]:
                self.speed = 5
            elif keys[pygame.K_DOWN]:
                self.speed = -5
            else:
                self.speed = 0

            if keys[pygame.K_LEFT]:
                self.angle += 5
            elif keys[pygame.K_RIGHT]:
                self.angle -= 5

            # Arabanın pozisyonunu ve yönünü güncelle
            self.rect.x += self.speed * math.cos(math.radians(self.angle))
            self.rect.y -= self.speed * math.sin(math.radians(self.angle))
            self.rect = self.rotated_image.get_rect(center=self.rect.center)

            # Işınların güncellenmesi
            for ray in self.rays:
                ray.start_x = self.rect.centerx
                ray.start_y = self.rect.centery
                ray.angle = self.angle + ray.angle
                ray.cast(getWalls())  # Çarpışma tespiti

    def draw(self, screen):
        """
        Araba ve ışınlarını çizdir
        """
        screen.blit(self.rotated_image, self.rect)

        # Tüm ray'leri (ışınları) çizdir
        for ray in self.rays:
            ray.draw(screen)

