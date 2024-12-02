import pygame
import math
from ray import Ray
from walls import getWalls

class Car:
    def __init__(self, image_path, scale_factor=0.1, start_x=0, start_y=0, start_angle=130):
        self.image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(self.image, (int(self.image.get_width() * scale_factor),
                                                         int(self.image.get_height() * scale_factor)))
        self.rotated_image = self.image
        self.rect = self.image.get_rect(center=(start_x, start_y))
        self.angle = start_angle
        self.speed = 0

        # Ray (ışın) tanımlamaları: ışınların başlangıç açısını kaydederiz
        self.ray_offsets = [-45, 45, 135, -135]
        self.rays = [
            Ray(self.rect.centerx, self.rect.centery, self.angle + offset) for offset in self.ray_offsets
        ]

    def update(self, keys):
        """
        Arabanın hareketini ve yönünü günceller.
        """
        rotation_speed = 3  # Dönüş hızı (derece)
        movement_speed = 5  # Hareket hızı (piksel)

        # Dönüş işlemi (A ve D tuşları)
        if keys[pygame.K_a]:  # Sola dön
            self.angle -= rotation_speed
        if keys[pygame.K_d]:  # Sağa dön
            self.angle += rotation_speed

        # Hareket işlemi (W ve S tuşları)
        if keys[pygame.K_w]:  # İleri hareket
            self.rect.x += movement_speed * math.sin(math.radians(self.angle))
            self.rect.y -= movement_speed * math.cos(math.radians(self.angle))
        if keys[pygame.K_s]:  # Geri hareket
            self.rect.x -= movement_speed * math.sin(math.radians(self.angle))
            self.rect.y += movement_speed * math.cos(math.radians(self.angle))

        # Arabayı yeni açıya göre döndür
        self.rotated_image = pygame.transform.rotate(self.image, -self.angle)
        self.rect = self.rotated_image.get_rect(center=self.rect.center)

        # Işınları arabanın etrafında doğru pozisyonda güncelle
        for i, ray in enumerate(self.rays):
            ray.start_x = self.rect.centerx
            ray.start_y = self.rect.centery
            ray.angle = self.angle + self.ray_offsets[i]  # Arabanın açısına offset ekleyerek güncelle
            ray.cast(getWalls())  # Çarpışma tespiti

    def draw(self, screen):
        """
        Araba ve ışınlarını ekrana çizdirir.
        """
        screen.blit(self.rotated_image, self.rect)

        # Tüm ray'leri (ışınları) çizdir
        for ray in self.rays:
            ray.draw(screen)
