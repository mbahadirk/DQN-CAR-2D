import math

import pygame


class Car:
    def __init__(self, image_path, scale_factor=0.001, start_x=100, start_y=100, start_angle=0):
        self.car_img = pygame.image.load(image_path)
        self.scale_factor = scale_factor

        self.start_x = start_x
        self.start_y = start_y
        self.start_angle = start_angle

        # Araba boyutlarını ölçekle
        self.car_width = int(self.car_img.get_width() * self.scale_factor)
        self.car_height = int(self.car_img.get_height() * self.scale_factor)
        self.car_image = pygame.transform.scale(self.car_img, (self.car_width, self.car_height))

        # Başlangıç konumu ve açısı
        self.x = start_x
        self.y = start_y
        self.angle = start_angle
        self.speed = 0
        self.acceleration = 0.5
        self.deceleration = 0.2
        self.max_speed = 8
        self.friction = 0.05

        # Collider için dikdörtgen
        self.rect = self.car_image.get_rect(center=(self.x, self.y))

    def draw(self, surface):
        # Arabayı çiz

        rotated_car = pygame.transform.rotate(self.car_image, -self.angle)
        rotated_car.set_alpha(255)
        rect = rotated_car.get_rect(center=(self.x, self.y))
        surface.blit(rotated_car, rect.topleft)

        # Yarı saydam kırmızı dikdörtgen (collider)

        collider_surface = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        # pygame.draw.rect(collider_surface, (127,0,0), collider_surface.get_rect())  # Yarı saydam kırmızı
        # Collider'ı döndür ve çiz
        rotated_collider = pygame.transform.rotate(collider_surface, -self.angle)
        collider_rect = rotated_collider.get_rect(center=rect.center)
        surface.blit(rotated_collider, collider_rect.topleft)


    def update(self, action):
        if action == 0:  # İleri gitme
            self.speed += self.acceleration
            if self.speed > self.max_speed:
                self.speed = self.max_speed
        if action == 1:  # Geri gitme
            self.speed -= self.deceleration
            if self.speed < -self.max_speed / 2:
                self.speed = -self.max_speed / 2

        # Sürtünme
        if self.speed > 0:
            self.speed -= self.friction
        elif self.speed < 0:
            self.speed += self.friction
        if abs(self.speed) < self.friction:
            self.speed = 0

        # Dönme
        if action == 2:
            self.angle += 3
        if action == 3:
            self.angle -= 3

        if action == 4:  # İleri sağa gitme
            self.speed += self.acceleration
            self.angle += 3
            if self.speed > self.max_speed:
                self.speed = self.max_speed

        if action == 5:  # İleri sola gitme
            self.speed += self.acceleration
            self.angle -= 3
            if self.speed > self.max_speed:
                self.speed = self.max_speed

        if action == 6:  # SAĞ Geri gitme
            self.speed -= self.deceleration
            self.angle += 3
            if self.speed < -self.max_speed / 2:
                self.speed = -self.max_speed / 2

        if action == 7:  # Sol Geri gitme
            self.speed -= self.deceleration
            self.angle -= 3
            if self.speed < -self.max_speed / 2:
                self.speed = -self.max_speed / 2

        if action == 8:
            pass

        # Araba hareketi
        self.x += self.speed * math.cos(math.radians(self.angle))
        self.y += self.speed * math.sin(math.radians(self.angle))

        # Collider pozisyonunu güncelle
        # self.rect.topleft = (self.x - self.car_width / 2, self.y - self.car_height / 2)


    def reset(self):
        # Arabayı başlangıç konumuna döndür
        self.x = self.start_x
        self.y = self.start_y
        self.angle = self.start_angle
        self.speed = 0
