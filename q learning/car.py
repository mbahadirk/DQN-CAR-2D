import math
import numpy as np
import pygame


class Car:
    def __init__(self, image_path, scale_factor=0.001, start_x=100, start_y=100, start_angle=0):
        self.car_img = pygame.image.load(image_path).convert_alpha()
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
        self.acceleration = 0.8
        self.deceleration = 0.5
        self.max_speed = 5
        self.friction = 0.05

        # Collider için dikdörtgen
        self.rect = self.car_image.get_rect(center=(self.x, self.y))

    def draw(self, surface):
        # Arabayı çiz
        rotated_car = pygame.transform.rotate(self.car_image, -self.angle)
        rect = rotated_car.get_rect(center=(self.x, self.y))
        surface.blit(rotated_car, rect.topleft)

    def update_with_action(self, action):
        """
        Q-learning aksiyonlarına göre arabayı güncelle.
        action: "accelerate", "decelerate", "turn_left_small", "turn_left_large",
                "turn_right_small", "turn_right_large" aksiyonlarından biri.
        """
        if action == "accelerate":  # Hızlan
            self.speed += self.acceleration
            if self.speed > self.max_speed:
                self.speed = self.max_speed
        elif action == "decelerate":  # Yavaşla
            self.speed -= self.deceleration
            if self.speed < self.max_speed / 2:  # Geri gitmeye sınır koy
                self.speed = self.max_speed / 2
        elif action == "turn_left_small":  # Hafif sola dön
            self.angle -= 5
        elif action == "turn_left_large":  # Keskin sola dön
            self.angle -= 10
        elif action == "turn_right_small":  # Hafif sağa dön
            self.angle += 5
        elif action == "turn_right_large":  # Keskin sağa dön
            self.angle += 10

        # Hızı uygula
        self.x += self.speed * np.cos(np.radians(self.angle))
        self.y += self.speed * np.sin(np.radians(self.angle))


        # Sürtünme
        if self.speed > 0:
            self.speed -= self.friction
        elif self.speed < 0:
            self.speed += self.friction
        if abs(self.speed) < self.friction:
            self.speed = 0

        # Araba hareketi
        self.x += self.speed * math.cos(math.radians(self.angle))
        self.y += self.speed * math.sin(math.radians(self.angle))

    def reset(self):
        # Arabayı başlangıç konumuna döndür
        self.x = self.start_x
        self.y = self.start_y
        self.angle = self.start_angle
        self.speed = 0
