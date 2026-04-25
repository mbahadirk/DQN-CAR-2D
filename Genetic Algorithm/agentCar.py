import math

import pygame


class Car:
    """
        This class represents a car that can be used in a game.
        The car has a position, an angle, a speed, an acceleration and a deceleration.
        The car can be drawn on a surface and can be moved.
    """
    def __init__(self, image_path="../images/car.png", scale_factor=0.1, start_x=100, start_y=100, start_angle=0):
        self.car_img = pygame.image.load(image_path)
        self.scale_factor = scale_factor

        self.start_x = start_x
        self.start_y = start_y
        self.start_angle = start_angle

        # Scale the car size
        self.car_width = int(self.car_img.get_width() * self.scale_factor)
        self.car_height = int(self.car_img.get_height() * self.scale_factor)
        self.car_image = pygame.transform.scale(self.car_img, (self.car_width, self.car_height))

        # Initial position and angle
        self.x = start_x
        self.y = start_y
        self.angle = start_angle
        self.speed = 3
        self.acceleration = 0.5
        self.deceleration = 0.4
        self.max_speed = 15
        self.friction = 0

        self.name = "Car"

        # Rectangle for collision detection
        self.rect = self.car_image.get_rect(center=(self.x, self.y))

    def draw(self, surface):
        # Draw the car

        rotated_car = pygame.transform.rotate(self.car_image, -self.angle)
        rotated_car.set_alpha(255)
        rect = rotated_car.get_rect(center=(self.x, self.y))
        surface.blit(rotated_car, rect.topleft)

        collider_surface = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        # pygame.draw.rect(collider_surface, (127,0,0), collider_surface.get_rect())  # Semi-transparent red
        # Rotate and draw the collider
        rotated_collider = pygame.transform.rotate(collider_surface, -self.angle)
        collider_rect = rotated_collider.get_rect(center=rect.center)
        surface.blit(rotated_collider, collider_rect.topleft)



    def update(self, action):
        if action == 0:  # Move forward
            self.speed += self.acceleration
            if self.speed > self.max_speed:
                self.speed = self.max_speed
        if action == 1:  # Move backward
            self.speed -= self.deceleration
            if self.speed <=0: self.speed = 0
            # if self.speed < -self.max_speed / 2:
            #     self.speed = -self.max_speed / 2

        # Friction
        if self.speed > 0:
            self.speed -= self.friction
        elif self.speed < 0:
            self.speed += self.friction
        if abs(self.speed) < self.friction:
            self.speed = 0

        # Steering
        if action == 2:
            self.angle += 16
        if action == 3:
            self.angle -= 16

        # if action == 4:  # Move forward right
        #     self.speed += self.acceleration
        #     self.angle += 5
        #     if self.speed > self.max_speed:
        #         self.speed = self.max_speed
        #
        # if action == 5:  # Move forward left
        #     self.speed += self.acceleration
        #     self.angle -= 5
        #     if self.speed > self.max_speed:
        #         self.speed = self.max_speed

        # if action == 6:  # Move backward right
        #     self.speed -= self.deceleration
        #     self.angle += 3
        #     if self.speed < -self.max_speed / 2:
        #         self.speed = -self.max_speed / 2
        #
        # if action == 7:  # Move backward left
        #     self.speed -= self.deceleration
        #     self.angle -= 3
        #     if self.speed < -self.max_speed / 2:
        #         self.speed = -self.max_speed / 2

        if action == 4:
            pass

        # Move the car
        self.x += self.speed * math.cos(math.radians(self.angle))
        self.y += self.speed * math.sin(math.radians(self.angle))

        # Update the collider position
        # self.rect.topleft = (self.x - self.car_width / 2, self.y - self.car_height / 2)


    def reset(self):
        # Reset the car to its initial position
        self.x = self.start_x
        self.y = self.start_y
        self.angle = self.start_angle
        self.speed = 0

