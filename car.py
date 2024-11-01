import pygame
from CFG import WHITE,WIDTH,HEIGHT,CAR_IMAGE_PATH, HALF_RED
import math


class myPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def rotate(origin,point,angle):
    qx = origin.x + math.cos(angle) * (point.x - origin.x) - math.sin(angle) * (point.y - origin.y)
    qy = origin.y + math.sin(angle) * (point.x - origin.x) + math.cos(angle) * (point.y - origin.y)
    q = myPoint(qx, qy)
    return q


def rotateRect(pt1, pt2, pt3, pt4, angle):

    pt_center = myPoint((pt1.x + pt3.x)/2, (pt1.y + pt3.y)/2)

    pt1 = rotate(pt_center,pt1,angle)
    pt2 = rotate(pt_center,pt2,angle)
    pt3 = rotate(pt_center,pt3,angle)
    pt4 = rotate(pt_center,pt4,angle)

    return pt1, pt2, pt3, pt4


class LineRect:
    def __init__(self, x, y, angle, width=3, length=100):
        self.x = x
        self.y = y
        self.angle = angle
        self.width = width
        self.length = length
        self.point = (0,0)

        # Dikdörtgen yüzeyi oluştur
        self.surface = pygame.Surface((self.length, self.width), pygame.SRCALPHA)
        self.surface.fill((255, 0, 0, 128))  # Yarı saydam kırmızı

        # Dikdörtgeni güncelle
        self.update_rect()

    def update(self, start_x, start_y, angle):
        self.x = start_x
        self.y = start_y
        self.angle = angle
        self.update_rect()

    def update_rect(self):
        # Yüzeyi döndür ve merkezi güncelle
        self.rotated_surface = pygame.transform.rotate(self.surface, -self.angle)
        self.rotated_rect = self.rotated_surface.get_rect(center=(self.x, self.y))

    def draw(self, screen, *pos):
        # Döndürülmüş dikdörtgeni ekran üzerinde çiz
        screen.blit(self.rotated_surface, self.rotated_rect.topleft)

        # draw collide point
        if self.point != (0, 0):
            pygame.draw.circle(screen, WHITE, (self.point), 5)



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
        self.acceleration = 0.2
        self.deceleration = 0.1
        self.max_speed = 8
        self.friction = 0.05

        # Collider için dikdörtgen
        self.rect = pygame.Rect(self.x, self.y, self.car_width, self.car_height)


        # front,back,rigth ,left sides
        self.topLeftPoint = myPoint(self.x - self.car_width / 2, self.y - self.car_height / 2)
        self.topRightPoint  = myPoint(self.x - self.car_width / 2, self.y + self.car_height / 2)
        self.backLeftPoint  = myPoint(self.x + self.car_width / 2, self.y - self.car_height / 2)
        self.backRightPoint = myPoint(self.x + self.car_width / 2, self.y + self.car_height / 2)


        # create ray
        self.topLeftRay = LineRect(self.topLeftPoint.x, self.topLeftPoint.y, self.angle)
        self.topRightRay = LineRect(self.topRightPoint.x, self.topRightPoint.y, self.angle)
        self.backRightRay = LineRect(self.backRightPoint.x, self.backRightPoint.y, self.angle)
        self.backLeftRay = LineRect(self.backLeftPoint.x, self.backLeftPoint.y, self.angle)

        self.rays = {
            "topLeft": {
                "offset": 100,
                "rayName": self.topLeftRay,
                "rayPoint": self.topLeftPoint,
                "angleOffset" : 45
            },
            "topRight": {
                "offset": 100,
                "rayName": self.topRightRay,
                "rayPoint": self.topRightPoint,
                "angleOffset": 45
            },
            "backRight": {
                "offset": -100,
                "rayName": self.backRightRay,
                "rayPoint": self.backRightPoint,
                "angleOffset": 0
            },
            "backLeft": {
                "offset": -100,
                "rayName": self.backLeftRay,
                "rayPoint": self.backLeftPoint,
                "angleOffset": 0
            }
        }

    def updatePoints(self):
        self.topLeftPoint = myPoint(self.x - self.car_width / 2, self.y - self.car_height / 2)
        self.backLeftPoint = myPoint(self.x + self.car_width / 2, self.y - self.car_height / 2)
        self.backRightPoint = myPoint(self.x + self.car_width / 2, self.y + self.car_height / 2)
        self.topRightPoint = myPoint(self.x - self.car_width / 2, self.y + self.car_height / 2)

        self.topLeftPoint, self.backLeftPoint, self.backRightPoint, self.topRightPoint =\
            rotateRect(self.topLeftPoint,
                        self.backLeftPoint,
                        self.backRightPoint,
                        self.topRightPoint,
                        self.angle / 180 * math.pi)
        self.rays = {
            "topLeft": {
                "offset": 100,
                "rayName": self.topLeftRay,
                "rayPoint": self.topLeftPoint,
                "angleOffset" : 0
            },
            "topRight": {
                "offset": 100,
                "rayName": self.topRightRay,
                "rayPoint": self.topRightPoint,
                "angleOffset": 0
            },
            "backRight": {
                "offset": -100,
                "rayName": self.backRightRay,
                "rayPoint": self.backRightPoint,
                "angleOffset": 0
            },
            "backLeft": {
                "offset": -100,
                "rayName": self.backLeftRay,
                "rayPoint": self.backLeftPoint,
                "angleOffset": 0
            }
        }

    def draw(self, surface):
        # Arabayı çiz

        rotated_car = pygame.transform.rotate(self.car_image, -self.angle)
        rotated_car.set_alpha(128)
        rect = rotated_car.get_rect(center=(self.x, self.y))
        surface.blit(rotated_car, rect.topleft)

        # Yarı saydam kırmızı dikdörtgen (collider)

        collider_surface = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        pygame.draw.rect(collider_surface, HALF_RED, collider_surface.get_rect())  # Yarı saydam kırmızı
        # Collider'ı döndür ve çiz
        rotated_collider = pygame.transform.rotate(collider_surface, -self.angle)
        collider_rect = rotated_collider.get_rect(center=rect.center)
        surface.blit(rotated_collider, collider_rect.topleft)

        # draw ray
        # self.topLeftRay.draw(surface)
        # self.topRightRay.draw(surface)
        # self.backRightRay.draw(surface)
        # self.backLeftRay.draw(surface)

        for ray_name, ray_info in self.rays.items():
            ray_info["rayName"].draw(surface)
            pygame.draw.circle(surface, WHITE, (ray_info['rayPoint'].x, ray_info['rayPoint'].y), 2)

        # draw side points
        #
        # pygame.draw.circle(surface, WHITE, (self.topLeftPoint.x, self.topLeftPoint.y), 5)
        # pygame.draw.circle(surface, WHITE, (self.backLeftPoint.x, self.backLeftPoint.y), 5)
        # pygame.draw.circle(surface, WHITE, (self.backRightPoint.x, self.backRightPoint.y), 5)
        # pygame.draw.circle(surface, WHITE, (self.topRightPoint.x, self.topRightPoint.y), 5)




    def update(self, keys):
        if keys[pygame.K_w]:  # İleri gitme
            self.speed += self.acceleration
            if self.speed > self.max_speed:
                self.speed = self.max_speed
        if keys[pygame.K_s]:  # Geri gitme
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
        if keys[pygame.K_a]:
            self.angle -= 3
        if keys[pygame.K_d]:
            self.angle += 3

        # Araba hareketi
        self.x += self.speed * math.cos(math.radians(self.angle))
        self.y += self.speed * math.sin(math.radians(self.angle))

        # Collider pozisyonunu güncelle
        # self.rect.topleft = (self.x - self.car_width / 2, self.y - self.car_height / 2)


        # updating side points of the car
        self.updatePoints()

        # updating ray
        for ray_name, ray_info in self.rays.items():
            self.updateRay(ray_info["offset"], ray_info["rayName"], ray_info["rayPoint"],angleOffset=ray_info["angleOffset"])


    def updateRay(self, offset, rayName, rayPos,angleOffset = 0):
        ray_start_x = rayPos.x + offset * math.cos(math.radians(self.angle))
        ray_start_y = rayPos.y + offset * math.sin(math.radians(self.angle))
        rayName.update(ray_start_x, ray_start_y, self.angle+angleOffset)



    def reset(self):
        # Arabayı başlangıç konumuna döndür
        self.x = self.start_x
        self.y = self.start_y
        self.angle = self.start_angle
        self.speed = 0
