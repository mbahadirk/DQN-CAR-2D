
import pygame
import math

class Ray:
    def __init__(self, start_x, start_y, angle):
        self.start_x = start_x
        self.start_y = start_y
        self.angle = angle
        self.end_x = start_x
        self.end_y = start_y
        self.length = 0  # Mesafe

    def cast(self, walls):
        """
        Duvarlarla çarpışma tespiti ve mesafe hesaplama
        """
        closest_point = None
        min_distance = float('inf')

        for wall in walls:
            # Duvarın iki ucu
            x1, y1, x2, y2 = wall.x1, wall.y1, wall.x2, wall.y2

            # Ray'in bitiş noktası hesaplanır
            x3, y3 = self.start_x, self.start_y
            x4 = self.start_x + math.cos(math.radians(self.angle)) * 1000
            y4 = self.start_y + math.sin(math.radians(self.angle)) * 1000

            # İki çizginin kesişip kesişmediğini kontrol et
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if den == 0:
                continue  # Paralel çizgiler

            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

            if 0 <= t <= 1 and u >= 0:
                # Kesişme noktası
                px = x1 + t * (x2 - x1)
                py = y1 + t * (y2 - y1)

                # Öklid mesafesi
                distance = math.hypot(px - self.start_x, py - self.start_y)

                if distance < min_distance:
                    min_distance = distance
                    closest_point = (px, py)

        # Kesişme noktasını ve mesafeyi güncelle
        if closest_point:
            self.end_x, self.end_y = closest_point
            self.length = min_distance
        else:
            self.end_x = self.start_x + math.cos(math.radians(self.angle)) * 1000
            self.end_y = self.start_y + math.sin(math.radians(self.angle)) * 1000
            self.length = float('inf')

    def draw(self, screen):
        """
        Ray'i ve çarpışma noktasını çizdir
        """
        pygame.draw.line(screen, (255, 0, 0), (self.start_x, self.start_y), (self.end_x, self.end_y), 2)
        if self.length < float('inf'):
            pygame.draw.circle(screen, (0, 255, 0), (int(self.end_x), int(self.end_y)), 5)



