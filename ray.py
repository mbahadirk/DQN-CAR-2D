import math
import pygame


class Ray:
    def __init__(self, angle, surface, name):
        self.angle = angle
        self.surface = surface
        self.name = name
        self.distance = 0  # Çarpma mesafesi.

    def draw_beam(self, pos, car_angle, flipped_masks, beam_surface, threshold_mask):
        adjusted_angle = self.angle + car_angle
        c = math.cos(math.radians(adjusted_angle))
        s = math.sin(math.radians(adjusted_angle))
        flip_x = c < 0
        flip_y = s < 0
        flipped_mask = flipped_masks[flip_x][flip_y]

        x_dest = beam_surface.get_width() * abs(c)
        y_dest = beam_surface.get_height() * abs(s)

        beam_surface.fill((0, 0, 0, 0))
        pygame.draw.line(beam_surface, (0, 0, 255), (0, 0), (x_dest, y_dest))
        beam_mask = pygame.mask.from_surface(beam_surface)

        offset_x = threshold_mask.get_size()[0] - 1 - pos[0] if flip_x else pos[0]
        offset_y = threshold_mask.get_size()[1] - 1 - pos[1] if flip_y else pos[1]
        hit = flipped_mask.overlap(beam_mask, (offset_x, offset_y))

        if hit is not None:
            hx = threshold_mask.get_size()[0] - 1 - hit[0] if flip_x else hit[0]
            hy = threshold_mask.get_size()[1] - 1 - hit[1] if flip_y else hit[1]
            hit_pos = (hx, hy)

            pygame.draw.line(self.surface, (0, 0, 255), pos, hit_pos)
            pygame.draw.circle(self.surface, (0, 255, 0), hit_pos, 3)
            self.distance = math.sqrt((hit_pos[0] - pos[0]) ** 2 + (hit_pos[1] - pos[1]) ** 2)
