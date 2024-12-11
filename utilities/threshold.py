import pygame


def apply_threshold(image, threshold=200):
    width, height = image.get_size()
    threshold_surface = pygame.Surface((width, height), pygame.SRCALPHA)

    for x in range(width):
        for y in range(height):
            r, g, b, a = image.get_at((x, y))
            if r >= threshold and g >= threshold and b >= threshold:
                threshold_surface.set_at((x, y), (255, 255, 255, 255))
            else:
                threshold_surface.set_at((x, y), (0, 0, 0, 0))

    return threshold_surface
