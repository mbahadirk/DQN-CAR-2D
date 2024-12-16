WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
HALF_RED  = (0, 250, 0, 128)
P_YELLOW = (255, 255, 0)


CAR_IMAGE_PATH = '../images/car.png'
ROAD_IMAGE_PATH = "../images/track2.png"


import pygame

road_img = pygame.image.load(ROAD_IMAGE_PATH)
WINDOW_SIZE = road_img.get_size()


