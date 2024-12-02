import cv2
import numpy as np
import pygame

from CFG import ROAD_IMAGE_PATH, WIDTH, HEIGHT


class Wall:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def draw(self, win):
        pygame.draw.line(win, (255, 255, 255), (self.x1, self.y1), (self.x2, self.y2), 5)


def generate_walls_from_image(image_path):
    # Görüntüyü yükle
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gürültü azaltmak için Gaussian Blur
    # blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    blurred = image
    # cv2.imshow("Blurred Image", blurred)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Kenar algılama (Canny)
    edges = cv2.Canny(blurred, 50, 150)

    # cv2.imshow("Edges Image", edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Konturların bulunması
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    walls = []
    for contour in contours:
        # Konturu çokgen şeklinde basitleştir
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Kontur noktalarını al ve çizgiler oluştur
        for i in range(len(approx)):
            x1, y1 = approx[i][0]
            x2, y2 = approx[(i + 1) % len(approx)][0]
            walls.append(Wall(x1, y1, x2, y2))

    return walls


if __name__ == "__main__":
    # Resim dosyasının yolunu ayarla
    image_path = ROAD_IMAGE_PATH
    walls = generate_walls_from_image(image_path)

    # Pygame'de görselleştirme (isteğe bağlı)
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    win.fill((0, 0, 0))

    for wall in walls:
        wall.draw(win)

    pygame.display.update()
    pygame.time.wait(3000)
    pygame.quit()