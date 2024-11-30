import pygame
import sys

# Pygame'i başlat
pygame.init()

# Ekran boyutunu ayarla
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Pygame - Araba ve Çizgi")

# Renkler (RGB)
white = (255, 255, 255)
blue = (0, 0, 255)
red = (255, 0, 0)

# Arabanın boyutu ve başlangıç pozisyonu
car_width = 100
car_height = 60
car_x = 350
car_y = 300

# FPS ayarı için saat
clock = pygame.time.Clock()

# Ana döngü
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Klavye ile araba hareketi
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        car_x -= 5
    if keys[pygame.K_RIGHT]:
        car_x += 5
    if keys[pygame.K_UP]:
        car_y -= 5
    if keys[pygame.K_DOWN]:
        car_y += 5

    # Araba ve çizgi için koordinatlar
    front_left = (car_x, car_y)  # Ön sol köşe
    front_right = (car_x + car_width, car_y)  # Ön sağ köşe
    center_top = (car_x + car_width // 2, car_y - 50)  # Çizginin bağlanacağı nokta

    # Ekranı beyazla temizle
    screen.fill(white)

    # Araba (dikdörtgen) çiz
    pygame.draw.rect(screen, red, (car_x, car_y, car_width, car_height))

    # Çizgiyi arabanın ön köşelerine bağla
    pygame.draw.lines(screen, blue, False, [front_left, center_top, front_right], 3)

    # Ekranı güncelle
    pygame.display.flip()

    # 60 FPS
    clock.tick(60)

# Pygame'i kapat
pygame.quit()
sys.exit()
