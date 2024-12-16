import numpy as np
import random
import pickle
import csv
import matplotlib.pyplot as plt
import pygame
from car import Car
from utilities.threshold import apply_threshold
from CFG import CAR_IMAGE_PATH, WINDOW_SIZE, P_YELLOW, GRAY
from ray_list import create_rays
from utilities.road_utils import find_closest_point, calculate_distance_from_start, load_road_points
from utilities.reorder_road_points import reorder_road_points
from track_lines import TrackLines, handle_collision_with_lines

# Oyun ve Ortam Ayarları
pygame.init()
window = pygame.display.set_mode(WINDOW_SIZE)
clock = pygame.time.Clock()
rays = create_rays(window)
track_lines = TrackLines()
ROAD_IMAGE_PATH = '../images/track2.png'

# Yolu yükle ve eşik uygula
road_image = pygame.image.load(ROAD_IMAGE_PATH).convert_alpha()
threshold_image = apply_threshold(road_image)
threshold_mask = pygame.mask.from_surface(threshold_image)

# Maskeleri hazırla
mask_fx = pygame.mask.from_surface(pygame.transform.flip(threshold_image, True, False))
mask_fy = pygame.mask.from_surface(pygame.transform.flip(threshold_image, False, True))
mask_fx_fy = pygame.mask.from_surface(pygame.transform.flip(threshold_image, True, True))
flipped_masks = [[threshold_mask, mask_fy], [mask_fx, mask_fx_fy]]
beam_surface = pygame.Surface((WINDOW_SIZE[0], WINDOW_SIZE[1]), pygame.SRCALPHA)

# Başlangıç konumları ve yolu yükle
start_pos = (60, 300)
road_points = load_road_points("../road_points.txt")
road_points = reorder_road_points(start_pos, road_points)

# Çoklu Ajan Yapılandırması
NUM_AGENTS = 5
cars = [Car(CAR_IMAGE_PATH, scale_factor=0.1, start_x=start_pos[0], start_y=start_pos[1], start_angle=90)
        for _ in range(NUM_AGENTS)]

# Q-learning Parametreleri
ACTIONS = ["accelerate","turn_left_small","turn_left_large", "turn_right_small", "turn_right_large"]
GRID_SIZE = (80, 40, 50)
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.05
ALPHA = 0.4
GAMMA = 0.9

# Her ajan için bağımsız Q-tabloları ve durum değişkenleri
Q_tables = [np.zeros((*GRID_SIZE, len(ACTIONS))) for _ in range(NUM_AGENTS)]
previous_distances = [0 for _ in range(NUM_AGENTS)]
epsilons = [1.0 for _ in range(NUM_AGENTS)]
scores = [0 for _ in range(NUM_AGENTS)]
collision_iterations = [0 for _ in range(NUM_AGENTS)]
max_scores = [0 for _ in range(NUM_AGENTS)]
agents_alive = [True for _ in range(NUM_AGENTS)]

def discretize_state(car):
    car_pos = (car.x, car.y)
    closest_point = find_closest_point(car_pos, road_points)
    distance = calculate_distance_from_start(road_points, closest_point)

    distance_index = int(distance // 50)
    distance_index = min(distance_index, GRID_SIZE[0] - 1)  # Sınırları kontrol et

    angle_index = int(car.angle % 180) // 30
    angle_index = min(angle_index, GRID_SIZE[1] - 1)  # Sınırları kontrol et


    speed_index = int(car.speed // 5) + (GRID_SIZE[2] // 2)
    speed_index = min(max(speed_index, 0), GRID_SIZE[2] - 1)  # Sınırları kontrol et

    return distance_index, angle_index, speed_index

def calculate_reward(car, collision, distance_from_start, previous_distance):
    if collision:
        return -100
    progress = distance_from_start - previous_distance
    if progress > 0:
        angle_reward = max(0, 1 - abs(car.angle % 360 - previous_distance % 360) / 180)
        return progress * 15 + angle_reward * 10
    return -10

def reset_agent(car, index, iteration):
    car.x, car.y = start_pos[0], start_pos[1]
    car.angle = 90
    car.speed = 0
    max_scores[index] = max(max_scores[index], previous_distances[index])
    collision_iterations[index] = iteration
    previous_distances[index] = 0
    scores[index] = 0
    agents_alive[index] = False
    print(f"Car {index} reset at iteration {iteration}!")

def save_q_tables():
    # for i, q_table in enumerate(Q_tables):
    #     with open(f"q_table_agent_{i}.pkl", "wb") as f:
    #         pickle.dump(q_table, f)
    print("Q-tables have been saved!")

def load_q_tables():
    for i in range(NUM_AGENTS):
        try:
            with open(f"q_table_agent_{i}.pkl", "rb") as f:
                Q_tables[i] = pickle.load(f)
            print(f"Q-table for agent {i} loaded!")
        except FileNotFoundError:
            print(f"No Q-table found for agent {i}, initializing empty Q-table.")

def save_training_results():
    with open("training_results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Iteration", "Agent", "Score", "Max Score", "Last Collision"])
        for i in range(NUM_AGENTS):
            writer.writerow([iteration, i, previous_distances[i], max_scores[i], collision_iterations[i]])
    print("Training results saved to training_results.csv!")

def plot_training_performance():
    for i in range(NUM_AGENTS):
        plt.plot(range(len(previous_distances)), previous_distances, label=f"Agent {i}")
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.title("Training Performance Over Iterations")
    plt.legend()
    plt.show()

load_q_tables()

iteration = 0
run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    all_dead = all(not alive for alive in agents_alive)
    if all_dead:
        iteration += 1
        agents_alive = [True for _ in range(NUM_AGENTS)]

    for i, car in enumerate(cars):
        if not agents_alive[i]:
            continue

        state = discretize_state(car)
        if not all(0 <= state[j] < GRID_SIZE[j] for j in range(len(GRID_SIZE))):
            reset_agent(car, i, iteration)
            continue

        if random.uniform(0, 1) < epsilons[i]:
            action = random.choice(ACTIONS)
        else:
            action = ACTIONS[np.argmax(Q_tables[i][state])]

        car.update_with_action(action)

        car_mask = pygame.mask.from_surface(car.car_image)
        car_offset = (int(car.x - car.rect.width / 2), int(car.y - car.rect.height / 2))
        collision = threshold_mask.overlap(car_mask, car_offset)

        if collision:
            reset_agent(car, i, iteration)
            continue

        car_pos = (car.x, car.y)
        closest_point = find_closest_point(car_pos, road_points)
        distance_from_start = calculate_distance_from_start(road_points, closest_point)
        reward = calculate_reward(car, collision, distance_from_start, previous_distances[i])
        previous_distances[i] = distance_from_start
        new_state = discretize_state(car)

        action_index = ACTIONS.index(action)
        max_next_q = np.max(Q_tables[i][new_state])
        Q_tables[i][state][action_index] += ALPHA * (reward + GAMMA * max_next_q - Q_tables[i][state][action_index])

        epsilons[i] = max(MIN_EPSILON, epsilons[i] * EPSILON_DECAY)

    window.fill((0, 0, 0))
    window.blit(threshold_image, threshold_image.get_rect(center=window.get_rect().center))

    for car in cars:
        car_pos = (car.x, car.y)
        for ray in rays:
            ray.draw_beam(car_pos, car.angle, flipped_masks, beam_surface, threshold_mask)
        car.draw(window)

    for point in road_points:
        pygame.draw.circle(window, P_YELLOW, point, 2)

    font = pygame.font.SysFont(None, 24)
    for i, car in enumerate(cars):
        text = font.render(f"Car {i} Score: {int(previous_distances[i])}, Max:{int(max_scores[i])}, Last Collision: {collision_iterations[i]}", True, GRAY)
        window.blit(text, (10, 10 + i * 20))

    pygame.draw.rect(window, (0, 255, 0), track_lines.start_line)
    pygame.draw.rect(window, (255, 0, 0), track_lines.mid_line)
    pygame.draw.rect(window, (0, 0, 255), track_lines.blue_line_rect)

    pygame.display.update()
    clock.tick(60)

pygame.quit()

save_q_tables()
save_training_results()
