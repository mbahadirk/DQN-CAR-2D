from datetime import datetime
import numpy as np
import pygame
import torch
import csv
import os

from Agent import DQNAgent
from CarEnvironment import CarEnvironment
from agentCar import Car
from ray_list import create_rays
from track_lines import TrackLines, handle_collision_with_lines
from utilities.reorder_road_points import reorder_road_points
from utilities.road_utils import find_closest_point, load_road_points, calculate_distance_from_start
from utilities.threshold import apply_threshold
from weightVisualizer import WeightVisualizer


track_lines = TrackLines()

start_pos = (250, 500)

batch_size = 128
tick_rate = 30


def reset_game(env):
    global car, track_lines
    car.x, car.y = start_pos[0], start_pos[1]
    car.speed = 3
    car.angle = 0
    track_lines = TrackLines()
    env.reward = 0


def draw_action_buttons(window, action):
    pos = [(750, 100), (750, 130), (780, 130), (720, 130)]
    for i, (x, y) in enumerate(pos):
        color = (0, 255, 0) if action == i else (128, 128, 128)
        pygame.draw.rect(window, color, (x, y, 20, 20))


def draw_debug_texts(window, score, gen, epsilon, speed, frame):
    font = pygame.font.SysFont(None, 24)
    stats = [
        f"Total Score: {int(score)}",
        f"Generation: {gen}",
        f"Epsilon: {epsilon:.3f}",
        f"Speed: {speed:.2f}",
        f"Time: {frame / 60:.1f}"
    ]
    for i, text in enumerate(stats):
        rendered = font.render(text, True, (128, 128, 128))
        window.blit(rendered, (window.get_width() - 250, 10 + i * 15))


def update_score_display(car_pos, road_points, window):
    closest_index = find_closest_point(car_pos, road_points)
    distance = calculate_distance_from_start(road_points, closest_index)
    return distance



car = Car('images/car.png', scale_factor=0.1, start_x=start_pos[0], start_y=start_pos[1], start_angle=0)


def main():
    pygame.init()
    win_size = (1000, 600)
    window = pygame.display.set_mode(win_size)
    clock = pygame.time.Clock()
    rays = create_rays(window)
    pygame.display.set_caption("Self Driving Car")
    # Load the track image and apply threshold
    road_image = pygame.image.load('images/track_basic.png').convert_alpha()
    threshold_image = apply_threshold(road_image)
    threshold_mask = pygame.mask.from_surface(threshold_image)

    # Flipped masks
    mask_fx = pygame.mask.from_surface(pygame.transform.flip(threshold_image, True, False))
    mask_fy = pygame.mask.from_surface(pygame.transform.flip(threshold_image, False, True))
    mask_fx_fy = pygame.mask.from_surface(pygame.transform.flip(threshold_image, True, True))
    flipped_masks = [[threshold_mask, mask_fy], [mask_fx, mask_fx_fy]]

    beam_surface = pygame.Surface((200, 200), pygame.SRCALPHA)

    # Environment setup
    road_points = load_road_points("road_points/road_points_road_basic.txt")
    road_points = reorder_road_points(start_pos, road_points)[:-2]

    env = CarEnvironment(car, track_lines, rays, 0)
    agent = DQNAgent(env.state_size, env.action_size)
    visualizer = WeightVisualizer(agent.model, use_cv2=False)

    # Training parameters
    collision_penalty = -200
    best_score = 0
    # CSV Logging setup
    csv_file = "training_log.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Generation", "Score", "Epsilon", "Frames"])

    gen = 0
    frame = 0
    episode_frames = 0
    pass_startline = False
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    print("Game manually stopped!")
                    env.reward -= 100
                    env.done = True
                    reset_game(env)
                    env.reward = 0

                elif event.key == pygame.K_s:
                    now = datetime.now()
                    torch.save(agent.model.state_dict(), f"model_weights_{now.month:02d}{now.day:02d}_{now.hour:02d}{now.minute:02d}.pt")
                    print('Model saved.')


        window.fill((0, 0, 0))
        window.blit(threshold_image, threshold_image.get_rect(center=window.get_rect().center))

        visualizer.visualize()

        car_pos = (car.x, car.y)

        # State before action
        state = env.get_state()
        action = agent.act(state)

        # Base step reward is what environment gave plus our custom logic
        step_reward = 0

        for ray in rays:
            ray.draw_beam(car_pos, car.angle, flipped_masks, beam_surface, threshold_mask)
            if ray.distance < ray.dangerous_distance:
                red_penalty = ray.dangerous_distance - ray.distance
                env.reward -= 0.04 * red_penalty
                step_reward -= 0.04 * red_penalty

        # Step environment
        next_state, env_reward, env_done = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)

        step_reward += env_reward

        # penalty for jittery movement
        if action != 2:
            step_reward -= 0.1

        # Yeni: Ödül hesapla
        car_pos = (car.x, car.y)
        new_score = update_score_display(car_pos, road_points, window)

        # Collision with lines
        pass_startline, _ = handle_collision_with_lines(
            car, track_lines.start_line_rect, track_lines.mid_line_rect,
            track_lines.blue_line_rect, pass_startline, )

        car_rect = car.car_image.get_rect(center=(car.x, car.y))
        if car_rect.colliderect(track_lines.reward_line_1_rect):
            track_lines.reward_line_1_rect = (0, 0, 0, 0)
            step_reward += 200
            print("reward line 1 passed")
        elif car_rect.colliderect(track_lines.reward_line_2_rect):
            track_lines.reward_line_2_rect = (0, 0, 0, 0)
            step_reward += 300
            print("reward line 2 passed")
        elif car_rect.colliderect(track_lines.reward_line_3_rect):
            track_lines.reward_line_3_rect = (0, 0, 0, 0)
            step_reward += 500
            print("reward line 3 passed")
        elif car_rect.colliderect(track_lines.reward_line_4_rect):
            track_lines.reward_line_4_rect = (0, 0, 0, 0)
            step_reward += 700
            print("reward line 4 passed")

        # Collision detection
        car_mask = pygame.mask.from_surface(car.car_image)
        car_offset = (int(car.x - car.rect.width / 2), int(car.y - car.rect.height / 2))
        collision = threshold_mask.overlap(car_mask, car_offset)

        if collision:
            print("Collision detected with the road.")
            step_reward += collision_penalty
            agent.step(state, action, step_reward, next_state, True)

            # Log metrics at end of episode
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([gen, env.reward, agent.epsilon, episode_frames])

            reset_game(env)
            new_score, env.score = 0, 0
            gen += 1
            episode_frames = 0

        else:
            progress = new_score - env.score
            step_reward += progress * 3
            env.score = new_score

            env.reward += step_reward

            if env.reward > best_score:
                best_score = env.reward
                collision_penalty *= 1 + best_score / 50000

            agent.step(state, action, step_reward, next_state, False)

        if len(agent.replay_buffer) > batch_size:
            agent.replay(batch_size)

        if frame % 100 == 0:
            agent.update_target_network()

        # Draw GUI elements
        score = env.reward
        draw_action_buttons(window, action)
        draw_debug_texts(window, score, gen, agent.epsilon, car.speed, frame)



        pygame.draw.rect(window, (0, 255, 0), track_lines.start_line)
        pygame.draw.rect(window, (255, 0, 0), track_lines.mid_line)
        pygame.draw.rect(window, (0, 0, 255), track_lines.blue_line_rect)
        pygame.draw.rect(window, (120, 180, 120), track_lines.reward_line_1_rect)
        pygame.draw.rect(window, (120, 180, 120), track_lines.reward_line_2_rect)
        pygame.draw.rect(window, (120, 180, 120), track_lines.reward_line_3_rect)
        pygame.draw.rect(window, (120, 180, 120), track_lines.reward_line_4_rect)

        car.draw(window)
        pygame.display.update()
        clock.tick(tick_rate)
        frame += 1
        episode_frames += 1


if __name__ == "__main__":
    main()
