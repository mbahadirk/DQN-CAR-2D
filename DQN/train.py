import numpy as np
import pygame
import torch

from DQN.Agent import DQNAgent
from DQN.CarEnvironment import CarEnvironment
from agentCar import Car
from ray_list import create_rays
from track_lines import TrackLines, handle_collision_with_lines
from utilities.reorder_road_points import reorder_road_points
from utilities.road_utils import find_closest_point, load_road_points, calculate_distance_from_start
from utilities.threshold import apply_threshold
from weightVisualizer import WeightVisualizer

track_lines = TrackLines()


def handle_reward_lines(car, reward_line_1_rect, reward_line_2_rect, reward_line_3_rect):
    car_rect = car.car_image.get_rect(center=(car.x, car.y))


    return 0

start_pos = (60, 300)

batch_size = 64
tick_rate = 30


def apply_collision_penalty(env, rays, score):
    base_penalty = -50

    env.reward += base_penalty
    env.done = True


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


def reset_game(env, reward):
    global car, track_lines
    car.x, car.y = start_pos[0], start_pos[1]
    car.speed = 3
    car.angle = 90
    track_lines = TrackLines()
    env.reward = 0


car = Car('../images/car.png', scale_factor=0.1, start_x=start_pos[0], start_y=start_pos[1], start_angle=90)


def main():
    pygame.init()
    win_size = (1000, 600)
    window = pygame.display.set_mode(win_size)
    clock = pygame.time.Clock()
    rays = create_rays(window)

    # Load the track image and apply threshold
    road_image = pygame.image.load('../images/track_basic.png').convert_alpha()
    threshold_image = apply_threshold(road_image)
    threshold_mask = pygame.mask.from_surface(threshold_image)

    # Flipped masks
    mask_fx = pygame.mask.from_surface(pygame.transform.flip(threshold_image, True, False))
    mask_fy = pygame.mask.from_surface(pygame.transform.flip(threshold_image, False, True))
    mask_fx_fy = pygame.mask.from_surface(pygame.transform.flip(threshold_image, True, True))
    flipped_masks = [[threshold_mask, mask_fy], [mask_fx, mask_fx_fy]]

    beam_surface = pygame.Surface((200, 200), pygame.SRCALPHA)

    # Environment setup
    road_points = load_road_points("../road_points_road_basic.txt")
    road_points = reorder_road_points(start_pos, road_points)[:-2]

    env = CarEnvironment(car, track_lines, rays, 0)
    agent = DQNAgent(env.state_size, env.action_size, learning_rate=0.01, epsilon=1.0,
                     epsilon_min=0.01, epsilon_decay=0.9999, buffer_size=2000)
    visualizer = WeightVisualizer(agent.model, use_cv2=True)

    # Training parameters
    gamma = 0.5
    gen = 0
    frame = 0
    max_time = 10
    total_reward = 0
    pass_startline = False
    reward = 0
    #todo: rewardları düzelt env.reward ve reward'ı eşitle şuan sorun çıkartıyor her resette -den başlıyor
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    print("Game manually stopped!")
                    env.reward -= 10
                    env.done = True
                    torch.save(agent.model.state_dict(), "model_weights.pth")
                    reset_game(env, reward)
                    env.reward = 0

                elif event.key == pygame.K_s:
                    torch.save(agent.model.state_dict(), "model_weights.pth")
                    print('Model saved.')

        keys = pygame.key.get_pressed()
        car.update(keys)

        window.fill((0, 0, 0))
        window.blit(threshold_image, threshold_image.get_rect(center=window.get_rect().center))

        visualizer.visualize()

        car_pos = (car.x, car.y)

        for ray in rays:
            ray.draw_beam(car_pos, car.angle, flipped_masks, beam_surface, threshold_mask)

        # Collision detection
        car_mask = pygame.mask.from_surface(car.car_image)
        car_offset = (int(car.x - car.rect.width / 2), int(car.y - car.rect.height / 2))
        collision = threshold_mask.overlap(car_mask, car_offset)

        if collision:
            print("Collision detected with the road.")
            gen += 1
            apply_collision_penalty(env, rays, env.score)
            reset_game(env, reward)
            frame = 0
            reward = 0
            env.reward = 0
            max_time += 0.1

        # if env.reward / 1000 < -1:
        #     print("Stuck penalty.")
        #     env.reward -= 1000
        #     env.reward /= 1000
        #     env.reset()
        #     reset_game()
        #     env.reward = 0

        # if frame >= tick_rate * 2 * max_time:
        #     print("Time limit exceeded.")
        #     apply_collision_penalty(env, rays, env.score)
        #     env.reset()
        #     reset_game()
        #     frame = 0
        #     env.reward = 0

        # Reward shaping
        # env.reward += car.speed * 70 / car.max_speed
        # if car.speed < 0.5:
        #     env.reward -= 1
        # if car.speed > 1:
        #     env.reward += 2
        # if env.score < 15:
        #     env.reward -= 0.03



        # State and action
        state = env.get_state()
        action = agent.act(state)

        # Yeni: Ödül hesapla
        new_score = update_score_display(car_pos, road_points, window)

        progress = new_score - env.score
        reward += progress * 10


        # Collision with lines
        pass_startline, _ = handle_collision_with_lines(
            car, track_lines.start_line_rect, track_lines.mid_line_rect,
            track_lines.blue_line_rect, pass_startline, )
        car_rect = car.car_image.get_rect(center=(car.x, car.y))
        if car_rect.colliderect(track_lines.reward_line_1_rect):
            track_lines.reward_line_1_rect = (0, 0, 0, 0)
            reward += 200
        if car_rect.colliderect(track_lines.reward_line_2_rect):
            track_lines.reward_line_2_rect = (0, 0, 0, 0)
            reward += 300
        if car_rect.colliderect(track_lines.reward_line_3_rect):
            track_lines.reward_line_3_rect = (0, 0, 0, 0)
            reward += 500


        env.reward += reward
        env.score = new_score

        next_state, _, done = env.step(action)  # reward parametresi kullanılmıyor
        next_state = np.array(next_state, dtype=np.float32)
        agent.step(state, action, reward, next_state, done)
        if len(agent.replay_buffer) > batch_size:
            agent.replay(batch_size)

        # Draw GUI elements
        score = update_score_display(car_pos, road_points, window)
        draw_action_buttons(window, action)
        draw_debug_texts(window, score, gen, agent.epsilon, car.speed, frame)



        pygame.draw.rect(window, (0, 255, 0), track_lines.start_line)
        pygame.draw.rect(window, (255, 0, 0), track_lines.mid_line)
        pygame.draw.rect(window, (0, 0, 255), track_lines.blue_line_rect)
        pygame.draw.rect(window, (120, 180, 120), track_lines.reward_line_1_rect)
        pygame.draw.rect(window, (120, 180, 120), track_lines.reward_line_2_rect)
        pygame.draw.rect(window, (120, 180, 120), track_lines.reward_line_3_rect)

        car.draw(window)
        pygame.display.update()
        clock.tick(tick_rate)
        frame += 1


if __name__ == "__main__":
    main()
