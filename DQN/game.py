import pygame
import numpy as np
import torch

from DQN.Agent import DQNAgent
from DQN.CarEnvironment import CarEnvironment
from agentCar import Car
from ray_list import create_rays
from track_lines import TrackLines, handle_collision_with_lines
from utilities.reorder_road_points import reorder_road_points
from utilities.road_utils import find_closest_point, load_road_points, calculate_distance_from_start

track_lines = TrackLines()

from utilities.threshold import apply_threshold


start_pos = (60, 230)

batch_size = 256
tick_rate = 30

def reset_game():
    global car
    car.x, car.y = start_pos[0], start_pos[1]
    car.speed = 0
    car.angle = 90


def start_training(car, rays, score, env, agent):
    # env = CarEnvironment(car, track_lines, reset_game, rays, score)
    # agent = DQNAgent(env.state_size, env.action_size)
    episodes = 10

    for e in range(episodes):
        state = env.reset()
        total_reward = 0

        for time in range(500):  # Maksimum adım sayısı
            # Rastgele bir aksiyon seç (ya da ajan hareketi)
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                print(f"Episode: {e+1}/{episodes}, Reward: {total_reward}")
                break

        # Ajanın öğrenmesi için replay fonksiyonunu çağır
        agent.replay(batch_size)  # Rastgele bir batch öğrenme




car = Car('../images/car.png', scale_factor=0.1, start_x=start_pos[0], start_y=start_pos[1], start_angle=90)


def main():
    pygame.init()
    win_size = (1000, 600)
    window = pygame.display.set_mode(win_size)
    clock = pygame.time.Clock()
    rays = create_rays(window)

    # Yolu yükle ve eşik uygulayın
    road_image = pygame.image.load('../images/track2.png').convert_alpha()
    threshold_image = apply_threshold(road_image)
    threshold_mask = pygame.mask.from_surface(threshold_image)

    # Yansıma maskelerini oluştur
    mask_fx = pygame.mask.from_surface(pygame.transform.flip(threshold_image, True, False))
    mask_fy = pygame.mask.from_surface(pygame.transform.flip(threshold_image, False, True))
    mask_fx_fy = pygame.mask.from_surface(pygame.transform.flip(threshold_image, True, True))
    flipped_masks = [[threshold_mask, mask_fy], [mask_fx, mask_fx_fy]]
    beam_surface = pygame.Surface((win_size[0], win_size[1]), pygame.SRCALPHA)

    run = True
    pass_startline = False
    score = 0
    collided = False

    road_points = load_road_points("../road_points.txt")
    road_points = reorder_road_points(start_pos, road_points)
    road_points = road_points[:-2]

    env = CarEnvironment(car, track_lines, reset_game, rays, score)
    agent = DQNAgent(env.state_size, env.action_size)
    gen = 0
    time = 0
    max_time = 6

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    print("game manually stopped!")

                    env.reward -= 200
                    env.reward = env.reward / 5000
                    env.done = True
                    torch.save(agent.model.state_dict(), "model_weights.pth")
                    start_training(car, rays, score, env, agent)  # Eğitim döngüsünü başlat
                    env.reset()
                    reset_game()
                    env.reward = 0
                if event.key == pygame.K_s:
                    torch.save(agent.model.state_dict(), "model_weights.pth")
                    print('model saved')
        keys = pygame.key.get_pressed()
        car.update(keys)

        window.fill((0, 0, 0))
        window.blit(threshold_image, threshold_image.get_rect(center=window.get_rect().center))

        car_pos = (car.x, car.y)

        for ray in rays:
            ray.draw_beam(car_pos, car.angle, flipped_masks, beam_surface, threshold_mask)
        car_mask = pygame.mask.from_surface(car.car_image)
        car_offset = (int(car.x - car.rect.width / 2), int(car.y - car.rect.height / 2))
        collision = threshold_mask.overlap(car_mask, car_offset)



        if collision :
            collided = True
        if env.reward/1000 < -1:
            env.reward -= 1000
            env.reward = env.reward / 1000
            print("stuck penalty\ntotal reward = ", env.reward)
            env.done = True
            start_training(car, rays, score, env, agent)  # Eğitim döngüsünü başlat
            env.reset()
            reset_game()
            env.reward = 0
        if collided:
            collided = False
            print("Collision detected with the road")
            gen += 1
            if score < 10:
                env.reward -= 500
                print("close engage penatly")
            elif score < 30:
                env.reward -= 100
                print("engage penalty before 30m penatly")
            if score <= 30:
                env.reward += score * 100
                print('under 50m reward')
            elif score > 30:
                env.reward += score * 400
                print("after 50m reward")
            # env.reward -= (1 - rays[1].distance) * 100
            if rays[1].distance < 15:
                env.reward -= 100
            env.reward = env.reward / 1000
            print("total reward =", env.reward)
            env.done = True
            start_training(car, rays, score, env, agent)  # Eğitim döngüsünü başlat
            env.reset()
            reset_game()
            time = 0
            env.reward = 0
            max_time += 0.5

        if time >= tick_rate*2 * max_time:
            collided = True
            print("Time limit exceeded")

        time += 1


        if score < 15:
            env.reward -= 0.03
        # env.reward -= np.sum([ray.distance * 0.01 for ray in rays])/2 # Ray mesafesi küçükse negatif ödül
        # env.reward += np.sum([ray.distance * 0.001 for ray in rays[:4]])
        env.reward += car.speed * 70 / car.max_speed
        if car.speed < 0.5:
            env.reward -= 1
        if car.speed > 1:
            env.reward += 2
        # finding lap distance
        closest_point = find_closest_point(car_pos, road_points)
        score = calculate_distance_from_start(road_points, closest_point)

        # draw the road points
        for point in road_points:
            pygame.draw.circle(window, (128,128,0), point, 2)

        # draw distance
        font = pygame.font.SysFont(None, 24)
        text = font.render(f"Total Score: {int(score)}", True, (128,128,128))
        window.blit(text, (10, 10))

        gen_text = font.render(f"Generation: {gen}", True, (128,128,128))
        window.blit(gen_text, (window.get_width() - 250, 10))

        epsilon_text = font.render(f"Epsilon: {agent.epsilon}", True, (128, 128, 128))
        window.blit(epsilon_text, (window.get_width() - 250, 25))

        speed_text = font.render(f"Speed: {car.speed}", True, (128, 128, 128))
        window.blit(speed_text, (window.get_width() - 250, 40))

        timer_text = font.render(f"Time: {time/60}", True, (128, 128, 128))
        window.blit(timer_text, (window.get_width() - 250, 55))

        # collision with lines and a flag
        pass_startline, lap_flag = handle_collision_with_lines(car, track_lines.start_line_rect,
                                                               track_lines.mid_line_rect, track_lines.blue_line_rect,
                                                               pass_startline)

        # draw the start line
        pygame.draw.rect(window, (0, 255, 0), track_lines.start_line)
        pygame.draw.rect(window, (255, 0, 0), track_lines.mid_line)
        pygame.draw.rect(window, (0, 0, 255), track_lines.blue_line_rect)

        # Oyun ve ajan etkileşimi
        state = env.get_state()  # Şu anki durumu al
        action = agent.act(state)  # Ajanı seç

        # draw actions as buttons
        pygame.draw.rect(window, (128,128,128), (750, 100, 20, 20))
        pygame.draw.rect(window, (128,128,128), (750, 130, 20, 20))
        pygame.draw.rect(window, (128,128,128), (720, 130, 20, 20))
        pygame.draw.rect(window, (128,128,128), (780, 130, 20, 20))

        if action == 0:
            pygame.draw.rect(window, (0,255,0), (750, 100, 20, 20))
            env.reward += 0.5
        if action == 1:
            pygame.draw.rect(window, (0,255,0), (750, 130, 20, 20))
        if action == 2:
            pygame.draw.rect(window, (0,255,0), (780, 130, 20, 20))
        if action == 3:
            pygame.draw.rect(window, (0,255,0), (720, 130, 20, 20))


        # Seçilen eylemi uygula ve durumu güncelle
        next_state, reward, done = env.step(action)

        car.draw(window)
        pygame.display.update()  # Ekran güncellemeyi unutmayın

        print(env.reward/1000)

        clock.tick(tick_rate)  # FPS'yi kontrol et


if __name__ == "__main__":
    main()
