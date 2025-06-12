import numpy as np
import pygame
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from Agent import DQNAgent
from CarEnvironment import CarEnvironment
from agentCar import Car
from ray_list import create_rays
from track_lines import TrackLines, handle_collision_with_lines
from utilities.reorder_road_points import reorder_road_points
from utilities.road_utils import find_closest_point, load_road_points, calculate_distance_from_start
from utilities.threshold import apply_threshold
from weightVisualizer import visualize_model


track_lines = TrackLines()

start_pos = (280, 530)

batch_size = 10
tick_rate = 30


def reset_agent(agent):
    agent.car.x, agent.car.y = start_pos[0], start_pos[1]
    agent.car.speed = 3
    agent.car.angle = 0
    agent.env.reward = 0


def stop_dead_agent(agent):
    agent.isDead = True
    agent.car.speed = 0
    agent.env.done = True


def draw_action_buttons(window, action):
    pos = [(750, 100), (750, 130), (780, 130), (720, 130)]
    for i, (x, y) in enumerate(pos):
        color = (0, 255, 0) if action == i else (128, 128, 128)
        pygame.draw.rect(window, color, (x, y, 20, 20))


def draw_debug_texts(window, score, gen, epsilon, alive_agent, frame):
    font = pygame.font.SysFont(None, 24)
    stats = [
        f"Best Score: {int(score)}",
        f"Generation: {gen}",
        f"Epsilon: {epsilon:.3f}",
        f"Alive Agent: {alive_agent}",
        f"Time: {frame / 60:.1f}"
    ]
    for i, text in enumerate(stats):
        rendered = font.render(text, True, (128, 128, 128))
        window.blit(rendered, (window.get_width() - 250, 10 + i * 15))


def update_score_display(car_pos, road_points, window):
    closest_index = find_closest_point(car_pos, road_points)
    distance = calculate_distance_from_start(road_points, closest_index)
    return distance



car_list = []
for _ in range(20):
    car_list.append(Car(start_x=start_pos[0], start_y=start_pos[1]))


def mutate_model(model, mutation_rate=0.02):
    """
    Model ağırlıklarına küçük gürültü ekler.
    mutation_rate: Mutasyonun şiddeti (0.01-0.05 gibi).
    """
    with torch.no_grad():
        # Ağırlıkları tek vektör yap
        weights = parameters_to_vector(model.parameters())

        # Gürültü (mutasyon) ekle
        noise = torch.randn_like(weights) * mutation_rate

        mutated_weights = weights + noise

        # Mutasyonlu ağırlıkları modele yükle
        vector_to_parameters(mutated_weights, model.parameters())


class AGENT:
    def __init__(self, car, env, agent):
        self.car = car
        self.env = env
        self.agent = agent
        self.car_pos = self.car.x, self.car.y
        self.isDead = False


def crossover(model1, model2):
    child_model = DQNAgent(model1.env.state_size, model1.env.action_size).model
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    child_state_dict = {}

    for key in state_dict1.keys():
        weights1 = state_dict1[key]
        weights2 = state_dict2[key]
        # Her bir parametrenin yarısını 1. modelden, diğer yarısını 2. modelden alalım
        mask = torch.rand_like(weights1) > 0.5
        child_weights = torch.where(mask, weights1, weights2)
        child_state_dict[key] = child_weights

    child_model.load_state_dict(child_state_dict)
    return child_model


def regenerate_agents(car_list, rays, generation, prev_agents=None):
    agent_list = []
    best_model_path = "models/best_model.pt"  # En iyi modelin kaydedildiği yer

    # Eğer öncekiler varsa, çiftler oluşturup crossover yap
    if prev_agents is not None and len(prev_agents) >= 2:
        for i in range(len(car_list)):
            car = car_list[i]
            env = CarEnvironment(car, track_lines, rays, 0)

            # Ebeveyn ajanlar
            parent1 = prev_agents[i % len(prev_agents)].agent.model
            parent2 = prev_agents[(i+1) % len(prev_agents)].agent.model

            child_model = crossover(parent1, parent2)
            agent = DQNAgent(env.state_size, env.action_size, learning_rate=1e-4,
                             epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, buffer_size=1000)
            agent.model.load_state_dict(child_model.state_dict())

            car_agent = AGENT(car, env, agent)
            reset_agent(car_agent)
            car.name = f"Car {i + 1}"
            agent_list.append(car_agent)
    else:
        # İlk generation ya da model yoksa klasik
        for car in car_list:
            env = CarEnvironment(car, track_lines, rays, 0)
            agent = DQNAgent(env.state_size, env.action_size, learning_rate=1e-4,
                             epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, buffer_size=1000)
            try:
                agent.model.load_state_dict(torch.load(best_model_path))
                print("💾 En iyi model yüklendi!")
            except FileNotFoundError:
                print("⚠ En iyi model bulunamadı, yeni model oluşturuluyor.")

            car_agent = AGENT(car, env, agent)
            reset_agent(car_agent)
            car.name = f"Car {len(agent_list) + 1}"
            agent_list.append(car_agent)

    return agent_list


def main():
    pygame.init()
    win_size = (1000, 600)
    window = pygame.display.set_mode(win_size)
    clock = pygame.time.Clock()
    rays = create_rays(window)
    pygame.display.set_caption("Self Driving Car")
    # Load the track image and apply threshold
    road_image = pygame.image.load('../images/track_hard.png').convert_alpha()
    threshold_image = apply_threshold(road_image)
    threshold_mask = pygame.mask.from_surface(threshold_image)

    # Flipped masks
    mask_fx = pygame.mask.from_surface(pygame.transform.flip(threshold_image, True, False))
    mask_fy = pygame.mask.from_surface(pygame.transform.flip(threshold_image, False, True))
    mask_fx_fy = pygame.mask.from_surface(pygame.transform.flip(threshold_image, True, True))
    flipped_masks = [[threshold_mask, mask_fy], [mask_fx, mask_fx_fy]]

    beam_surface = pygame.Surface((200, 200), pygame.SRCALPHA)

    # Environment setup
    road_points = load_road_points("../road_points/road_points_road_hard.txt")
    road_points = reorder_road_points(start_pos, road_points)[:-2]




    # Training parameters
    collision_penalty = -500
    best_score = 0
    generation = 0
    frame = 0
    pass_startline = False
    run = True
    fitness_score = 0
    time = 0

    agent_list = regenerate_agents(car_list, rays, generation)


    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False


        window.fill((0, 0, 0))
        window.blit(threshold_image, threshold_image.get_rect(center=window.get_rect().center))

        # visualizer.visualize()

        for agent in agent_list:
            agent.car_pos = (agent.car.x, agent.car.y)

            for ray in rays:
                ray.draw_beam(agent.car_pos, agent.car.angle, flipped_masks, beam_surface, threshold_mask)
                if ray.distance < ray.dangerous_distance:
                    red_penalty = ray.dangerous_distance - ray.distance
                    agent.env.reward -= 0.002 * red_penalty
            # Collision detection
            car_mask = pygame.mask.from_surface(agent.car.car_image)
            car_offset = (int(agent.car.x - agent.car.rect.width / 2), int(agent.car.y - agent.car.rect.height / 2))
            collision = threshold_mask.overlap(car_mask, car_offset)


            # State and action
            state = agent.env.get_state()
            action = agent.agent.act(state)

            # penalty for jittery movement
            if action != 2:
                agent.env.reward -= 0.03
            if agent.car.speed < 2:
                agent.env.reward -= 0.08
            else: agent.env.reward += 0.08

            # Yeni: Ödül hesapla
            distance_score = update_score_display(agent.car_pos, road_points, window)
            new_score = distance_score
            # Collision with lines
            pass_startline, _ = handle_collision_with_lines(
                agent.car, track_lines.start_line_rect, track_lines.mid_line_rect,
                track_lines.blue_line_rect, pass_startline, )

            car_rect = agent.car.car_image.get_rect(center=(agent.car.x, agent.car.y))
            if car_rect.colliderect(track_lines.reward_line_1_rect):
                track_lines.reward_line_1_rect = (0, 0, 0, 0)
                agent.env.reward += 50
                print("reward line 1 passed")
            elif car_rect.colliderect(track_lines.reward_line_2_rect):
                track_lines.reward_line_2_rect = (0, 0, 0, 0)
                agent.env.reward += 100
                print("reward line 2 passed")
            elif car_rect.colliderect(track_lines.reward_line_3_rect):
                track_lines.reward_line_3_rect = (0, 0, 0, 0)
                agent.env.reward += 200
                print("reward line 3 passed")
            elif car_rect.colliderect(track_lines.reward_line_4_rect):
                track_lines.reward_line_4_rect = (0, 0, 0, 0)
                agent.env.reward += 200
                print("reward line 4 passed")

            if collision:
                print("Collision detected with the road.")
                agent.env.reward = collision_penalty
                agent.agent.step(state, action, agent.env.reward, next_state, True)
                new_score, agent.env.score = 0, 0
                stop_dead_agent(agent)
            elif time >= 10000:
                agent.agent.step(state, action, agent.env.reward, next_state, True)
                new_score, agent.env.score = 0, 0
                stop_dead_agent(agent)
            else:
                progress = new_score - agent.env.score
                agent.env.reward += progress * 0.6
                agent.env.score = new_score

                if agent.env.reward > best_score:
                    best_score = agent.env.reward
                    collision_penalty *= 1 + best_score / 50000

                next_state, reward, done = agent.env.step(action)
                next_state = np.array(next_state, dtype=np.float32)
                agent.agent.step(state, action, agent.env.reward, next_state, done)
                if len(agent.agent.replay_buffer) > batch_size:
                    agent.agent.replay(batch_size)
                    agent.env.done = True

            pygame.draw.rect(window, (0, 255, 0), track_lines.start_line)
            pygame.draw.rect(window, (255, 0, 0), track_lines.mid_line)
            pygame.draw.rect(window, (0, 0, 255), track_lines.blue_line_rect)
            pygame.draw.rect(window, (120, 180, 120), track_lines.reward_line_1_rect)
            pygame.draw.rect(window, (120, 180, 120), track_lines.reward_line_2_rect)
            pygame.draw.rect(window, (120, 180, 120), track_lines.reward_line_3_rect)
            pygame.draw.rect(window, (120, 180, 120), track_lines.reward_line_4_rect)

            for point in road_points:
                pygame.draw.circle(window, (220, 125, 160), point, 2)

            draw_debug_texts(window, fitness_score, generation, agent.agent.epsilon, len(agent_list), frame*10 )


            if agent.isDead:
                if distance_score > fitness_score:
                    fitness_score = distance_score
                    torch.save(agent.agent.model.state_dict(),
                               f"models/best_model.pt")
                    print('Model saved.')
                agent_list.pop(agent_list.index(agent))


        for car in car_list:
            car.draw(window)
        pygame.display.update()
        clock.tick(tick_rate)

        if len(agent_list) == 0:
            agent_list = regenerate_agents(car_list, rays, generation, prev_agents=agent_list)
            generation += 1
            time = 0



        frame += 1
        time += 10


if __name__ == "__main__":
    main()
