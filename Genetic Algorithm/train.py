import copy
import math

import pygame
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from Agent import DQNAgent
from CarEnvironment import CarEnvironment
from agentCar import Car
from ray_list import create_rays
from track_lines import TrackLines
from utilities.reorder_road_points import reorder_road_points
from utilities.road_utils import find_closest_point, load_road_points, calculate_distance_from_start
from utilities.threshold import apply_threshold


START_POS = (280, 530)
POPULATION = 30
TICK_RATE = 30
MAX_TIME = 999999999999
MUTATION_RATE = 0.05
EPSILON_DECAY = 0.9998
EPSILON_SLOW_THRESHOLD = 0.25
EPSILON_SLOW_DECAY = 0.99995
# Eğer ajan bu kadar frame içinde en az PROGRESS_MIN ilerleme kaydedemezse öldür
STALL_FRAMES = 90       # ~3 saniye (30fps'de)
PROGRESS_MIN = 3        # bu kadar road-point ilerlemesi bekleniyor
N_PARENTS = 2           # bir sonraki nesli üretmek için kullanılan en iyi ajan sayısı
SPEED_TARGET = 3.5      # bu hızın altında kalmak slow_frames sayacını artırır
JITTER_PENALTY = 0.4    # fitness'tan düşülecek miktar (her yön değiştirmede)
SLOW_PENALTY   = 0.03   # fitness'tan düşülecek miktar (her yavaş frame'de)
WALL_PENALTY_SCALE  = 0.0003  # tehlikeli ray başına frame başına ceza
CORNER_BONUS_SCALE  = 0.25    # aktif viraj + ilerleme başına bonus
CORNER_ANGLE_THRESHOLD = 2.0  # bir frame'de bu kadar derece dönüş = viraj sayılır

# --- Parkur konfigürasyonu ---
# Farklı parkurda eğitim için bu iki satırı değiştir:
TRACK_IMAGE      = '../images/track2.png'
ROAD_POINTS_FILE = '../road_points/road_points_track2.txt'

# Önceki eğitimden devam etmek için en iyi modelin yolunu gir.
# None → sıfırdan başla, 'models/best_model.pt' → kaydedilmiş modelden başla.
SEED_MODEL = "models/best_model.pt" #None


car_list = [Car(start_x=START_POS[0], start_y=START_POS[1]) for _ in range(POPULATION)]


class AGENT:
    def __init__(self, car, env, agent):
        self.car = car
        self.env = env
        self.agent = agent
        self.isDead = False
        self.stall_frames = 0
        self.last_checkpoint_score = 0
        self.lap_count = 0             # tamamlanan tur sayısı
        self.collected_rewards = set() # hangi reward line'lardan bonus alındı (per-agent)
        self.prev_action = 4           # bir önceki action (jitter tespiti için)
        self.jitter_count = 0          # art arda zıt dönüş sayısı
        self.slow_frames = 0           # SPEED_TARGET altında geçen frame sayısı
        self.wall_proximity_penalty = 0.0  # duvara yakınlık birikimli ceza
        self.corner_bonus = 0.0            # viraj bonusu
        self.prev_angle = car.angle        # bir önceki frame açısı


def reset_agent(agent):
    agent.car.x, agent.car.y = START_POS
    agent.car.speed = 3
    agent.car.angle = 0
    agent.env.reward = 0
    agent.env.score = 0
    agent.env.pass_startline = False
    agent.env.lap_cooldown = 0
    agent.isDead = False
    agent.stall_frames = 0
    agent.last_checkpoint_score = 0
    agent.lap_count = 0
    agent.collected_rewards = set()
    agent.prev_action = 4
    agent.jitter_count = 0
    agent.slow_frames = 0
    agent.wall_proximity_penalty = 0.0
    agent.corner_bonus = 0.0
    agent.prev_angle = agent.car.angle


def stop_dead_agent(agent):
    agent.isDead = True
    agent.car.speed = 0


def mutate_model(model, mutation_rate=MUTATION_RATE):
    with torch.no_grad():
        weights = parameters_to_vector(model.parameters())
        noise = torch.randn_like(weights) * mutation_rate
        vector_to_parameters(weights + noise, model.parameters())


def crossover(model1, model2):
    """Crossover two nn.Sequential models gene-by-gene, returning a new child model."""
    child = copy.deepcopy(model1)
    sd1 = model1.state_dict()
    sd2 = model2.state_dict()
    child_sd = {}
    for key in sd1:
        w1, w2 = sd1[key].float(), sd2[key].float()
        mask = torch.rand_like(w1) > 0.5
        child_sd[key] = torch.where(mask, w1, w2)
    child.load_state_dict(child_sd)
    return child


def make_agent(car, track_lines, rays, epsilon=0.05):
    env = CarEnvironment(car, track_lines, rays, 0)
    agent = DQNAgent(env.state_size, env.action_size, epsilon=epsilon)
    return AGENT(car, env, agent)


def regenerate_agents(car_list, track_lines, rays, scored_dead=None, seed_model_path=None, current_epsilon=0.05):
    """
    scored_dead:      list of (fitness_score, nn.Sequential model), one per dead agent.
                      Selects top ELITE_FRACTION as parents, fills next gen via crossover + mutation.
    seed_model_path:  yol verilirse ilk nesil bu modelin mutasyonlu kopyalarından oluşur.
    """
    agent_list = []

    if scored_dead and len(scored_dead) >= 2:
        # Sort by fitness score descending
        scored_dead.sort(key=lambda x: x[0], reverse=True)
        parents = [model for _, model in scored_dead[:N_PARENTS]]
        print(f"  Parents: {[round(s, 1) for s, _ in scored_dead[:N_PARENTS]]}")

        for i, car in enumerate(car_list):
            ca = make_agent(car, track_lines, rays, epsilon=current_epsilon)
            
            if i == 0:
                # ---------------- CHAMPION PROTECTION ----------------
                # Car 1 inherits the absolute best model without mutation
                ca.agent.model.load_state_dict(copy.deepcopy(parents[0].state_dict()))
                car.name = "Champion"
            else:
                # Other cars crossover and mutate normally
                p1 = parents[i % N_PARENTS]
                p2 = parents[(i + 1) % N_PARENTS]
                child_model = crossover(p1, p2)
                mutate_model(child_model)
                ca.agent.model.load_state_dict(child_model.state_dict())
                car.name = f"Car {i + 1}"
                
            reset_agent(ca)
            agent_list.append(ca)
    else:
        # Seed model yükle (varsa) — tüm popülasyon onun mutasyonlu kopyaları olur
        seed_weights = None
        if seed_model_path:
            try:
                seed_weights = torch.load(seed_model_path, weights_only=True)
                print(f"Seed model yüklendi: {seed_model_path}")
            except FileNotFoundError:
                print(f"Seed model bulunamadı ({seed_model_path}), sıfırdan başlanıyor.")

        for i, car in enumerate(car_list):
            ca = make_agent(car, track_lines, rays, epsilon=current_epsilon)
            reset_agent(ca)
            if seed_weights:
                ca.agent.model.load_state_dict(copy.deepcopy(seed_weights))
                mutate_model(ca.agent.model, mutation_rate=MUTATION_RATE)
            car.name = f"Car {i + 1}"
            agent_list.append(ca)

    return agent_list


def draw_debug_texts(window, font, best_score, generation, alive_count, time_val, epsilon):
    stats = [
        f"Best Score: {int(best_score)}",
        f"Generation: {generation}",
        f"Alive: {alive_count}",
        f"Time: {time_val / 60:.1f}s",
        f"Epsilon: {epsilon:.3f}",
    ]
    for i, text in enumerate(stats):
        rendered = font.render(text, True, (200, 200, 200))
        window.blit(rendered, (window.get_width() - 250, 10 + i * 18))


def update_distance(car_pos, road_points):
    closest_index = find_closest_point(car_pos, road_points)
    return calculate_distance_from_start(road_points, closest_index)


def main():
    pygame.init()
    win_size = (1000, 600)
    window = pygame.display.set_mode(win_size)
    clock = pygame.time.Clock()
    pygame.display.set_caption("Self Driving Car - Genetic Algorithm")

    road_image = pygame.image.load(TRACK_IMAGE).convert_alpha()
    threshold_image = apply_threshold(road_image)
    threshold_mask = pygame.mask.from_surface(threshold_image)

    mask_fx = pygame.mask.from_surface(pygame.transform.flip(threshold_image, True, False))
    mask_fy = pygame.mask.from_surface(pygame.transform.flip(threshold_image, False, True))
    mask_fx_fy = pygame.mask.from_surface(pygame.transform.flip(threshold_image, True, True))
    flipped_masks = [[threshold_mask, mask_fy], [mask_fx, mask_fx_fy]]

    beam_surface = pygame.Surface((200, 200), pygame.SRCALPHA)

    road_points = load_road_points(ROAD_POINTS_FILE)
    road_points = reorder_road_points(START_POS, road_points)[:-2]
    max_pts = len(road_points)  # bir tam turun road-point sayısı

    rays = create_rays(window)
    track_lines = TrackLines()

    # Pre-render road points once to a surface — avoid 250 draw.circle calls per frame
    road_surface = pygame.Surface(win_size, pygame.SRCALPHA)
    for point in road_points:
        pygame.draw.circle(road_surface, (220, 125, 160), point, 2)

    # Cache font — SysFont is expensive to create per frame
    font = pygame.font.SysFont(None, 24)

    best_score = 0
    best_model_weights = None  # en yüksek skoru yapan ajanın ağırlıkları
    generation = 0
    frame = 0
    time = 0
    run = True
    global_epsilon = 0.05 if SEED_MODEL else 0.1
    agent_list = regenerate_agents(car_list, track_lines, rays, seed_model_path=SEED_MODEL, current_epsilon=global_epsilon)
    # (fitness_score, nn.Sequential model) pairs collected each generation
    dead_agents: list[tuple[float, torch.nn.Sequential]] = []
    global_elites: list[tuple[float, torch.nn.Sequential]] = []

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    if best_model_weights is not None:
                        torch.save(best_model_weights, "models/best_model.pt")
                        print(f"[S] Model kaydedildi. En iyi skor: {best_score:.1f}")
                elif event.key == pygame.K_UP:
                    global_epsilon = min(1.0, global_epsilon + 0.05)
                    for ag in agent_list:
                        ag.agent.epsilon = global_epsilon
                    print(f"Epsilon artırıldı: {global_epsilon:.2f}")
                elif event.key == pygame.K_DOWN:
                    global_epsilon = max(0.0, global_epsilon - 0.05)
                    for ag in agent_list:
                        ag.agent.epsilon = global_epsilon
                    print(f"Epsilon azaltıldı: {global_epsilon:.2f}")
        if global_epsilon > 0.01:
            current_decay = EPSILON_DECAY if global_epsilon > EPSILON_SLOW_THRESHOLD else EPSILON_SLOW_DECAY
            global_epsilon = max(0.01, global_epsilon * current_decay)
            for ag in agent_list:
                ag.agent.epsilon = global_epsilon

        window.fill((0, 0, 0))
        window.blit(threshold_image, threshold_image.get_rect(center=window.get_rect().center))

        for agent in list(agent_list):
            car_pos = (agent.car.x, agent.car.y)

            # Draw rays from the correct point on the car's surface, not the center
            cos_ca = math.cos(math.radians(agent.car.angle))
            sin_ca = math.sin(math.radians(agent.car.angle))
            half_fwd = agent.car.rect.width / 2   # nose-to-tail half-length
            half_lat = agent.car.rect.height / 2  # side-to-side half-width
            for ray in rays:
                r = math.radians(ray.angle)
                lx = math.cos(r) * half_fwd  # forward component in car space
                ly = math.sin(r) * half_lat  # lateral component in car space
                origin = (
                    agent.car.x + lx * cos_ca - ly * sin_ca,
                    agent.car.y + lx * sin_ca + ly * cos_ca,
                )
                ray.draw_beam(origin, agent.car.angle, flipped_masks, beam_surface, threshold_mask)
                if ray.distance < ray.dangerous_distance:
                    agent.wall_proximity_penalty += WALL_PENALTY_SCALE * (ray.dangerous_distance - ray.distance)

            # Pixel collision with properly rotated mask
            rotated_car_img = pygame.transform.rotate(agent.car.car_image, -agent.car.angle)
            car_mask = pygame.mask.from_surface(rotated_car_img)
            rotated_rect = rotated_car_img.get_rect(center=(agent.car.x, agent.car.y))
            car_offset = (rotated_rect.left, rotated_rect.top)
            collision = threshold_mask.overlap(car_mask, car_offset)

            state = agent.env.get_state()
            action = agent.agent.act(state)

            # Jitter tespiti: art arda zıt dönüş (2↔3) → sayaç artır
            if action in (2, 3) and agent.prev_action in (2, 3) and action != agent.prev_action:
                agent.jitter_count += 1
            agent.prev_action = action

            # Yavaşlık tespiti
            if agent.car.speed < SPEED_TARGET:
                agent.slow_frames += 1

            distance_score = update_distance(car_pos, road_points)
            # Birikimli mesafe: tur sınırında sıfırlanmaz, fitness doğru kalır
            cumulative = agent.lap_count * max_pts + distance_score

            # Viraj bonusu: ileri giderken aktif dönüş yapıyorsa bonus ver
            angle_delta = abs(agent.car.angle - agent.prev_angle)
            if angle_delta > 180:
                angle_delta = 360 - angle_delta
            if angle_delta >= CORNER_ANGLE_THRESHOLD and cumulative > agent.env.score:
                agent.corner_bonus += CORNER_BONUS_SCALE
            agent.prev_angle = agent.car.angle

            # Per-ajan reward line bonusları — rect'ler silinmez, her ajan kendi flag'ini tutar
            car_rect = agent.car.car_image.get_rect(center=(agent.car.x, agent.car.y))
            for line_rect, idx, bonus in (
                (track_lines.reward_line_1_rect, 0,  50),
                (track_lines.reward_line_2_rect, 1, 100),
                (track_lines.reward_line_3_rect, 2, 200),
                (track_lines.reward_line_4_rect, 3, 200),
            ):
                if idx not in agent.collected_rewards and car_rect.colliderect(line_rect):
                    agent.collected_rewards.add(idx)
                    agent.env.reward += bonus

            # İlerleme takibi — cumulative kullanılır, tur geçişinde yanlış alarm vermez
            agent.stall_frames += 1
            if agent.stall_frames >= STALL_FRAMES:
                if cumulative - agent.last_checkpoint_score < PROGRESS_MIN:
                    collision = True  # stall → çarpışmış gibi işle
                else:
                    agent.last_checkpoint_score = cumulative
                    agent.stall_frames = 0

            # Gerçek fitness: mesafe + viraj bonusu - jitter - yavaşlık - duvar yakınlığı
            fitness = max(0.1, cumulative
                          + agent.corner_bonus
                          - agent.jitter_count        * JITTER_PENALTY
                          - agent.slow_frames         * SLOW_PENALTY
                          - agent.wall_proximity_penalty)

            if collision or time >= MAX_TIME:
                if fitness > best_score:
                    best_score = fitness
                    best_model_weights = copy.deepcopy(agent.agent.model.state_dict())
                    print(f"  New best: {best_score:.1f} (gen {generation}) — S ile kaydet")
                dead_agents.append((fitness, copy.deepcopy(agent.agent.model)))
                stop_dead_agent(agent)
                agent_list.remove(agent)
            else:
                progress = cumulative - agent.env.score
                agent.env.reward += progress * 0.6
                agent.env.score = cumulative
                agent.env.step(action)
                if agent.env.lap_flag:
                    agent.lap_count += 1
                # Hayattayken de best_score guncelle — olum bekleme
                if fitness > best_score:
                    best_score = fitness
                    best_model_weights = copy.deepcopy(agent.agent.model.state_dict())

        # Draw track markers
        pygame.draw.rect(window, (0, 255, 0), track_lines.start_line)
        pygame.draw.rect(window, (255, 0, 0), track_lines.mid_line)
        pygame.draw.rect(window, (0, 0, 255), track_lines.blue_line_rect)
        pygame.draw.rect(window, (120, 180, 120), track_lines.reward_line_1_rect)
        pygame.draw.rect(window, (120, 180, 120), track_lines.reward_line_2_rect)
        pygame.draw.rect(window, (120, 180, 120), track_lines.reward_line_3_rect)
        pygame.draw.rect(window, (120, 180, 120), track_lines.reward_line_4_rect)

        window.blit(road_surface, (0, 0))

        current_epsilon = agent_list[0].agent.epsilon if agent_list else 0.05
        draw_debug_texts(window, font, best_score, generation, len(agent_list), frame, current_epsilon)

        for car in car_list:
            car.draw(window)

        pygame.display.update()
        clock.tick(TICK_RATE)

        if len(agent_list) == 0:
            generation += 1
            print(f"\n=== Generation {generation} ===")
            track_lines = TrackLines()  # reset reward lines
            
            # --- ELITISM UPDATE ---
            combined_pool = global_elites + dead_agents
            combined_pool.sort(key=lambda x: x[0], reverse=True)
            # Keep only the all-time best N_PARENTS
            global_elites = combined_pool[:max(N_PARENTS, 2)]
            
            agent_list = regenerate_agents(car_list, track_lines, rays, scored_dead=global_elites, current_epsilon=global_epsilon)
            dead_agents = []
            time = 0

        frame += 1
        time += 10

    pygame.quit()


if __name__ == "__main__":
    main()
