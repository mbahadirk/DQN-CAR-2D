import numpy as np

from track_lines import handle_collision_with_lines


class CarEnvironment:
    def __init__(self, car, track_lines, reset_function,rays,score):
        self.car = car
        self.track_lines = track_lines
        self.reset_function = reset_function
        self.state_size = 15  # [ışınlar*9, hız,score]
        self.action_size = 8  # [ileri, geri, sağ, sol, sağ ileri, sol ileri, sağ geri, sol geri, hiçbiri]
        self.pass_startline = False
        self.lap_flag = False
        self.rays = rays
        self.score = score
        self.reward = -1

    def reset(self):
        self.reset_function()
        return self.get_state()

    def get_state(self):
        ray_distances = [ray.distance/500 for ray in self.rays]
        # ray_distances = np.array(ray_distances)
        # ray_distances = ray_distances / np.max(ray_distances)
        # normalized_angle = self.car.angle / 360.0  # 0-1 aralığında
        normalized_speed = self.car.speed / self.car.max_speed
        return np.array([
            *ray_distances, normalized_speed, self.score])

    def step(self, action):
        # Arabayı güncelle
        self.car.update(action)  # Bu metodu `Car` sınıfına uygun şekilde değiştir

        # Çizgi çarpışmalarını kontrol et
        self.pass_startline, self.lap_flag = handle_collision_with_lines(
            self.car,
            self.track_lines.start_line_rect,
            self.track_lines.mid_line_rect,
            self.track_lines.blue_line_rect,
            self.pass_startline
        )

        # self.reward = self.score*2
        # Ödül hesapla
        if self.lap_flag:
            self.reward += 1000 # Tur tamamlandığında ödül

        # Oyunun bitip bitmediğini kontrol et
        done = False

        # Yeni durum döndür
        state = self.get_state()
        return state, self.reward, done
