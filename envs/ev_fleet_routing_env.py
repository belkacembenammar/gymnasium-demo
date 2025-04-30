import gymnasium as gym
from gymnasium import spaces
import numpy as np

class EVFleetRoutingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # State: [battery_level, distance_to_next_stop, station_available (0/1)]
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0]), high=np.array([1.0, 1.0, 1]), dtype=np.float32)

        # Actions: 0 = continuer sans recharge, 1 = aller à une station de recharge
        self.action_space = spaces.Discrete(2)

        self.battery_level = 1.0  # plein au départ
        self.distance_to_next_stop = 0.5  # distance normalisée
        self.station_available = 1  # station disponible au départ

    def reset(self, seed=None, options=None):
        self.battery_level = 1.0
        self.distance_to_next_stop = np.random.uniform(0.2, 0.8)
        self.station_available = np.random.choice([0, 1])
        obs = np.array([self.battery_level, self.distance_to_next_stop, self.station_available], dtype=np.float32)
        return obs, {}

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        if action == 0:  # continuer
            energy_used = self.distance_to_next_stop * 0.7
            self.battery_level -= energy_used
            if self.battery_level <= 0:
                reward = -10  # panne
                terminated = True
            else:
                reward = 1  # progression
        elif action == 1:  # aller se recharger
            if self.station_available:
                self.battery_level = 1.0
                reward = -0.5  # perte de temps
            else:
                reward = -2  # déplacement inutile

        # Simuler prochaine étape
        self.distance_to_next_stop = np.random.uniform(0.2, 0.8)
        self.station_available = np.random.choice([0, 1])
        obs = np.array([self.battery_level, self.distance_to_next_stop, self.station_available], dtype=np.float32)
        return obs, reward, terminated, truncated, {}

    def render(self):
        print(f"🔋 Battery: {self.battery_level:.2f} | 📍 Dist: {self.distance_to_next_stop:.2f} | ⚡ Station: {self.station_available}")