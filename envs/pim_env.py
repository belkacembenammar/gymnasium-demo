import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Define the same environment class so we can load the model
class PimEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Environment parameters
        self.squads = ['PAS', 'CHG', 'BIL', 'PLA', 'COR']
        self.ticket_valid = True

        # The state now has 3 elements:
        # 1. If the ticket Jira is valid (1 = valid, 0 = invalid)
        # 2. If the developer is part of an authorized squad (1 = yes, 0 = no)
        # 3. If the developer already has a pending request (1 = yes, 0 = no)
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        self.ticket_valid = np.random.choice([0, 1])
        self.developer_squad = np.random.choice(self.squads)
        self.developer_in_squad = 1 if self.developer_squad in self.squads else 0
        self.request_pending = np.random.choice([0, 1])
        
        return np.array([self.ticket_valid, self.developer_in_squad, self.request_pending], dtype=np.float32), {}

    def step(self, action):
        if action == 0:  # Accept
            if self.ticket_valid == 1 and self.developer_in_squad == 1 and self.request_pending == 0:
                reward = 100 # Correct to accept valid requests
                self.request_pending = 1
            else:
                reward = -10 # Penalty for accepting invalid requests or when there's already a pending request
        else:  # Reject
            if self.ticket_valid == 1 and self.developer_in_squad == 1 and self.request_pending == 0:
                reward = -100 # Penalty for rejecting a valid request
            else:
                reward = 50  # Correct to reject invalid requests or when there's already a pending request
        
        self.done = False
        return np.array([self.ticket_valid, self.developer_in_squad, self.request_pending], dtype=np.float32), reward, self.done, False, {}

    def render(self):
        print(f"Ticket valid: {self.ticket_valid}")
        print(f"Authorized squad: {self.developer_in_squad}")
        print(f"Request pending: {self.request_pending}")
