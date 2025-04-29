import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Définir l'environnement
class PIMAccessControlEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Paramètres de l'environnement
        self.squads = ['PAS', 'CHG', 'BIL', 'PLA', 'COR']  # Liste mise à jour des squads autorisées
        self.risk_threshold = 0.5  # Seuil de risque pour accepter une demande
        self.ticket_valid = True  # Supposons que le ticket Jira est valide si fourni
        self.max_requests_per_day = 5  # Maximum de demandes acceptées par jour pour un développeur

        # L'état est constitué de 4 éléments :
        # 1. Si le ticket Jira est valide (1 = valide, 0 = invalide)
        # 2. Si le développeur fait partie d'une squad autorisée (1 = oui, 0 = non)
        # 3. Le risque de la demande (valeur entre 0 et 1)
        # 4. Le nombre de demandes d'accès déjà faites aujourd'hui par le développeur
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        # Actions possibles : accepter (0) ou rejeter (1) la demande
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        # Initialisation aléatoire de l'état du système
        self.ticket_valid = np.random.choice([0, 1])  # 0 = Ticket invalide, 1 = Ticket valide
        self.developer_squad = np.random.choice(self.squads)  # Sélectionner aléatoirement une squad
        self.developer_in_squad = 1 if self.developer_squad in self.squads else 0  # 1 si dans une squad autorisée
        self.risk_level = np.random.uniform(0, 1)  # Risque entre 0 (faible) et 1 (élevé)
        self.requests_today = np.random.randint(0, self.max_requests_per_day + 1)  # Nombre de demandes d'accès aujourd'hui
        
        return np.array([self.ticket_valid, self.developer_in_squad, self.risk_level, self.requests_today], dtype=np.float32), {}

    def step(self, action):
        total_reward = 0
        if action == 0:  # Action : accepter
            # Accepter la demande si toutes les conditions sont respectées
            if self.ticket_valid == 1 and self.developer_in_squad == 1 and self.risk_level < self.risk_threshold and self.requests_today < self.max_requests_per_day:
                reward = 1
                self.requests_today += 1
            else:
                reward = -1
        else:  # Action : rejeter
            # Rejeter la demande si elle est invalide
            if self.ticket_valid == 1 and self.developer_in_squad == 1 and self.risk_level < self.risk_threshold:
                reward = 0.5
            else:
                reward = 0
        
        self.done = False  # On suppose qu'il n'y a pas de condition d'arrêt
        return np.array([self.ticket_valid, self.developer_in_squad, self.risk_level, self.requests_today], dtype=np.float32), reward, self.done, False, {}

    def render(self):
        print(f"Ticket valide : {self.ticket_valid}")
        print(f"Squad autorisée : {self.developer_in_squad}")
        print(f"Niveau de risque : {self.risk_level}")
        print(f"Demandes d'accès aujourd'hui : {self.requests_today}")


# Créer l'environnement
env = PIMAccessControlEnv()

# Vectoriser l'environnement
env = DummyVecEnv([lambda: env])

# Choisir l'algorithme PPO pour l'entraînement
model = PPO("MlpPolicy", env, verbose=1)

# Entraîner le modèle
model.learn(total_timesteps=100000)  # Nombre d'étapes à entraîner, tu peux augmenter ce nombre

# Sauvegarder le modèle entraîné
model.save("pim_pam_poum_model")

# Tester le modèle entraîné
obs = env.reset()
for _ in range(1000):  # Tester sur 1000 étapes
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()  # Affiche les résultats
    if dones:
        break
