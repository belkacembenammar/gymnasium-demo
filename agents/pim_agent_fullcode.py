import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import random
import os

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

        # Actions: 0 = reject, 1 = accept
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        self.ticket_valid = np.random.choice([0, 1])
        self.developer_squad = np.random.choice(self.squads)
        self.developer_in_squad = 1 if self.developer_squad in self.squads else 0
        self.request_pending = np.random.choice([0, 1])
        
        return np.array([self.ticket_valid, self.developer_in_squad, self.request_pending], dtype=np.float32), {}

    def step(self, action):
        if action == 1:  # Accept
            if self.ticket_valid == 1 and self.developer_in_squad == 1 and self.request_pending == 0:
                reward = 200 # Correct to accept valid requests
            else:
                reward = -1500 # Penalty for accepting invalid requests or when there's already a pending request
        else:  # Reject
            if self.ticket_valid == 1 and self.developer_in_squad == 1 and self.request_pending == 0:
                reward = 200 # Penalty for rejecting a valid request
            else:
                reward = -2000  # Correct to reject invalid requests or when there's already a pending request
        
        self.done = False
        return np.array([self.ticket_valid, self.developer_in_squad, self.request_pending], dtype=np.float32), reward, self.done, False, {}

    def render(self):
        print(f"Ticket valid: {self.ticket_valid}")
        print(f"Authorized squad: {self.developer_in_squad}")
        print(f"Request pending: {self.request_pending}")


# Create a PIM Access Control Agent class
class PimAgent:
    def __init__(self, model_path="models/pim_pam_poum_model"):
        # Load the trained model
        self.env = PimEnv()
        self.model = PPO.load(model_path)
        print(f"Model loaded from {model_path}")
        
    def process_request(self, ticket_valid, developer_squad, request_pending):
        # Create observation based on request parameters
        developer_in_squad = 1 if developer_squad in self.env.squads else 0
        obs = np.array([ticket_valid, developer_in_squad, request_pending], dtype=np.float32)
        
        # Get model prediction
        action, _states = self.model.predict(obs)
        
        # Convert action to decision
        decision = "ACCEPT" if action == 0 else "REJECT"
        
        # Print request details and decision
        print("\n--- PIM Access Request Details ---")
        print(f"Ticket valid: {'Yes' if ticket_valid == 1 else 'No'}")
        print(f"Developer squad: {developer_squad}")
        print(f"Request already pending: {'Yes' if request_pending == 1 else 'No'}")
        print(f"\nDECISION: {decision}")
        
        return decision, action


# Example usage
if __name__ == "__main__":
    # Create models directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")
    
    model_path = "models/pim_pam_poum_model"
    
    # Ask if user wants to train a new model or use existing one
    train_new = input("Do you want to train a new model? (y/n): ").lower() == 'y'
    
    if train_new:
        print("\n=== Training New PIM Access Control Model ===")
        
        # Create and vectorize the environment for training
        env = PimEnv()
        vec_env = DummyVecEnv([lambda: env])
        
        # Ask for training parameters
        try:
            timesteps = int(input("Enter training timesteps (default: 100000): ") or "100000")
        except ValueError:
            timesteps = 100000
            print(f"Invalid input. Using default: {timesteps}")
        
        # Create and train the model
        model = PPO("MlpPolicy", vec_env, verbose=1)
        print(f"\nTraining for {timesteps} timesteps...")
        model.learn(total_timesteps=timesteps)
        
        # Save the trained model
        model.save(model_path)
        print(f"Model saved to {model_path}")
    
    # Create the agent with the model (either existing or newly trained)
    print("\n=== Creating PIM Access Control Agent ===")
    agent = PimAgent(model_path=model_path)
    
    print("\n=== PIM Access Control Agent Demo ===")
    
    # Test with various scenarios
    test_cases = [
        # Typical valid request (should be ACCEPT)
        {"ticket_valid": 1, "developer_squad": "PAS", "request_pending": 0},
        
        # Invalid ticket (should be REJECT)
        {"ticket_valid": 0, "developer_squad": "PAS", "request_pending": 0},
        
        # Unauthorized squad (should be REJECT)
        {"ticket_valid": 1, "developer_squad": "EXT", "request_pending": 0},
        
        # Already has pending request (should be REJECT)
        {"ticket_valid": 1, "developer_squad": "PAS", "request_pending": 1},
        
        # Edge case (valid but unusual squad)
        {"ticket_valid": 1, "developer_squad": "COR", "request_pending": 0},
    ]
    
    # Process each test case
    for i, case in enumerate(test_cases):
        print(f"\n========= Test Case {i+1} =========")
        decision, action = agent.process_request(**case)
        
        # Explain expected behavior
        expected = "ACCEPT" if (
            case["ticket_valid"] == 1 and 
            case["developer_squad"] in ["PAS", "CHG", "BIL", "PLA", "COR"] and
            case["request_pending"] == 0
        ) else "REJECT"
        
        if decision == expected:
            print(f"✅ Agent decision matches expected behavior ({expected})")
        else:
            print(f"❌ Agent decision ({decision}) differs from expected behavior ({expected})")
    
    print("\n=== Interactive Mode ===")
    print("Now you can test your own scenarios")
    
    # Interactive testing
    while True:
        try:
            print("\nEnter request details (or press Ctrl+C to quit):")
            ticket_valid = int(input("Ticket valid (1=Yes, 0=No): "))
            
            squad_list = ", ".join(agent.env.squads)
            developer_squad = input(f"Developer squad ({squad_list} or other): ")
            
            request_pending = int(input("Request already pending (1=Yes, 0=No): "))
            
            decision, action = agent.process_request(ticket_valid, developer_squad, request_pending)
            
        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break
        except ValueError:
            print("Invalid input. Please try again.")