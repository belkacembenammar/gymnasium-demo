import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
import random

# Define the same environment class so we can load the model
class PIMAccessControlEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Environment parameters
        self.squads = ['PAS', 'CHG', 'BIL', 'PLA', 'COR']
        self.risk_threshold = 0.5
        self.ticket_valid = True
        self.max_requests_per_day = 5

        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        self.ticket_valid = np.random.choice([0, 1])
        self.developer_squad = np.random.choice(self.squads)
        self.developer_in_squad = 1 if self.developer_squad in self.squads else 0
        self.risk_level = np.random.uniform(0, 1)
        self.requests_today = np.random.randint(0, self.max_requests_per_day + 1)
        
        return np.array([self.ticket_valid, self.developer_in_squad, self.risk_level, self.requests_today], dtype=np.float32), {}

    def step(self, action):
        if action == 0:  # Accept
            if self.ticket_valid == 1 and self.developer_in_squad == 1 and self.risk_level < self.risk_threshold and self.requests_today < self.max_requests_per_day:
                reward = 1
                self.requests_today += 1
            else:
                reward = -1
        else:  # Reject
            if self.ticket_valid == 1 and self.developer_in_squad == 1 and self.risk_level < self.risk_threshold:
                reward = 0.5
            else:
                reward = 0
        
        self.done = False
        return np.array([self.ticket_valid, self.developer_in_squad, self.risk_level, self.requests_today], dtype=np.float32), reward, self.done, False, {}

    def render(self):
        print(f"Ticket valid: {self.ticket_valid}")
        print(f"Authorized squad: {self.developer_in_squad}")
        print(f"Risk level: {self.risk_level:.2f}")
        print(f"Access requests today: {self.requests_today}")


# Create a PIM Access Control Agent class
class PIMAccessAgent:
    def __init__(self, model_path="models/pim_pam_poum_model"):
        # Load the trained model
        self.env = PIMAccessControlEnv()
        self.model = PPO.load(model_path)
        print(f"Model loaded from {model_path}")
        
    def process_request(self, ticket_valid, developer_squad, risk_level, requests_today):
        # Create observation based on request parameters
        developer_in_squad = 1 if developer_squad in self.env.squads else 0
        obs = np.array([ticket_valid, developer_in_squad, risk_level, requests_today], dtype=np.float32)
        
        # Get model prediction
        action, _states = self.model.predict(obs)
        
        # Convert action to decision
        decision = "ACCEPT" if action == 0 else "REJECT"
        
        # Print request details and decision
        print("\n--- PIM Access Request Details ---")
        print(f"Ticket valid: {'Yes' if ticket_valid == 1 else 'No'}")
        print(f"Developer squad: {developer_squad}")
        print(f"Risk level: {risk_level:.2f}")
        print(f"Previous requests today: {requests_today}")
        print(f"\nDECISION: {decision}")
        
        return decision, action


# Example usage
if __name__ == "__main__":
    # Create the agent
    agent = PIMAccessAgent()
    
    print("=== PIM Access Control Agent Demo ===\n")
    
    # Test with various scenarios
    test_cases = [
        # Typical valid request (should be ACCEPT)
        {"ticket_valid": 1, "developer_squad": "PAS", "risk_level": 0.2, "requests_today": 2},
        
        # Invalid ticket (should be REJECT)
        {"ticket_valid": 0, "developer_squad": "PAS", "risk_level": 0.2, "requests_today": 2},
        
        # Unauthorized squad (should be REJECT)
        {"ticket_valid": 1, "developer_squad": "EXT", "risk_level": 0.2, "requests_today": 2},
        
        # High risk (should be REJECT)
        {"ticket_valid": 1, "developer_squad": "PAS", "risk_level": 0.8, "requests_today": 2},
        
        # Too many requests (should be REJECT)
        {"ticket_valid": 1, "developer_squad": "PAS", "risk_level": 0.2, "requests_today": 5},
        
        # Edge case (borderline risk)
        {"ticket_valid": 1, "developer_squad": "BIL", "risk_level": 0.45, "requests_today": 3},
    ]
    
    # Process each test case
    for i, case in enumerate(test_cases):
        print(f"\n========= Test Case {i+1} =========")
        decision, action = agent.process_request(**case)
        
        # Explain expected behavior
        expected = "ACCEPT" if (
            case["ticket_valid"] == 1 and 
            case["developer_squad"] in ["PAS", "CHG", "BIL", "PLA", "COR"] and
            case["risk_level"] < 0.5 and
            case["requests_today"] < 5
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
            
            risk_level = float(input("Risk level (0.0-1.0): "))
            requests_today = int(input("Previous requests today (0-5): "))
            
            decision, action = agent.process_request(ticket_valid, developer_squad, risk_level, requests_today)
            
        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break
        except ValueError:
            print("Invalid input. Please try again.")