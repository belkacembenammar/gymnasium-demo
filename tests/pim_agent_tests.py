import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import random
import os

from agents.pim_agent import PimAgent
from envs.pim_env import PimEnv

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