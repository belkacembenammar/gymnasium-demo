from stable_baselines3 import PPO
from pim_env import PimEnv

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
