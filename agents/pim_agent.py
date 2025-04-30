from stable_baselines3 import PPO
from env import PIMEnv


class PIMAgent:
    def __init__(self, model_path="models/pim_pam_poum_model"):
        # Load the trained model
        self.env = PIMEnv()
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