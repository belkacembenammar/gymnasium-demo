# Example usage
from pim_agent import PIMAgent


if __name__ == "__main__":
    # Create the agent
    agent = PIMAgent()
    
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