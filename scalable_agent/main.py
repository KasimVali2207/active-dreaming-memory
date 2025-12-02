import threading
import time
from sprint.server import run_server
from scalable_agent.agent import LifelongAgent

def start_server():
    run_server(port=8081) # Use different port to avoid conflict

def main():
    print("Starting Phase 5: Scalable Architecture Prototype...")
    
    # 1. Start Server
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(1)
    
    # 2. Initialize Agent
    agent = LifelongAgent()
    agent.store.clear() # Start fresh
    
    task = "Write a Python script to fetch data from http://localhost:8081. Print the response."
    
    print("\n\n--- RUN 1 (Naive Agent) ---")
    success_1 = agent.run_task(task)
    
    print("\n\n--- RUN 2 (Experienced Agent) ---")
    # The agent should have consolidated the rule during the sleep phase of Run 1
    success_2 = agent.run_task(task)
    
    print("\n\n=== PROTOTYPE RESULTS ===")
    print(f"Run 1 Success: {success_1}")
    print(f"Run 2 Success: {success_2}")
    
    if not success_1 and success_2:
        print("\nSUCCESS: The Scalable Architecture successfully demonstrated Active Dreaming!")
    else:
        print("\nFAILURE: Something went wrong.")

if __name__ == "__main__":
    main()
