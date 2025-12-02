import threading
import time
import sys
import os
from sprint.server import run_server
from scalable_agent.agent import LifelongAgent

# Ensure we can import modules from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def start_server():
    run_server(port=8082) # Use port 8082 for ablation

def run_trial(condition_name: str, enable_sleep: bool, enable_symbolic: bool) -> bool:
    print(f"\n\n>>> STARTING CONDITION: {condition_name} <<<")
    print(f"Flags: Sleep={enable_sleep}, Symbolic={enable_symbolic}")
    
    agent = LifelongAgent(enable_sleep=enable_sleep, enable_symbolic=enable_symbolic)
    agent.store.clear() # Start fresh for each condition
    
    task = "Write a Python script to fetch data from http://localhost:8082. Print the response."
    
    print("\n--- RUN 1 (Naive) ---")
    success_1 = agent.run_task(task)
    
    # Simulate time passing for consolidation (if enabled)
    if enable_sleep:
        print("Sleeping...")
        time.sleep(2) 
    
    print("\n--- RUN 2 (Experienced) ---")
    success_2 = agent.run_task(task)
    
    result = "SUCCESS" if success_2 else "FAILURE"
    print(f"\n>>> CONDITION {condition_name} RESULT: {result}")
    return success_2

def main():
    print("Starting Phase 7: Ablation Studies...")
    
    # 1. Start Server
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(1)
    
    results = {}
    
    # 2. Run Conditions
    
    # Condition A: No Sleep (Baseline)
    # Expectation: Run 2 fails because rule was never consolidated.
    results["No Sleep"] = run_trial("No Sleep", enable_sleep=False, enable_symbolic=True)
    
    # Condition B: No Symbolic (Ablation)
    # Expectation: Run 2 might fail if dense retrieval is noisy, or succeed if lucky.
    # In this simple prototype, it might still succeed, but we test the mechanism.
    results["No Symbolic"] = run_trial("No Symbolic", enable_sleep=True, enable_symbolic=False)
    
    # Condition C: Full System
    # Expectation: Run 2 Succeeds.
    results["Full System"] = run_trial("Full System", enable_sleep=True, enable_symbolic=True)
    
    # 3. Report
    print("\n\n=== ABLATION STUDY RESULTS ===")
    print("| Condition | Run 2 Outcome |")
    print("| :--- | :--- |")
    for cond, outcome in results.items():
        status = "SUCCESS" if outcome else "FAILURE"
        print(f"| {cond} | {status} |")

if __name__ == "__main__":
    main()
