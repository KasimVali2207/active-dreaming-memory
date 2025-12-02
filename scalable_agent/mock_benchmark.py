import re
from typing import List, Dict
from .adapter import LifelongAgentAdapter

class MockBenchmark:
    """
    Simulates the LifelongAgentBench environment.
    Provides a sequence of tasks and evaluates the agent's response.
    """
    def __init__(self):
        self.tasks = [
            {
                "id": "task_sql_1",
                "prompt": "Write a SQL query to select all users from the 'users' table where age > 30. Return only the SQL code.",
                "expected_patterns": [r"SELECT.*FROM\s+users.*WHERE.*age\s*>\s*30", r"select.*from\s+users.*where.*age\s*>\s*30"],
                "error_msg": "SyntaxError: Missing WHERE clause."
            },
            {
                "id": "task_python_1",
                "prompt": "Write a Python function called 'factorial' to calculate the factorial of a number. Return only the Python code.",
                "expected_patterns": [r"def\s+factorial", r"factorial\s*=\s*lambda"],
                "error_msg": "NameError: function 'factorial' not defined"
            },
            {
                "id": "task_api_1",
                "prompt": "Write Python code to fetch data from https://api.example.com/data using the requests library. Return only the Python code.",
                "expected_patterns": [r"requests\.get", r"requests\.request", r"import\s+requests"],
                "error_msg": "ModuleNotFoundError: No module named 'requests'"
            }
        ]
        
    def run_evaluation(self, agent: LifelongAgentAdapter) -> Dict[str, float]:
        """
        Runs the agent on the task suite.
        Returns metrics (Success Rate).
        """
        print("\n=== Starting Mock Benchmark Evaluation ===")
        success_count = 0
        
        for i, task in enumerate(self.tasks):
            print(f"\n[Benchmark] Task {i+1}: {task['prompt']}")
            
            # 1. Agent Step
            code = agent.step(task['prompt'])
            print(f"[Agent] Action:\n{code[:200]}..." if len(code) > 200 else f"[Agent] Action:\n{code}")
            
            # 2. Evaluate (Simulated)
            # Check if any of the expected patterns match
            success = any(re.search(pattern, code, re.IGNORECASE | re.DOTALL) for pattern in task['expected_patterns'])
            
            result_msg = "Execution Successful" if success else task['error_msg']
            print(f"[Benchmark] Result: {result_msg}")
            
            # 3. Feedback Loop (Lifelong Learning)
            agent.learn(task['prompt'], code, result_msg, success)
            
            if success:
                success_count += 1
                
        sr = success_count / len(self.tasks)
        print(f"\n=== Evaluation Complete. Success Rate: {sr*100:.1f}% ===")
        return {"success_rate": sr}

if __name__ == "__main__":
    # Test the mock benchmark
    adapter = LifelongAgentAdapter()
    bench = MockBenchmark()
    bench.run_evaluation(adapter)

