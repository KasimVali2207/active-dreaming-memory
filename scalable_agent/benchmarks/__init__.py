"""
Multi-Domain Benchmark for Active Dreaming Memory
Implements the 60-task evaluation described in the research paper.

Domains: SQL, Python, API, Dialogue, Navigation, STEM (10 tasks each)
"""

import re
from typing import List, Dict, Tuple
from ..adapter import LifelongAgentAdapter

class MultiDomainBenchmark:
    def __init__(self):
        self.tasks = self._create_tasks()
    
    def _create_tasks(self) -> List[Dict]:
        """Create 60 tasks across 6 domains (10 per domain)."""
        tasks = []
        
        # SQL Domain (10 tasks)
        sql_tasks = [
            {
                "id": "sql_1",
                "domain": "SQL",
                "prompt": "Write a SQL query to select all users from 'users' table where age > 30",
                "expected_patterns": [r"SELECT.*FROM\s+users.*WHERE.*age\s*>\s*30"],
                "difficulty": "easy"
            },
            {
                "id": "sql_2",
                "domain": "SQL",
                "prompt": "Write a SQL query with INNER JOIN to get orders with customer names from 'orders' and 'customers' tables",
                "expected_patterns": [r"INNER\s+JOIN", r"JOIN.*customers", r"orders.*customers"],
                "difficulty": "medium"
            },
            {
                "id": "sql_3",
                "domain": "SQL",
                "prompt": "Write a SQL query to handle NULL values using IS NULL operator in 'products' table",
                "expected_patterns": [r"IS\s+NULL", r"IS\s+NOT\s+NULL"],
                "difficulty": "medium"
            },
            {
                "id": "sql_4",
                "domain": "SQL",
                "prompt": "Write a SQL query with GROUP BY and COUNT to count users per city",
                "expected_patterns": [r"GROUP\s+BY", r"COUNT\("],
                "difficulty": "medium"
            },
            {
                "id": "sql_5",
                "domain": "SQL",
                "prompt": "Write a SQL query with LEFT JOIN to include all customers even without orders",
                "expected_patterns": [r"LEFT\s+JOIN", r"LEFT\s+OUTER\s+JOIN"],
                "difficulty": "medium"
            },
            {
                "id": "sql_6",
                "domain": "SQL",
                "prompt": "Write a SQL query with HAVING clause to filter groups with count > 5",
                "expected_patterns": [r"HAVING", r"COUNT.*>\s*5"],
                "difficulty": "hard"
            },
            {
                "id": "sql_7",
                "domain": "SQL",
                "prompt": "Write a SQL query with subquery to find users with above-average salary",
                "expected_patterns": [r"SELECT.*FROM.*WHERE.*>\s*\(SELECT\s+AVG"],
                "difficulty": "hard"
            },
            {
                "id": "sql_8",
                "domain": "SQL",
                "prompt": "Write a SQL query with UNION to combine results from two tables",
                "expected_patterns": [r"UNION"],
                "difficulty": "medium"
            },
            {
                "id": "sql_9",
                "domain": "SQL",
                "prompt": "Write a SQL query with CASE statement for conditional logic",
                "expected_patterns": [r"CASE\s+WHEN", r"CASE.*WHEN.*THEN"],
                "difficulty": "hard"
            },
            {
                "id": "sql_10",
                "domain": "SQL",
                "prompt": "Write a SQL query with window function ROW_NUMBER() OVER",
                "expected_patterns": [r"ROW_NUMBER\(\)", r"OVER\s*\("],
                "difficulty": "hard"
            }
        ]
        
        # Python Domain (10 tasks)
        python_tasks = [
            {
                "id": "python_1",
                "domain": "Python",
                "prompt": "Write a Python function 'factorial' to calculate factorial of a number",
                "expected_patterns": [r"def\s+factorial", r"factorial\s*=\s*lambda"],
                "difficulty": "easy"
            },
            {
                "id": "python_2",
                "domain": "Python",
                "prompt": "Write a Python function to reverse a string",
                "expected_patterns": [r"def.*reverse", r"\[::-1\]", r"reversed\("],
                "difficulty": "easy"
            },
            {
                "id": "python_3",
                "domain": "Python",
                "prompt": "Write a Python function to check if a number is prime",
                "expected_patterns": [r"def.*prime", r"for.*in\s+range", r"%\s*==\s*0"],
                "difficulty": "medium"
            },
            {
                "id": "python_4",
                "domain": "Python",
                "prompt": "Write a Python function to implement binary search on a sorted list",
                "expected_patterns": [r"def.*binary", r"while.*left.*right", r"mid\s*="],
                "difficulty": "medium"
            },
            {
                "id": "python_5",
                "domain": "Python",
                "prompt": "Write a Python function to merge two sorted lists",
                "expected_patterns": [r"def.*merge", r"while.*and", r"append"],
                "difficulty": "medium"
            },
            {
                "id": "python_6",
                "domain": "Python",
                "prompt": "Write a Python function with try-except to handle KeyError when accessing dictionary",
                "expected_patterns": [r"try:", r"except\s+KeyError", r"\.get\("],
                "difficulty": "medium"
            },
            {
                "id": "python_7",
                "domain": "Python",
                "prompt": "Write a Python class with __init__ and __str__ methods",
                "expected_patterns": [r"class\s+\w+", r"def\s+__init__", r"def\s+__str__"],
                "difficulty": "medium"
            },
            {
                "id": "python_8",
                "domain": "Python",
                "prompt": "Write a Python generator function using yield",
                "expected_patterns": [r"def\s+\w+", r"yield"],
                "difficulty": "hard"
            },
            {
                "id": "python_9",
                "domain": "Python",
                "prompt": "Write a Python decorator function",
                "expected_patterns": [r"def\s+\w+\(func\)", r"def\s+wrapper", r"return\s+wrapper"],
                "difficulty": "hard"
            },
            {
                "id": "python_10",
                "domain": "Python",
                "prompt": "Write a Python function using list comprehension with conditional",
                "expected_patterns": [r"\[.*for.*in.*if.*\]"],
                "difficulty": "medium"
            }
        ]
        
        # API Domain (10 tasks)
        api_tasks = [
            {
                "id": "api_1",
                "domain": "API",
                "prompt": "Write Python code to make a GET request using requests library",
                "expected_patterns": [r"import\s+requests", r"requests\.get\("],
                "difficulty": "easy"
            },
            {
                "id": "api_2",
                "domain": "API",
                "prompt": "Write Python code to make a POST request with JSON data",
                "expected_patterns": [r"requests\.post\(", r"json\s*=", r"headers"],
                "difficulty": "medium"
            },
            {
                "id": "api_3",
                "domain": "API",
                "prompt": "Write Python code to handle HTTP authentication with headers",
                "expected_patterns": [r"headers\s*=", r"Authorization", r"Bearer|Token"],
                "difficulty": "medium"
            },
            {
                "id": "api_4",
                "domain": "API",
                "prompt": "Write Python code with try-except to handle requests.exceptions.RequestException",
                "expected_patterns": [r"try:", r"except.*RequestException", r"requests\."],
                "difficulty": "medium"
            },
            {
                "id": "api_5",
                "domain": "API",
                "prompt": "Write Python code to check response.status_code before processing",
                "expected_patterns": [r"status_code", r"==\s*200", r"if.*status"],
                "difficulty": "easy"
            },
            {
                "id": "api_6",
                "domain": "API",
                "prompt": "Write Python code to parse JSON response with response.json()",
                "expected_patterns": [r"\.json\(\)", r"response\.json"],
                "difficulty": "easy"
            },
            {
                "id": "api_7",
                "domain": "API",
                "prompt": "Write Python code to handle API rate limiting with retry logic",
                "expected_patterns": [r"while|for.*range", r"sleep\(", r"retry|attempt"],
                "difficulty": "hard"
            },
            {
                "id": "api_8",
                "domain": "API",
                "prompt": "Write Python code to make API request with query parameters",
                "expected_patterns": [r"params\s*=", r"requests\.get.*params"],
                "difficulty": "medium"
            },
            {
                "id": "api_9",
                "domain": "API",
                "prompt": "Write Python code to handle timeout in API requests",
                "expected_patterns": [r"timeout\s*=", r"requests\..*timeout"],
                "difficulty": "medium"
            },
            {
                "id": "api_10",
                "domain": "API",
                "prompt": "Write Python code to upload file using requests with files parameter",
                "expected_patterns": [r"files\s*=", r"open\(.*'rb'\)", r"requests\.post.*files"],
                "difficulty": "hard"
            }
        ]
        
        # Simplified Dialogue, Navigation, STEM tasks
        dialogue_tasks = [{"id": f"dialogue_{i+1}", "domain": "Dialogue", "prompt": f"Write a chatbot response for: {topic}", "expected_patterns": [r"def|class|response"], "difficulty": "medium"} 
                          for i, topic in enumerate(["greeting", "help request", "complaint", "question", "farewell", "confusion", "thanks", "apology", "feedback", "escalation"])]
        
        navigation_tasks = [{"id": f"nav_{i+1}", "domain": "Navigation", "prompt": f"Write pathfinding code for: {scenario}", "expected_patterns": [r"def|class|path|move"], "difficulty": "medium"}
                           for i, scenario in enumerate(["grid navigation", "obstacle avoidance", "shortest path", "A* algorithm", "BFS", "DFS", "dijkstra", "maze solving", "waypoint following", "collision detection"])]
        
        stem_tasks = [{"id": f"stem_{i+1}", "domain": "STEM", "prompt": f"Write code to solve: {problem}", "expected_patterns": [r"def|class|\*\*|math"], "difficulty": "hard"}
                     for i, problem in enumerate(["quadratic equation", "physics kinematics", "statistics mean/std", "matrix multiplication", "derivative calculation", "integral approximation", "probability calculation", "combinatorics", "geometry area", "trigonometry"])]
        
        tasks.extend(sql_tasks)
        tasks.extend(python_tasks)
        tasks.extend(api_tasks)
        tasks.extend(dialogue_tasks)
        tasks.extend(navigation_tasks)
        tasks.extend(stem_tasks)
        
        return tasks
    
    def run_evaluation(self, agent: LifelongAgentAdapter) -> Dict:
        """
        Run full 60-task evaluation.
        Returns comprehensive metrics.
        """
        print("\n" + "="*70)
        print("MULTI-DOMAIN BENCHMARK EVALUATION (60 Tasks)")
        print("="*70)
        
        results_by_domain = {}
        all_results = []
        
        for task in self.tasks:
            domain = task['domain']
            if domain not in results_by_domain:
                results_by_domain[domain] = {"success": 0, "total": 0}
            
            print(f"\n[{task['id']}] {task['prompt'][:60]}...")
            
            # Agent generates code
            code = agent.step(task['prompt'])
            
            # Evaluate
            success = any(re.search(pattern, code, re.IGNORECASE | re.DOTALL) 
                         for pattern in task['expected_patterns'])
            
            result_msg = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"[Result] {result_msg}")
            
            # Learn from result
            agent.learn(task['prompt'], code, result_msg, success)
            
            # Track results
            results_by_domain[domain]["total"] += 1
            if success:
                results_by_domain[domain]["success"] += 1
            
            all_results.append({
                "task_id": task['id'],
                "domain": domain,
                "success": success,
                "difficulty": task.get('difficulty', 'medium')
            })
        
        # Calculate metrics
        overall_success = sum(r["success"] for r in all_results)
        overall_total = len(all_results)
        overall_rate = (overall_success / overall_total) * 100
        
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        print(f"{'Domain':<15} {'Success Rate':<15} {'Tasks':<10}")
        print("-"*70)
        
        for domain, stats in results_by_domain.items():
            rate = (stats['success'] / stats['total']) * 100
            print(f"{domain:<15} {rate:>6.1f}%{'':<8} {stats['success']}/{stats['total']}")
        
        print("-"*70)
        print(f"{'OVERALL':<15} {overall_rate:>6.1f}%{'':<8} {overall_success}/{overall_total}")
        print("="*70)
        
        return {
            "overall_success_rate": overall_rate,
            "by_domain": results_by_domain,
            "all_results": all_results,
            "total_tasks": overall_total
        }


if __name__ == "__main__":
    from ..adapter import LifelongAgentAdapter
    
    adapter = LifelongAgentAdapter()
    benchmark = MultiDomainBenchmark()
    results = benchmark.run_evaluation(adapter)
    
    print(f"\n✓ Benchmark complete. Success rate: {results['overall_success_rate']:.1f}%")
