"""
Baseline Agent Implementations for Comparative Evaluation

This module implements three baseline agents:
1. No Memory: Standard LLM without any memory
2. RAG Only: Simple retrieval-augmented generation without consolidation
3. ADM (Full): Our proposed Active Dreaming Memory system
"""

from openai import OpenAI
import json
import os
from typing import List, Dict, Optional

class NoMemoryAgent:
    """Baseline 1: Standard LLM without memory (stateless)"""
    
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY", "your-api-key-here")
        self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        self.model = "llama-3.3-70b-versatile"
        
    def run_task(self, task: str) -> bool:
        """Execute task without any memory"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a coding assistant. Generate only executable Python code."},
                    {"role": "user", "content": task}
                ],
                temperature=0.7
            )
            code = response.choices[0].message.content
            
            # Execute code (simplified)
            exec_globals = {}
            exec(code, exec_globals)
            return True
        except Exception as e:
            print(f"[NoMemory] Failed: {e}")
            return False


class RAGOnlyAgent:
    """Baseline 2: RAG without consolidation (stores raw episodes)"""
    
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY", "your-api-key-here")
        self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        self.model = "llama-3.3-70b-versatile"
        self.memory = []  # Simple list of past experiences
        
    def run_task(self, task: str) -> bool:
        """Execute task with simple RAG (no consolidation)"""
        # Retrieve similar past experiences (simple string matching)
        context = self._retrieve_context(task)
        
        # Generate code with context
        prompt = f"Task: {task}\n\nPast experiences:\n{context}\n\nGenerate Python code:"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a coding assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            code = response.choices[0].message.content
            
            # Execute
            exec_globals = {}
            exec(code, exec_globals)
            
            # Store raw experience
            self.memory.append({
                "task": task,
                "code": code,
                "success": True
            })
            return True
            
        except Exception as e:
            # Store failure
            self.memory.append({
                "task": task,
                "error": str(e),
                "success": False
            })
            print(f"[RAG] Failed: {e}")
            return False
    
    def _retrieve_context(self, task: str) -> str:
        """Simple keyword-based retrieval"""
        relevant = [m for m in self.memory if any(word in task.lower() for word in m.get("task", "").lower().split())]
        return "\n".join([f"- {m.get('task', 'N/A')}: {'Success' if m.get('success') else m.get('error', 'Failed')}" 
                         for m in relevant[:3]])


def run_comparative_benchmark():
    """Run all three agents on the same benchmark"""
    from scalable_agent.mock_benchmark import MockBenchmark
    from scalable_agent.adapter import LifelongAgentAdapter
    
    results = {}
    
    print("\n" + "="*60)
    print("COMPARATIVE BENCHMARK EVALUATION")
    print("="*60)
    
    # Baseline 1: No Memory
    print("\n[1/3] Running No Memory Agent...")
    no_mem = NoMemoryAgent()
    bench1 = MockBenchmark()
    # Note: MockBenchmark expects an adapter, so we'll run simplified version
    success_count = 0
    for task in bench1.tasks:
        if no_mem.run_task(task['prompt']):
            success_count += 1
    results['No Memory'] = success_count / len(bench1.tasks)
    print(f"Result: {results['No Memory']*100:.1f}%")
    
    # Baseline 2: RAG Only
    print("\n[2/3] Running RAG-Only Agent...")
    rag = RAGOnlyAgent()
    bench2 = MockBenchmark()
    success_count = 0
    for task in bench2.tasks:
        if rag.run_task(task['prompt']):
            success_count += 1
    results['RAG Only'] = success_count / len(bench2.tasks)
    print(f"Result: {results['RAG Only']*100:.1f}%")
    
    # Our System: ADM
    print("\n[3/3] Running ADM (Full System)...")
    adapter = LifelongAgentAdapter()
    bench3 = MockBenchmark()
    result = bench3.run_evaluation(adapter)
    results['ADM (Ours)'] = result['success_rate']
    
    # Print comparison table
    print("\n" + "="*60)
    print("COMPARATIVE RESULTS")
    print("="*60)
    print(f"{'Agent':<20} {'Success Rate':<15} {'Improvement':<15}")
    print("-"*60)
    
    baseline = results['No Memory']
    for agent, sr in results.items():
        improvement = f"+{(sr - baseline)*100:.1f}%" if sr > baseline else f"{(sr - baseline)*100:.1f}%"
        print(f"{agent:<20} {sr*100:.1f}%{'':<10} {improvement:<15}")
    
    return results


if __name__ == "__main__":
    results = run_comparative_benchmark()
    
    # Save results
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to baseline_results.json")
