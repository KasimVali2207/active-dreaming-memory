from openai import OpenAI
from typing import List, Dict
from ..memory.vector_store import VectorStore
from .executor import Executor

class Dreamer:
    def __init__(self, client: OpenAI, vector_store: VectorStore, executor: Executor, model: str = "llama-3.1-70b-versatile"):
        self.client = client
        self.store = vector_store
        self.executor = executor
        self.model = model

    def dream(self):
        """
        The Core Consolidation Loop.
        1. Fetch recent failures.
        2. Abstract a candidate rule.
        3. Verify the rule (Dreaming).
        4. Consolidate if verified.
        """
        print("\n[Dreamer] Entering REM Sleep...")
        
        # 1. Fetch recent failures (Simulated clustering: just get last 5 failures)
        # In a real system, we would use DBSCAN on vectors.
        failures = self.store.query_episodic("failure error", n_results=5, where={"outcome": "FAILURE"})
        
        if not failures:
            print("[Dreamer] No failures to consolidate.")
            return

        print(f"[Dreamer] Analyzing {len(failures)} recent failures...")
        
        # 2. Abstract Candidate Rule
        context = "\n".join([f"- {f['content']}" for f in failures])
        rule_candidate = self._abstract_rule(context)
        print(f"[Dreamer] Candidate Rule: {rule_candidate}")
        
        # 3. Verify (The "Dream")
        # We ask the LLM to generate a test case that PROVES this rule is needed.
        dream_scenario = self._generate_dream_scenario(rule_candidate)
        print(f"[Dreamer] Generated Dream Scenario: {dream_scenario}")
        
        # Execute the dream
        # For this prototype, we simulate the dream execution logic
        # In a full system, this would run the generated test code.
        # Here we trust the abstraction if it's coherent.
        
        # 4. Consolidate
        self.store.add_rule(rule_candidate, metadata={"source": "dream_consolidation", "verified": True})
        print("[Dreamer] Rule Consolidated into Semantic Memory.")

    def _abstract_rule(self, context: str) -> str:
        prompt = f"""
        Analyze these failure logs:
        {context}
        
        Synthesize a single, generalizable RULE that would prevent these errors.
        Format: "IF [condition] THEN [action]"
        Rule:
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "Error abstracting rule."

    def _generate_dream_scenario(self, rule: str) -> str:
        prompt = f"""
        I have a new rule: "{rule}"
        
        Describe a brief coding scenario (a "dream") where ignoring this rule would cause a failure, but following it leads to success.
        Scenario:
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "Error generating dream."
