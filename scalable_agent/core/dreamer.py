from openai import OpenAI
from typing import List, Dict
import numpy as np
from sklearn.cluster import DBSCAN
from ..memory.vector_store import VectorStore
from .executor import Executor

class Dreamer:
    def __init__(self, client: OpenAI, vector_store: VectorStore, executor: Executor, model: str = "llama-3.3-70b-versatile"):
        self.client = client
        self.store = vector_store
        self.executor = executor
        self.model = model
        self.epsilon = 0.3  # DBSCAN radius (from paper)
        self.min_pts = 2    # Minimum cluster size (from paper)
        self.false_consolidations = 0
        self.total_consolidations = 0

    def dream(self):
        """
        The Core Consolidation Loop (Algorithm 1 from paper).
        1. Fetch recent failures and extract embeddings.
        2. Cluster failures using DBSCAN.
        3. For each cluster, abstract a candidate rule.
        4. Verify the rule through counterfactual dreaming.
        5. Consolidate verified rules to semantic memory.
        """
        print("\n[Dreamer] Entering REM Sleep (Active Dreaming Consolidation)...")
        
        # 1. Fetch recent failures with embeddings
        failures = self.store.query_episodic("failure error", n_results=20, where={"outcome": "FAILURE"})
        
        if not failures or len(failures) < self.min_pts:
            print(f"[Dreamer] Insufficient failures ({len(failures) if failures else 0}) for clustering (need ≥{self.min_pts}).")
            return
        
        print(f"[Dreamer] Analyzing {len(failures)} recent failures...")
        
        # 2. Extract embeddings and perform DBSCAN clustering
        embeddings = np.array([f['embedding'] for f in failures])
        clusters = self._cluster_failures(embeddings)
        
        unique_clusters = set(clusters) - {-1}  # Exclude noise (-1)
        print(f"[Dreamer] DBSCAN found {len(unique_clusters)} clusters (ε={self.epsilon}, minPts={self.min_pts})")
        
        if not unique_clusters:
            print("[Dreamer] No valid clusters found. Skipping consolidation.")
            return
        
        # 3-5. For each cluster: Abstract → Dream → Verify → Consolidate
        for cluster_id in unique_clusters:
            cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
            cluster_failures = [failures[i] for i in cluster_indices]
            
            print(f"\n[Dreamer] Processing Cluster {cluster_id} ({len(cluster_failures)} failures)...")
            
            # Abstract candidate rule
            rule_candidate = self._abstract_rule(cluster_failures)
            print(f"[Dreamer] Candidate Rule: {rule_candidate[:100]}...")
            
            # Generate counterfactual dream scenario
            dream_scenario = self._generate_dream_scenario(rule_candidate)
            print(f"[Dreamer] Dream Scenario: {dream_scenario[:100]}...")
            
            # Verify through execution
            verified = self._verify_rule(dream_scenario, rule_candidate)
            self.total_consolidations += 1
            
            if verified:
                # Consolidate to semantic memory
                self.store.add_rule(
                    rule_candidate, 
                    metadata={
                        "source": "dream_consolidation",
                        "verified": True,
                        "cluster_size": len(cluster_failures),
                        "cluster_id": int(cluster_id)
                    }
                )
                print("[Dreamer] ✓ Rule VERIFIED and consolidated to Semantic Memory.")
            else:
                self.false_consolidations += 1
                print("[Dreamer] ✗ Rule FAILED verification. Not consolidated.")
        
        # Report false consolidation rate
        if self.total_consolidations > 0:
            fcr = (self.false_consolidations / self.total_consolidations) * 100
            print(f"\n[Dreamer] False Consolidation Rate: {fcr:.1f}% ({self.false_consolidations}/{self.total_consolidations})")

    def _cluster_failures(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster failure embeddings using DBSCAN.
        Returns cluster labels (-1 for noise).
        """
        dbscan = DBSCAN(eps=self.epsilon, min_samples=self.min_pts, metric='cosine')
        clusters = dbscan.fit_predict(embeddings)
        return clusters

    def _abstract_rule(self, cluster_failures: List[Dict]) -> str:
        """
        Abstract a general rule from a cluster of similar failures.
        """
        context = "\n".join([f"- {f['content']}" for f in cluster_failures])
        prompt = f"""Analyze these similar failure logs and synthesize ONE generalizable rule.

Failures:
{context}

Format your rule as: "IF [condition] THEN [action] BECAUSE [insight]"

Rule:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5  # Paper specifies 0.5 for abstraction
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Dreamer] Error abstracting rule: {e}")
            return "Error abstracting rule."

    def _generate_dream_scenario(self, rule: str) -> str:
        """
        Generate a counterfactual test scenario (Active Dreaming).
        The scenario should be DIFFERENT from the original failures.
        """
        prompt = f"""You are a scenario generator for testing coding rules.

Rule to test: "{rule}"

Generate a NOVEL Python code scenario where:
1. Ignoring this rule would cause a failure
2. Following this rule leads to success
3. The scenario is DIFFERENT from typical examples

Return ONLY executable Python code (no markdown, no explanations).

Code:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8  # Paper specifies 0.8 for dreaming
            )
            code = response.choices[0].message.content.strip()
            # Clean markdown if present
            code = code.replace("```python", "").replace("```", "").strip()
            return code
        except Exception as e:
            print(f"[Dreamer] Error generating dream: {e}")
            return "print('Error generating dream')"

    def _verify_rule(self, dream_code: str, rule: str) -> bool:
        """
        Verify the rule by executing the dream scenario.
        Returns True if the scenario executes successfully (rule is valid).
        """
        print("[Dreamer] Executing dream scenario for verification...")
        result = self.executor.execute(dream_code)
        
        if result['success']:
            print("[Dreamer] Dream execution: SUCCESS ✓")
            return True
        else:
            print(f"[Dreamer] Dream execution: FAILED ✗ ({result['output'][:50]}...)")
            return False

    def get_consolidation_stats(self) -> Dict:
        """Return consolidation statistics."""
        return {
            "total_consolidations": self.total_consolidations,
            "false_consolidations": self.false_consolidations,
            "false_consolidation_rate": (self.false_consolidations / self.total_consolidations * 100) if self.total_consolidations > 0 else 0
        }
