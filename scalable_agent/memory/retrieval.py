from typing import List, Dict, Any
from .vector_store import VectorStore

class HybridRetriever:
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store

    def retrieve_context(self, query: str, error_type: str = None) -> str:
        """
        Implements the Hybrid Retrieval Operator H(q, M).
        1. Dense Search: Find semantically similar past episodes.
        2. Symbolic Filter: If 'error_type' is provided, filter by it.
        3. Rule Retrieval: Always fetch relevant semantic rules.
        """
        
        # 1. Retrieve Rules (Semantic Memory) - High Priority
        rules = self.store.query_semantic(query, n_results=2)
        
        # 2. Retrieve Episodes (Episodic Memory)
        where_filter = {"error_type": error_type} if error_type else None
        episodes = self.store.query_episodic(query, n_results=3, where=where_filter)
        
        # 3. Format Context
        context_parts = []
        
        if rules:
            context_parts.append("--- ESTABLISHED RULES ---")
            for r in rules:
                context_parts.append(f"RULE: {r['content']}")
                
        if episodes:
            context_parts.append("\n--- RELEVANT PAST EXPERIENCES ---")
            for e in episodes:
                meta = e['metadata']
                context_parts.append(f"EPISODE: {e['content']}\nOUTCOME: {meta.get('outcome', 'Unknown')}")
                
        return "\n".join(context_parts)
