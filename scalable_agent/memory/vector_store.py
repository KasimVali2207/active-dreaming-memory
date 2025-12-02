import chromadb
from chromadb.config import Settings
import uuid
import os
from typing import List, Dict, Any, Optional

class VectorStore:
    def __init__(self, persist_path: str = "chroma_db"):
        """
        Initialize ChromaDB client.
        """
        self.client = chromadb.PersistentClient(path=persist_path)
        
        # Create or get collections
        self.episodic = self.client.get_or_create_collection(name="episodic_memory")
        self.semantic = self.client.get_or_create_collection(name="semantic_memory")
        
        print(f"[VectorStore] Initialized at {persist_path}")

    def add_episode(self, text: str, metadata: Dict[str, Any]):
        """
        Stores a raw episode trace.
        """
        self.episodic.add(
            documents=[text],
            metadatas=[metadata],
            ids=[str(uuid.uuid4())]
        )

    def add_rule(self, rule: str, metadata: Dict[str, Any]):
        """
        Stores a consolidated rule.
        """
        self.semantic.add(
            documents=[rule],
            metadatas=[metadata],
            ids=[str(uuid.uuid4())]
        )

    def query_episodic(self, query: str, n_results: int = 3, where: Optional[Dict] = None) -> List[Dict]:
        """
        Searches episodic memory.
        """
        results = self.episodic.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        return self._format_results(results)

    def query_semantic(self, query: str, n_results: int = 3) -> List[Dict]:
        """
        Searches semantic memory (rules).
        """
        results = self.semantic.query(
            query_texts=[query],
            n_results=n_results
        )
        return self._format_results(results)
    
    def _format_results(self, results) -> List[Dict]:
        """
        Helper to format ChromaDB results into a clean list of dicts.
        """
        formatted = []
        if not results['documents']:
            return []
            
        for i in range(len(results['documents'][0])):
            formatted.append({
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if results['distances'] else None
            })
        return formatted

    def clear(self):
        """
        Resets the database (for testing).
        """
        self.client.delete_collection("episodic_memory")
        self.client.delete_collection("semantic_memory")
        self.episodic = self.client.get_or_create_collection(name="episodic_memory")
        self.semantic = self.client.get_or_create_collection(name="semantic_memory")
