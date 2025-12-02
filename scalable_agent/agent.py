import os
from openai import OpenAI
from .memory.vector_store import VectorStore
from .memory.retrieval import HybridRetriever
from .core.executor import Executor
from .core.reflector import Reflector
from .core.dreamer import Dreamer

class LifelongAgent:
    def __init__(self, enable_sleep: bool = True, enable_symbolic: bool = True):
        # Initialize API Client (Groq)
        api_key = os.getenv("GROQ_API_KEY", "your-api-key-here")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model = "llama-3.3-70b-versatile"
        
        self.enable_sleep = enable_sleep
        self.enable_symbolic = enable_symbolic

        # Initialize Components
        self.store = VectorStore(persist_path="chroma_db")
        self.retriever = HybridRetriever(self.store)
        self.executor = Executor()
        self.reflector = Reflector(self.client, self.model)
        self.dreamer = Dreamer(self.client, self.store, self.executor, self.model)
        
        print(f"[LifelongAgent] Online. Sleep={self.enable_sleep}, Symbolic={self.enable_symbolic}")

    def run_task(self, task: str):
        print(f"\n=== New Task: {task} ===")
        
        # 1. Wake Phase: Retrieve
        # If symbolic is disabled, we pass None for error_type to force dense-only retrieval
        error_type_filter = "RuntimeError" if self.enable_symbolic else None
        # Note: In a real system, error_type would be extracted from the task/context. 
        # Here we assume we are looking for similar errors.
        
        context = self.retriever.retrieve_context(task, error_type=error_type_filter)
        if context:
            print(f"[Agent] Retrieved Context:\n{context}")
        
        # 2. Wake Phase: Act (Generate Code)
        code = self._generate_code(task, context)
        print(f"[Agent] Generated Code:\n{code}")
        
        # 3. Wake Phase: Execute
        result = self.executor.execute(code)
        print(f"[Agent] Execution Result: {result['success']}")
        
        # 4. Wake Phase: Reflect & Store
        if not result['success']:
            print("[Agent] Failure Detected. Reflecting...")
            insight = self.reflector.reflect(task, code, result['output'])
            print(f"[Agent] Insight: {insight}")
            
            # Store Episode
            self.store.add_episode(
                text=f"Task: {task}\nCode: {code}\nOutput: {result['output']}",
                metadata={
                    "outcome": "FAILURE",
                    "insight": insight,
                    "error_type": "RuntimeError" # Simplified for prototype
                }
            )
            
            # Trigger Sleep/Consolidation
            if self.enable_sleep:
                self.sleep()
            else:
                print("[Agent] Sleep Disabled. Skipping consolidation.")
            return False
        else:
            print("[Agent] Success!")
            # Store Success Episode
            self.store.add_episode(
                text=f"Task: {task}\nCode: {code}",
                metadata={"outcome": "SUCCESS"}
            )
            return True

    def sleep(self):
        """
        Triggers the consolidation process.
        """
        self.dreamer.dream()

    def _generate_code(self, task: str, context: str) -> str:
        system_prompt = "You are a Python coding agent. Write a COMPLETE, RUNNABLE Python script. Print the final result. Do not use markdown blocks."
        user_prompt = f"Task: {task}\n\nContext: {context}\n\nWrite the code:"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0
            )
            code = response.choices[0].message.content
            code = code.replace("```python", "").replace("```", "").strip()
            return code
        except Exception as e:
            print(f"[Agent] Generation Error: {e}")
            return "print('Error')"
