from .agent import LifelongAgent

class LifelongAgentAdapter:
    """
    Adapter to make LifelongAgent compatible with standard benchmark interfaces.
    """
    def __init__(self):
        self.agent = LifelongAgent(enable_sleep=True, enable_symbolic=True)
        
    def reset(self):
        """
        Resets the agent's short-term state but KEEPS long-term memory.
        """
        # In a real benchmark, we might clear context window here.
        pass
        
    def step(self, observation: str) -> str:
        """
        Receives an observation (task) and returns an action (code).
        """
        # For the benchmark, we want to return the GENERATED CODE, not execute it immediately
        # (The benchmark harness usually handles execution).
        
        # 1. Retrieve
        context = self.agent.retriever.retrieve_context(observation)
        
        # 2. Generate
        code = self.agent._generate_code(observation, context)
        
        return code

    def learn(self, task: str, code: str, result: str, success: bool):
        """
        Explicit feedback channel for the benchmark to tell the agent the result.
        This allows the agent to reflect and consolidate.
        """
        if not success:
            insight = self.agent.reflector.reflect(task, code, result)
            self.agent.store.add_episode(
                text=f"Task: {task}\nCode: {code}\nOutput: {result}",
                metadata={
                    "outcome": "FAILURE",
                    "insight": insight,
                    "error_type": "BenchmarkError"
                }
            )
            self.agent.sleep() # Consolidate
        else:
            self.agent.store.add_episode(
                text=f"Task: {task}\nCode: {code}",
                metadata={"outcome": "SUCCESS"}
            )
