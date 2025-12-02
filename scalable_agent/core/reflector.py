from openai import OpenAI

class Reflector:
    def __init__(self, client: OpenAI, model: str = "llama-3.1-70b-versatile"):
        self.client = client
        self.model = model

    def reflect(self, task: str, code: str, output: str) -> str:
        """
        Analyzes a failure and produces a concise insight.
        """
        prompt = f"""
        I tried to execute a task and failed.
        
        Task: {task}
        Code:
        {code}
        
        Execution Output:
        {output}
        
        Analyze the error. What is the root cause? 
        Provide a concise, actionable 'Insight' that I can store in memory to avoid this mistake next time.
        Do not explain the code, just give the insight.
        Insight:
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Reflector] Error: {e}")
            return "Error reflecting on failure."
