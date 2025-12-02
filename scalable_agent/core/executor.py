import subprocess
import sys
import os

class Executor:
    def execute(self, code: str, timeout: int = 10) -> dict:
        """
        Executes the provided Python code in a separate process.
        Returns a dict with 'success', 'output', and 'error'.
        """
        # Save code to a temp file
        temp_filename = f"temp_exec_{os.getpid()}.py"
        with open(temp_filename, "w", encoding="utf-8") as f:
            f.write(code)
            
        try:
            result = subprocess.run(
                [sys.executable, temp_filename],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            stdout = result.stdout
            stderr = result.stderr
            
            # Determine success based on return code and stderr
            success = (result.returncode == 0)
            
            return {
                "success": success,
                "output": stdout + stderr,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "Execution Timed Out",
                "return_code": -1
            }
        except Exception as e:
            return {
                "success": False,
                "output": str(e),
                "return_code": -1
            }
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
