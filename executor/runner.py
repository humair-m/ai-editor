# executor/runner.py

import subprocess
import tempfile
import os

def run_code(code: str) -> dict:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=10
        )

        return {
            "status": "success" if result.returncode == 0 else "error",
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "exit_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "stdout": "", "stderr": "Execution timed out", "exit_code": -1}
    except Exception as e:
        return {"status": "crash", "stdout": "", "stderr": str(e), "exit_code": -1}
    finally:
        os.remove(tmp_path)
