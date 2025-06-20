# test_debugger.py

from debugger.tracer import explain_error
from executor.runner import run_code

# Buggy code: divide by zero
code = """
def divide(a, b):
    return a / b

print(divide(10, 0))
"""

result = run_code(code)
print(f"\n[Raw Error]\n{result['stderr']}\n")

if result['status'] != "success":
    explanation = explain_error(code, result['stderr'], model="deepseek")
    print("\n[LLM Explanation]\n", explanation)
else:
    print("[No error found]")

