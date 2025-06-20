# test_runner.py

from executor.runner import run_code

code = """
def divide(a, b):
    return a / b

print(divide(10, 2))
"""

result = run_code(code)
print("\n[Execution Result]")
print(f"Status   : {result['status']}")
print(f"Output   : {result['stdout']}")
print(f"Error    : {result['stderr']}")
print(f"Exit Code: {result['exit_code']}")

