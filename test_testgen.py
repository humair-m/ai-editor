# test_testgen.py
import sys
sys.path.insert(0, "/home/humair/models")  # Adjust path as needed


from tests_c.testgen import generate_tests


code = """
def divide(a, b):
    if b == 0:
        return None
    return a / b
"""

tests = generate_tests(code, model="qwen")
print("\n[Generated Test Cases]\n", tests)

