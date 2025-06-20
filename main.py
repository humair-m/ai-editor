# main.py

import fire
from rich import print

# models/pollinations_api.py

import requests
import urllib.parse

POLLINATIONS_API = "https://text.pollinations.ai"

AVAILABLE_MODELS = {
    "deepseek": "deepseek-v3",
    "grok": "grok-3-mini",
    "qwen": "qwen2.5-coder-32b-instruct"
}

def call_model(model: str, prompt: str) -> str:
    model_id = AVAILABLE_MODELS.get(model.lower())
    if not model_id:
        raise ValueError(f"Unknown model '{model}'. Choose from: {list(AVAILABLE_MODELS.keys())}")
    
    encoded_prompt = urllib.parse.quote(prompt)
    url = f"{POLLINATIONS_API}/{encoded_prompt}?model={model_id}"

    try:
        res = requests.get(url, timeout=60)
        res.raise_for_status()
        return res.text.strip()
    except Exception as e:
        return f"[Error] Failed to call model: {str(e)}"



# ===================================================
from executor.runner import run_code
from debugger.tracer import explain_error
from tests_c.testgen import generate_tests

def read_file(file_path: str) -> str:
    with open(file_path, "r") as f:
        return f.read()

def write_file(file_path: str, content: str):
    with open(file_path, "w") as f:
        f.write(content)

def edit(file: str, model: str = "deepseek"):
    code = read_file(file)
    prompt = f"Improve or fix this Python code:\n\n{code}"
    result = call_model(model, prompt)
    print("\n[LLM Edited Code]\n")
    print(result)
    confirm = input("\n[?] Overwrite original file? (y/n): ")
    if confirm.lower() == "y":
        write_file(file, result)
        print("[✔] File updated.")

def run(file: str):
    code = read_file(file)
    result = run_code(code)
    print("\n[Run Result]")
    print(f"Status   : {result['status']}")
    print(f"Output   : {result['stdout']}")
    print(f"Error    : {result['stderr']}")
    print(f"Exit Code: {result['exit_code']}")

def debug(file: str, model: str = "deepseek"):
    code = read_file(file)
    result = run_code(code)
    if result["status"] == "success":
        print("\n✅ No errors to debug.")
    else:
        print(f"\n[Execution Error]\n{result['stderr']}")
        explanation = explain_error(code, result["stderr"], model)
        print("\n[LLM Debug Explanation]\n", explanation)

def test(file: str, model: str = "qwen"):
    code = read_file(file)
    tests = generate_tests(code, model)
    print("\n[Generated Test Cases]\n", tests)
    save = input("\n[?] Save to test_<name>.py file? (y/n): ")
    i

