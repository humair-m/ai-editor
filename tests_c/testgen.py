# tests/testgen.py


# test_testgen.py
import sys
sys.path.insert(0, "/home/humair/models")  # Adjust path as needed



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



# ====================================================================================================================

def generate_tests(code: str, model="qwen") -> str:
    prompt = f"""
You're a Python test assistant. Given this function, generate relevant test cases using `assert` statements.

[Function]
{code}

Output only the test cases. Do not repeat the function.
"""
    return call_model(model, prompt)
