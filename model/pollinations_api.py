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



