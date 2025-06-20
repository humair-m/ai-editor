# models/pollinations_api.py

import requests
import urllib.parse
import json
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

POLLINATIONS_API = "https://text.pollinations.ai"

class ModelType(Enum):
    DEEPSEEK = "deepseek-v3"
    GROK = "grok-3-mini"
    QWEN = "qwen2.5-coder-32b-instruct"
    CLAUDE = "claude-3-sonnet"
    GPT4 = "gpt-4"

AVAILABLE_MODELS = {
    "deepseek": ModelType.DEEPSEEK.value,
    "grok": ModelType.GROK.value,
    "qwen": ModelType.QWEN.value,
    "claude": ModelType.CLAUDE.value,
    "gpt4": ModelType.GPT4.value
}

@dataclass
class ModelResponse:
    content: str
    model: str
    success: bool
    error: Optional[str] = None
    response_time: float = 0.0
    tokens_used: Optional[int] = None

class PollinationsAPI:
    def __init__(self, timeout: int = 60, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.call_history: List[Dict[str, Any]] = []
    
    def call_model(self, model: str, prompt: str, temperature: float = 0.7, 
                   max_tokens: Optional[int] = None) -> ModelResponse:
        """Enhanced model calling with retry logic and response tracking"""
        model_id = AVAILABLE_MODELS.get(model.lower())
        if not model_id:
            return ModelResponse(
                content="", 
                model=model, 
                success=False, 
                error=f"Unknown model '{model}'. Choose from: {list(AVAILABLE_MODELS.keys())}"
            )
        
        start_time = time.time()
        encoded_prompt = urllib.parse.quote(prompt)
        
        # Build URL with parameters
        url = f"{POLLINATIONS_API}/{encoded_prompt}?model={model_id}"
        if temperature != 0.7:
            url += f"&temperature={temperature}"
        if max_tokens:
            url += f"&max_tokens={max_tokens}"
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                response_time = time.time() - start_time
                content = response.text.strip()
                
                # Track call history
                self.call_history.append({
                    "model": model,
                    "prompt_length": len(prompt),
                    "response_length": len(content),
                    "response_time": response_time,
                    "attempt": attempt + 1,
                    "success": True
                })
                
                return ModelResponse(
                    content=content,
                    model=model,
                    success=True,
                    response_time=response_time,
                    tokens_used=self._estimate_tokens(prompt + content)
                )
                
            except requests.exceptions.Timeout:
                last_error = f"Request timed out after {self.timeout}s"
            except requests.exceptions.RequestException as e:
                last_error = f"Request failed: {str(e)}"
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
            
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
        
        # All retries failed
        response_time = time.time() - start_time
        self.call_history.append({
            "model": model,
            "prompt_length": len(prompt),
            "response_length": 0,
            "response_time": response_time,
            "attempt": self.max_retries,
            "success": False,
            "error": last_error
        })
        
        return ModelResponse(
            content="",
            model=model,
            success=False,
            error=f"Failed after {self.max_retries} attempts: {last_error}",
            response_time=response_time
        )
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)"""
        return len(text) // 4
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about available models"""
        return {
            "deepseek": "DeepSeek V3 - Advanced reasoning and code generation",
            "grok": "Grok 3 Mini - Fast and efficient responses",
            "qwen": "Qwen 2.5 Coder - Specialized for programming tasks",
            "claude": "Claude 3 Sonnet - Balanced performance and safety",
            "gpt4": "GPT-4 - General purpose with strong reasoning"
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        if not self.call_history:
            return {"total_calls": 0}
        
        successful_calls = [call for call in self.call_history if call["success"]]
        failed_calls = [call for call in self.call_history if not call["success"]]
        
        return {
            "total_calls": len(self.call_history),
            "successful_calls": len(successful_calls),
            "failed_calls": len(failed_calls),
            "average_response_time": sum(call["response_time"] for call in successful_calls) / len(successful_calls) if successful_calls else 0,
            "total_tokens_estimated": sum(call.get("tokens_used", 0) for call in self.call_history),
            "models_used": list(set(call["model"] for call in self.call_history))
        }
    
    def clear_history(self):
        """Clear call history"""
        self.call_history.clear()

# Global instance for backward compatibility
api_client = PollinationsAPI()

def call_model(model: str, prompt: str) -> str:
    """Legacy function for backward compatibility"""
    response = api_client.call_model(model, prompt)
    if response.success:
        return response.content
    else:
        return f"[Error] {response.error}"
