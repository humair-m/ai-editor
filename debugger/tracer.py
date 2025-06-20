# debugger/tracer.py

import ast
import traceback
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from models.pollinations_api import PollinationsAPI

class ErrorType(Enum):
    SYNTAX_ERROR = "syntax_error"
    NAME_ERROR = "name_error"
    TYPE_ERROR = "type_error"
    VALUE_ERROR = "value_error"
    INDEX_ERROR = "index_error"
    KEY_ERROR = "key_error"
    ATTRIBUTE_ERROR = "attribute_error"
    IMPORT_ERROR = "import_error"
    LOGIC_ERROR = "logic_error"
    RUNTIME_ERROR = "runtime_error"
    UNKNOWN = "unknown"

@dataclass
class ErrorAnalysis:
    error_type: ErrorType
    line_number: Optional[int]
    column_number: Optional[int]
    error_message: str
    suggested_fixes: List[str]
    code_context: str
    severity: str  # "low", "medium", "high", "critical"
    confidence: float  # 0.0 to 1.0

@dataclass
class DebugResult:
    analysis: ErrorAnalysis
    ai_explanation: str
    fixed_code: Optional[str]
    prevention_tips: List[str]
    related_documentation: List[str]

class SmartDebugger:
    def __init__(self, api_client: Optional[PollinationsAPI] = None):
        self.api_client = api_client or PollinationsAPI()
        self.error_patterns = self._initialize_patterns()
        self.debug_history: List[Dict[str, Any]] = []
    
    def _initialize_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize common error patterns and their solutions"""
        return {
            r"ZeroDivisionError": {
                "type": ErrorType.VALUE_ERROR,
                "severity": "high",
                "common_fixes": [
                    "Add a check: if divisor != 0:",
                    "Use try-except block",
                    "Validate input before division"
                ]
            },
            r"NameError: name '(\w+)' is not defined": {
                "type": ErrorType.NAME_ERROR,
                "severity": "medium",
                "common_fixes": [
                    "Check variable spelling",
                    "Ensure variable is defined before use",
                    "Check import statements"
                ]
            },
            r"TypeError: unsupported operand type": {
                "type": ErrorType.TYPE_ERROR,
                "severity": "medium",
                "common_fixes": [
                    "Check data types of operands",
                    "Add type conversion",
                    "Validate input types"
                ]
            },
            r"IndentationError": {
                "type": ErrorType.SYNTAX_ERROR,
                "severity": "high",
                "common_fixes": [
                    "Fix indentation levels",
                    "Use consistent spaces or tabs",
                    "Check block structure"
                ]
            },
            r"IndexError: list index out of range": {
                "type": ErrorType.INDEX_ERROR,
                "severity": "medium",
                "common_fixes": [
                    "Check list length before accessing",
                    "Use len() to validate index",
                    "Add bounds checking"
                ]
            }
        }
    
    def analyze_error(self, code: str, error_message: str) -> ErrorAnalysis:
        """Analyze error and provide structured information"""
        
        # Extract line number from error
        line_match = re.search(r'line (\d+)', error_message)
        line_number = int(line_match.group(1)) if line_match else None
        
        # Determine error type
        error_type = ErrorType.UNKNOWN
        suggested_fixes = []
        severity = "medium"
        confidence = 0.5
        
        for pattern, info in self.error_patterns.items():
            if re.search(pattern, error_message, re.IGNORECASE):
                error_type = info["type"]
                suggested_fixes = info["common_fixes"]
                severity = info["severity"]
                confidence = 0.8
                break
        
        # Get code context
        lines = code.split('\n')
        if line_number and line_number <= len(lines):
            start = max(0, line_number - 3)
            end = min(len(lines), line_number + 2)
            context_lines = lines[start:end]
            code_context = '\n'.join(f"{i+start+1:3}: {line}" for i, line in enumerate(context_lines))
        else:
            code_context = code[:200] + "..." if len(code) > 200 else code
        
        return ErrorAnalysis(
            error_type=error_type,
            line_number=line_number,
            column_number=None,
            error_message=error_message,
            suggested_fixes=suggested_fixes,
            code_context=code_context,
            severity=severity,
            confidence=confidence
        )
    
    def get_ai_explanation(self, code: str, error_message: str, model: str = "deepseek") -> str:
        """Get AI-powered explanation of the error"""
        prompt = f"""
You are an expert Python debugger. Analyze this error and provide a clear, helpful explanation.

CODE:
```python
{code}
```

ERROR:
{error_message}

Provide:
1. What went wrong (in simple terms)
2. Why it happened
3. How to fix it (specific steps)
4. How to prevent similar errors

Be concise but thorough. Focus on actionable advice.
"""
        
        response = self.api_client.call_model(model, prompt)
        return response.content if response.success else f"AI explanation unavailable: {response.error}"
    
    def suggest_fix(self, code: str, error_message: str, model: str = "qwen") -> Optional[str]:
        """Generate fixed code using AI"""
        prompt = f"""
Fix this Python code that has an error. Return ONLY the corrected code, no explanations.

ORIGINAL CODE:
```python
{code}
```

ERROR:
{error_message}

FIXED CODE:
"""
        
        response = self.api_client.call_model(model, prompt)
        if response.success:
            # Extract code from response
            fixed_code = response.content
            # Remove markdown code blocks if present
            if "```python" in fixed_code:
                fixed_code = re.sub(r'```python\n(.*?)\n```', r'\1', fixed_code, flags=re.DOTALL)
            elif "```" in fixed_code:
                fixed_code = re.sub(r'```\n(.*?)\n```', r'\1', fixed_code, flags=re.DOTALL)
            return fixed_code.strip()
        return None
    
    def get_prevention_tips(self, error_type: ErrorType) -> List[str]:
        """Get tips to prevent similar errors in the future"""
        tips_map = {
            ErrorType.NAME_ERROR: [
                "Use descriptive variable names",
                "Initialize variables before use",
                "Use IDEs with syntax highlighting",
                "Follow consistent naming conventions"
            ],
            ErrorType.TYPE_ERROR: [
                "Use type hints in function definitions",
                "Validate input types early",
                "Use isinstance() for type checking",
                "Consider using dataclasses for structured data"
            ],
            ErrorType.INDEX_ERROR: [
                "Always check array bounds",
                "Use enumerate() when you need indices",
                "Consider using get() method for dictionaries",
                "Validate input ranges"
            ],
            ErrorType.VALUE_ERROR: [
                "Validate inputs at function entry",
                "Use defensive programming practices",
                "Add meaningful error messages",
                "Consider edge cases in your logic"
            ],
            ErrorType.SYNTAX_ERROR: [
                "Use consistent indentation",
                "Use a code formatter (black, autopep8)",
                "Enable syntax checking in your editor",
                "Review code before running"
            ]
        }
        
        return tips_map.get(error_type, [
            "Write unit tests for your functions",
            "Use version control to track changes",
            "Read error messages carefully",
            "Use debugging tools and print statements"
        ])
    
    def debug(self, code: str, stderr: str, model: str = "deepseek") -> DebugResult:
        """Complete debugging analysis"""
        if not stderr.strip():
            return DebugResult(
                analysis=ErrorAnalysis(
                    error_type=ErrorType.UNKNOWN,
                    line_number=None,
                    column_number=None,
                    error_message="No error detected",
                    suggested_fixes=[],
                    code_context="",
                    severity="low",
                    confidence=1.0
                ),
                ai_explanation="No error to debug - code executed successfully!",
                fixed_code=None,
                prevention_tips=[],
                related_documentation=[]
            )
        
        # Analyze the error
        analysis = self.analyze_error(code, stderr)
        
        # Get AI explanation
        ai_explanation = self.get_ai_explanation(code, stderr, model)
        
        # Suggest fix
        fixed_code = self.suggest_fix(code, stderr, model)
        
        # Get prevention tips
        prevention_tips = self.get_prevention_tips(analysis.error_type)
        
        # Related documentation (placeholder - could be enhanced)
        related_docs = [
            "https://docs.python.org/3/tutorial/errors.html",
            "https://docs.python.org/3/library/exceptions.html"
        ]
        
        result = DebugResult(
            analysis=analysis,
            ai_explanation=ai_explanation,
            fixed_code=fixed_code,
            prevention_tips=prevention_tips,
            related_documentation=related_docs
        )
        
        # Log for analytics
        self.debug_history.append({
            "timestamp": __import__("time").time(),
            "error_type": analysis.error_type.value,
            "severity": analysis.severity,
            "confidence": analysis.confidence,
            "model_used": model,
            "fix_provided": fixed_code is not None
        })
        
        return result
    
    def get_debug_stats(self) -> Dict[str, Any]:
        """Get debugging statistics"""
        if not self.debug_history:
            return {"total_debugs": 0}
        
        error_types = {}
        for entry in self.debug_history:
            error_type = entry["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_debugs": len(self.debug_history),
            "error_type_breakdown": error_types,
            "average_confidence": sum(entry["confidence"] for entry in self.debug_history) / len(self.debug_history),
            "fixes_provided": sum(1 for entry in self.debug_history if entry["fix_provided"]),
            "most_common_error": max(error_types, key=error_types.get) if error_types else None
        }

# Global instance
debugger = SmartDebugger()

def explain_error(code: str, stderr: str, model: str = "deepseek") -> str:
    """Legacy function for backward compatibility"""
    result = debugger.debug(code, stderr, model)
    return result.ai_explanation
