# tests/testgen.py

import ast
import inspect
import re
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
from models.pollinations_api import PollinationsAPI

class TestType(Enum):
    UNIT = "unit"
    INTEGRATION = "integration"
    EDGE_CASE = "edge_case"
    PERFORMANCE = "performance"
    SECURITY = "security"

@dataclass
class TestCase:
    name: str
    description: str
    test_type: TestType
    input_params: Dict[str, Any]
    expected_output: Any
    expected_exception: Optional[str] = None
    test_code: str = ""
    priority: str = "medium"  # low, medium, high, critical

@dataclass
class TestSuite:
    function_name: str
    test_cases: List[TestCase]
    coverage_analysis: Dict[str, Any]
    performance_benchmarks: List[Dict[str, Any]]
    generated_code: str

class SmartTestGenerator:
    def __init__(self, api_client: Optional[PollinationsAPI] = None):
        self.api_client = api_client or PollinationsAPI()
        self.test_templates = self._initialize_templates()
        self.generation_history: List[Dict[str, Any]] = []
    
    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize test templates for different scenarios"""
        return {
            "basic_test": """
def test_{function_name}_{test_name}():
    # {description}
    result = {function_name}({input_params})
    assert result == {expected_output}, f"Expected {expected_output}, got {{result}}"
""",
            "exception_test": """
def test_{function_name}_{test_name}():
    # {description}
    with pytest.raises({exception_type}):
        {function_name}({input_params})
""",
            "parametrized_test": """
@pytest.mark.parametrize("input_params,expected", [
{test_parameters}
])
def test_{function_name}_parametrized(input_params, expected):
    # Parametrized test for {function_name}
    result = {function_name}(*input_params) if isinstance(input_params, tuple) else {function_name}(input_params)
    assert result == expected, f"Expected {{expected}}, got {{result}}"
""",
            "performance_test": """
import time

def test_{function_name}_performance():
    # Performance test for {function_name}
    start_time = time.time()
    for _ in range(1000):
        {function_name}({input_params})
    execution_time = time.time() - start_time
    assert execution_time < {max_time}, f"Function too slow: {{execution_time}}s"
"""
        }
    
    def analyze_function(self, code: str) -> Dict[str, Any]:
        """Analyze function to understand its structure and requirements"""
        try:
            tree = ast.parse(code)
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "returns": ast.get_source_segment(code, node.returns) if node.returns else None,
                        "docstring": ast.get_docstring(node),
                        "line_count": node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0,
                        "complexity": self._calculate_complexity(node),
                        "has_loops": self._has_loops(node),
                        "has_conditions": self._has_conditions(node),
                        "has_exceptions": self._has_exceptions(node)
                    }
                    functions.append(func_info)
            
            return {
                "functions": functions,
                "imports": self._extract_imports(tree),
                "classes": [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            }
        except Exception as e:
            return {"error": f"Failed to analyze code: {str(e)}"}
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    def _has_loops(self, node: ast.FunctionDef) -> bool:
        """Check if function contains loops"""
        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While, ast.AsyncFor)):
                return True
        return False
    
    def _has_conditions(self, node: ast.FunctionDef) -> bool:
        """Check if function contains conditional statements"""
        for child in ast.walk(node):
            if isinstance(child, ast.If):
                return True
        return False
    
    def _has_exceptions(self, node: ast.FunctionDef) -> bool:
        """Check if function handles exceptions"""
        for child in ast.walk(node):
            if isinstance(child, (ast.Try, ast.Raise, ast.ExceptHandler)):
                return True
        return False
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.extend([f"{module}.{alias.name}" for alias in node.names])
        return imports
    
    def generate_test_cases(self, func_info: Dict[str, Any]) -> List[TestCase]:
        """Generate test cases based on function analysis"""
        test_cases = []
        func_name = func_info["name"]
        args = func_info["args"]
        
        # Basic functionality test
        test_cases.append(TestCase(
            name="basic_functionality",
            description=f"Test basic functionality of {func_name}",
            test_type=TestType.UNIT,
            input_params={arg: self._generate_sample_value(arg) for arg in args},
            expected_output="expected_result",
            priority="high"
        ))
        
        # Edge cases
        if args:
            # Empty/None values
            test_cases.append(TestCase(
                name="empty_input",
                description=f"Test {func_name} with empty/None inputs",
                test_type=TestType.EDGE_CASE,
                input_params={args[0]: None} if args else {},
                expected_output="expected_result",
                priority="medium"
            ))
            
            # Boundary values
            test_cases.append(TestCase(
                name="boundary_values",
                description=f"Test {func_name} with boundary values",
                test_type=TestType.EDGE_CASE,
                input_params={arg: self._generate_boundary_value(arg) for arg in args},
                expected_output="expected_result",
                priority="high"
            ))
        
        # Type validation tests
        if args:
            test_cases.append(TestCase(
                name="invalid_type",
                description=f"Test {func_name} with invalid input types",
                test_type=TestType.EDGE_CASE,
                input_params={args[0]: "invalid_type"},
                expected_output=None,
                expected_exception="TypeError",
                priority="medium"
            ))
        
        # Performance test for complex functions
        if func_info.get("complexity", 0) > 5:
            test_cases.append(TestCase(
                name="performance",
                description=f"Test {func_name} performance",
                test_type=TestType.PERFORMANCE,
                input_params={arg: self._generate_large_value(arg) for arg in args},
                expected_output="performance_check",
                priority="low"
            ))
        
        return test_cases
    
    def _generate_sample_value(self, param_name: str) -> Any:
        """Generate sample values based on parameter name"""
        if "num" in param_name.lower() or "count" in param_name.lower():
            return 5
        elif "str" in param_name.lower() or "text" in param_name.lower():
            return "test_string"
        elif "list" in param_name.lower() or "arr" in param_name.lower():
            return [1, 2, 3]
        elif "dict" in param_name.lower():
            return {"key": "value"}
        elif "bool" in param_name.lower():
            return True
        else:
            return "sample_value"
    
    def _generate_boundary_value(self, param_name: str) -> Any:
        """Generate boundary values for testing"""
        if "num" in param_name.lower():
            return 0
        elif "list" in param_name.lower():
            return []
        elif "str" in param_name.lower():
            return ""
        else:
            return None
    
    def _generate_large_value(self, param_name: str) -> Any:
        """Generate large values for performance testing"""
        if "num" in param_name.lower():
            return 1000000
        elif "list" in param_name.lower():
            return list(range(1000))
        elif "str" in param_name.lower():
            return "x" * 1000
        else:
            return "large_value"
    
    def generate_ai_tests(self, code: str, model: str = "qwen") -> str:
        """Generate comprehensive tests using AI"""
        prompt = f"""
Generate comprehensive Python test cases for this function using pytest.

CODE:
```python
{code}
```

Generate tests that cover:
1. Normal functionality
2. Edge cases (empty inputs, None values, boundary conditions)
3. Error cases (invalid inputs, type errors)
4. Performance considerations if applicable

Return ONLY the test code with proper assertions. Use descriptive test names.
Include import statements needed.

FORMAT:
```python
import pytest
from your_module import function_name

# Your test cases here
```
"""
        
        response = self.api_client.call_model(model, prompt)
        if response.success:
            # Clean up the response
            test_code = response.content
            if "```python" in test_code:
                test_code = re.search(r'```python\n(.*?)\n```', test_code, re.DOTALL)
                if test_code:
                    test_code = test_code.group(1)
            elif "```" in test_code:
                test_code = re.search(r'```\n(.*?)\n```', test_code, re.DOTALL)
                if test_code:
                    test_code = test_code.group(1)
            
            return test_code.strip() if isinstance(test_code, str) else response.content
        
        return f"# Error generating tests: {response.error}"
    
    def generate_test_suite(self, code: str, model: str = "qwen") -> TestSuite:
        """Generate complete test suite with analysis"""
        analysis = self.analyze_function(code)
        
        if "error" in analysis:
            return TestSuite(
                function_name="unknown",
                test_cases=[],
                coverage_analysis={"error": analysis["error"]},
                performance_benchmarks=[],
                generated_code=f"# Error: {analysis['error']}"
            )
        
        if not analysis.get("functions"):
            return TestSuite(
                function_name="unknown",
                test_cases=[],
                coverage_analysis={"error": "No functions found in code"},
                performance_benchmarks=[],
                generated_code="# No functions found to test"
            )
        
        func_info = analysis["functions"][0]  # Take first function
        test_cases = self.generate_test_cases(func_info)
        
        # Generate AI-powered tests
        ai_tests = self.generate_ai_tests(code, model)
        
        # Coverage analysis
        coverage_analysis = {
            "function_complexity": func_info.get("complexity", 0),
            "has_loops": func_info.get("has_loops", False),
            "has_conditions": func_info.get("has_conditions", False),
            "has_exceptions": func_info.get("has_exceptions", False),
            "test_types_covered": list(set(tc.test_type.value for tc in test_cases)),
            "estimated_coverage": min(95, len(test_cases) * 15)  # Rough estimate
        }
        
        # Performance benchmarks (placeholder)
        performance_benchmarks = [
            {"metric": "execution_time", "target": "< 1ms", "priority": "medium"},
            {"metric": "memory_usage", "target": "< 1MB", "priority": "low"}
        ]
        
        # Log generation
        self.generation_history.append({
            "timestamp": __import__("time").time(),
            "function_name": func_info["name"],
            "test_count": len(test_cases),
            "complexity": func_info.get("complexity", 0),
            "model_used": model,
            "success": True
        })
        
        return TestSuite(
            function_name=func_info["name"],
            test_cases=test_cases,
            coverage_analysis=coverage_analysis,
            performance_benchmarks=performance_benchmarks,
            generated_code=ai_tests
        )
    
    def generate_mock_data(self, data_type: str, count: int = 10) -> List[Any]:
        """Generate mock data for testing"""
        if data_type == "integers":
            return [__import__("random").randint(1, 100) for _ in range(count)]
        elif data_type == "strings":
            import string
            import random
            return [''.join(random.choices(string.ascii_letters, k=8)) for _ in range(count)]
        elif data_type == "floats":
            return [__import__("random").uniform(0.1, 100.0) for _ in range(count)]
        elif data_type == "booleans":
            return [__import__("random").choice([True, False]) for _ in range(count)]
        else:
            return [f"mock_data_{i}" for i in range(count)]
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get test generation statistics"""
        if not self.generation_history:
            return {"total_generations": 0}
        
        successful = [h for h in self.generation_history if h.get("success", False)]
        
        return {
            "total_generations": len(self.generation_history),
            "successful_generations": len(successful),
            "average_tests_per_function": sum(h["test_count"] for h in successful) / len(successful) if successful else 0,
            "functions_tested": [h["function_name"] for h in successful],
            "complexity_distribution": {
                "simple": len([h for h in successful if h["complexity"] <= 3]),
                "moderate": len([h for h in successful if 3 < h["complexity"] <= 7]),
                "complex": len([h for h in successful if h["complexity"] > 7])
            }
        }

# Global instance
test_generator = SmartTestGenerator()

def generate_tests(code: str, model: str = "qwen") -> str:
    """Legacy function for backward compatibility"""
    return test_generator.generate_ai_tests(code, model)
