# check.py - Enhanced Code Analysis and Management System

import fire
import os
import sys
import json
import time
import ast
import subprocess
import tempfile
import traceback
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import requests
import hashlib
import difflib
from datetime import datetime

# Rich imports for beautiful CLI
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeColumn
    from rich.syntax import Syntax
    from rich.prompt import Confirm, Prompt, IntPrompt
    from rich.tree import Tree
    from rich.layout import Layout
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.text import Text
    from rich.columns import Columns
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallback console
    class Console:
        def print(self, *args, **kwargs):
            print(*args)

console = Console()

# Data models
class ErrorType(Enum):
    SYNTAX = "syntax"
    RUNTIME = "runtime"
    LOGICAL = "logical"
    PERFORMANCE = "performance"
    SECURITY = "security"
    STYLE = "style"

class ExecutionStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    MEMORY_LIMIT = "memory_limit"

@dataclass
class ExecutionResult:
    status: ExecutionStatus
    stdout: str
    stderr: str
    execution_time: float
    memory_used: float
    exit_code: int
    security_warnings: List[str]

@dataclass
class AnalysisResult:
    error_type: ErrorType
    severity: str
    confidence: float
    line_number: Optional[int]
    code_context: Optional[str]
    suggested_fixes: List[str]

@dataclass
class APIResponse:
    success: bool
    content: str
    error: Optional[str] = None
    response_time: float = 0.0

class MockAPI:
    """Mock API for testing when real API is not available"""
    
    def __init__(self):
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "response_times": []
        }
    
    def call_model(self, model: str, prompt: str) -> APIResponse:
        """Mock API call with realistic responses"""
        self.stats["total_calls"] += 1
        time.sleep(0.5)  # Simulate API delay
        
        # Generate mock responses based on prompt content
        if "debug" in prompt.lower() or "error" in prompt.lower():
            response = self._generate_debug_response(prompt)
        elif "test" in prompt.lower():
            response = self._generate_test_response(prompt)
        elif "improve" in prompt.lower() or "optimize" in prompt.lower():
            response = self._generate_improvement_response(prompt)
        elif "analyze" in prompt.lower():
            response = self._generate_analysis_response(prompt)
        else:
            response = "# Improved code with better error handling and documentation\n" + prompt.split("Code:")[-1] if "Code:" in prompt else "# Enhanced code implementation"
        
        self.stats["successful_calls"] += 1
        self.stats["response_times"].append(0.5)
        
        return APIResponse(success=True, content=response, response_time=0.5)
    
    def _generate_debug_response(self, prompt: str) -> str:
        return """
The error appears to be caused by:
1. Missing imports or undefined variables
2. Incorrect function calls or syntax
3. Type mismatches

Suggested fixes:
- Add proper imports at the top of the file
- Check variable names for typos
- Ensure all functions are properly defined
- Add error handling with try-except blocks

Example fix:
```python
try:
    # Your code here
    result = risky_operation()
except Exception as e:
    print(f"Error occurred: {e}")
    result = None
```
"""
    
    def _generate_test_response(self, prompt: str) -> str:
        return """
import unittest
import sys
import os

class TestCode(unittest.TestCase):
    
    def test_basic_functionality(self):
        \"\"\"Test basic functionality\"\"\"
        # Add your basic test cases here
        self.assertTrue(True)
    
    def test_edge_cases(self):
        \"\"\"Test edge cases\"\"\"
        # Test with empty inputs
        # Test with large inputs
        # Test with invalid inputs
        pass
    
    def test_error_handling(self):
        \"\"\"Test error handling\"\"\"
        # Test how code handles errors
        with self.assertRaises(ValueError):
            # Code that should raise ValueError
            pass
    
    def test_performance(self):
        \"\"\"Test performance requirements\"\"\"
        import time
        start = time.time()
        # Your code here
        end = time.time()
        self.assertLess(end - start, 1.0)  # Should complete within 1 second

if __name__ == '__main__':
    unittest.main()
"""
    
    def _generate_improvement_response(self, prompt: str) -> str:
        return """
# Improved and optimized code with best practices

import logging
from typing import Optional, List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedCode:
    \"\"\"
    Enhanced version with:
    - Better error handling
    - Type hints
    - Documentation
    - Performance optimizations
    - Security improvements
    \"\"\"
    
    def __init__(self):
        self.data = {}
    
    def process_data(self, input_data: Any) -> Optional[Dict]:
        \"\"\"
        Process input data with proper error handling
        
        Args:
            input_data: The data to process
            
        Returns:
            Processed data or None if error
        \"\"\"
        try:
            if not input_data:
                logger.warning("Empty input data received")
                return None
            
            # Process the data
            result = self._internal_process(input_data)
            logger.info("Data processed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return None
    
    def _internal_process(self, data: Any) -> Dict:
        \"\"\"Internal processing logic\"\"\"
        return {"processed": True, "data": data}

# Usage example
if __name__ == "__main__":
    processor = ImprovedCode()
    result = processor.process_data("sample data")
    print(result)
"""
    
    def _generate_analysis_response(self, prompt: str) -> str:
        return """
## Code Quality Analysis

### Strengths:
- Good use of functions and classes
- Reasonable error handling in some areas
- Clear variable naming

### Areas for Improvement:
1. **Documentation**: Add more docstrings and comments
2. **Error Handling**: Implement comprehensive try-catch blocks
3. **Type Hints**: Add type annotations for better code clarity
4. **Testing**: Create unit tests for all functions
5. **Performance**: Consider optimizing loops and data structures

### Security Considerations:
- Validate all input data
- Use secure methods for file operations
- Avoid executing arbitrary code
- Implement proper logging for security events

### Best Practices Recommendations:
- Follow PEP 8 style guidelines
- Use context managers for resource handling
- Implement proper logging instead of print statements
- Consider using dataclasses for structured data
- Add comprehensive error messages
"""
    
    def get_stats(self) -> Dict:
        avg_time = sum(self.stats["response_times"]) / len(self.stats["response_times"]) if self.stats["response_times"] else 0
        return {
            **self.stats,
            "average_response_time": avg_time
        }
    
    def get_model_info(self) -> Dict[str, str]:
        return {
            "gpt-4": "Advanced language model for complex tasks",
            "gpt-3.5": "Fast and efficient model for general tasks",
            "claude": "Anthropic's AI assistant",
            "deepseek": "Code-specialized model",
            "qwen": "Alibaba's multilingual model"
        }

class SecureCodeRunner:
    """Secure code execution with monitoring and limits"""
    
    def __init__(self, timeout: int = 10, memory_limit_mb: int = 100):
        self.timeout = timeout
        self.memory_limit = memory_limit_mb * 1024 * 1024  # Convert to bytes
        self.stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "execution_times": []
        }
    
    def security_check(self, code: str) -> List[str]:
        """Check code for potential security issues"""
        warnings = []
        
        dangerous_patterns = [
            ("eval(", "Use of eval() can execute arbitrary code"),
            ("exec(", "Use of exec() can execute arbitrary code"),
            ("__import__", "Dynamic imports can be dangerous"),
            ("subprocess", "Subprocess calls can execute system commands"),
            ("os.system", "System calls can be dangerous"),
            ("open(", "File operations - ensure proper validation"),
            ("input(", "User input - ensure proper validation"),
            ("pickle.load", "Pickle can execute arbitrary code"),
        ]
        
        for pattern, warning in dangerous_patterns:
            if pattern in code:
                warnings.append(warning)
        
        return warnings
    
    def run_code(self, code: str, input_data: str = "") -> ExecutionResult:
        """Execute code with security and resource monitoring"""
        self.stats["total_executions"] += 1
        start_time = time.time()
        
        # Security check
        security_warnings = self.security_check(code)
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Execute with timeout and monitoring
            process = subprocess.Popen(
                [sys.executable, temp_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(input=input_data, timeout=self.timeout)
                exit_code = process.returncode
                execution_time = time.time() - start_time
                
                # Estimate memory usage (simplified)
                memory_used = psutil.Process().memory_info().rss / 1024  # KB
                
                status = ExecutionStatus.SUCCESS if exit_code == 0 else ExecutionStatus.ERROR
                
                if status == ExecutionStatus.SUCCESS:
                    self.stats["successful_executions"] += 1
                    self.stats["execution_times"].append(execution_time)
                
                return ExecutionResult(
                    status=status,
                    stdout=stdout,
                    stderr=stderr,
                    execution_time=execution_time,
                    memory_used=memory_used,
                    exit_code=exit_code,
                    security_warnings=security_warnings
                )
                
            except subprocess.TimeoutExpired:
                process.kill()
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    stdout="",
                    stderr=f"Process timed out after {self.timeout} seconds",
                    execution_time=self.timeout,
                    memory_used=0,
                    exit_code=-1,
                    security_warnings=security_warnings
                )
            
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                stdout="",
                stderr=str(e),
                execution_time=time.time() - start_time,
                memory_used=0,
                exit_code=-1,
                security_warnings=security_warnings
            )
        
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def get_stats(self) -> Dict:
        avg_time = sum(self.stats["execution_times"]) / len(self.stats["execution_times"]) if self.stats["execution_times"] else 0
        return {
            **self.stats,
            "average_execution_time": avg_time
        }

class SmartDebugger:
    """Advanced debugging with AI assistance"""
    
    def debug(self, code: str, error_msg: str, model: str = "gpt-4") -> Dict:
        """Debug code with comprehensive analysis"""
        analysis = self._analyze_error(code, error_msg)
        
        # Mock AI explanation for debugging
        ai_explanation = f"""
Based on the error analysis, this appears to be a {analysis.error_type.value} error.

Error Details:
- Type: {analysis.error_type.value}
- Severity: {analysis.severity}
- Confidence: {analysis.confidence:.1%}

The error likely occurs because:
1. Variable or function is not defined
2. Incorrect syntax or indentation
3. Type mismatch or invalid operation
4. Missing imports or dependencies

Recommended fixes:
{chr(10).join(f"- {fix}" for fix in analysis.suggested_fixes)}
"""
        
        prevention_tips = [
            "Use a linter like pylint or flake8",
            "Add type hints to catch type errors early",
            "Write unit tests for your functions",
            "Use an IDE with syntax highlighting",
            "Add logging to track variable values"
        ]
        
        # Generate fixed code (simplified)
        fixed_code = self._attempt_fix(code, error_msg)
        
        return {
            "analysis": analysis,
            "ai_explanation": ai_explanation,
            "prevention_tips": prevention_tips,
            "fixed_code": fixed_code
        }
    
    def _analyze_error(self, code: str, error_msg: str) -> AnalysisResult:
        """Analyze error and provide structured feedback"""
        
        # Determine error type
        if "SyntaxError" in error_msg:
            error_type = ErrorType.SYNTAX
            severity = "High"
        elif "NameError" in error_msg:
            error_type = ErrorType.RUNTIME
            severity = "Medium"
        elif "TypeError" in error_msg:
            error_type = ErrorType.RUNTIME
            severity = "Medium"
        elif "ImportError" in error_msg or "ModuleNotFoundError" in error_msg:
            error_type = ErrorType.RUNTIME
            severity = "Medium"
        else:
            error_type = ErrorType.LOGICAL
            severity = "Low"
        
        # Extract line number
        line_number = None
        try:
            if "line " in error_msg:
                line_part = error_msg.split("line ")[1].split(",")[0]
                line_number = int(line_part)
        except:
            pass
        
        # Get code context
        code_context = None
        if line_number:
            lines = code.split('\n')
            if 0 < line_number <= len(lines):
                start = max(0, line_number - 3)
                end = min(len(lines), line_number + 2)
                context_lines = lines[start:end]
                code_context = '\n'.join(f"{i+start+1:3d}: {line}" for i, line in enumerate(context_lines))
        
        # Generate suggested fixes
        suggested_fixes = []
        if "NameError" in error_msg:
            var_name = error_msg.split("'")[1] if "'" in error_msg else "variable"
            suggested_fixes = [
                f"Define the variable '{var_name}' before using it",
                "Check for typos in variable names",
                "Ensure the variable is in the correct scope"
            ]
        elif "SyntaxError" in error_msg:
            suggested_fixes = [
                "Check for missing colons, parentheses, or brackets",
                "Verify proper indentation",
                "Look for unclosed strings or comments"
            ]
        elif "ImportError" in error_msg:
            suggested_fixes = [
                "Install the required package using pip",
                "Check the module name spelling",
                "Verify the module is in the Python path"
            ]
        
        return AnalysisResult(
            error_type=error_type,
            severity=severity,
            confidence=0.8,
            line_number=line_number,
            code_context=code_context,
            suggested_fixes=suggested_fixes
        )
    
    def _attempt_fix(self, code: str, error_msg: str) -> str:
        """Attempt to generate fixed code"""
        
        # Simple fixes for common errors
        fixed_code = code
        
        if "NameError" in error_msg and "'" in error_msg:
            # Try to add a simple variable definition
            var_name = error_msg.split("'")[1]
            fixed_code = f"{var_name} = None  # Added missing variable\n" + code
        
        elif "ImportError" in error_msg or "ModuleNotFoundError" in error_msg:
            # Add common imports
            imports = [
                "import os",
                "import sys", 
                "import json",
                "from typing import Optional, List, Dict"
            ]
            fixed_code = "\n".join(imports) + "\n\n" + code
        
        return fixed_code

class CodeAnalyzer:
    """Main code analyzer with comprehensive features"""
    
    def __init__(self):
        self.api_client = MockAPI()
        self.runner = SecureCodeRunner()
        self.debugger = SmartDebugger()
        self.session_history = []
        self.config = self._load_config()
        self.file_cache = {}
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        config_file = Path.home() / ".code_analyzer_config.json"
        default_config = {
            "default_model": "gpt-4",
            "timeout": 10,
            "memory_limit_mb": 100,
            "auto_save_results": True,
            "theme": "dark",
            "security_checks": True,
            "test_model": "gpt-3.5",
            "debug_model": "gpt-4",
            "auto_backup": True,
            "max_backups": 5
        }
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except:
                pass
        
        # Save default config
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def _save_config(self):
        """Save current configuration"""
        config_file = Path.home() / ".code_analyzer_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _create_backup(self, file_path: str, content: str) -> str:
        """Create backup of file"""
        if not self.config["auto_backup"]:
            return None
            
        backup_dir = Path(file_path).parent / ".backups"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_stem = Path(file_path).stem
        backup_path = backup_dir / f"{file_stem}_backup_{timestamp}.py"
        
        try:
            with open(backup_path, 'w') as f:
                f.write(content)
            
            # Clean old backups
            self._cleanup_backups(backup_dir, file_stem)
            
            return str(backup_path)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not create backup: {e}[/yellow]")
            return None
    
    def _cleanup_backups(self, backup_dir: Path, file_stem: str):
        """Remove old backups, keeping only max_backups"""
        pattern = f"{file_stem}_backup_*.py"
        backups = sorted(backup_dir.glob(pattern), key=os.path.getmtime, reverse=True)
        
        for backup in backups[self.config["max_backups"]:]:
            try:
                backup.unlink()
            except:
                pass
    
    def read_file(self, file_path: str) -> str:
        """Read file with caching and error handling"""
        try:
            path = Path(file_path)
            if not path.exists():
                console.print(f"[red]Error: File '{file_path}' not found[/red]")
                return None
            
            # Check cache
            file_stat = path.stat()
            cache_key = f"{file_path}_{file_stat.st_mtime}"
            
            if cache_key in self.file_cache:
                return self.file_cache[cache_key]
            
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Cache the content
            self.file_cache[cache_key] = content
            return content
            
        except Exception as e:
            console.print(f"[red]Error reading file: {str(e)}[/red]")
            return None
    
    def write_file(self, file_path: str, content: str):
        """Write file with backup and error handling"""
        try:
            # Create backup if file exists
            if Path(file_path).exists():
                original_content = self.read_file(file_path)
                if original_content:
                    backup_path = self._create_backup(file_path, original_content)
                    if backup_path:
                        console.print(f"[blue]📁 Backup created: {backup_path}[/blue]")
            
            # Create directory if it doesn't exist
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            # Clear cache for this file
            self.file_cache = {k: v for k, v in self.file_cache.items() if not k.startswith(file_path)}
            
            console.print(f"[green]✅ File saved: {file_path}[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Error writing file: {str(e)}[/red]")
    
    def edit(self, file: str, model: str = None, interactive: bool = True, instruction: str = None, 
             diff: bool = False, validate: bool = True):
        """Enhanced code editing with AI assistance"""
        model = model or self.config["default_model"]
        code = self.read_file(file)
        if code is None:
            return
        
        if RICH_AVAILABLE:
            console.print(Panel(f"[bold blue]🤖 AI Code Editor[/bold blue]\nFile: {file}\nModel: {model}"))
        else:
            print(f"AI Code Editor - File: {file}, Model: {model}")
        
        if interactive and not instruction:
            if RICH_AVAILABLE:
                # Show current code
                syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
                console.print(Panel(syntax, title="📄 Current Code"))
            else:
                print("Current Code:")
                print(code)
            
            # Get editing instructions
            instruction = input("Enter editing instructions (or press Enter for general improvement): ").strip()
        
        if instruction:
            prompt = f"Modify this Python code based on these instructions: {instruction}\n\nCode:\n{code}"
        else:
            prompt = f"Improve and optimize this Python code:\n\n{code}"
        
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("🔄 Getting AI suggestions...", total=None)
                response = self.api_client.call_model(model, prompt)
                progress.stop()
        else:
            print("Getting AI suggestions...")
            response = self.api_client.call_model(model, prompt)
        
        if not response.success:
            console.print(f"[red]❌ Error: {response.error}[/red]")
            return
        
        # Validate the improved code
        if validate:
            validation_result = self._validate_code(response.content)
            if not validation_result["valid"]:
                console.print(f"[yellow]⚠️  Code validation issues: {validation_result['error']}[/yellow]")
                if not Confirm.ask("Continue anyway?"):
                    return
        
        # Show diff if requested
        if diff:
            self._show_diff(code, response.content, file)
        
        # Display improved code
        if RICH_AVAILABLE:
            console.print("\n[bold green]✨ AI Improved Code:[/bold green]")
            improved_syntax = Syntax(response.content, "python", theme="monokai", line_numbers=True)
            console.print(Panel(improved_syntax, title="🔧 Improved Code"))
        else:
            print("\nAI Improved Code:")
            print(response.content)
        
        if interactive:
            if input("Save the improved code? (y/N): ").lower().startswith('y'):
                self.write_file(file, response.content)
        else:
            print("\nUse --interactive flag to save changes")
        
        # Log session
        self.session_history.append({
            "action": "edit",
            "file": file,
            "model": model,
            "timestamp": time.time(),
            "success": response.success,
            "instruction": instruction
        })
    
    def _validate_code(self, code: str) -> Dict[str, Any]:
        """Validate Python code syntax"""
        try:
            ast.parse(code)
            return {"valid": True}
        except SyntaxError as e:
            return {"valid": False, "error": str(e)}
    
    def _show_diff(self, original: str, modified: str, filename: str):
        """Show diff between original and modified code"""
        diff = list(difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=f"{filename} (original)",
            tofile=f"{filename} (modified)",
            n=3
        ))
        
        if diff:
            console.print("\n[bold yellow]📝 Changes Preview:[/bold yellow]")
            diff_text = ''.join(diff)
            if RICH_AVAILABLE:
                console.print(Panel(diff_text, title="Diff"))
            else:
                print(diff_text)
    
    def run(self, file: str, input_data: str = "", verbose: bool = False, 
            benchmark: bool = False, profile: bool = False, watch: bool = False):
        """Enhanced code execution with comprehensive monitoring"""
        code = self.read_file(file)
        if code is None:
            return
        
        if RICH_AVAILABLE:
            console.print(Panel(f"[bold blue]🚀 Code Execution[/bold blue]\nFile: {file}"))
        else:
            print(f"Code Execution - File: {file}")
        
        # Watch mode - monitor file changes
        if watch:
            self._watch_and_run(file, input_data, verbose)
            return
        
        # Security check if enabled
        if self.config["security_checks"]:
            warnings = self.runner.security_check(code)
            if warnings:
                console.print("[yellow]🔒 Security Warnings:[/yellow]")
                for warning in warnings:
                    console.print(f"  ⚠️  {warning}")
                
                if not input("Continue execution despite warnings? (y/N): ").lower().startswith('y'):
                    console.print("[red]❌ Execution cancelled[/red]")
                    return
        
        # Benchmark mode - multiple runs
        if benchmark:
            self._run_benchmark(code, input_data, file)
            return
        
        # Profile mode
        if profile:
            self._run_with_profiling(code, input_data, file)
            return
        
        # Regular execution
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("⚡ Executing code...", total=None)
                result = self.runner.run_code(code, input_data)
                progress.stop()
        else:
            print("Executing code...")
            result = self.runner.run_code(code, input_data)
        
        # Display results
        self._display_execution_results(result, file, verbose)
        
        # Log session
        self.session_history.append({
            "action": "run",
            "file": file,
            "status": result.status.value,
            "execution_time": result.execution_time,
            "timestamp": time.time()
        })
    
    def _run_benchmark(self, code: str, input_data: str, filename: str):
        """Run benchmark with multiple iterations"""
        runs = 5
        console.print(f"[cyan]🏁 Running benchmark with {runs} iterations...[/cyan]")
        results = []
        
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                task = progress.add_task("Running benchmark...", total=runs)
                
                for i in range(runs):
                    result = self.runner.run_code(code, input_data)
                    results.append(result)
                    progress.advance(task)
        else:
            for i in range(runs):
                print(f"Run {i+1}/{runs}")
                result = self.runner.run_code(code, input_data)
                results.append(result)
        
        # Analyze benchmark results
        self._analyze_benchmark_results(results, filename)
    
    def _analyze_benchmark_results(self, results: List[ExecutionResult], filename: str):
        """Analyze and display benchmark results"""
        execution_times = [r.execution_time for r in results if r.status.value == "success"]
        
        if not execution_times:
            console.print("[red]❌ No successful runs to analyze[/red]")
            return
