# executor/runner.py

import subprocess
import tempfile
import os
import time
import resource
import signal
import sys
import ast
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json

class ExecutionStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    MEMORY_LIMIT = "memory_limit"
    SECURITY_VIOLATION = "security_violation"
    CRASH = "crash"

@dataclass
class ExecutionResult:
    status: ExecutionStatus
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float = 0.0
    memory_used: int = 0  # in KB
    security_warnings: List[str] = None
    
    def __post_init__(self):
        if self.security_warnings is None:
            self.security_warnings = []

class SecureCodeRunner:
    def __init__(self, 
                 timeout: int = 10, 
                 memory_limit_mb: int = 100,
                 enable_security_check: bool = True):
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
        self.enable_security_check = enable_security_check
        self.execution_history: List[Dict[str, Any]] = []
        
        # Dangerous imports/functions to detect
        self.dangerous_patterns = [
            'import os', 'import subprocess', 'import sys', 'import socket',
            'import requests', 'import urllib', 'import shutil', 'import glob',
            'open(', 'exec(', 'eval(', '__import__', 'compile(',
            'os.system', 'os.popen', 'os.remove', 'os.rmdir',
            'subprocess.', 'shutil.', 'socket.', 'urllib.'
        ]
    
    def security_check(self, code: str) -> List[str]:
        """Check code for potentially dangerous operations"""
        if not self.enable_security_check:
            return []
        
        warnings = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            for pattern in self.dangerous_patterns:
                if pattern in line_stripped:
                    warnings.append(f"Line {i}: Potentially dangerous operation '{pattern}'")
        
        # Check for file operations
        if any(keyword in code for keyword in ['open(', 'file(', 'with open']):
            warnings.append("File operations detected - ensure proper file handling")
        
        # Check for network operations
        if any(keyword in code for keyword in ['socket', 'urllib', 'requests', 'http']):
            warnings.append("Network operations detected - potential security risk")
        
        return warnings
    
    def syntax_check(self, code: str) -> Optional[str]:
        """Check Python syntax before execution"""
        try:
            ast.parse(code)
            return None
        except SyntaxError as e:
            return f"Syntax Error: {e.msg} at line {e.lineno}"
    
    def run_code(self, code: str, input_data: str = "") -> ExecutionResult:
        """Execute Python code with security checks and resource monitoring"""
        start_time = time.time()
        
        # Syntax check
        syntax_error = self.syntax_check(code)
        if syntax_error:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                stdout="",
                stderr=syntax_error,
                exit_code=1,
                execution_time=time.time() - start_time
            )
        
        # Security check
        security_warnings = self.security_check(code)
        
        # If high-risk operations detected, warn but continue
        if security_warnings and any("dangerous" in warning.lower() for warning in security_warnings):
            print("⚠️  Security Warning: Potentially dangerous operations detected!")
            for warning in security_warnings:
                print(f"   {warning}")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        
        try:
            # Set up resource limits (Unix/Linux only)
            def set_limits():
                try:
                    # Memory limit (in bytes)
                    resource.setrlimit(resource.RLIMIT_AS, 
                                     (self.memory_limit_mb * 1024 * 1024, 
                                      self.memory_limit_mb * 1024 * 1024))
                except:
                    pass  # Windows doesn't support this
            
            # Execute with monitoring
            process = subprocess.Popen(
                [sys.executable, tmp_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=set_limits if os.name != 'nt' else None
            )
            
            try:
                stdout, stderr = process.communicate(input=input_data, timeout=self.timeout)
                execution_time = time.time() - start_time
                
                # Determine status
                if process.returncode == 0:
                    status = ExecutionStatus.SUCCESS
                else:
                    status = ExecutionStatus.ERROR
                
                result = ExecutionResult(
                    status=status,
                    stdout=stdout.strip(),
                    stderr=stderr.strip(),
                    exit_code=process.returncode,
                    execution_time=execution_time,
                    security_warnings=security_warnings
                )
                
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                result = ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    stdout="",
                    stderr=f"Execution timed out after {self.timeout} seconds",
                    exit_code=-1,
                    execution_time=self.timeout,
                    security_warnings=security_warnings
                )
        
        except MemoryError:
            result = ExecutionResult(
                status=ExecutionStatus.MEMORY_LIMIT,
                stdout="",
                stderr=f"Memory limit exceeded ({self.memory_limit_mb}MB)",
                exit_code=-1,
                execution_time=time.time() - start_time,
                security_warnings=security_warnings
            )
        
        except Exception as e:
            result = ExecutionResult(
                status=ExecutionStatus.CRASH,
                stdout="",
                stderr=f"Execution crashed: {str(e)}",
                exit_code=-1,
                execution_time=time.time() - start_time,
                security_warnings=security_warnings
            )
        
        finally:
            # Clean up
            try:
                os.remove(tmp_path)
            except:
                pass
        
        # Log execution
        self.execution_history.append({
            "timestamp": time.time(),
            "status": result.status.value,
            "execution_time": result.execution_time,
            "memory_used": result.memory_used,
            "code_length": len(code),
            "security_warnings_count": len(security_warnings)
        })
        
        return result
    
    def run_with_tests(self, code: str, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run code with multiple test cases"""
        results = []
        
        for i, test_case in enumerate(test_cases):
            input_data = test_case.get("input", "")
            expected_output = test_case.get("expected_output", "")
            
            result = self.run_code(code, input_data)
            
            test_result = {
                "test_case": i + 1,
                "input": input_data,
                "expected": expected_output,
                "actual": result.stdout,
                "passed": result.stdout.strip() == expected_output.strip() if expected_output else result.status == ExecutionStatus.SUCCESS,
                "execution_time": result.execution_time,
                "status": result.status.value
            }
            
            results.append(test_result)
        
        passed_tests = sum(1 for r in results if r["passed"])
        
        return {
            "total_tests": len(test_cases),
            "passed_tests": passed_tests,
            "failed_tests": len(test_cases) - passed_tests,
            "success_rate": passed_tests / len(test_cases) if test_cases else 0,
            "results": results
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {"total_executions": 0}
        
        successful = [h for h in self.execution_history if h["status"] == "success"]
        
        return {
            "total_executions": len(self.execution_history),
            "successful_executions": len(successful),
            "average_execution_time": sum(h["execution_time"] for h in successful) / len(successful) if successful else 0,
            "total_security_warnings": sum(h["security_warnings_count"] for h in self.execution_history),
            "status_breakdown": {
                status: len([h for h in self.execution_history if h["status"] == status])
                for status in set(h["status"] for h in self.execution_history)
            }
        }

# Global instance for backward compatibility
runner = SecureCodeRunner()

def run_code(code: str) -> dict:
    """Legacy function for backward compatibility"""
    result = runner.run_code(code)
    return {
        "status": result.status.value,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.exit_code
    }
