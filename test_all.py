#!/usr/bin/env python3
"""
Comprehensive test suite for the Code Analyzer application.
Tests main functionality, components, and edge cases.
"""

import unittest
import tempfile
import os
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

# Add the project root to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from main import CodeAnalyzer
    from models.pollinations_api import PollinationsAPI, APIResponse
    from executor.runner import SecureCodeRunner, ExecutionResult, ExecutionStatus
    from debugger.tracer import SmartDebugger, ErrorType
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")
    IMPORTS_AVAILABLE = False


class TestCodeAnalyzer(unittest.TestCase):
    """Test the main CodeAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
        
        self.analyzer = CodeAnalyzer()
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test_code.py")
        
        # Sample test code
        self.sample_code = '''def hello_world():
    """A simple hello world function"""
    print("Hello, World!")
    return "Hello, World!"

if __name__ == "__main__":
    hello_world()
'''
        
        # Create test file
        with open(self.test_file, 'w') as f:
            f.write(self.sample_code)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test CodeAnalyzer initialization"""
        self.assertIsInstance(self.analyzer.api_client, PollinationsAPI)
        self.assertIsInstance(self.analyzer.runner, SecureCodeRunner)
        self.assertIsInstance(self.analyzer.debugger, SmartDebugger)
        self.assertIsInstance(self.analyzer.config, dict)
        self.assertEqual(self.analyzer.session_history, [])
    
    def test_config_loading(self):
        """Test configuration loading and defaults"""
        config = self.analyzer._load_config()
        
        # Check required config keys exist
        required_keys = [
            "default_model", "timeout", "memory_limit_mb", 
            "auto_save_results", "theme", "security_checks"
        ]
        for key in required_keys:
            self.assertIn(key, config)
        
        # Check default values
        self.assertEqual(config["default_model"], "deepseek")
        self.assertEqual(config["timeout"], 10)
        self.assertTrue(isinstance(config["security_checks"], bool))
    
    def test_read_file(self):
        """Test file reading functionality"""
        content = self.analyzer.read_file(self.test_file)
        self.assertEqual(content, self.sample_code)
    
    def test_read_file_not_found(self):
        """Test reading non-existent file"""
        with self.assertRaises(SystemExit):
            self.analyzer.read_file("nonexistent_file.py")
    
    def test_write_file(self):
        """Test file writing functionality"""
        test_content = "# Test content"
        test_path = os.path.join(self.temp_dir, "write_test.py")
        
        self.analyzer.write_file(test_path, test_content)
        
        # Verify file was created and content is correct
        self.assertTrue(os.path.exists(test_path))
        with open(test_path, 'r') as f:
            self.assertEqual(f.read(), test_content)
    
    def test_config_set_and_show(self):
        """Test configuration management"""
        # Test setting config
        self.analyzer.config_set("test_key", "test_value")
        self.assertEqual(self.analyzer.config["test_key"], "test_value")
        
        # Test boolean conversion
        self.analyzer.config_set("test_bool", "true")
        self.assertTrue(self.analyzer.config["test_bool"])
        
        # Test integer conversion
        self.analyzer.config_set("test_int", "42")
        self.assertEqual(self.analyzer.config["test_int"], 42)
    
    @patch('main.console')
    def test_analyze_basic(self, mock_console):
        """Test basic code analysis"""
        self.analyzer.analyze(self.test_file)
        
        # Verify console output was called (analysis was performed)
        self.assertTrue(mock_console.print.called)
    
    @patch('main.console')
    def test_analyze_comprehensive(self, mock_console):
        """Test comprehensive code analysis with mocked API"""
        with patch.object(self.analyzer.api_client, 'call_model') as mock_api:
            mock_response = Mock()
            mock_response.success = True
            mock_response.content = "Code looks good with minor improvements possible."
            mock_api.return_value = mock_response
            
            self.analyzer.analyze(self.test_file, comprehensive=True)
            
            # Verify API was called for comprehensive analysis
            mock_api.assert_called_once()
            self.assertTrue(mock_console.print.called)


class TestAPIResponse(unittest.TestCase):
    """Test API response handling"""
    
    def test_api_response_success(self):
        """Test successful API response"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
        
        response = APIResponse(
            success=True,
            content="Test response",
            error=None,
            response_time=0.5
        )
        
        self.assertTrue(response.success)
        self.assertEqual(response.content, "Test response")
        self.assertIsNone(response.error)
        self.assertEqual(response.response_time, 0.5)
    
    def test_api_response_error(self):
        """Test error API response"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
        
        response = APIResponse(
            success=False,
            content=None,
            error="API Error",
            response_time=0.1
        )
        
        self.assertFalse(response.success)
        self.assertIsNone(response.content)
        self.assertEqual(response.error, "API Error")


class TestExecutionResult(unittest.TestCase):
    """Test code execution results"""
    
    def test_execution_result_success(self):
        """Test successful execution result"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
        
        result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            stdout="Hello, World!",
            stderr="",
            exit_code=0,
            execution_time=0.01,
            memory_used=1024
        )
        
        self.assertEqual(result.status, ExecutionStatus.SUCCESS)
        self.assertEqual(result.stdout, "Hello, World!")
        self.assertEqual(result.stderr, "")
        self.assertEqual(result.exit_code, 0)
        self.assertGreater(result.execution_time, 0)
    
    def test_execution_result_error(self):
        """Test error execution result"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
        
        result = ExecutionResult(
            status=ExecutionStatus.ERROR,
            stdout="",
            stderr="NameError: name 'undefined_var' is not defined",
            exit_code=1,
            execution_time=0.01,
            memory_used=512
        )
        
        self.assertEqual(result.status, ExecutionStatus.ERROR)
        self.assertEqual(result.exit_code, 1)
        self.assertIn("NameError", result.stderr)


class TestSecurityChecks(unittest.TestCase):
    """Test security checking functionality"""
    
    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
        self.runner = SecureCodeRunner()
    
    def test_security_check_safe_code(self):
        """Test security check on safe code"""
        safe_code = '''
def add_numbers(a, b):
    return a + b

result = add_numbers(2, 3)
print(result)
'''
        warnings = self.runner.security_check(safe_code)
        # Safe code should have no warnings or minimal warnings
        self.assertIsInstance(warnings, list)
    
    def test_security_check_dangerous_code(self):
        """Test security check on potentially dangerous code"""
        dangerous_code = '''
import os
os.system("rm -rf /")
exec("print('dangerous')")
'''
        warnings = self.runner.security_check(dangerous_code)
        self.assertIsInstance(warnings, list)
        # Should detect dangerous patterns
        self.assertGreater(len(warnings), 0)


class TestFileOperations(unittest.TestCase):
    """Test file operations and edge cases"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_create_nested_directories(self):
        """Test creating files in nested directories"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
        
        analyzer = CodeAnalyzer()
        nested_path = os.path.join(self.temp_dir, "deep", "nested", "test.py")
        test_content = "# Nested file test"
        
        analyzer.write_file(nested_path, test_content)
        
        self.assertTrue(os.path.exists(nested_path))
        with open(nested_path, 'r') as f:
            self.assertEqual(f.read(), test_content)
    
    def test_unicode_content(self):
        """Test handling Unicode content"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
        
        analyzer = CodeAnalyzer()
        unicode_content = '''# -*- coding: utf-8 -*-
def greet():
    print("Hello 世界! 🌍")
    return "Γεια σας κόσμε!"
'''
        test_path = os.path.join(self.temp_dir, "unicode_test.py")
        
        analyzer.write_file(test_path, unicode_content)
        read_content = analyzer.read_file(test_path)
        
        self.assertEqual(read_content, unicode_content)


class TestSessionHistory(unittest.TestCase):
    """Test session history tracking"""
    
    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
        self.analyzer = CodeAnalyzer()
    
    def test_session_history_tracking(self):
        """Test that session history is properly tracked"""
        # Initially empty
        self.assertEqual(len(self.analyzer.session_history), 0)
        
        # Add a mock session entry
        self.analyzer.session_history.append({
            "action": "test_action",
            "file": "test.py",
            "timestamp": time.time(),
            "success": True
        })
        
        self.assertEqual(len(self.analyzer.session_history), 1)
        entry = self.analyzer.session_history[0]
        self.assertEqual(entry["action"], "test_action")
        self.assertEqual(entry["file"], "test.py")
        self.assertTrue(entry["success"])
        self.assertIsInstance(entry["timestamp"], float)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def test_malformed_python_code(self):
        """Test handling of malformed Python code"""
        malformed_code = '''
def broken_function(
    # Missing closing parenthesis and colon
    print("This is broken"
'''
        # Test that AST parsing fails gracefully
        import ast
        with self.assertRaises(SyntaxError):
            ast.parse(malformed_code)
    
    def test_empty_file_handling(self):
        """Test handling of empty files"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("")  # Empty file
            empty_file = f.name
        
        try:
            analyzer = CodeAnalyzer()
            content = analyzer.read_file(empty_file)
            self.assertEqual(content, "")
        finally:
            os.unlink(empty_file)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows"""
    
    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
        
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = CodeAnalyzer()
        
        # Create a more complex test file
        self.complex_code = '''
import sys
import os
from typing import List, Optional

class Calculator:
    """A simple calculator class"""
    
    def __init__(self):
        self.history = []
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers"""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def divide(self, a: float, b: float) -> float:
        """Divide two numbers"""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result
    
    def get_history(self) -> List[str]:
        """Get calculation history"""
        return self.history.copy()

def main():
    calc = Calculator()
    print(calc.add(5, 3))
    print(calc.divide(10, 2))
    
    try:
        calc.divide(1, 0)
    except ValueError as e:
        print(f"Error: {e}")
    
    print("History:", calc.get_history())

if __name__ == "__main__":
    main()
'''
        
        self.test_file = os.path.join(self.temp_dir, "calculator.py")
        with open(self.test_file, 'w') as f:
            f.write(self.complex_code)
    
    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('main.console')
    def test_full_analysis_workflow(self, mock_console):
        """Test complete analysis workflow"""
        # Test basic analysis
        self.analyzer.analyze(self.test_file)
        self.assertTrue(mock_console.print.called)
        
        # Test that session history was updated
        # Note: analyze method should add to session history
        # This would need to be implemented in the actual method
    
    def test_code_structure_analysis(self):
        """Test analysis of code structure"""
        import ast
        
        tree = ast.parse(self.complex_code)
        
        # Check that we can identify classes and functions
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        self.assertGreater(len(functions), 0)
        self.assertGreater(len(classes), 0)
        
        # Check specific elements
        function_names = [f.name for f in functions]
        class_names = [c.name for c in classes]
        
        self.assertIn("add", function_names)
        self.assertIn("divide", function_names)
        self.assertIn("Calculator", class_names)


class TestPerformance(unittest.TestCase):
    """Performance and benchmark tests"""
    
    def test_large_file_handling(self):
        """Test handling of large code files"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
        
        # Generate a large code file
        large_code = "# Large file test\n" + "\n".join([
            f"def function_{i}():\n    return {i}"
            for i in range(1000)
        ])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(large_code)
            large_file = f.name
        
        try:
            analyzer = CodeAnalyzer()
            start_time = time.time()
            content = analyzer.read_file(large_file)
            read_time = time.time() - start_time
            
            self.assertEqual(len(content), len(large_code))
            self.assertLess(read_time, 1.0)  # Should read large file quickly
        finally:
            os.unlink(large_file)
    
    def test_multiple_operations_performance(self):
        """Test performance of multiple operations"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required modules not available")
        
        analyzer = CodeAnalyzer()
        
        # Test multiple config operations
        start_time = time.time()
        for i in range(100):
            analyzer.config_set(f"test_key_{i}", f"value_{i}")
        config_time = time.time() - start_time
        
        self.assertLess(config_time, 1.0)  # Should handle config operations quickly


def run_specific_test_suite():
    """Run specific test categories based on available modules"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Always run basic tests
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    
    if IMPORTS_AVAILABLE:
        # Add tests that require imports
        suite.addTests(loader.loadTestsFromTestCase(TestCodeAnalyzer))
        suite.addTests(loader.loadTestsFromTestCase(TestAPIResponse))
        suite.addTests(loader.loadTestsFromTestCase(TestExecutionResult))
        suite.addTests(loader.loadTestsFromTestCase(TestSecurityChecks))
        suite.addTests(loader.loadTestsFromTestCase(TestFileOperations))
        suite.addTests(loader.loadTestsFromTestCase(TestSessionHistory))
        suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
        suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    else:
        print("Warning: Some tests skipped due to missing modules")
    
    return suite


def main():
    """Main test runner with detailed output"""
    print("=" * 60)
    print("Code Analyzer Test Suite")
    print("=" * 60)
    
    # Check for required modules
    if not IMPORTS_AVAILABLE:
        print("⚠️  Warning: Not all modules are available for testing")
        print("   Some tests will be skipped")
    
    print(f"Python version: {sys.version}")
    print(f"Test directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("-" * 60)
    
    # Run tests
    suite = run_specific_test_suite()
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Exception:')[-1].strip()}")
    
    # Return exit code
    return len(result.failures) + len(result.errors)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
