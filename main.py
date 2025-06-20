# main.py

import fire
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.prompt import Confirm, Prompt
from rich.tree import Tree
from rich.layout import Layout
from rich.live import Live
import tempfile

# Import our enhanced modules
from models.pollinations_api import PollinationsAPI, api_client
from executor.runner import SecureCodeRunner, runner
from debugger.tracer import SmartDebugger, debugger
from tests.testgen import SmartTestGenerator

# Initialize rich console
console = Console()

class CodeAnalyzer:
    """Enhanced code analysis and management system"""
    
    def __init__(self):
        self.api_client = PollinationsAPI()
        self.runner = SecureCodeRunner()
        self.debugger = SmartDebugger()
        self.test_generator = SmartTestGenerator()
        self.session_history = []
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        config_file = Path.home() / ".code_analyzer_config.json"
        default_config = {
            "default_model": "deepseek",
            "timeout": 10,
            "memory_limit_mb": 100,
            "auto_save_results": True,
            "theme": "dark",
            "security_checks": True,
            "test_model": "qwen",
            "debug_model": "deepseek"
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
    
    def read_file(self, file_path: str) -> str:
        """Read file with error handling"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            console.print(f"[red]Error: File '{file_path}' not found[/red]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Error reading file: {str(e)}[/red]")
            sys.exit(1)
    
    def write_file(self, file_path: str, content: str):
        """Write file with error handling"""
        try:
            # Create directory if it doesn't exist
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            console.print(f"[green]✓ File saved: {file_path}[/green]")
        except Exception as e:
            console.print(f"[red]Error writing file: {str(e)}[/red]")
    
    def edit(self, file: str, model: str = None, interactive: bool = True, instruction: str = None):
        """Enhanced code editing with AI assistance"""
        model = model or self.config["default_model"]
        code = self.read_file(file)
        
        console.print(Panel(f"[bold blue]AI Code Editor[/bold blue]\nFile: {file}\nModel: {model}"))
        
        if interactive and not instruction:
            # Show current code
            syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
            console.print(Panel(syntax, title="Current Code"))
            
            # Get editing instructions
            instruction = Prompt.ask("Enter editing instructions (or press Enter for general improvement)")
        
        if instruction:
            prompt = f"Modify this Python code based on these instructions: {instruction}\n\nCode:\n{code}"
        else:
            prompt = f"Improve and optimize this Python code:\n\n{code}"
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Getting AI suggestions...", total=None)
            
            response = self.api_client.call_model(model, prompt)
            progress.stop()
        
        if not response.success:
            console.print(f"[red]Error: {response.error}[/red]")
            return
        
        # Display improved code
        console.print("\n[bold green]AI Improved Code:[/bold green]")
        improved_syntax = Syntax(response.content, "python", theme="monokai", line_numbers=True)
        console.print(Panel(improved_syntax, title="Improved Code"))
        
        if interactive:
            if Confirm.ask("Save the improved code?"):
                # Backup original
                backup_path = f"{file}.backup_{int(time.time())}"
                self.write_file(backup_path, code)
                console.print(f"[yellow]Original code backed up to: {backup_path}[/yellow]")
                
                # Save improved code
                self.write_file(file, response.content)
        else:
            console.print("\n[yellow]Use --interactive flag to save changes[/yellow]")
        
        # Log session
        self.session_history.append({
            "action": "edit",
            "file": file,
            "model": model,
            "timestamp": time.time(),
            "success": response.success
        })
    
    def run(self, file: str, input_data: str = "", verbose: bool = False, benchmark: bool = False):
        """Enhanced code execution with detailed output"""
        code = self.read_file(file)
        
        console.print(Panel(f"[bold blue]Code Execution[/bold blue]\nFile: {file}"))
        
        # Security check if enabled
        if self.config["security_checks"]:
            warnings = self.runner.security_check(code)
            if warnings:
                console.print("[yellow]Security Warnings:[/yellow]")
                for warning in warnings:
                    console.print(f"  ⚠️  {warning}")
                
                if not Confirm.ask("Continue execution despite warnings?"):
                    console.print("[red]Execution cancelled[/red]")
                    return
        
        # Benchmark mode - multiple runs
        if benchmark:
            runs = 5
            console.print(f"[cyan]Running benchmark with {runs} iterations...[/cyan]")
            results = []
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Running benchmark...", total=runs)
                
                for i in range(runs):
                    result = self.runner.run_code(code, input_data)
                    results.append(result)
                    progress.advance(task)
            
            # Analyze benchmark results
            execution_times = [r.execution_time for r in results if r.status.value == "success"]
            if execution_times:
                avg_time = sum(execution_times) / len(execution_times)
                min_time = min(execution_times)
                max_time = max(execution_times)
                
                benchmark_table = Table(title="Benchmark Results")
                benchmark_table.add_column("Metric", style="cyan")
                benchmark_table.add_column("Value", style="magenta")
                
                benchmark_table.add_row("Successful Runs", f"{len(execution_times)}/{runs}")
                benchmark_table.add_row("Average Time", f"{avg_time:.4f}s")
                benchmark_table.add_row("Min Time", f"{min_time:.4f}s")
                benchmark_table.add_row("Max Time", f"{max_time:.4f}s")
                benchmark_table.add_row("Std Deviation", f"{(sum((t - avg_time) ** 2 for t in execution_times) / len(execution_times)) ** 0.5:.4f}s")
                
                console.print(benchmark_table)
            
            result = results[0]  # Show first result details
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Executing code...", total=None)
                result = self.runner.run_code(code, input_data)
                progress.stop()
        
        # Create results table
        table = Table(title="Execution Results")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        table.add_row("Status", f"[green]{result.status.value}[/green]" if result.status.value == "success" else f"[red]{result.status.value}[/red]")
        table.add_row("Execution Time", f"{result.execution_time:.4f}s")
        table.add_row("Memory Used", f"{result.memory_used} KB")
        table.add_row("Exit Code", str(result.exit_code))
        
        console.print(table)
        
        if result.stdout:
            console.print(Panel(result.stdout, title="[green]Output[/green]"))
        
        if result.stderr:
            console.print(Panel(result.stderr, title="[red]Error[/red]"))
        
        if result.security_warnings and verbose:
            console.print("\n[yellow]Security Analysis:[/yellow]")
            for warning in result.security_warnings:
                console.print(f"  ⚠️  {warning}")
        
        # Log session
        self.session_history.append({
            "action": "run",
            "file": file,
            "status": result.status.value,
            "execution_time": result.execution_time,
            "timestamp": time.time()
        })
    
    def debug(self, file: str, model: str = None, fix: bool = False, interactive: bool = True):
        """Enhanced debugging with comprehensive analysis"""
        model = model or self.config["debug_model"]
        code = self.read_file(file)
        
        console.print(Panel(f"[bold blue]AI Debugger[/bold blue]\nFile: {file}\nModel: {model}"))
        
        # Run code to get errors
        result = self.runner.run_code(code)
        
        if result.status.value == "success":
            console.print("[green]✅ No errors found - code executed successfully![/green]")
            
            # Offer static analysis
            if interactive and Confirm.ask("Perform static code analysis for potential issues?"):
                self.analyze(file, comprehensive=True)
            return
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing error...", total=None)
            debug_result = self.debugger.debug(code, result.stderr, model)
            progress.stop()
        
        # Display error analysis
        analysis = debug_result.analysis
        
        error_table = Table(title="Error Analysis")
        error_table.add_column("Property", style="cyan")
        error_table.add_column("Value", style="magenta")
        
        error_table.add_row("Error Type", analysis.error_type.value)
        error_table.add_row("Severity", analysis.severity)
        error_table.add_row("Confidence", f"{analysis.confidence:.1%}")
        if analysis.line_number:
            error_table.add_row("Line Number", str(analysis.line_number))
        
        console.print(error_table)
        
        # Show code context
        if analysis.code_context:
            console.print(Panel(analysis.code_context, title="[yellow]Code Context[/yellow]"))
        
        # Show AI explanation
        console.print(Panel(debug_result.ai_explanation, title="[blue]AI Analysis[/blue]"))
        
        # Show suggested fixes
        if analysis.suggested_fixes:
            console.print("\n[bold cyan]Suggested Fixes:[/bold cyan]")
            for i, fix_suggestion in enumerate(analysis.suggested_fixes, 1):
                console.print(f"  {i}. {fix_suggestion}")
        
        # Show prevention tips
        if debug_result.prevention_tips:
            console.print("\n[bold green]Prevention Tips:[/bold green]")
            for tip in debug_result.prevention_tips:
                console.print(f"  💡 {tip}")
        
        # Offer to apply fix
        if fix and debug_result.fixed_code:
            console.print("\n[bold yellow]AI Generated Fix:[/bold yellow]")
            fixed_syntax = Syntax(debug_result.fixed_code, "python", theme="monokai", line_numbers=True)
            console.print(Panel(fixed_syntax, title="Fixed Code"))
            
            if interactive and Confirm.ask("Apply this fix?"):
                backup_path = f"{file}.backup_{int(time.time())}"
                self.write_file(backup_path, code)
                self.write_file(file, debug_result.fixed_code)
                console.print(f"[green]Fix applied! Original backed up to: {backup_path}[/green]")
                
                # Test the fix
                if Confirm.ask("Test the fixed code?"):
                    self.run(file)
        
        # Log session
        self.session_history.append({
            "action": "debug",
            "file": file,
            "model": model,
            "error_type": analysis.error_type.value,
            "severity": analysis.severity,
            "timestamp": time.time()
        })
    
    def test(self, file: str, model: str = None, save: bool = None, run_tests: bool = False):
        """Enhanced test generation with comprehensive analysis"""
        model = model or self.config["test_model"]
        code = self.read_file(file)
        
        console.print(Panel(f"[bold blue]AI Test Generator[/bold blue]\nFile: {file}\nModel: {model}"))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating comprehensive test suite...", total=None)
            # For now, use basic test generation until test_generator is complete
            prompt = f"""
Generate comprehensive test cases for this Python code. Include:
1. Normal test cases
2. Edge cases
3. Error cases
4. Performance considerations

Code:
{code}

Return only the test code with proper imports and assertions.
"""
            response = self.api_client.call_model(model, prompt)
            progress.stop()
        
        if not response.success:
            console.print(f"[red]Error generating tests: {response.error}[/red]")
            return
        
        generated_tests = response.content
        
        # Display generated test code
        console.print("\n[bold green]Generated Test Code:[/bold green]")
        test_syntax = Syntax(generated_tests, "python", theme="monokai", line_numbers=True)
        console.print(Panel(test_syntax, title="Test Cases"))
        
        # Save option
        should_save = save
        if should_save is None:
            should_save = Confirm.ask("Save test file?")
        
        if should_save:
            file_stem = Path(file).stem
            test_file = f"test_{file_stem}.py"
            self.write_file(test_file, generated_tests)
            
            # Run tests if requested
            if run_tests and Confirm.ask("Run the generated tests?"):
                self.run(test_file)
        
        # Log session
        self.session_history.append({
            "action": "test",
            "file": file,
            "model": model,
            "timestamp": time.time()
        })
    
    def analyze(self, file: str, comprehensive: bool = False, output_format: str = "tree"):
        """Comprehensive code analysis"""
        code = self.read_file(file)
        
        console.print(Panel(f"[bold blue]Code Analysis[/bold blue]\nFile: {file}"))
        
        # Basic analysis
        lines = code.split('\n')
        line_count = len(lines)
        char_count = len(code)
        blank_lines = sum(1 for line in lines if not line.strip())
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        
        # Function and class analysis
        import ast
        try:
            tree = ast.parse(code)
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        except SyntaxError:
            functions = classes = imports = []
        
        if output_format == "tree":
            # Create analysis tree
            analysis_tree = Tree("📁 Code Analysis")
            
            # Basic metrics
            metrics_branch = analysis_tree.add("📊 Basic Metrics")
            metrics_branch.add(f"Total lines: {line_count}")
            metrics_branch.add(f"Code lines: {line_count - blank_lines - comment_lines}")
            metrics_branch.add(f"Blank lines: {blank_lines}")
            metrics_branch.add(f"Comment lines: {comment_lines}")
            metrics_branch.add(f"Characters: {char_count}")
            
            # Structure analysis
            structure_branch = analysis_tree.add("🏗️ Structure")
            structure_branch.add(f"Functions: {len(functions)}")
            structure_branch.add(f"Classes: {len(classes)}")
            structure_branch.add(f"Imports: {len(imports)}")
            
            # Function details
            if functions:
                funcs_branch = analysis_tree.add("🔧 Functions")
                for func in functions:
                    func_name = func.name
                    args_count = len(func.args.args)
                    funcs_branch.add(f"def {func_name}() - {args_count} args")
            
            # Security analysis
            if comprehensive:
                security_warnings = self.runner.security_check(code)
                if security_warnings:
                    security_branch = analysis_tree.add("🔒 Security Issues")
                    for warning in security_warnings:
                        security_branch.add(f"⚠️ {warning}")
                else:
                    analysis_tree.add("🔒 Security: No issues detected")
        
            console.print(analysis_tree)
        
        else:  # table format
            analysis_table = Table(title="Code Analysis")
            analysis_table.add_column("Metric", style="cyan")
            analysis_table.add_column("Value", style="magenta")
            
            analysis_table.add_row("Total Lines", str(line_count))
            analysis_table.add_row("Code Lines", str(line_count - blank_lines - comment_lines))
            analysis_table.add_row("Functions", str(len(functions)))
            analysis_table.add_row("Classes", str(len(classes)))
            analysis_table.add_row("Imports", str(len(imports)))
            
            console.print(analysis_table)
        
        # Comprehensive analysis with AI
        if comprehensive:
            console.print("\n[bold cyan]Getting AI code review...[/bold cyan]")
            
            prompt = f"""
Analyze this Python code and provide insights on:
1. Code quality and style
2. Performance considerations  
3. Potential bugs or issues
4. Suggestions for improvement
5. Best practices compliance

Code:
{code}
"""
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Getting AI analysis...", total=None)
                response = self.api_client.call_model("deepseek", prompt)
                progress.stop()
            
            if response.success:
                console.print(Panel(response.content, title="[blue]AI Code Review[/blue]"))
    
    def stats(self):
        """Show session statistics and system info"""
        console.print(Panel("[bold blue]Session Statistics[/bold blue]"))
        
        if not self.session_history:
            console.print("[yellow]No actions performed in this session[/yellow]")
            return
        
        # Session stats
        actions = {}
        for entry in self.session_history:
            action = entry["action"]
            actions[action] = actions.get(action, 0) + 1
        
        session_table = Table(title="Session Activity")
        session_table.add_column("Action", style="cyan")
        session_table.add_column("Count", style="magenta")
        
        for action, count in actions.items():
            session_table.add_row(action.title(), str(count))
        
        console.print(session_table)
        
        # API stats
        api_stats = self.api_client.get_stats()
        if api_stats["total_calls"] > 0:
            api_table = Table(title="API Usage")
            api_table.add_column("Metric", style="cyan")
            api_table.add_column("Value", style="magenta")
            
            api_table.add_row("Total API Calls", str(api_stats["total_calls"]))
            api_table.add_row("Successful Calls", str(api_stats["successful_calls"]))
            api_table.add_row("Failed Calls", str(api_stats["failed_calls"]))
            api_table.add_row("Avg Response Time", f"{api_stats.get('average_response_time', 0):.2f}s")
            
            console.print(api_table)
        
        # System stats
        runner_stats = self.runner.get_stats()
        if runner_stats["total_executions"] > 0:
            exec_table = Table(title="Code Execution Stats")
            exec_table.add_column("Metric", style="cyan")
            exec_table.add_column("Value", style="magenta")
            
            exec_table.add_row("Total Executions", str(runner_stats["total_executions"]))
            exec_table.add_row("Successful", str(runner_stats["successful_executions"]))
            exec_table.add_row("Avg Execution Time", f"{runner_stats.get('average_execution_time', 0):.4f}s")
            
            console.print(exec_table)
    
    def config_set(self, key: str, value: str):
        """Set configuration option"""
        # Type conversion
        if value.lower() in ['true', 'false']:
            value = value.lower() == 'true'
        elif value.isdigit():
            value = int(value)
        elif value.replace('.', '').isdigit():
            value = float(value)
        
        self.config[key] = value
        self._save_config()
        console.print(f"[green]✓ Configuration updated: {key} = {value}[/green]")
    
    def config_show(self):
        """Show current configuration"""
        config_table = Table(title="Configuration")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="magenta")
        
        for key, value in self.config.items():
            config_table.add_row(key, str(value))
        
        console.print(config_table)

# Initialize the analyzer
analyzer = CodeAnalyzer()

# CLI Functions
def edit(file: str, model: str = None, interactive: bool = True, instruction: str = None):
    """Edit/improve code with AI assistance"""
    analyzer.edit(file, model, interactive, instruction)

def run(file: str, input_data: str = "", verbose: bool = False, benchmark: bool = False):
    """Execute Python code with monitoring"""
    analyzer.run(file, input_data, verbose, benchmark)

def debug(file: str, model: str = None, fix: bool = False, interactive: bool = True):
    """Debug code with AI analysis"""
    analyzer.debug(file, model, fix, interactive)

def test(file: str, model: str = None, save: bool = None, run_tests: bool = False):
    """Generate test cases for code"""
    analyzer.test(file, model, save, run_tests)

def analyze(file: str, comprehensive: bool = False, format: str = "tree"):
    """Analyze code structure and quality"""
    analyzer.analyze(file, comprehensive, format)

def stats():
    """Show session and system statistics"""
    analyzer.stats()

def config(action: str = "show", key: str = None, value: str = None):
    """Manage configuration (show, set)"""
    if action == "show":
        analyzer.config_show()
    elif action == "set" and key and value:
        analyzer.config_set(key, value)
    else:
        console.print("[red]Usage: config show | config set <key> <value>[/red]")

def models():
    """Show available AI models"""
    models_info = analyzer.api_client.get_model_info()
    
    models_table = Table(title="Available AI Models")
    models_table.add_column("Model", style="cyan")
    models_table.add_column("Description", style="magenta")
    
    for model, desc in models_info.items():
        models_table.add_row(model, desc)
    
    console.print(models_table)

def interactive():
    """Start interactive mode"""
    console.print(Panel("[bold green]Interactive Code Analysis Mode[/bold green]\nType 'help' for commands, 'exit' to quit"))
    
    while True:
        try:
            command = Prompt.ask("[bold cyan]analyzer>[/bold cyan]").strip()
            
            if command == "exit":
                console.print("[yellow]Goodbye![/yellow]")
                break
            elif command == "help":
                console.print("""
[bold cyan]Available Commands:[/bold cyan]
  edit <file>         - Edit code with AI
  run <file>          - Execute code
  debug <file>        - Debug code
  test <file>         - Generate tests
  analyze <file>      - Analyze code
  stats               - Show statistics
  config              - Show configuration
  models              - List AI models
  clear               - Clear screen
  exit                - Exit interactive mode
                """)
            elif command == "clear":
                os.system('clear' if os.name == 'posix' else 'cls')
            elif command == "stats":
                stats()
            elif command == "config":
                config()
            elif command == "models":
                models()
            elif command.startswith("edit "):
                file = command.split(" ", 1)[1]
                edit(file)
            elif command.startswith("run "):
                file = command.split(" ", 1)[1]
                run(file)
            elif command.startswith("debug "):
                file = command.split(" ", 1)[1]
                debug(file)
            elif command.startswith("test "):
                file = command.split(" ", 1)[1]
                test(file)
            elif command.startswith("analyze "):
                file = command.split(" ", 1)[1]
                analyze(file)
            else:
                console.print("[red]Unknown command. Type 'help' for available commands.[/red]")
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Use 'exit' to quit[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")

if __name__ == "__main__":
    try:
        fire.Fire({
            'edit': edit,
            'run': run,
            'debug': debug,
            'test': test,
            'analyze': analyze,
            'stats': stats,
            'config': config,
            'models': models,
            'interactive': interactive
        })
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled[/yellow]")
    except Exception as e:
        console.print(f"[red]Unexpected error: {str(e)}[/red]")
        sys.exit(1)
