# Code Analyzer Usage Guide

A comprehensive AI-powered code analysis, debugging, and improvement tool with advanced features for Python development.

## 🚀 Quick Start

```bash
# Basic usage
python main.py <command> <file>

# Interactive mode
python main.py interactive

# Get help
python main.py --help
```

## 📋 Commands Overview

### 1. **edit** - AI-Powered Code Editing
Improve and optimize your code with AI assistance.

```bash
# Interactive editing with AI suggestions
python main.py edit script.py

# Non-interactive editing with specific instructions
python main.py edit script.py --instruction "Add error handling and logging"

# Use specific AI model
python main.py edit script.py --model deepseek

# Automated improvement (no prompts)
python main.py edit script.py --interactive=False
```

**Features:**
- AI-powered code optimization
- Interactive editing with previews
- Automatic backup creation
- Support for multiple AI models
- Custom instruction support

### 2. **run** - Enhanced Code Execution
Execute Python code with comprehensive monitoring and analysis.

```bash
# Basic execution
python main.py run script.py

# With input data
python main.py run script.py --input_data "test input"

# Verbose output with security analysis
python main.py run script.py --verbose

# Performance benchmarking (5 runs)
python main.py run script.py --benchmark
```

**Features:**
- Security warnings and checks
- Execution time and memory monitoring
- Performance benchmarking
- Detailed error reporting
- Input data support

### 3. **debug** - AI Debugging Assistant
Intelligent debugging with AI-powered error analysis and fixes.

```bash
# Analyze and explain errors
python main.py debug buggy_script.py

# Get AI-generated fixes
python main.py debug buggy_script.py --fix

# Use specific debugging model
python main.py debug buggy_script.py --model qwen

# Non-interactive debugging
python main.py debug buggy_script.py --interactive=False
```

**Features:**
- Intelligent error categorization
- AI-powered error explanations
- Automatic fix generation
- Code context analysis
- Prevention tips and best practices

### 4. **test** - Smart Test Generation
Generate comprehensive test suites with AI assistance.

```bash
# Generate test cases
python main.py test mymodule.py

# Auto-save test file
python main.py test mymodule.py --save

# Generate and run tests immediately
python main.py test mymodule.py --run_tests

# Use specific model for test generation
python main.py test mymodule.py --model qwen
```

**Features:**
- Comprehensive test case generation
- Edge case identification
- Error condition testing
- Performance test considerations
- Automatic test file creation

### 5. **analyze** - Code Analysis & Review
Perform detailed code analysis and get AI-powered reviews.

```bash
# Basic code analysis
python main.py analyze script.py

# Comprehensive analysis with AI review
python main.py analyze script.py --comprehensive

# Tree view format (default)
python main.py analyze script.py --format tree

# Table format
python main.py analyze script.py --format table
```

**Features:**
- Code metrics and statistics
- Structure analysis (functions, classes, imports)
- Security vulnerability detection
- AI-powered code review
- Best practices compliance check

### 6. **stats** - Session Statistics
View detailed statistics about your analysis session.

```bash
python main.py stats
```

**Shows:**
- Session activity summary
- API usage statistics
- Code execution metrics
- Performance data

### 7. **config** - Configuration Management
Manage tool settings and preferences.

```bash
# Show current configuration
python main.py config

# Set configuration values
python main.py config set default_model deepseek
python main.py config set timeout 15
python main.py config set security_checks true
```

**Available Settings:**
- `default_model`: Default AI model for editing
- `debug_model`: Model for debugging tasks
- `test_model`: Model for test generation
- `timeout`: Execution timeout in seconds
- `memory_limit_mb`: Memory limit for code execution
- `security_checks`: Enable/disable security warnings
- `auto_save_results`: Auto-save analysis results
- `theme`: UI theme (dark/light)

### 8. **models** - Available AI Models
List all available AI models and their descriptions.

```bash
python main.py models
```

### 9. **interactive** - Interactive Mode
Start an interactive session for continuous code analysis.

```bash
python main.py interactive
```

**Interactive Commands:**
- `edit <file>` - Edit code interactively
- `run <file>` - Execute code
- `debug <file>` - Debug code
- `test <file>` - Generate tests
- `analyze <file>` - Analyze code
- `stats` - Show statistics
- `config` - Show configuration
- `models` - List models
- `clear` - Clear screen
- `help` - Show help
- `exit` - Exit interactive mode

## 🛠️ Advanced Usage Examples

### Workflow 1: Complete Code Development Cycle
```bash
# 1. Analyze existing code
python main.py analyze my_project.py --comprehensive

# 2. Improve code with AI
python main.py edit my_project.py --instruction "Optimize performance and add type hints"

# 3. Generate comprehensive tests
python main.py test my_project.py --save --run_tests

# 4. Debug any issues
python main.py debug my_project.py --fix

# 5. Final execution with benchmarking
python main.py run my_project.py --benchmark --verbose
```

### Workflow 2: Error Investigation
```bash
# 1. Run code to identify errors
python main.py run problematic_code.py --verbose

# 2. Debug with AI assistance
python main.py debug problematic_code.py --fix --interactive

# 3. Verify fix
python main.py run problematic_code.py
```

### Workflow 3: Code Quality Assessment
```bash
# 1. Comprehensive analysis
python main.py analyze legacy_code.py --comprehensive

# 2. Get AI improvement suggestions
python main.py edit legacy_code.py --instruction "Modernize code and improve readability"

# 3. Generate tests for refactored code
python main.py test legacy_code.py --save
```

## 🔧 Configuration Options

### Security Settings
```bash
# Enable/disable security checks
python main.py config set security_checks true

# Set memory limits
python main.py config set memory_limit_mb 200
```

### Model Preferences
```bash
# Set default models for different tasks
python main.py config set default_model deepseek
python main.py config set debug_model qwen
python main.py config set test_model deepseek
```

### Performance Settings
```bash
# Set execution timeout
python main.py config set timeout 30

# Enable auto-save
python main.py config set auto_save_results true
```

## 📁 File Management

### Automatic Backups
- Original files are automatically backed up before modifications
- Backup files are named with timestamps: `script.py.backup_1640995200`

### Output Files
- Test files are saved as `test_<filename>.py`
- Analysis results can be auto-saved (configurable)
- Configuration stored in `~/.code_analyzer_config.json`

## 🔍 Understanding Output

### Execution Results
- **Status**: Success/Error/Timeout
- **Execution Time**: Runtime in seconds
- **Memory Used**: Peak memory usage in KB
- **Exit Code**: Process exit code

### Debug Analysis
- **Error Type**: Categorized error classification
- **Severity**: Impact level assessment
- **Confidence**: AI confidence in analysis
- **Line Number**: Specific error location
- **Code Context**: Relevant code snippet
- **Suggested Fixes**: Actionable solutions

### Code Analysis
- **Basic Metrics**: Lines, functions, classes, imports
- **Structure Analysis**: Code organization assessment
- **Security Issues**: Potential vulnerabilities
- **AI Review**: Comprehensive code quality assessment

## 🚨 Security Features

### Automatic Security Checks
- Detection of potentially dangerous operations
- Warning for file system access
- Network operation alerts
- Subprocess execution warnings

### Safe Execution Environment
- Memory usage monitoring
- Execution time limits
- Sandboxed code execution
- Input validation

## 💡 Tips & Best Practices

### Getting Better AI Results
1. **Be Specific**: Use detailed instructions for editing
2. **Iterative Improvement**: Use multiple small improvements vs. one large change
3. **Model Selection**: Choose appropriate models for different tasks
4. **Context Matters**: Provide relevant context in instructions

### Optimal Workflow
1. **Analyze First**: Always start with code analysis
2. **Test Early**: Generate tests before major changes
3. **Debug Incrementally**: Fix issues one at a time
4. **Benchmark Performance**: Use benchmark mode for optimization

### Configuration Tips
1. **Set Model Preferences**: Configure models for different use cases
2. **Enable Security**: Keep security checks enabled
3. **Adjust Timeouts**: Set appropriate limits for your system
4. **Use Auto-save**: Enable for important analysis sessions

## 🔗 Integration Examples

### CI/CD Integration
```bash
# Pre-commit hook example
python main.py analyze $1 --comprehensive
python main.py test $1 --save
```

### IDE Integration
```bash
# Quick debug command
python main.py debug current_file.py --fix --interactive=false
```

### Batch Processing
```bash
# Process multiple files
for file in *.py; do
    python main.py analyze "$file" --comprehensive
done
```

## 📊 Performance Monitoring

### Benchmark Mode
- Runs code multiple times (default: 5)
- Provides statistical analysis
- Shows min/max/average execution times
- Calculates standard deviation

### Memory Monitoring
- Tracks peak memory usage
- Warns about memory limit violations
- Provides memory optimization suggestions

### API Usage Tracking
- Monitors API call statistics
- Tracks response times
- Shows success/failure rates

---

## 🆘 Troubleshooting

### Common Issues

**"File not found" errors**
- Ensure file path is correct
- Use absolute paths if needed
- Check file permissions

**API connection issues**
- Verify internet connection
- Check model availability
- Review API rate limits

**Execution timeouts**
- Increase timeout in config
- Optimize code for performance
- Check for infinite loops

**Memory limit exceeded**
- Increase memory limit in config
- Optimize memory usage
- Use generators for large data

### Getting Help
- Use `--help` flag with any command
- Check `python main.py interactive` for guided usage
- Review configuration with `python main.py config`
- Use `python main.py stats` to monitor tool performance

---

*This tool provides AI-powered assistance for code development. Always review AI suggestions before applying them to production code.*
