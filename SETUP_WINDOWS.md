# Legal LLM Evaluation Framework - Windows Setup Guide

This guide provides step-by-step instructions for setting up and running the Legal LLM Evaluation Framework on Windows systems.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running Evaluations](#running-evaluations)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## Prerequisites

### System Requirements
- **Operating System**: Windows 10 or later (Windows 11 recommended)
- **Python**: Python 3.8 or later (Python 3.11 recommended)
- **Memory**: Minimum 8GB RAM (16GB recommended for large models)
- **Storage**: At least 2GB free space for dependencies
- **Internet**: Required for package installation and API-based models

### Required Software

#### 1. Install Python
Download and install Python from [python.org](https://www.python.org/downloads/windows/):

1. Download the latest Python 3.11.x installer for Windows
2. **Important**: Check "Add Python to PATH" during installation
3. Choose "Install for all users" if you have admin rights
4. Verify installation by opening Command Prompt and running:
   ```cmd
   python --version
   python -m pip --version
   ```

#### 2. Install Git (Optional but Recommended)
Download Git from [git-scm.com](https://git-scm.com/download/win):

1. Install with default settings
2. Verify installation:
   ```cmd
   git --version
   ```

#### 3. Install Windows Terminal (Recommended)
Install from Microsoft Store or download from [GitHub](https://github.com/microsoft/terminal):
- Provides better command-line experience
- Supports multiple terminal types
- Better Unicode support for evaluation outputs

## Installation

### Option 1: Clone from Repository (if available)
```cmd
git clone <repository-url>
cd legal_llm_eval
```

### Option 2: Manual Setup
If you don't have Git or the repository URL, create the directory structure manually:

```cmd
mkdir legal_llm_eval
cd legal_llm_eval
```

Then copy all the framework files to this directory.

### Set up Python Virtual Environment

1. **Create virtual environment**:
   ```cmd
   python -m venv legal_eval_env
   ```

2. **Activate the virtual environment**:
   ```cmd
   # Command Prompt
   legal_eval_env\Scripts\activate.bat
   
   # PowerShell
   legal_eval_env\Scripts\Activate.ps1
   
   # Git Bash
   source legal_eval_env/Scripts/activate
   ```

3. **Verify activation** (you should see `(legal_eval_env)` in your prompt):
   ```cmd
   (legal_eval_env) C:\path\to\legal_llm_eval>
   ```

4. **Upgrade pip**:
   ```cmd
   python -m pip install --upgrade pip
   ```

### Install Dependencies

Choose one of the following based on your needs:

#### Minimal Installation (Mock LLM only)
For testing and development without API costs:
```cmd
pip install -r requirements_minimal.txt
```

#### Full Installation (All LLM Providers)
For complete functionality with all supported models:
```cmd
pip install -r requirements.txt
```

#### Custom Installation
Install only specific providers:
```cmd
# Core dependencies
pip install pandas numpy requests aiohttp python-dotenv typing-extensions jsonlines

# For OpenAI models
pip install openai

# For Anthropic models  
pip install anthropic

# For local models (HuggingFace/Ollama)
pip install transformers torch huggingface-hub

# For text processing
pip install nltk spacy scikit-learn
```

## Configuration

### 1. Create Configuration File

Create a configuration file for your setup:

#### For Free/Mock Models (No API Keys Required)
Create `config_windows.json`:
```json
{
  "models": {
    "mock-legal": {
      "provider": "mock",
      "description": "Mock LLM for testing"
    }
  },
  "evaluation": {
    "k_values": [1, 3, 5],
    "max_samples": 10,
    "batch_size": 3
  },
  "datasets": ["legal_humaneval"],
  "output_dir": "results"
}
```

#### For Paid API Models
Create `config_api.json`:
```json
{
  "models": {
    "gpt-4": {
      "provider": "openai", 
      "api_key": "your-openai-api-key-here"
    },
    "claude-3": {
      "provider": "anthropic",
      "api_key": "your-anthropic-api-key-here"  
    },
    "deepseek": {
      "provider": "deepseek",
      "api_key": "your-deepseek-api-key-here"
    }
  },
  "evaluation": {
    "k_values": [1, 5, 10],
    "max_samples": 50,
    "batch_size": 5
  },
  "datasets": ["legal_humaneval"],
  "output_dir": "results"
}
```

### 2. Environment Variables (Optional)
Create a `.env` file for sensitive configuration:
```
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
DEEPSEEK_API_KEY=your-deepseek-key
```

### 3. Verify Installation
Test your setup:
```cmd
python -c "import pandas, numpy, requests; print('Core dependencies installed successfully')"
```

## Running Evaluations

### Basic Usage

1. **Ensure virtual environment is activated**:
   ```cmd
   legal_eval_env\Scripts\activate.bat
   ```

2. **Run basic evaluation**:
   ```cmd
   python main.py --benchmark benchmarks/legal_humaneval.jsonl --config config_windows.json --max-samples 5
   ```

3. **Check results**:
   ```cmd
   type results\evaluation_results_report.txt
   ```

### Advanced Usage Examples

#### 1. Evaluate Specific Legal Domains
```cmd
# Corporate law evaluation
python main.py --benchmark benchmarks/corporate_law_humaneval.jsonl --config config_windows.json --max-samples 3

# Intellectual property evaluation  
python main.py --benchmark benchmarks/intellectual_property_humaneval.jsonl --config config_windows.json --max-samples 3

# Employment law evaluation
python main.py --benchmark benchmarks/employment_law_humaneval.jsonl --config config_windows.json --max-samples 3
```

#### 2. Multiple Models Comparison
```cmd
python main.py --benchmark benchmarks/legal_humaneval.jsonl --config config_api.json --models gpt-4 claude-3 --max-samples 10
```

#### 3. Different Prompt Strategies
```cmd
python main.py --benchmark benchmarks/legal_humaneval.jsonl --config config_windows.json --strategies role_based chain_of_thought --max-samples 5
```

#### 4. Batch Processing Multiple Benchmarks
Create a batch script `run_all_evaluations.bat`:
```batch
@echo off
call legal_eval_env\Scripts\activate.bat

echo Running Legal HumanEval...
python main.py --benchmark benchmarks/legal_humaneval.jsonl --config config_windows.json --max-samples 5 --output results/legal_humaneval_results.json

echo Running Corporate Law Eval...
python main.py --benchmark benchmarks/corporate_law_humaneval.jsonl --config config_windows.json --max-samples 3 --output results/corporate_law_results.json

echo Running IP Law Eval...
python main.py --benchmark benchmarks/intellectual_property_humaneval.jsonl --config config_windows.json --max-samples 3 --output results/ip_law_results.json

echo Running Employment Law Eval...
python main.py --benchmark benchmarks/employment_law_humaneval.jsonl --config config_windows.json --max-samples 3 --output results/employment_law_results.json

echo All evaluations complete!
pause
```

Run the batch script:
```cmd
run_all_evaluations.bat
```

### PowerShell Scripts

For PowerShell users, create `run_evaluation.ps1`:
```powershell
# Activate virtual environment
& .\legal_eval_env\Scripts\Activate.ps1

# Run evaluation
python main.py --benchmark benchmarks/legal_humaneval.jsonl --config config_windows.json --max-samples 5

# Display results
Get-Content results\evaluation_results_report.txt
```

Run with:
```powershell
PowerShell -ExecutionPolicy Bypass -File run_evaluation.ps1
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Python Not Found
**Error**: `'python' is not recognized as an internal or external command`

**Solutions**:
- Reinstall Python and check "Add Python to PATH"
- Use `py` instead of `python`:
  ```cmd
  py --version
  py -m venv legal_eval_env
  ```
- Manually add Python to PATH:
  1. Find Python installation (usually `C:\Users\{username}\AppData\Local\Programs\Python\Python311\`)
  2. Add to System PATH environment variable

#### 2. Virtual Environment Activation Issues
**Error**: Execution policies prevent script execution

**Solutions**:
```powershell
# For PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or use Command Prompt instead
legal_eval_env\Scripts\activate.bat
```

#### 3. Package Installation Failures
**Error**: Package installation fails with SSL/certificate errors

**Solutions**:
```cmd
# Upgrade pip with trusted hosts
python -m pip install --upgrade pip --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org

# Install packages with trusted hosts
pip install -r requirements_minimal.txt --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org
```

#### 4. Memory Issues with Large Models
**Error**: Out of memory when loading models

**Solutions**:
- Use minimal requirements for mock models only
- Reduce batch size in configuration:
  ```json
  "evaluation": {
    "batch_size": 1
  }
  ```
- Close other applications during evaluation

#### 5. Path Issues
**Error**: Files not found or incorrect paths

**Solutions**:
- Use forward slashes or double backslashes in paths:
  ```cmd
  python main.py --benchmark benchmarks/legal_humaneval.jsonl
  # or
  python main.py --benchmark benchmarks\\legal_humaneval.jsonl
  ```

#### 6. Encoding Issues
**Error**: Unicode/encoding errors in output

**Solutions**:
```cmd
# Set environment variables
set PYTHONIOENCODING=utf-8
chcp 65001

# Or run with explicit encoding
python -X utf8 main.py --benchmark benchmarks/legal_humaneval.jsonl --config config_windows.json
```

### Performance Optimization

#### 1. Faster Evaluation
```json
{
  "evaluation": {
    "k_values": [1],
    "max_samples": 5,
    "batch_size": 1
  }
}
```

#### 2. Reduce Dependencies
Use minimal requirements and remove unused features:
```cmd
pip install pandas numpy requests aiohttp jsonlines
```

#### 3. Use SSD Storage
Install on SSD drive for faster I/O operations.

## Validating Your Setup

Create a test script `test_setup.py`:
```python
#!/usr/bin/env python3
"""
Test script to validate Legal LLM Evaluation Framework setup on Windows
"""
import sys
import subprocess
import importlib

def test_python_version():
    """Test Python version compatibility"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version compatible")
        return True
    else:
        print("❌ Python version too old (requires 3.8+)")
        return False

def test_dependencies():
    """Test required dependencies"""
    required = [
        'pandas', 'numpy', 'requests', 'aiohttp', 
        'jsonlines', 'typing_extensions'
    ]
    
    missing = []
    for package in required:
        try:
            importlib.import_module(package)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} missing")
            missing.append(package)
    
    return len(missing) == 0

def test_framework_files():
    """Test framework file structure"""
    import os
    
    required_files = [
        'main.py',
        'prompts/prompt_optimizer.py',
        'evaluator/legal_evaluator.py', 
        'models/llm_interface.py',
        'datasets/dataset_loader.py',
        'benchmarks/legal_humaneval.jsonl'
    ]
    
    missing = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} found")
        else:
            print(f"❌ {file_path} missing")
            missing.append(file_path)
    
    return len(missing) == 0

def test_basic_import():
    """Test basic framework imports"""
    try:
        from prompts.prompt_optimizer import LegalPromptOptimizer
        from evaluator.legal_evaluator import LegalEvaluator
        from models.llm_interface import LLMEvaluationManager
        print("✅ Framework imports successful")
        return True
    except ImportError as e:
        print(f"❌ Framework import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Legal LLM Evaluation Framework - Windows Setup Test")
    print("=" * 50)
    
    tests = [
        ("Python Version", test_python_version),
        ("Dependencies", test_dependencies), 
        ("Framework Files", test_framework_files),
        ("Basic Imports", test_basic_import)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    if all(results):
        print("🎉 All tests passed! Framework is ready to use.")
        print("\nNext steps:")
        print("1. Run: python main.py --benchmark benchmarks/legal_humaneval.jsonl --config config_windows.json --max-samples 3")
        print("2. Check results in: results/evaluation_results_report.txt")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- pip install -r requirements_minimal.txt")
        print("- Ensure all framework files are in the correct locations")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

Run the test:
```cmd
python test_setup.py
```

## Next Steps

After successful setup:

1. **Run Quick Test**:
   ```cmd
   python main.py --benchmark benchmarks/legal_humaneval.jsonl --config config_windows.json --max-samples 3
   ```

2. **Explore Results**:
   - JSON results: `results/evaluation_results.json`
   - Human-readable report: `results/evaluation_results_report.txt`

3. **Try Different Domains**:
   - Corporate law: `benchmarks/corporate_law_humaneval.jsonl`
   - IP law: `benchmarks/intellectual_property_humaneval.jsonl`
   - Employment law: `benchmarks/employment_law_humaneval.jsonl`

4. **Add Real LLM Providers**:
   - Get API keys from OpenAI, Anthropic, or DeepSeek
   - Update configuration files
   - Run comparative evaluations

5. **Customize for Your Use Case**:
   - Add new legal task types in `prompts/prompt_optimizer.py`
   - Create custom benchmark datasets
   - Implement new evaluation metrics in `evaluator/legal_evaluator.py`

## Support

For issues specific to Windows setup:
- Check the troubleshooting section above
- Ensure you're using a supported Python version (3.8+)
- Try using Command Prompt if PowerShell has execution policy issues
- Use Windows Terminal for better Unicode support

For framework-specific issues:
- Review the main README.md
- Check the code documentation in each module
- Validate your configuration files are properly formatted JSON