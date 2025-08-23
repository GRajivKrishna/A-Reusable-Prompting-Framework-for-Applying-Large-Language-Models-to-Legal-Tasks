# Legal LLM Evaluation Framework - Windows PowerShell Runner
# This script automates the setup and execution process on Windows using PowerShell

Write-Host "Legal LLM Evaluation Framework - Windows PowerShell Runner" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Function to check if a command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Check if Python is installed
if (-not (Test-Command "python")) {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://python.org" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Python found: " -NoNewline
python --version

# Check if virtual environment exists
if (-not (Test-Path "legal_eval_env")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv legal_eval_env
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow

# Check PowerShell execution policy
$executionPolicy = Get-ExecutionPolicy
if ($executionPolicy -eq "Restricted") {
    Write-Host "PowerShell execution policy is restricted. Attempting to set for current user..." -ForegroundColor Yellow
    try {
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
        Write-Host "Execution policy updated successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "WARNING: Could not update execution policy. You may need to run as administrator or use Command Prompt instead." -ForegroundColor Yellow
        Write-Host "Alternative: Run 'run_windows.bat' from Command Prompt" -ForegroundColor Yellow
    }
}

# Activate virtual environment
try {
    & .\legal_eval_env\Scripts\Activate.ps1
    Write-Host "Virtual environment activated" -ForegroundColor Green
}
catch {
    Write-Host "WARNING: Could not activate virtual environment using PowerShell script" -ForegroundColor Yellow
    Write-Host "Trying alternative activation method..." -ForegroundColor Yellow
    
    # Try using cmd to activate
    cmd /c "legal_eval_env\Scripts\activate.bat && python --version"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to activate virtual environment" -ForegroundColor Red
        Write-Host "Please try running 'run_windows.bat' from Command Prompt instead" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Check if dependencies are installed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
python -c "import pandas, numpy, requests" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    
    if (Test-Path "requirements_minimal.txt") {
        pip install -r requirements_minimal.txt
    }
    else {
        Write-Host "requirements_minimal.txt not found, installing core dependencies..." -ForegroundColor Yellow
        pip install pandas numpy requests aiohttp python-dotenv typing-extensions jsonlines
    }
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "Dependencies installed successfully" -ForegroundColor Green
}
else {
    Write-Host "Dependencies already installed" -ForegroundColor Green
}

# Verify framework files exist
$requiredFiles = @("main.py", "benchmarks\legal_humaneval.jsonl")
foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        Write-Host "ERROR: Required file '$file' not found" -ForegroundColor Red
        Write-Host "Please ensure you are in the legal_llm_eval directory with all framework files" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Create results directory if it doesn't exist
if (-not (Test-Path "results")) {
    New-Item -ItemType Directory -Path "results" | Out-Null
    Write-Host "Created results directory" -ForegroundColor Green
}

# Create default configuration if it doesn't exist
if (-not (Test-Path "config_windows.json")) {
    Write-Host "Creating default Windows configuration..." -ForegroundColor Yellow
    
    $config = @{
        models = @{
            "mock-legal" = @{
                provider = "mock"
                description = "Mock LLM for testing"
            }
        }
        evaluation = @{
            k_values = @(1, 3, 5)
            max_samples = 10
            batch_size = 3
        }
        datasets = @("legal_humaneval")
        output_dir = "results"
    }
    
    $config | ConvertTo-Json -Depth 3 | Out-File -FilePath "config_windows.json" -Encoding UTF8
    Write-Host "Configuration file created: config_windows.json" -ForegroundColor Green
}

Write-Host ""
Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "- Virtual environment: legal_eval_env" -ForegroundColor White
Write-Host "- Configuration file: config_windows.json" -ForegroundColor White
Write-Host "- Output directory: results" -ForegroundColor White
Write-Host ""

# Get user input for evaluation parameters
$benchmark = Read-Host "Enter benchmark file (default: benchmarks/legal_humaneval.jsonl)"
if ([string]::IsNullOrWhiteSpace($benchmark)) {
    $benchmark = "benchmarks/legal_humaneval.jsonl"
}

$maxSamples = Read-Host "Enter max samples (default: 5)"
if ([string]::IsNullOrWhiteSpace($maxSamples)) {
    $maxSamples = "5"
}

Write-Host ""
Write-Host "Starting evaluation..." -ForegroundColor Cyan
Write-Host "- Benchmark: $benchmark" -ForegroundColor White
Write-Host "- Max samples: $maxSamples" -ForegroundColor White
Write-Host "- Configuration: config_windows.json" -ForegroundColor White
Write-Host ""

# Run the evaluation
python main.py --benchmark $benchmark --config config_windows.json --max-samples $maxSamples

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Evaluation failed" -ForegroundColor Red
    Write-Host "Check the error messages above for details" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Evaluation completed successfully!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

# Display results
if (Test-Path "results\evaluation_results_report.txt") {
    Write-Host "Results Summary:" -ForegroundColor Cyan
    Write-Host "----------------" -ForegroundColor Cyan
    Get-Content "results\evaluation_results_report.txt"
    Write-Host ""
    Write-Host "Full results saved to: results\evaluation_results.json" -ForegroundColor Green
    Write-Host "Human-readable report: results\evaluation_results_report.txt" -ForegroundColor Green
}
else {
    Write-Host "WARNING: Results file not found" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Additional commands you can try:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Corporate Law Evaluation:" -ForegroundColor Yellow
Write-Host "   python main.py --benchmark benchmarks/corporate_law_humaneval.jsonl --config config_windows.json --max-samples 3" -ForegroundColor White
Write-Host ""
Write-Host "2. Intellectual Property Evaluation:" -ForegroundColor Yellow
Write-Host "   python main.py --benchmark benchmarks/intellectual_property_humaneval.jsonl --config config_windows.json --max-samples 3" -ForegroundColor White
Write-Host ""
Write-Host "3. Employment Law Evaluation:" -ForegroundColor Yellow
Write-Host "   python main.py --benchmark benchmarks/employment_law_humaneval.jsonl --config config_windows.json --max-samples 3" -ForegroundColor White
Write-Host ""

# Function to run additional evaluations
function Start-AdditionalEvaluation {
    param(
        [string]$EvalType,
        [string]$BenchmarkFile
    )
    
    $response = Read-Host "Would you like to run $EvalType evaluation now? (y/n)"
    if ($response -eq "y" -or $response -eq "Y") {
        Write-Host "Starting $EvalType evaluation..." -ForegroundColor Cyan
        python main.py --benchmark $BenchmarkFile --config config_windows.json --max-samples 3
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "$EvalType evaluation completed successfully!" -ForegroundColor Green
        }
        else {
            Write-Host "$EvalType evaluation failed" -ForegroundColor Red
        }
        Write-Host ""
    }
}

# Offer to run additional evaluations
Start-AdditionalEvaluation "Corporate Law" "benchmarks/corporate_law_humaneval.jsonl"
Start-AdditionalEvaluation "Intellectual Property" "benchmarks/intellectual_property_humaneval.jsonl"
Start-AdditionalEvaluation "Employment Law" "benchmarks/employment_law_humaneval.jsonl"

Read-Host "Press Enter to exit"