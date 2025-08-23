@echo off
REM Legal LLM Evaluation Framework - Windows Batch Runner
REM This script automates the setup and execution process on Windows

echo Legal LLM Evaluation Framework - Windows Runner
echo ================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python found: 
python --version

REM Check if virtual environment exists
if not exist "legal_eval_env" (
    echo Creating virtual environment...
    python -m venv legal_eval_env
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call legal_eval_env\Scripts\activate.bat

REM Check if dependencies are installed
python -c "import pandas, numpy, requests" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements_minimal.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        echo Trying with full requirements...
        pip install pandas numpy requests aiohttp python-dotenv typing-extensions jsonlines
        if errorlevel 1 (
            echo ERROR: Failed to install minimal dependencies
            pause
            exit /b 1
        )
    )
)

REM Verify framework files exist
if not exist "main.py" (
    echo ERROR: main.py not found in current directory
    echo Please ensure you are in the legal_llm_eval directory
    pause
    exit /b 1
)

if not exist "benchmarks\legal_humaneval.jsonl" (
    echo ERROR: Benchmark file not found
    echo Please ensure benchmarks directory exists with legal_humaneval.jsonl
    pause
    exit /b 1
)

REM Create results directory if it doesn't exist
if not exist "results" mkdir results

REM Create default configuration if it doesn't exist
if not exist "config_windows.json" (
    echo Creating default Windows configuration...
    echo { > config_windows.json
    echo   "models": { >> config_windows.json
    echo     "mock-legal": { >> config_windows.json
    echo       "provider": "mock", >> config_windows.json
    echo       "description": "Mock LLM for testing" >> config_windows.json
    echo     } >> config_windows.json
    echo   }, >> config_windows.json
    echo   "evaluation": { >> config_windows.json
    echo     "k_values": [1, 3, 5], >> config_windows.json
    echo     "max_samples": 10, >> config_windows.json
    echo     "batch_size": 3 >> config_windows.json
    echo   }, >> config_windows.json
    echo   "datasets": ["legal_humaneval"], >> config_windows.json
    echo   "output_dir": "results" >> config_windows.json
    echo } >> config_windows.json
)

echo.
echo Configuration:
echo - Virtual environment: legal_eval_env
echo - Configuration file: config_windows.json
echo - Output directory: results
echo.

REM Get user input for evaluation parameters
set /p BENCHMARK="Enter benchmark file (default: benchmarks/legal_humaneval.jsonl): "
if "%BENCHMARK%"=="" set BENCHMARK=benchmarks/legal_humaneval.jsonl

set /p MAX_SAMPLES="Enter max samples (default: 5): "
if "%MAX_SAMPLES%"=="" set MAX_SAMPLES=5

echo.
echo Starting evaluation...
echo - Benchmark: %BENCHMARK%
echo - Max samples: %MAX_SAMPLES%
echo - Configuration: config_windows.json
echo.

REM Run the evaluation
python main.py --benchmark %BENCHMARK% --config config_windows.json --max-samples %MAX_SAMPLES%

if errorlevel 1 (
    echo.
    echo ERROR: Evaluation failed
    echo Check the error messages above for details
    pause
    exit /b 1
)

echo.
echo ================================================
echo Evaluation completed successfully!
echo ================================================
echo.

REM Display results
if exist "results\evaluation_results_report.txt" (
    echo Results Summary:
    echo ----------------
    type results\evaluation_results_report.txt
    echo.
    echo Full results saved to: results\evaluation_results.json
    echo Human-readable report: results\evaluation_results_report.txt
) else (
    echo WARNING: Results file not found
)

echo.
echo Additional commands you can try:
echo.
echo 1. Corporate Law Evaluation:
echo    python main.py --benchmark benchmarks/corporate_law_humaneval.jsonl --config config_windows.json --max-samples 3
echo.
echo 2. Intellectual Property Evaluation:
echo    python main.py --benchmark benchmarks/intellectual_property_humaneval.jsonl --config config_windows.json --max-samples 3
echo.
echo 3. Employment Law Evaluation:
echo    python main.py --benchmark benchmarks/employment_law_humaneval.jsonl --config config_windows.json --max-samples 3
echo.

pause