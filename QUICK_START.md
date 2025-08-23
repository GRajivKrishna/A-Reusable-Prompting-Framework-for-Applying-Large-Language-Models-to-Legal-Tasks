# Legal LLM Evaluation Framework - Quick Start Guide

Get the Legal LLM Evaluation Framework running in 5 minutes on Windows, Mac, or Linux.

## 🚀 Super Quick Start (Windows)

### Option 1: Automated Setup (Recommended)
1. **Download/extract the framework files to a folder**
2. **Open Command Prompt in that folder**
3. **Run the automated installer**:
   ```cmd
   run_windows.bat
   ```
4. **Follow the prompts** - the script will:
   - Install Python virtual environment
   - Install all dependencies
   - Create default configuration
   - Run a sample evaluation
   - Show you the results

### Option 2: PowerShell (Alternative)
1. **Open PowerShell in the framework folder**
2. **Run**:
   ```powershell
   .\run_windows.ps1
   ```
3. **If you get execution policy errors**:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   .\run_windows.ps1
   ```

## 🐧 Quick Start (Linux/Mac)

```bash
# 1. Clone or download framework files
cd legal_llm_eval

# 2. Set up virtual environment
python3 -m venv legal_eval_env
source legal_eval_env/bin/activate

# 3. Install dependencies
pip install -r requirements_minimal.txt

# 4. Run evaluation
python main.py --benchmark benchmarks/legal_humaneval.jsonl --config config_free.json --max-samples 5

# 5. View results
cat results/evaluation_results_report.txt
```

## 📊 What You'll See

After running the evaluation, you'll get results like this:

```
Legal LLM Evaluation Report
==================================================

Benchmark: benchmarks/legal_humaneval.jsonl
Total samples: 5
Models evaluated: mock-legal

Model Performance Summary:
------------------------------

mock-legal:
  Strategy: role_based
    pass@1: 0.600
    pass@3: 0.800
    avg_citation_accuracy: 0.400
    avg_reasoning_coherence: 0.360
    avg_completeness: 0.200

Efficiency Metrics:
mock-legal:
  Avg response time: 0.50s
  Avg tokens: 187
  Error rate: 0.0%
```

## 🎯 Next Steps

### Try Different Legal Domains
```cmd
# Corporate law (M&A, securities, governance)
python main.py --benchmark benchmarks/corporate_law_humaneval.jsonl --config config_free.json --max-samples 3

# Intellectual property (patents, trademarks, copyright)
python main.py --benchmark benchmarks/intellectual_property_humaneval.jsonl --config config_free.json --max-samples 3

# Employment law (discrimination, wages, benefits)
python main.py --benchmark benchmarks/employment_law_humaneval.jsonl --config config_free.json --max-samples 3
```

### Add Real LLM Models
1. **Get API keys** from OpenAI, Anthropic, or DeepSeek
2. **Create config file** `config_api.json`:
   ```json
   {
     "models": {
       "gpt-4": {
         "provider": "openai", 
         "api_key": "your-openai-key"
       },
       "claude-3": {
         "provider": "anthropic",
         "api_key": "your-anthropic-key"
       }
     },
     "evaluation": {
       "k_values": [1, 5, 10],
       "max_samples": 20,
       "batch_size": 5
     }
   }
   ```
3. **Run comparative evaluation**:
   ```cmd
   python main.py --benchmark benchmarks/legal_humaneval.jsonl --config config_api.json --models gpt-4 claude-3
   ```

### Compare Prompt Strategies
```cmd
python main.py --benchmark benchmarks/legal_humaneval.jsonl --config config_free.json --strategies role_based chain_of_thought --max-samples 5
```

## 🛠️ Troubleshooting

### Windows Issues
- **"Python not found"**: Install Python from [python.org](https://python.org) and check "Add to PATH"
- **"Execution policy"**: Run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` in PowerShell
- **Virtual environment fails**: Use Command Prompt instead of PowerShell
- **Dependencies fail**: Try `pip install pandas numpy requests aiohttp jsonlines`

### General Issues
- **"Module not found"**: Make sure virtual environment is activated
- **"File not found"**: Ensure you're in the correct directory with framework files
- **API errors**: Check your API keys and internet connection
- **Memory issues**: Reduce `max_samples` to 1-3 and `batch_size` to 1

### Getting Help
1. **Check error messages** - they usually indicate the specific issue
2. **Try the test script**:
   ```cmd
   python test_setup.py
   ```
3. **Use minimal setup** if full installation fails:
   ```cmd
   pip install pandas numpy requests aiohttp jsonlines
   ```

## 📁 File Structure Overview

```
legal_llm_eval/
├── run_windows.bat         # ← Automated Windows setup
├── run_windows.ps1         # ← PowerShell alternative
├── main.py                 # ← Main evaluation script
├── config_free.json        # ← Free models configuration
├── benchmarks/             # ← Legal task datasets
│   ├── legal_humaneval.jsonl
│   ├── corporate_law_humaneval.jsonl
│   ├── intellectual_property_humaneval.jsonl
│   └── employment_law_humaneval.jsonl
├── results/                # ← Output directory
├── prompts/                # ← Prompt optimization
├── evaluator/              # ← Evaluation engine
├── models/                 # ← LLM interfaces
└── datasets/               # ← Data loading utilities
```

## 🎉 Success Indicators

You know everything is working when:
- ✅ Virtual environment activates without errors
- ✅ Dependencies install successfully  
- ✅ Evaluation runs and produces results files
- ✅ You see legal metrics like citation accuracy and reasoning coherence
- ✅ Results are saved to `results/evaluation_results_report.txt`

## 🚀 Ready to Go Further?

Once you have the basic setup working:
- Read `SETUP_WINDOWS.md` for detailed configuration options
- Check `REPLICATION_GUIDE.md` to understand the implementation
- Explore `README.md` for comprehensive documentation
- Add your own legal datasets and evaluation metrics
- Integrate with your preferred LLM providers

**Congratulations!** You now have a working Legal LLM Evaluation Framework that can systematically assess legal reasoning capabilities across multiple domains and models. 🎊