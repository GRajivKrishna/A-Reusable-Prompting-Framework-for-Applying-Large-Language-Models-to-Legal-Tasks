# Legal LLM Evaluation Framework

A comprehensive evaluation framework for legal Large Language Models (LLMs) inspired by HumanEval. This framework systematically evaluates LLMs across various legal reasoning tasks including contract analysis, statutory interpretation, case summarization, and legal Q&A.

## Features

- **Multi-LLM Support**: Evaluate OpenAI GPT, Anthropic Claude, DeepSeek, and other models
- **Legal Task Coverage**: Contract review, statutory interpretation, case summarization, precedent retrieval, legal Q&A
- **HumanEval-Inspired**: Systematic evaluation with pass@k metrics adapted for legal reasoning
- **Prompt Optimization**: Domain-specific prompting strategies (role-based, chain-of-thought, context-layered)
- **Comprehensive Metrics**: Citation accuracy, reasoning coherence, completeness, hallucination detection
- **Dataset Integration**: Support for LegalBench, CUAD, LexGLUE, and custom legal datasets

## Installation

```bash
git clone <repository-url>
cd legal_llm_eval
pip install -r requirements.txt
```

## Quick Start

1. **Setup API Keys**: Configure your LLM API keys in a config file:

```json
{
  "models": {
    "gpt-4": {"provider": "openai", "api_key": "your-openai-key"},
    "claude-3": {"provider": "anthropic", "api_key": "your-anthropic-key"},
    "deepseek": {"provider": "deepseek", "api_key": "your-deepseek-key"}
  }
}
```

2. **Run Evaluation**: 

```bash
python main.py --benchmark benchmarks/legal_humaneval.jsonl --config config.json
```

3. **View Results**: Check the generated report in `results/evaluation_results_report.txt`

## Architecture

```
legal_llm_eval/
├── prompts/             # Prompt optimization framework
│   └── prompt_optimizer.py
├── evaluator/           # Legal evaluation engine  
│   └── legal_evaluator.py
├── models/              # LLM integration layer
│   └── llm_interface.py
├── datasets/            # Dataset loading and processing
│   └── dataset_loader.py
├── benchmarks/          # Legal benchmark datasets
│   └── legal_humaneval.jsonl
└── main.py             # Main execution pipeline
```

## Legal Tasks Supported

### 1. Contract Review
- Clause identification and analysis
- Risk assessment and recommendations
- Legal compliance checking

### 2. Statutory Interpretation
- Legislative text analysis
- Jurisdictional application
- Element-by-element breakdown

### 3. Case Summarization
- Key holdings extraction
- Legal significance analysis
- Precedent impact assessment

### 4. Legal Q&A
- Context-aware legal guidance
- Multi-jurisdictional considerations
- Practical recommendations

### 5. Precedent Retrieval
- Relevant case identification
- Legal principle matching
- Citation accuracy verification

## Evaluation Metrics

### Core Metrics (adapted from HumanEval)
- **pass@k**: Percentage of tasks passed in top k attempts
- **citation_accuracy@k**: Correct legal citations percentage
- **reasoning_coherence@k**: Logical consistency scoring
- **completeness@k**: Coverage of required legal elements

### Legal-Specific Metrics
- **Hallucination Rate**: Detection of false legal claims
- **Jurisdictional Accuracy**: Correct application of relevant law
- **Risk Assessment Quality**: Practical legal risk identification

## Prompt Strategies

### 1. Role-Based Prompting
```python
"You are an experienced contract attorney with 15 years of practice..."
```

### 2. Chain-of-Thought
```python
"Let's analyze this contract step by step:
Step 1: Identify the contract type..."
```

### 3. Context-Layered
```python
"Context: You are interpreting California law.
Legal Question: ...
Analysis Framework: ..."
```

## Dataset Integration

### Supported Datasets
- **LegalBench**: 162 reasoning tasks for general legal evaluation
- **CUAD**: 510 contracts with 13,000+ labeled clauses
- **LexGLUE**: 7 sub-datasets for multi-task legal NLP
- **Custom Legal HumanEval**: 10 hand-crafted legal reasoning tasks

### Loading Custom Datasets
```python
from datasets.dataset_loader import LegalDatasetManager

manager = LegalDatasetManager()
samples = manager.load_dataset("legalbench", "/path/to/data")
```

## Advanced Usage

### Custom Model Integration
```python
from models.llm_interface import LLMEvaluationManager, BaseLLMProvider

class CustomLLMProvider(BaseLLMProvider):
    async def generate_response(self, prompt, system_prompt=None):
        # Your custom LLM implementation
        pass

manager = LLMEvaluationManager()
manager.add_provider("custom_model", CustomLLMProvider())
```

### Prompt Strategy Development
```python
from prompts.prompt_optimizer import LegalPromptOptimizer, LegalPromptTemplate

optimizer = LegalPromptOptimizer()
prompt = optimizer.generate_prompt(
    LegalTaskType.CONTRACT_REVIEW,
    PromptStrategy.CHAIN_OF_THOUGHT,
    contract_text="Your contract text here"
)
```

### Batch Evaluation
```python
results = await pipeline.run_evaluation(
    benchmark_path="benchmarks/legal_humaneval.jsonl",
    models=["gpt-4", "claude-3"],
    prompt_strategies=["role_based", "chain_of_thought"],
    max_samples=100
)
```

## Evaluation Results

### Sample Output
```
Legal LLM Evaluation Report
==================================================

Benchmark: benchmarks/legal_humaneval.jsonl
Total samples: 10
Models evaluated: gpt-4, claude-3

Model Performance Summary:
------------------------------

gpt-4:
  Strategy: role_based
    pass@1: 0.800
    pass@5: 0.900
    avg_citation_accuracy: 0.850
    avg_reasoning_coherence: 0.780

claude-3:
  Strategy: role_based
    pass@1: 0.750
    pass@5: 0.850
    avg_citation_accuracy: 0.820
    avg_reasoning_coherence: 0.790

Comparative Analysis:
--------------------
Best pass@1: gpt-4 (0.800)
Best pass@5: gpt-4 (0.900)
```

## Research Applications

This framework enables systematic research into:

- **Legal Reasoning Capabilities**: How well do LLMs understand legal concepts?
- **Prompt Engineering**: Which prompting strategies work best for legal tasks?
- **Model Comparison**: Objective comparison across legal reasoning tasks
- **Domain Adaptation**: Effectiveness of legal-specific fine-tuning
- **Hallucination Patterns**: Understanding where legal LLMs fail

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements (new datasets, evaluation metrics, prompting strategies)
4. Submit a pull request

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{legal_llm_eval,
  title={Legal LLM Evaluation Framework: HumanEval for Legal Reasoning},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-repo]/legal_llm_eval}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Inspired by OpenAI's HumanEval framework
- Built on legal datasets from LegalBench, CUAD, and LexGLUE projects
- Incorporates legal reasoning methodologies from legal AI research