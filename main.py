"""
Legal LLM Evaluation Framework - Main Execution Script
HumanEval-inspired evaluation system for legal reasoning tasks
"""

import asyncio
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from prompts.prompt_optimizer import LegalPromptOptimizer, LegalTaskType, PromptStrategy
from evaluator.legal_evaluator import LegalEvaluator, LegalTaskSample
from models.llm_interface import LLMEvaluationManager, OpenAIProvider, AnthropicProvider, DeepSeekProvider
from models.free_llm_providers import MockLLMProvider, FREE_MODEL_CONFIGS
from datasets.dataset_loader import LegalDatasetManager


class LegalLLMEvaluationPipeline:
    """Main pipeline for legal LLM evaluation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.prompt_optimizer = LegalPromptOptimizer()
        self.evaluator = LegalEvaluator()
        self.llm_manager = LLMEvaluationManager()
        self.dataset_manager = LegalDatasetManager()
        
        self._setup_llm_providers()
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            "models": {
                "gpt-4": {"provider": "openai", "api_key": ""},
                "claude-3": {"provider": "anthropic", "api_key": ""},
                "deepseek": {"provider": "deepseek", "api_key": ""}
            },
            "evaluation": {
                "k_values": [1, 5, 10],
                "max_samples": 100,
                "batch_size": 5
            },
            "datasets": ["legal_humaneval"],
            "output_dir": "results"
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                for key in default_config:
                    if key not in config:
                        config[key] = default_config[key]
                return config
        
        return default_config
    
    def _setup_llm_providers(self):
        """Setup LLM providers based on configuration"""
        for model_name, model_config in self.config["models"].items():
            try:
                provider_type = model_config.get("provider", "")
                
                if provider_type == "mock":
                    provider = MockLLMProvider(model_name)
                elif provider_type == "openai" and model_config.get("api_key"):
                    provider = OpenAIProvider(
                        api_key=model_config["api_key"],
                        model=model_name
                    )
                elif provider_type == "anthropic" and model_config.get("api_key"):
                    provider = AnthropicProvider(
                        api_key=model_config["api_key"],
                        model=model_name
                    )
                elif provider_type == "deepseek" and model_config.get("api_key"):
                    provider = DeepSeekProvider(
                        api_key=model_config["api_key"],
                        model=model_name
                    )
                elif model_name in FREE_MODEL_CONFIGS:
                    # Use free model configurations
                    provider_class = FREE_MODEL_CONFIGS[model_name]["provider"]
                    provider = provider_class() if callable(provider_class) else provider_class
                else:
                    print(f"Unknown provider or missing API key for {model_name}, skipping...")
                    continue
                
                self.llm_manager.add_provider(model_name, provider)
                print(f"Added LLM provider: {model_name}")
                
            except Exception as e:
                print(f"Failed to setup {model_name}: {e}")
    
    def load_benchmark_data(self, benchmark_path: str) -> List[LegalTaskSample]:
        """Load legal benchmark dataset"""
        samples = []
        
        with open(benchmark_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                sample = LegalTaskSample(**data)
                samples.append(sample)
        
        return samples
    
    async def run_evaluation(self, 
                           benchmark_path: str,
                           models: Optional[List[str]] = None,
                           prompt_strategies: Optional[List[str]] = None,
                           max_samples: Optional[int] = None) -> Dict:
        """Run complete legal LLM evaluation"""
        
        print("Loading benchmark data...")
        samples = self.load_benchmark_data(benchmark_path)
        
        if max_samples:
            samples = samples[:max_samples]
        
        print(f"Loaded {len(samples)} samples")
        
        # Filter models
        available_models = list(self.llm_manager.providers.keys())
        if models:
            models = [m for m in models if m in available_models]
        else:
            models = available_models
        
        if not models:
            raise ValueError("No valid models available for evaluation")
        
        print(f"Evaluating models: {models}")
        
        # Prepare evaluation results
        results = {
            "benchmark_info": {
                "path": benchmark_path,
                "total_samples": len(samples),
                "models_evaluated": models
            },
            "model_results": {},
            "comparative_analysis": {}
        }
        
        # Evaluate each model
        for model_name in models:
            print(f"\nEvaluating {model_name}...")
            
            model_results = await self._evaluate_model(
                model_name, samples, prompt_strategies
            )
            results["model_results"][model_name] = model_results
            
            print(f"Completed evaluation for {model_name}")
        
        # Generate comparative analysis
        results["comparative_analysis"] = self._generate_comparative_analysis(
            results["model_results"]
        )
        
        return results
    
    async def _evaluate_model(self, 
                            model_name: str,
                            samples: List[LegalTaskSample],
                            prompt_strategies: Optional[List[str]] = None) -> Dict:
        """Evaluate a single model on the benchmark"""
        
        if prompt_strategies is None:
            prompt_strategies = ["role_based"]  # Default strategy
        
        model_results = {}
        
        for strategy in prompt_strategies:
            print(f"  Using {strategy} prompting strategy...")
            
            # Generate prompts
            prompts = []
            system_prompts = []
            
            for sample in samples:
                try:
                    task_type = LegalTaskType(sample.task_type)
                    strategy_enum = PromptStrategy(strategy)
                    
                    # Create prompt using the optimizer
                    prompt = self.prompt_optimizer.generate_prompt(
                        task_type, 
                        strategy_enum,
                        contract_text=sample.context,
                        statute_text=sample.context,
                        case_text=sample.context,
                        question=sample.prompt
                    )
                    
                    prompts.append(prompt)
                    system_prompts.append("You are an expert legal assistant.")
                    
                except Exception as e:
                    # Fallback to original prompt if optimization fails
                    prompts.append(sample.prompt)
                    system_prompts.append("You are an expert legal assistant.")
            
            # Get LLM responses
            llm_responses = await self.llm_manager.evaluate_benchmark(
                prompts, system_prompts, [model_name]
            )
            
            # Extract responses for this model
            model_responses = llm_responses.get(model_name, [])
            response_texts = [r.content for r in model_responses]
            
            # Evaluate responses
            evaluation_results = self.evaluator.evaluate_benchmark(
                samples, 
                response_texts,
                self.config["evaluation"]["k_values"]
            )
            
            # Add performance metrics
            evaluation_results["performance_metrics"] = self.llm_manager.get_comparative_metrics({
                model_name: model_responses
            })[model_name]
            
            model_results[strategy] = evaluation_results
        
        return model_results
    
    def _generate_comparative_analysis(self, model_results: Dict) -> Dict:
        """Generate comparative analysis across models"""
        
        analysis = {
            "best_performing_models": {},
            "task_type_performance": {},
            "efficiency_metrics": {}
        }
        
        # Find best performing models for each metric
        for metric in ["pass@1", "pass@5", "pass@10"]:
            best_model = None
            best_score = 0
            
            for model_name, model_data in model_results.items():
                # Use the first strategy's results for comparison
                strategy_results = list(model_data.values())[0]
                score = strategy_results.get("pass_at_k", {}).get(metric, 0)
                
                if score > best_score:
                    best_score = score
                    best_model = model_name
            
            if best_model:
                analysis["best_performing_models"][metric] = {
                    "model": best_model,
                    "score": best_score
                }
        
        # Efficiency analysis
        for model_name, model_data in model_results.items():
            strategy_results = list(model_data.values())[0]
            perf_metrics = strategy_results.get("performance_metrics", {})
            
            analysis["efficiency_metrics"][model_name] = {
                "avg_response_time": perf_metrics.get("avg_response_time", 0),
                "avg_tokens_per_response": perf_metrics.get("avg_tokens_per_response", 0),
                "error_rate": perf_metrics.get("error_rate", 0)
            }
        
        return analysis
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results"""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate and save report
        report = self._generate_report(results)
        report_path = output_path.replace('.json', '_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Results saved to: {output_path}")
        print(f"Report saved to: {report_path}")
    
    def _generate_report(self, results: Dict) -> str:
        """Generate human-readable evaluation report"""
        
        report = "Legal LLM Evaluation Report\n"
        report += "=" * 50 + "\n\n"
        
        # Benchmark info
        benchmark_info = results["benchmark_info"]
        report += f"Benchmark: {benchmark_info['path']}\n"
        report += f"Total samples: {benchmark_info['total_samples']}\n"
        report += f"Models evaluated: {', '.join(benchmark_info['models_evaluated'])}\n\n"
        
        # Model performance
        report += "Model Performance Summary:\n"
        report += "-" * 30 + "\n"
        
        for model_name, model_data in results["model_results"].items():
            report += f"\n{model_name}:\n"
            
            for strategy, strategy_results in model_data.items():
                report += f"  Strategy: {strategy}\n"
                
                # Pass@k metrics
                pass_at_k = strategy_results.get("pass_at_k", {})
                for metric, score in pass_at_k.items():
                    report += f"    {metric}: {score:.3f}\n"
                
                # Aggregate metrics
                agg_metrics = strategy_results.get("aggregate_metrics", {})
                for metric, score in agg_metrics.items():
                    report += f"    {metric}: {score:.3f}\n"
        
        # Comparative analysis
        comparative = results.get("comparative_analysis", {})
        if comparative:
            report += "\nComparative Analysis:\n"
            report += "-" * 20 + "\n"
            
            best_models = comparative.get("best_performing_models", {})
            for metric, data in best_models.items():
                report += f"Best {metric}: {data['model']} ({data['score']:.3f})\n"
            
            # Efficiency metrics
            efficiency = comparative.get("efficiency_metrics", {})
            if efficiency:
                report += "\nEfficiency Metrics:\n"
                for model, metrics in efficiency.items():
                    report += f"{model}:\n"
                    report += f"  Avg response time: {metrics['avg_response_time']:.2f}s\n"
                    report += f"  Avg tokens: {metrics['avg_tokens_per_response']:.0f}\n"
                    report += f"  Error rate: {metrics['error_rate']:.1%}\n"
        
        return report


async def main():
    parser = argparse.ArgumentParser(description="Legal LLM Evaluation Framework")
    parser.add_argument("--benchmark", "-b", 
                       default="benchmarks/legal_humaneval.jsonl",
                       help="Path to benchmark dataset")
    parser.add_argument("--config", "-c",
                       help="Path to configuration file")
    parser.add_argument("--models", "-m", nargs="+",
                       help="Models to evaluate (default: all available)")
    parser.add_argument("--strategies", "-s", nargs="+",
                       default=["role_based"],
                       help="Prompt strategies to use")
    parser.add_argument("--max-samples", type=int,
                       help="Maximum number of samples to evaluate")
    parser.add_argument("--output", "-o",
                       default="results/evaluation_results.json",
                       help="Output path for results")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = LegalLLMEvaluationPipeline(args.config)
    
    try:
        # Run evaluation
        results = await pipeline.run_evaluation(
            benchmark_path=args.benchmark,
            models=args.models,
            prompt_strategies=args.strategies,
            max_samples=args.max_samples
        )
        
        # Save results
        pipeline.save_results(results, args.output)
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))