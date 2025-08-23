"""
Legal LLM Evaluator - HumanEval inspired framework for legal reasoning assessment
"""

import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from pathlib import Path


class EvaluationResult(Enum):
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"
    INVALID = "invalid"


@dataclass
class LegalTaskSample:
    task_id: str
    task_type: str
    prompt: str
    context: str
    expected_output: Dict
    test_cases: List[Dict]
    jurisdiction: str
    complexity: str
    completion: Optional[str] = None
    evaluation_result: Optional[EvaluationResult] = None
    metrics: Optional[Dict] = None


@dataclass
class EvaluationMetrics:
    citation_accuracy: float
    reasoning_coherence: float
    completeness: float
    hallucination_rate: float
    response_time: float
    token_count: int


class LegalEvaluator:
    def __init__(self, safety_mode: bool = True):
        self.safety_mode = safety_mode
        self.evaluation_results = []
    
    def load_legal_benchmark(self, benchmark_path: str) -> List[LegalTaskSample]:
        """Load legal benchmark dataset in HumanEval .jsonl format"""
        samples = []
        
        with open(benchmark_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                sample = LegalTaskSample(**data)
                samples.append(sample)
        
        return samples
    
    def evaluate_citation_accuracy(self, 
                                 response: str, 
                                 expected_citations: List[str]) -> float:
        """Evaluate accuracy of legal citations in response"""
        if not expected_citations:
            return 1.0
        
        found_citations = 0
        for citation in expected_citations:
            # Simple citation matching - can be enhanced with regex
            if citation.lower() in response.lower():
                found_citations += 1
        
        return found_citations / len(expected_citations)
    
    def evaluate_reasoning_coherence(self, response: str) -> float:
        """Evaluate logical coherence of legal reasoning"""
        
        # Check for key reasoning indicators
        reasoning_indicators = [
            "therefore", "because", "thus", "consequently", 
            "as a result", "follows that", "given that"
        ]
        
        logical_structure_score = 0.0
        
        # Basic coherence checks
        sentences = response.split('.')
        if len(sentences) > 1:
            logical_structure_score += 0.3
        
        # Check for reasoning indicators
        indicator_count = sum(1 for indicator in reasoning_indicators 
                            if indicator in response.lower())
        if indicator_count > 0:
            logical_structure_score += min(0.4, indicator_count * 0.1)
        
        # Check for conclusion/summary
        if any(word in response.lower() for word in ["conclusion", "summary", "therefore"]):
            logical_structure_score += 0.3
        
        return min(1.0, logical_structure_score)
    
    def evaluate_completeness(self, 
                            response: str, 
                            required_elements: List[str]) -> float:
        """Evaluate completeness of legal analysis"""
        if not required_elements:
            return 1.0
        
        covered_elements = 0
        for element in required_elements:
            if element.lower() in response.lower():
                covered_elements += 1
        
        return covered_elements / len(required_elements)
    
    def detect_hallucinations(self, 
                            response: str, 
                            known_facts: List[str]) -> float:
        """Detect potential legal hallucinations"""
        
        # Simplified hallucination detection
        hallucination_indicators = [
            "according to a recent study",
            "it is widely known that",
            "experts agree that",
            "studies show that"
        ]
        
        hallucination_count = sum(1 for indicator in hallucination_indicators 
                                if indicator in response.lower())
        
        # Check for contradictions with known facts
        contradiction_count = 0
        for fact in known_facts:
            # Simple contradiction detection - can be enhanced
            if f"not {fact.lower()}" in response.lower():
                contradiction_count += 1
        
        total_issues = hallucination_count + contradiction_count
        
        # Return rate (lower is better)
        if len(response.split()) == 0:
            return 1.0
        
        return min(1.0, total_issues / max(1, len(response.split()) / 100))
    
    def evaluate_single_sample(self, 
                             sample: LegalTaskSample,
                             llm_response: str,
                             response_time: float = 0.0,
                             token_count: int = 0) -> EvaluationMetrics:
        """Evaluate a single legal task sample"""
        
        # Extract test criteria
        test_cases = sample.test_cases[0] if sample.test_cases else {}
        
        citation_accuracy = self.evaluate_citation_accuracy(
            llm_response, 
            test_cases.get("expected_citations", [])
        )
        
        reasoning_coherence = self.evaluate_reasoning_coherence(llm_response)
        
        completeness = self.evaluate_completeness(
            llm_response,
            test_cases.get("required_elements", [])
        )
        
        hallucination_rate = self.detect_hallucinations(
            llm_response,
            test_cases.get("known_facts", [])
        )
        
        return EvaluationMetrics(
            citation_accuracy=citation_accuracy,
            reasoning_coherence=reasoning_coherence,
            completeness=completeness,
            hallucination_rate=hallucination_rate,
            response_time=response_time,
            token_count=token_count
        )
    
    def calculate_pass_at_k(self, 
                          results: List[EvaluationResult], 
                          k: int) -> float:
        """Calculate pass@k metric similar to HumanEval"""
        
        if k > len(results):
            k = len(results)
        
        if k == 0:
            return 0.0
        
        passed_count = sum(1 for result in results[:k] 
                          if result == EvaluationResult.PASSED)
        
        return passed_count / k
    
    def evaluate_benchmark(self, 
                         samples: List[LegalTaskSample],
                         llm_responses: List[str],
                         k_values: List[int] = [1, 5, 10]) -> Dict:
        """Evaluate full benchmark dataset"""
        
        if len(samples) != len(llm_responses):
            raise ValueError("Number of samples must match number of responses")
        
        evaluation_results = []
        detailed_metrics = []
        
        for sample, response in zip(samples, llm_responses):
            metrics = self.evaluate_single_sample(sample, response)
            detailed_metrics.append(metrics)
            
            # Determine pass/fail based on composite score
            composite_score = (
                metrics.citation_accuracy * 0.3 +
                metrics.reasoning_coherence * 0.3 +
                metrics.completeness * 0.3 +
                (1 - metrics.hallucination_rate) * 0.1
            )
            
            if composite_score >= 0.7:
                result = EvaluationResult.PASSED
            elif composite_score >= 0.4:
                result = EvaluationResult.PARTIAL
            else:
                result = EvaluationResult.FAILED
            
            evaluation_results.append(result)
        
        # Calculate pass@k metrics
        pass_at_k = {}
        for k in k_values:
            if k <= len(evaluation_results):
                pass_at_k[f"pass@{k}"] = self.calculate_pass_at_k(evaluation_results, k)
        
        # Calculate aggregate metrics
        avg_metrics = {}
        if detailed_metrics:
            for metric_name in ["citation_accuracy", "reasoning_coherence", 
                              "completeness", "hallucination_rate"]:
                values = [getattr(m, metric_name) for m in detailed_metrics]
                avg_metrics[f"avg_{metric_name}"] = sum(values) / len(values)
        
        return {
            "pass_at_k": pass_at_k,
            "aggregate_metrics": avg_metrics,
            "detailed_results": [asdict(m) for m in detailed_metrics],
            "task_results": [r.value for r in evaluation_results]
        }
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def generate_report(self, results: Dict) -> str:
        """Generate human-readable evaluation report"""
        
        report = "Legal LLM Evaluation Report\n"
        report += "=" * 50 + "\n\n"
        
        # Pass@k results
        report += "Pass@k Metrics:\n"
        for metric, value in results["pass_at_k"].items():
            report += f"  {metric}: {value:.3f}\n"
        
        report += "\nAggregate Performance:\n"
        for metric, value in results["aggregate_metrics"].items():
            report += f"  {metric}: {value:.3f}\n"
        
        # Task type breakdown
        task_results = results["task_results"]
        total_tasks = len(task_results)
        passed = task_results.count("passed")
        failed = task_results.count("failed")
        partial = task_results.count("partial")
        
        report += f"\nTask Results Summary:\n"
        report += f"  Total tasks: {total_tasks}\n"
        report += f"  Passed: {passed} ({passed/total_tasks:.1%})\n"
        report += f"  Partial: {partial} ({partial/total_tasks:.1%})\n"
        report += f"  Failed: {failed} ({failed/total_tasks:.1%})\n"
        
        return report


if __name__ == "__main__":
    # Example usage
    evaluator = LegalEvaluator()
    
    # Create sample data
    sample_data = [
        {
            "task_id": "legal/contract_review/001",
            "task_type": "contract_review", 
            "prompt": "Review this employment contract...",
            "context": "Sample contract text",
            "expected_output": {"key_clauses": ["non-compete", "termination"]},
            "test_cases": [{"expected_citations": ["17 USC 101"], "required_elements": ["termination clause"]}],
            "jurisdiction": "US_Federal",
            "complexity": "intermediate"
        }
    ]
    
    # Mock LLM responses
    responses = ["This contract contains a termination clause per 17 USC 101..."]
    
    # Convert to samples
    samples = [LegalTaskSample(**data) for data in sample_data]
    
    # Evaluate
    results = evaluator.evaluate_benchmark(samples, responses)
    
    print(evaluator.generate_report(results))