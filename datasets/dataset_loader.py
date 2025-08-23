"""
Legal Dataset Integration Pipeline
Supports loading and processing multiple legal datasets for LLM evaluation
"""

import json
import pandas as pd
from typing import Dict, List, Optional, Union, Iterator
from dataclasses import dataclass, asdict
from pathlib import Path
import requests
import zipfile
from abc import ABC, abstractmethod


@dataclass
class LegalDatasetSample:
    task_id: str
    task_type: str
    prompt: str
    context: str
    expected_output: Dict
    test_cases: List[Dict]
    jurisdiction: str
    complexity: str
    source_dataset: str
    metadata: Optional[Dict] = None


class BaseDatasetLoader(ABC):
    """Abstract base class for legal dataset loaders"""
    
    @abstractmethod
    def load_dataset(self, path: str) -> List[LegalDatasetSample]:
        pass
    
    @abstractmethod
    def get_dataset_info(self) -> Dict:
        pass


class LegalBenchLoader(BaseDatasetLoader):
    """Loader for LegalBench dataset (162 reasoning tasks)"""
    
    def __init__(self):
        self.dataset_name = "LegalBench"
        self.task_types = {
            "abercrombie": "statutory_interpretation",
            "contract_qa": "contract_review", 
            "case_holding": "case_summarization",
            "legal_reasoning": "legal_qa"
        }
    
    def load_dataset(self, path: str) -> List[LegalDatasetSample]:
        """Load LegalBench dataset from JSON files"""
        samples = []
        dataset_path = Path(path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {path}")
        
        # Process each task file
        for task_file in dataset_path.glob("*.json"):
            task_name = task_file.stem
            task_type = self.task_types.get(task_name, "legal_reasoning")
            
            with open(task_file, 'r') as f:
                task_data = json.load(f)
            
            for idx, item in enumerate(task_data):
                sample = LegalDatasetSample(
                    task_id=f"legalbench/{task_name}/{idx:03d}",
                    task_type=task_type,
                    prompt=item.get("text", ""),
                    context=item.get("context", ""),
                    expected_output={"answer": item.get("answer", "")},
                    test_cases=[{
                        "expected_answer": item.get("answer", ""),
                        "evaluation_criteria": ["accuracy", "legal_reasoning"]
                    }],
                    jurisdiction=item.get("jurisdiction", "US_General"),
                    complexity="intermediate",
                    source_dataset="LegalBench",
                    metadata={"original_task": task_name}
                )
                samples.append(sample)
        
        return samples
    
    def get_dataset_info(self) -> Dict:
        return {
            "name": "LegalBench",
            "description": "162 reasoning tasks for legal LLM evaluation",
            "tasks": len(self.task_types),
            "domain": "General legal reasoning",
            "citation": "Guha et al. (2023)"
        }


class CUADLoader(BaseDatasetLoader):
    """Loader for CUAD dataset (Contract Understanding Atticus Dataset)"""
    
    def __init__(self):
        self.dataset_name = "CUAD"
        self.clause_types = [
            "parties", "agreement_date", "effective_date", "expiration_date",
            "renewal_term", "notice_period", "governing_law", "most_favored_nation",
            "non_compete", "exclusivity", "no_solicit_of_customers",
            "competitive_restriction_exception", "no_solicit_of_employees"
        ]
    
    def load_dataset(self, path: str) -> List[LegalDatasetSample]:
        """Load CUAD dataset from JSON format"""
        samples = []
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        for idx, item in enumerate(data.get("data", [])):
            contract_text = item.get("title", "") + "\n" + item.get("paragraphs", [{}])[0].get("context", "")
            
            for qa in item.get("paragraphs", [{}])[0].get("qas", []):
                clause_type = self._extract_clause_type(qa.get("question", ""))
                
                sample = LegalDatasetSample(
                    task_id=f"cuad/contract_analysis/{idx:03d}_{qa.get('id', '')}",
                    task_type="contract_review",
                    prompt=f"Analyze the following contract for {clause_type}:\n\nQuestion: {qa.get('question', '')}\n\nContract:\n{contract_text}",
                    context=contract_text,
                    expected_output={
                        "answer": qa.get("answers", [{}])[0].get("text", ""),
                        "clause_type": clause_type
                    },
                    test_cases=[{
                        "expected_clauses": [clause_type],
                        "required_elements": ["clause_identification", "legal_analysis"],
                        "evaluation_criteria": ["accuracy", "completeness"]
                    }],
                    jurisdiction="US_Commercial",
                    complexity="advanced",
                    source_dataset="CUAD",
                    metadata={
                        "contract_id": item.get("title", ""),
                        "question_id": qa.get("id", "")
                    }
                )
                samples.append(sample)
        
        return samples
    
    def _extract_clause_type(self, question: str) -> str:
        """Extract clause type from question text"""
        question_lower = question.lower()
        for clause_type in self.clause_types:
            if clause_type.replace("_", " ") in question_lower:
                return clause_type
        return "general_clause"
    
    def get_dataset_info(self) -> Dict:
        return {
            "name": "CUAD",
            "description": "Contract Understanding Atticus Dataset - 510 contracts with 13,000+ labels",
            "domain": "Contract analysis and clause extraction",
            "clause_types": len(self.clause_types),
            "citation": "Hendrycks et al. (2021)"
        }


class LexGLUELoader(BaseDatasetLoader):
    """Loader for LexGLUE dataset (Legal General Language Understanding Evaluation)"""
    
    def __init__(self):
        self.dataset_name = "LexGLUE"
        self.subtasks = {
            "ecthr_a": "case_classification",
            "eurlex": "statutory_interpretation", 
            "scotus": "case_summarization",
            "ledgar": "contract_review",
            "unfair_tos": "contract_review",
            "case_hold": "legal_qa"
        }
    
    def load_dataset(self, path: str) -> List[LegalDatasetSample]:
        """Load LexGLUE dataset"""
        samples = []
        dataset_path = Path(path)
        
        for subtask_dir in dataset_path.iterdir():
            if subtask_dir.is_dir() and subtask_dir.name in self.subtasks:
                subtask_samples = self._load_subtask(subtask_dir, subtask_dir.name)
                samples.extend(subtask_samples)
        
        return samples
    
    def _load_subtask(self, subtask_path: Path, subtask_name: str) -> List[LegalDatasetSample]:
        """Load individual LexGLUE subtask"""
        samples = []
        task_type = self.subtasks[subtask_name]
        
        # Look for train/test/dev files
        for split_file in subtask_path.glob("*.jsonl"):
            with open(split_file, 'r') as f:
                for idx, line in enumerate(f):
                    item = json.loads(line.strip())
                    
                    sample = LegalDatasetSample(
                        task_id=f"lexglue/{subtask_name}/{split_file.stem}/{idx:03d}",
                        task_type=task_type,
                        prompt=self._create_prompt(item, subtask_name),
                        context=item.get("text", ""),
                        expected_output={"label": item.get("label", "")},
                        test_cases=[{
                            "expected_classification": item.get("label", ""),
                            "evaluation_criteria": ["classification_accuracy"]
                        }],
                        jurisdiction=self._get_jurisdiction(subtask_name),
                        complexity="intermediate",
                        source_dataset="LexGLUE",
                        metadata={"subtask": subtask_name, "split": split_file.stem}
                    )
                    samples.append(sample)
        
        return samples
    
    def _create_prompt(self, item: Dict, subtask_name: str) -> str:
        """Create task-specific prompt"""
        text = item.get("text", "")
        
        if subtask_name == "ecthr_a":
            return f"Classify the following legal case according to the European Convention on Human Rights articles:\n\n{text}"
        elif subtask_name == "eurlex":
            return f"Classify this EU legal document:\n\n{text}"
        elif subtask_name == "scotus":
            return f"Classify this Supreme Court case by legal issue area:\n\n{text}"
        else:
            return f"Analyze the following legal text:\n\n{text}"
    
    def _get_jurisdiction(self, subtask_name: str) -> str:
        """Get jurisdiction for subtask"""
        jurisdiction_map = {
            "ecthr_a": "ECHR",
            "eurlex": "EU",
            "scotus": "US_Federal",
            "ledgar": "US_Commercial",
            "unfair_tos": "US_Consumer",
            "case_hold": "US_General"
        }
        return jurisdiction_map.get(subtask_name, "General")
    
    def get_dataset_info(self) -> Dict:
        return {
            "name": "LexGLUE",
            "description": "Legal General Language Understanding Evaluation - 7 sub-datasets",
            "subtasks": list(self.subtasks.keys()),
            "domain": "Multi-task legal NLP",
            "citation": "Chalkidis et al. (2022)"
        }


class LegalDatasetManager:
    """Manages multiple legal datasets for evaluation"""
    
    def __init__(self):
        self.loaders = {
            "legalbench": LegalBenchLoader(),
            "cuad": CUADLoader(),
            "lexglue": LexGLUELoader()
        }
        self.loaded_datasets = {}
    
    def load_dataset(self, dataset_name: str, path: str) -> List[LegalDatasetSample]:
        """Load a specific dataset"""
        if dataset_name not in self.loaders:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        samples = self.loaders[dataset_name].load_dataset(path)
        self.loaded_datasets[dataset_name] = samples
        return samples
    
    def combine_datasets(self, 
                        dataset_names: List[str],
                        max_samples_per_dataset: Optional[int] = None) -> List[LegalDatasetSample]:
        """Combine multiple datasets for evaluation"""
        combined_samples = []
        
        for dataset_name in dataset_names:
            if dataset_name in self.loaded_datasets:
                samples = self.loaded_datasets[dataset_name]
                if max_samples_per_dataset:
                    samples = samples[:max_samples_per_dataset]
                combined_samples.extend(samples)
        
        return combined_samples
    
    def filter_by_criteria(self, 
                          samples: List[LegalDatasetSample],
                          task_types: Optional[List[str]] = None,
                          jurisdictions: Optional[List[str]] = None,
                          complexity: Optional[List[str]] = None) -> List[LegalDatasetSample]:
        """Filter samples by various criteria"""
        filtered = samples
        
        if task_types:
            filtered = [s for s in filtered if s.task_type in task_types]
        
        if jurisdictions:
            filtered = [s for s in filtered if s.jurisdiction in jurisdictions]
        
        if complexity:
            filtered = [s for s in filtered if s.complexity in complexity]
        
        return filtered
    
    def export_to_jsonl(self, 
                       samples: List[LegalDatasetSample],
                       output_path: str):
        """Export samples to HumanEval-style JSONL format"""
        with open(output_path, 'w') as f:
            for sample in samples:
                json.dump(asdict(sample), f)
                f.write('\n')
    
    def get_dataset_statistics(self, samples: List[LegalDatasetSample]) -> Dict:
        """Generate statistics about the dataset"""
        stats = {
            "total_samples": len(samples),
            "task_types": {},
            "jurisdictions": {},
            "complexity": {},
            "source_datasets": {}
        }
        
        for sample in samples:
            # Count task types
            stats["task_types"][sample.task_type] = stats["task_types"].get(sample.task_type, 0) + 1
            
            # Count jurisdictions
            stats["jurisdictions"][sample.jurisdiction] = stats["jurisdictions"].get(sample.jurisdiction, 0) + 1
            
            # Count complexity levels
            stats["complexity"][sample.complexity] = stats["complexity"].get(sample.complexity, 0) + 1
            
            # Count source datasets
            stats["source_datasets"][sample.source_dataset] = stats["source_datasets"].get(sample.source_dataset, 0) + 1
        
        return stats


if __name__ == "__main__":
    # Example usage
    manager = LegalDatasetManager()
    
    # Load datasets (paths would be actual dataset locations)
    # samples = manager.load_dataset("legalbench", "/path/to/legalbench")
    
    # Get dataset info
    for name, loader in manager.loaders.items():
        print(f"{name}: {loader.get_dataset_info()}")
    
    print("Dataset integration pipeline ready.")