"""
LLM Integration Layer for Legal Evaluation
Supports multiple LLM providers (OpenAI, Anthropic, DeepSeek, etc.)
"""

import asyncio
import time
from typing import Dict, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
# Optional imports for paid providers
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class LLMResponse:
    content: str
    model_name: str
    tokens_used: int
    response_time: float
    metadata: Dict


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate_response(self, 
                              prompt: str, 
                              system_prompt: Optional[str] = None,
                              **kwargs) -> LLMResponse:
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        pass


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, 
                 api_key: str,
                 model: str = "gpt-4",
                 max_tokens: int = 2000):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
    
    async def generate_response(self, 
                              prompt: str,
                              system_prompt: Optional[str] = None,
                              **kwargs) -> LLMResponse:
        start_time = time.time()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                **kwargs
            )
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model_name=self.model,
                tokens_used=response.usage.total_tokens,
                response_time=response_time,
                metadata={"finish_reason": response.choices[0].finish_reason}
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return LLMResponse(
                content=f"Error: {str(e)}",
                model_name=self.model,
                tokens_used=0,
                response_time=response_time,
                metadata={"error": str(e)}
            )
    
    def get_model_name(self) -> str:
        return self.model


class AnthropicProvider(BaseLLMProvider):
    def __init__(self, 
                 api_key: str,
                 model: str = "claude-3-sonnet-20240229",
                 max_tokens: int = 2000):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic library not installed. Install with: pip install anthropic")
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
    
    async def generate_response(self, 
                              prompt: str,
                              system_prompt: Optional[str] = None,
                              **kwargs) -> LLMResponse:
        start_time = time.time()
        
        try:
            # Run synchronous Anthropic client in executor to make it async
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system_prompt or "You are a helpful legal assistant.",
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs
                )
            )
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response.content[0].text,
                model_name=self.model,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                response_time=response_time,
                metadata={"stop_reason": response.stop_reason}
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return LLMResponse(
                content=f"Error: {str(e)}",
                model_name=self.model,
                tokens_used=0,
                response_time=response_time,
                metadata={"error": str(e)}
            )
    
    def get_model_name(self) -> str:
        return self.model


class DeepSeekProvider(BaseLLMProvider):
    def __init__(self, 
                 api_key: str,
                 model: str = "deepseek-chat",
                 max_tokens: int = 2000,
                 base_url: str = "https://api.deepseek.com"):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.max_tokens = max_tokens
    
    async def generate_response(self, 
                              prompt: str,
                              system_prompt: Optional[str] = None,
                              **kwargs) -> LLMResponse:
        start_time = time.time()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                **kwargs
            )
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model_name=self.model,
                tokens_used=response.usage.total_tokens,
                response_time=response_time,
                metadata={"finish_reason": response.choices[0].finish_reason}
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return LLMResponse(
                content=f"Error: {str(e)}",
                model_name=self.model,
                tokens_used=0,
                response_time=response_time,
                metadata={"error": str(e)}
            )
    
    def get_model_name(self) -> str:
        return self.model


class LLMEvaluationManager:
    """Manages multiple LLM providers for comparative evaluation"""
    
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
    
    def add_provider(self, name: str, provider: BaseLLMProvider):
        """Add an LLM provider to the evaluation pool"""
        self.providers[name] = provider
    
    async def evaluate_single_prompt(self, 
                                   prompt: str,
                                   system_prompt: Optional[str] = None,
                                   provider_names: Optional[List[str]] = None) -> Dict[str, LLMResponse]:
        """Evaluate a single prompt across multiple LLM providers"""
        
        if provider_names is None:
            provider_names = list(self.providers.keys())
        
        tasks = []
        for name in provider_names:
            if name in self.providers:
                task = self.providers[name].generate_response(prompt, system_prompt)
                tasks.append((name, task))
        
        results = {}
        responses = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        for (name, _), response in zip(tasks, responses):
            if isinstance(response, Exception):
                results[name] = LLMResponse(
                    content=f"Error: {str(response)}",
                    model_name=name,
                    tokens_used=0,
                    response_time=0.0,
                    metadata={"error": str(response)}
                )
            else:
                results[name] = response
        
        return results
    
    async def evaluate_benchmark(self, 
                               prompts: List[str],
                               system_prompts: Optional[List[str]] = None,
                               provider_names: Optional[List[str]] = None) -> Dict[str, List[LLMResponse]]:
        """Evaluate a full benchmark across multiple providers"""
        
        if system_prompts is None:
            system_prompts = [None] * len(prompts)
        
        if len(system_prompts) != len(prompts):
            raise ValueError("Number of system prompts must match number of prompts")
        
        all_results = {name: [] for name in (provider_names or self.providers.keys())}
        
        # Process prompts in batches to avoid rate limiting
        batch_size = 5
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_systems = system_prompts[i:i+batch_size]
            
            batch_tasks = []
            for prompt, system in zip(batch_prompts, batch_systems):
                task = self.evaluate_single_prompt(prompt, system, provider_names)
                batch_tasks.append(task)
            
            batch_results = await asyncio.gather(*batch_tasks)
            
            # Organize results by provider
            for result_dict in batch_results:
                for provider_name, response in result_dict.items():
                    if provider_name in all_results:
                        all_results[provider_name].append(response)
            
            # Add delay between batches to respect rate limits
            if i + batch_size < len(prompts):
                await asyncio.sleep(1)
        
        return all_results
    
    def get_comparative_metrics(self, 
                              results: Dict[str, List[LLMResponse]]) -> Dict[str, Dict]:
        """Calculate comparative metrics across providers"""
        
        metrics = {}
        
        for provider_name, responses in results.items():
            total_tokens = sum(r.tokens_used for r in responses)
            total_time = sum(r.response_time for r in responses)
            error_count = sum(1 for r in responses if "error" in r.metadata)
            
            metrics[provider_name] = {
                "total_tokens": total_tokens,
                "avg_tokens_per_response": total_tokens / len(responses) if responses else 0,
                "total_response_time": total_time,
                "avg_response_time": total_time / len(responses) if responses else 0,
                "error_rate": error_count / len(responses) if responses else 0,
                "total_responses": len(responses)
            }
        
        return metrics


def create_legal_system_prompt() -> str:
    """Create a system prompt optimized for legal tasks"""
    return """You are an expert legal assistant with deep knowledge of law across multiple jurisdictions. 

Your responses should:
1. Be accurate and cite relevant legal authorities
2. Acknowledge jurisdictional limitations when applicable
3. Provide clear, structured analysis
4. Avoid giving specific legal advice
5. Note when professional legal consultation is recommended

Always maintain objectivity and present multiple perspectives when relevant."""


if __name__ == "__main__":
    # Example usage
    async def main():
        manager = LLMEvaluationManager()
        
        # Add providers (API keys would be from environment variables)
        # manager.add_provider("gpt-4", OpenAIProvider("your-api-key"))
        # manager.add_provider("claude", AnthropicProvider("your-api-key"))
        # manager.add_provider("deepseek", DeepSeekProvider("your-api-key"))
        
        # Example evaluation
        test_prompt = "Analyze the key elements of a valid contract under common law."
        system_prompt = create_legal_system_prompt()
        
        # results = await manager.evaluate_single_prompt(test_prompt, system_prompt)
        # print(f"Evaluated prompt across {len(results)} providers")
    
    # asyncio.run(main())