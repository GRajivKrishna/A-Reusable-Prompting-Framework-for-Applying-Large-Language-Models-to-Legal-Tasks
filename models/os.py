"""
Free and Open Source LLM Providers
No API keys required - uses local/free models
"""

import time
import requests
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from .llm_interface import BaseLLMProvider, LLMResponse


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing without API costs"""
    
    def __init__(self, model_name: str = "mock-legal-llm"):
        self.model_name = model_name
    
    async def generate_response(self, 
                              prompt: str,
                              system_prompt: Optional[str] = None,
                              **kwargs) -> LLMResponse:
        start_time = time.time()
        
        # Simulate processing time
        await self._simulate_processing()
        
        # Generate mock legal response based on prompt content
        response_content = self._generate_mock_legal_response(prompt)
        
        response_time = time.time() - start_time
        
        return LLMResponse(
            content=response_content,
            model_name=self.model_name,
            tokens_used=len(response_content.split()) * 1.3,  # Rough token estimate
            response_time=response_time,
            metadata={"mock": True, "simulated": True}
        )
    
    async def _simulate_processing(self):
        """Simulate realistic LLM response time"""
        import asyncio
        await asyncio.sleep(0.5)  # Simulate 0.5 second processing
    
    def _generate_mock_legal_response(self, prompt: str) -> str:
        """Generate realistic mock legal responses based on prompt content"""
        
        prompt_lower = prompt.lower()
        
        if "contract" in prompt_lower and "review" in prompt_lower:
            return """Based on my analysis of this contract, I've identified several key areas of concern:

1. **Non-Compete Clause**: The 18-month non-compete period with a 50-mile radius appears overly broad and may not be enforceable in all jurisdictions, particularly California where such clauses are generally prohibited under Business and Professions Code Section 16600.

2. **Intellectual Property Provisions**: The clause stating "all work product created during employment belongs to Company, including inventions made at home" is excessively broad and may be unenforceable as it lacks reasonable limitations.

3. **Termination Clause**: The at-will termination provision with only 30 days notice provides minimal job security for the employee.

**Recommendations**:
- Narrow the non-compete clause to specific competing activities
- Limit IP assignments to work-related inventions
- Consider adding severance provisions for termination without cause

**Risk Level**: Medium to High
**Citations**: California Business and Professions Code § 16600, common law employment principles"""

        elif "statute" in prompt_lower or "interpretation" in prompt_lower:
            return """**Statutory Interpretation Analysis**

The statute in question establishes a clear framework for trade secret protection. Applying this to the software compilation process case:

**Element 1 - Economic Value**: The compilation process likely derives independent economic value if it provides competitive advantages through faster compilation, better optimization, or unique methodologies.

**Element 2 - Not Generally Known**: If the specific compilation techniques are proprietary and not publicly available, this element would be satisfied.

**Element 3 - Reasonable Secrecy Efforts**: The company must demonstrate reasonable measures to maintain secrecy, such as employee confidentiality agreements, access controls, and marking proprietary processes.

**Conclusion**: A software compilation process can qualify as a trade secret under this statute if all three elements are met. The analysis would be fact-specific regarding the particular methods used and secrecy measures employed.

**Relevant Authority**: Uniform Trade Secrets Act, Economic Espionage Act of 1996"""

        elif "case" in prompt_lower and ("summary" in prompt_lower or "summariz" in prompt_lower):
            return """**Case Summary: Miranda v. Arizona (1966)**

**Citation**: 384 U.S. 436 (1966)

**Key Holdings**:
1. Fifth Amendment privilege against self-incrimination requires law enforcement to advise suspects of their constitutional rights before custodial interrogation
2. Statements obtained during incommunicado interrogation without proper warnings are inadmissible in court
3. The warnings must include the right to remain silent and the right to have an attorney present

**Legal Significance**: This landmark decision established the "Miranda rights" that fundamentally changed police interrogation procedures across the United States.

**Impact**: Created mandatory procedural safeguards that balance law enforcement needs with constitutional protections, affecting millions of criminal cases since 1966.

**Procedural Requirements**: Custody + Interrogation = Miranda warnings required"""

        elif "precedent" in prompt_lower or "relevant case" in prompt_lower:
            return """**Relevant Precedent Analysis - Fourth Amendment Vehicle Search**

**Key Precedents**:

1. **Carroll v. United States (1925)** - 267 U.S. 132
   - Established the automobile exception to the warrant requirement
   - Rationale: Mobility of vehicles and reduced expectation of privacy

2. **Pennsylvania v. Labron (1996)** - 518 U.S. 938
   - Confirmed that probable cause alone justifies warrantless vehicle search
   - No requirement to show exigent circumstances beyond mobility

3. **California v. Acevedo (1991)** - 500 U.S. 565
   - Police may search containers in vehicles if they have probable cause
   - Eliminated distinction between searching vehicle and containers within

4. **Arizona v. Gant (2009)** - 556 U.S. 332
   - Limited scope of search incident to arrest for vehicle occupants
   - Search must be justified by officer safety or evidence preservation

**Application**: The plain view observation of drug paraphernalia likely provides probable cause for the vehicle search under the automobile exception. The search of the locked glove compartment would be permissible under Acevedo if supported by probable cause."""

        elif any(word in prompt_lower for word in ["question", "ask", "legal advice"]):
            return """**Legal Analysis - Employment Law Question**

**Short Answer**: No, a broad non-disclosure agreement prohibiting all salary discussions is not enforceable and violates federal labor law.

**Legal Reasoning**: 
The National Labor Relations Act (NLRA) Section 7 protects employees' rights to engage in "concerted activities" for mutual aid and protection, including discussing wages and working conditions. This applies to both unionized and non-unionized workplaces.

**Relevant Law**: 
- 29 U.S.C. § 157 (NLRA Section 7)
- NLRB precedent consistently invalidates broad salary confidentiality clauses

**Exceptions**: 
Limited confidentiality restrictions may be permissible for:
- Managerial employees with access to confidential business information
- HR personnel with access to payroll data
- Specific proprietary compensation formulas (not individual salaries)

**Recommendations**:
1. Remove broad salary confidentiality clauses
2. Limit restrictions to legitimate business information
3. Ensure any confidentiality clauses are narrowly tailored
4. Consider California's additional protections under Labor Code § 232

**Note**: This analysis is for informational purposes. Consult with employment counsel for specific legal advice."""

        else:
            return """**Legal Analysis**

Based on the provided information, this matter requires careful legal analysis considering applicable statutes, case law, and regulatory guidance.

**Key Considerations**:
1. Jurisdictional requirements and applicable law
2. Statutory elements and their application to the facts
3. Relevant precedent and case law analysis
4. Practical implications and recommendations

**Analysis Framework**:
- Legal standards and burden of proof
- Factual elements and evidence requirements
- Risk assessment and mitigation strategies
- Compliance considerations

**Recommendation**: This analysis requires review of specific facts and applicable law. Professional legal consultation is recommended for definitive guidance.

**Disclaimer**: This response is for informational purposes only and does not constitute legal advice."""
    
    def get_model_name(self) -> str:
        return self.model_name


class OllamaProvider(BaseLLMProvider):
    """Provider for Ollama - free local LLM runner"""
    
    def __init__(self, 
                 model: str = "llama2",
                 base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    async def generate_response(self, 
                              prompt: str,
                              system_prompt: Optional[str] = None,
                              **kwargs) -> LLMResponse:
        start_time = time.time()
        
        try:
            # Prepare the request for Ollama API
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
            
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("response", "")
                
                response_time = time.time() - start_time
                
                return LLMResponse(
                    content=content,
                    model_name=f"ollama-{self.model}",
                    tokens_used=len(content.split()) * 1.3,
                    response_time=response_time,
                    metadata={"ollama": True, "model": self.model}
                )
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            response_time = time.time() - start_time
            return LLMResponse(
                content=f"Error connecting to Ollama: {str(e)}. Please ensure Ollama is running with: ollama serve",
                model_name=f"ollama-{self.model}",
                tokens_used=0,
                response_time=response_time,
                metadata={"error": str(e), "ollama": True}
            )
    
    def get_model_name(self) -> str:
        return f"ollama-{self.model}"


class HuggingFaceProvider(BaseLLMProvider):
    """Provider for Hugging Face free inference API"""
    
    def __init__(self, 
                 model: str = "microsoft/DialoGPT-large",
                 api_url: str = "https://api-inference.huggingface.co/models/"):
        self.model = model
        self.api_url = api_url + model
    
    async def generate_response(self, 
                              prompt: str,
                              system_prompt: Optional[str] = None,
                              **kwargs) -> LLMResponse:
        start_time = time.time()
        
        try:
            # Prepare the prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            payload = {
                "inputs": full_prompt,
                "parameters": {
                    "max_new_tokens": 500,
                    "temperature": 0.7,
                    "return_full_text": False
                }
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Handle different response formats
                if isinstance(result, list) and len(result) > 0:
                    content = result[0].get("generated_text", "")
                elif isinstance(result, dict):
                    content = result.get("generated_text", "")
                else:
                    content = str(result)
                
                response_time = time.time() - start_time
                
                return LLMResponse(
                    content=content,
                    model_name=f"hf-{self.model.split('/')[-1]}",
                    tokens_used=len(content.split()) * 1.3,
                    response_time=response_time,
                    metadata={"huggingface": True, "model": self.model}
                )
            else:
                raise Exception(f"HuggingFace API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            response_time = time.time() - start_time
            return LLMResponse(
                content=f"Error with HuggingFace API: {str(e)}. Note: HuggingFace free tier has limitations.",
                model_name=f"hf-{self.model.split('/')[-1]}",
                tokens_used=0,
                response_time=response_time,
                metadata={"error": str(e), "huggingface": True}
            )
    
    def get_model_name(self) -> str:
        return f"hf-{self.model.split('/')[-1]}"


# Free model configurations
FREE_MODEL_CONFIGS = {
    "mock-legal": {
        "provider": MockLLMProvider,
        "description": "Mock LLM for testing - generates realistic legal responses"
    },
    "ollama-llama2": {
        "provider": lambda: OllamaProvider("llama2"),
        "description": "Llama2 via Ollama (requires local installation)"
    },
    "ollama-codellama": {
        "provider": lambda: OllamaProvider("codellama"),
        "description": "CodeLlama via Ollama (requires local installation)"
    },
    "hf-legal-bert": {
        "provider": lambda: HuggingFaceProvider("nlpaueb/legal-bert-base-uncased"),
        "description": "Legal-BERT via HuggingFace (free tier)"
    }
}
