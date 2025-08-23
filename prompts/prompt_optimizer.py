"""
Legal LLM Prompt Optimization Framework
Inspired by HumanEval's systematic evaluation approach
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json


class LegalTaskType(Enum):
    STATUTORY_INTERPRETATION = "statutory_interpretation"
    CONTRACT_REVIEW = "contract_review"
    CASE_SUMMARIZATION = "case_summarization"
    LEGAL_QA = "legal_qa"
    PRECEDENT_RETRIEVAL = "precedent_retrieval"
    CLAUSE_EXTRACTION = "clause_extraction"


class PromptStrategy(Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    ROLE_BASED = "role_based"
    CONTEXT_LAYERED = "context_layered"


@dataclass
class LegalPromptTemplate:
    task_type: LegalTaskType
    strategy: PromptStrategy
    template: str
    role_context: Optional[str] = None
    reasoning_steps: Optional[List[str]] = None
    examples: Optional[List[Dict]] = None


class LegalPromptOptimizer:
    def __init__(self):
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[LegalTaskType, Dict[PromptStrategy, LegalPromptTemplate]]:
        return {
            LegalTaskType.CONTRACT_REVIEW: {
                PromptStrategy.ROLE_BASED: LegalPromptTemplate(
                    task_type=LegalTaskType.CONTRACT_REVIEW,
                    strategy=PromptStrategy.ROLE_BASED,
                    template="""You are an experienced contract attorney with 15 years of practice. 
                    
Review the following contract and provide a comprehensive analysis:

Contract: {contract_text}

Please analyze:
1. Key terms and obligations
2. Potential risks and liabilities
3. Missing or problematic clauses
4. Recommendations for amendments

Provide specific legal citations where applicable.""",
                    role_context="Experienced contract attorney with expertise in commercial law"
                ),
                
                PromptStrategy.CHAIN_OF_THOUGHT: LegalPromptTemplate(
                    task_type=LegalTaskType.CONTRACT_REVIEW,
                    strategy=PromptStrategy.CHAIN_OF_THOUGHT,
                    template="""Let's analyze this contract step by step:

Contract: {contract_text}

Step 1: Identify the contract type and governing law
Step 2: Extract key parties and their obligations
Step 3: Analyze terms for completeness and clarity
Step 4: Identify potential legal risks
Step 5: Provide recommendations

Think through each step carefully and explain your reasoning.""",
                    reasoning_steps=[
                        "Identify contract type and governing law",
                        "Extract key parties and obligations", 
                        "Analyze terms for completeness",
                        "Identify potential risks",
                        "Provide recommendations"
                    ]
                )
            },
            
            LegalTaskType.STATUTORY_INTERPRETATION: {
                PromptStrategy.CONTEXT_LAYERED: LegalPromptTemplate(
                    task_type=LegalTaskType.STATUTORY_INTERPRETATION,
                    strategy=PromptStrategy.CONTEXT_LAYERED,
                    template="""Context: You are interpreting {jurisdiction} law.

Statute: {statute_text}

Legal Question: {question}

Analysis Framework:
1. Plain meaning interpretation
2. Legislative intent and history
3. Relevant case law precedents
4. Policy considerations

Provide a detailed interpretation with supporting citations.""",
                    role_context="Legal scholar specializing in statutory interpretation"
                )
            },
            
            LegalTaskType.CASE_SUMMARIZATION: {
                PromptStrategy.FEW_SHOT: LegalPromptTemplate(
                    task_type=LegalTaskType.CASE_SUMMARIZATION,
                    strategy=PromptStrategy.FEW_SHOT,
                    template="""Summarize the following legal case following this format:

Example 1:
Case: Brown v. Board of Education, 347 U.S. 483 (1954)
Summary: Supreme Court ruled that racial segregation in public schools violated the Equal Protection Clause, overturning Plessy v. Ferguson.
Key Holdings: Separate educational facilities are inherently unequal.
Impact: Landmark civil rights decision ending legal segregation.

Example 2:
Case: Miranda v. Arizona, 384 U.S. 436 (1966)  
Summary: Supreme Court established that suspects must be informed of their rights before interrogation.
Key Holdings: Fifth Amendment protections require warning of right to remain silent and right to counsel.
Impact: Created Miranda rights standard for police procedures.

Now summarize:
Case: {case_text}""",
                    examples=[
                        {
                            "case": "Brown v. Board of Education",
                            "summary": "Supreme Court ruled racial segregation unconstitutional",
                            "holdings": "Separate educational facilities are inherently unequal",
                            "impact": "Ended legal segregation"
                        }
                    ]
                )
            },
            
            LegalTaskType.LEGAL_QA: {
                PromptStrategy.ROLE_BASED: LegalPromptTemplate(
                    task_type=LegalTaskType.LEGAL_QA,
                    strategy=PromptStrategy.ROLE_BASED,
                    template="""You are a seasoned legal advisor with expertise across multiple practice areas.

Question: {question}

Context: {context}

Please provide a comprehensive legal analysis that includes:
1. Direct answer to the question
2. Relevant legal principles and authorities
3. Potential risks and considerations
4. Practical recommendations
5. Any jurisdictional limitations

Ensure your response is accurate, cites relevant authorities, and acknowledges any uncertainties.""",
                    role_context="Experienced legal advisor with multi-jurisdictional expertise"
                ),
                
                PromptStrategy.CHAIN_OF_THOUGHT: LegalPromptTemplate(
                    task_type=LegalTaskType.LEGAL_QA,
                    strategy=PromptStrategy.CHAIN_OF_THOUGHT,
                    template="""Let me work through this legal question systematically:

Question: {question}
Context: {context}

Step 1: Identify the legal issue(s)
- What is the core legal question being asked?
- What area(s) of law does this involve?

Step 2: Gather relevant legal authorities  
- What statutes, regulations, or case law apply?
- Are there jurisdictional considerations?

Step 3: Apply legal principles to the facts
- How do the authorities address this specific situation?
- Are there any distinguishing factors?

Step 4: Consider practical implications
- What are the risks and benefits of different approaches?
- What would I recommend to a client?

Step 5: Provide clear answer with caveats
- What is my conclusion?
- What limitations or uncertainties exist?

Let me think through each step...""",
                    reasoning_steps=[
                        "Identify legal issues",
                        "Gather relevant authorities",
                        "Apply principles to facts", 
                        "Consider practical implications",
                        "Provide clear answer with caveats"
                    ]
                ),
                
                PromptStrategy.CONTEXT_LAYERED: LegalPromptTemplate(
                    task_type=LegalTaskType.LEGAL_QA,
                    strategy=PromptStrategy.CONTEXT_LAYERED,
                    template="""Legal Context: {context}

Jurisdiction: {jurisdiction}
Practice Area: {practice_area}

Question: {question}

Analysis Framework:
1. Legal Standards: What laws, regulations, and precedents apply?
2. Factual Application: How do these standards apply to the specific facts?
3. Risk Assessment: What are the potential legal exposures?
4. Strategic Considerations: What practical options are available?

Please provide a thorough analysis following this framework.""",
                    role_context="Legal expert analyzing within specific jurisdictional context"
                )
            },
            
            LegalTaskType.PRECEDENT_RETRIEVAL: {
                PromptStrategy.ROLE_BASED: LegalPromptTemplate(
                    task_type=LegalTaskType.PRECEDENT_RETRIEVAL,
                    strategy=PromptStrategy.ROLE_BASED,
                    template="""You are a legal research specialist with extensive experience in case law analysis.

Research Request: {question}
Legal Context: {context}

Please identify relevant precedents and provide:
1. Most relevant case citations with brief summaries
2. Key legal principles established by each case
3. How each precedent applies to the current situation
4. Any distinguishing factors or limitations
5. Hierarchy of authority (Supreme Court, Circuit, State, etc.)

Focus on binding authority first, then persuasive authority. Include both favorable and potentially adverse precedents.""",
                    role_context="Legal research specialist with expertise in precedent analysis"
                ),
                
                PromptStrategy.CHAIN_OF_THOUGHT: LegalPromptTemplate(
                    task_type=LegalTaskType.PRECEDENT_RETRIEVAL,
                    strategy=PromptStrategy.CHAIN_OF_THOUGHT,
                    template="""Let me systematically search for relevant precedents:

Research Query: {question}
Context: {context}

Step 1: Identify key legal concepts and search terms
- What are the main legal issues?
- What specific doctrines or principles are involved?

Step 2: Determine relevant jurisdictions and courts
- What jurisdiction governs this matter?
- What level of courts should I prioritize?

Step 3: Search for binding precedents
- What cases from higher courts directly address this issue?
- Are there circuit splits or conflicting authority?

Step 4: Identify persuasive authority
- What cases from other jurisdictions are relevant?
- Are there influential district court decisions?

Step 5: Analyze precedential value
- Which cases are most analogous to our facts?
- How strong is the precedential support?

Let me work through this research systematically...""",
                    reasoning_steps=[
                        "Identify key legal concepts",
                        "Determine relevant jurisdictions", 
                        "Search for binding precedents",
                        "Identify persuasive authority",
                        "Analyze precedential value"
                    ]
                ),
                
                PromptStrategy.FEW_SHOT: LegalPromptTemplate(
                    task_type=LegalTaskType.PRECEDENT_RETRIEVAL,
                    strategy=PromptStrategy.FEW_SHOT,
                    template="""Find relevant precedents following these examples:

Example 1:
Query: Fourth Amendment search and seizure in vehicles
Relevant Precedents:
- United States v. Ross, 456 U.S. 798 (1982): Automobile exception allows warrantless search if probable cause exists
- Pennsylvania v. Labron, 518 U.S. 938 (1996): No separate exigent circumstances required for vehicle searches
- Arizona v. Gant, 556 U.S. 332 (2009): Limited search incident to arrest for vehicles

Example 2:  
Query: Employment discrimination based on pregnancy
Relevant Precedents:
- General Electric Co. v. Gilbert, 429 U.S. 125 (1976): Pregnancy discrimination not sex discrimination under Title VII
- Pregnancy Discrimination Act of 1978: Overruled Gilbert, prohibits pregnancy discrimination
- Young v. UPS, 575 U.S. 206 (2015): Clarified burden-shifting framework for pregnancy accommodation claims

Now find precedents for:
Query: {question}
Context: {context}""",
                    examples=[
                        {
                            "query": "Fourth Amendment vehicle searches",
                            "precedents": ["United States v. Ross", "Pennsylvania v. Labron", "Arizona v. Gant"]
                        }
                    ]
                )
            },
            
            LegalTaskType.CLAUSE_EXTRACTION: {
                PromptStrategy.ROLE_BASED: LegalPromptTemplate(
                    task_type=LegalTaskType.CLAUSE_EXTRACTION,
                    strategy=PromptStrategy.ROLE_BASED,
                    template="""You are a contract analysis expert specializing in clause identification and extraction.

Document: {contract_text}

Please extract and categorize the following types of clauses:
1. Essential Terms (parties, consideration, subject matter)
2. Performance Obligations (deliverables, timelines, standards)
3. Risk Allocation (indemnification, liability limits, insurance)
4. Dispute Resolution (governing law, arbitration, jurisdiction)
5. Termination Provisions (grounds, notice, effect)
6. Special Clauses (non-compete, confidentiality, force majeure)

For each identified clause:
- Provide the exact text
- Categorize the clause type
- Note any unusual or concerning provisions
- Assess enforceability risks""",
                    role_context="Contract analysis expert with experience in clause identification"
                ),
                
                PromptStrategy.CHAIN_OF_THOUGHT: LegalPromptTemplate(
                    task_type=LegalTaskType.CLAUSE_EXTRACTION,
                    strategy=PromptStrategy.CHAIN_OF_THOUGHT,
                    template="""Let me systematically extract clauses from this document:

Document: {contract_text}

Step 1: Scan for essential contractual elements
- Who are the parties?
- What is the consideration?
- What is the subject matter?

Step 2: Identify performance obligations
- What must each party do?
- When must they do it?
- What standards apply?

Step 3: Look for risk allocation provisions
- How are risks divided between parties?
- Are there liability limitations?
- What insurance requirements exist?

Step 4: Find dispute resolution mechanisms
- What law governs?
- How are disputes resolved?
- Where can litigation occur?

Step 5: Extract special or unusual clauses
- Are there restrictive covenants?
- How does the contract terminate?
- Are there any unique provisions?

Let me work through each section carefully...""",
                    reasoning_steps=[
                        "Scan for essential elements",
                        "Identify performance obligations",
                        "Look for risk allocation",
                        "Find dispute resolution",
                        "Extract special clauses"
                    ]
                ),
                
                PromptStrategy.FEW_SHOT: LegalPromptTemplate(
                    task_type=LegalTaskType.CLAUSE_EXTRACTION,
                    strategy=PromptStrategy.FEW_SHOT,
                    template="""Extract clauses following these examples:

Example 1:
Document excerpt: "Company shall indemnify and hold harmless Client from any third-party claims arising from Company's negligent performance..."
Extracted Clause:
- Type: Indemnification 
- Text: "Company shall indemnify and hold harmless Client from any third-party claims arising from Company's negligent performance"
- Analysis: One-way indemnification favoring Client, limited to negligent acts

Example 2:
Document excerpt: "This Agreement shall terminate automatically upon 30 days written notice by either party..."  
Extracted Clause:
- Type: Termination
- Text: "This Agreement shall terminate automatically upon 30 days written notice by either party"
- Analysis: Termination for convenience with reasonable notice period

Now extract clauses from:
Document: {contract_text}""",
                    examples=[
                        {
                            "text": "indemnification clause",
                            "type": "Risk Allocation",
                            "analysis": "One-way indemnification"
                        }
                    ]
                )
            }
        }
    
    def generate_prompt(self, 
                       task_type: LegalTaskType, 
                       strategy: PromptStrategy,
                       **kwargs) -> str:
        """Generate optimized prompt for specific legal task and strategy"""
        
        if task_type not in self.templates:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        if strategy not in self.templates[task_type]:
            raise ValueError(f"Unsupported strategy {strategy} for task {task_type}")
        
        template = self.templates[task_type][strategy]
        return template.template.format(**kwargs)
    
    def get_multi_strategy_prompts(self, 
                                  task_type: LegalTaskType, 
                                  **kwargs) -> Dict[PromptStrategy, str]:
        """Generate prompts using all available strategies for a task"""
        
        results = {}
        if task_type in self.templates:
            for strategy in self.templates[task_type]:
                try:
                    results[strategy] = self.generate_prompt(task_type, strategy, **kwargs)
                except KeyError as e:
                    print(f"Missing parameter for {strategy}: {e}")
        
        return results
    
    def optimize_for_domain(self, 
                           domain: str, 
                           sample_tasks: List[Dict]) -> Dict:
        """Optimize prompts for specific legal domain based on sample performance"""
        
        optimization_results = {
            "domain": domain,
            "best_strategies": {},
            "performance_metrics": {}
        }
        
        for task in sample_tasks:
            task_type = LegalTaskType(task["task_type"])
            
            # Generate prompts with different strategies
            prompts = self.get_multi_strategy_prompts(task_type, **task.get("inputs", {}))
            
            # Track which strategies are available for this domain
            optimization_results["best_strategies"][task_type.value] = list(prompts.keys())
        
        return optimization_results


def create_legal_evaluation_prompt(task_data: Dict) -> str:
    """Create evaluation prompt in HumanEval style for legal tasks"""
    
    return f"""
Legal Evaluation Task: {task_data['task_id']}

Task Type: {task_data['task_type']}
Jurisdiction: {task_data.get('jurisdiction', 'General')}
Complexity: {task_data.get('complexity', 'intermediate')}

{task_data['prompt']}

Context:
{task_data.get('context', '')}

Expected Analysis Framework:
{chr(10).join(f"- {criterion}" for criterion in task_data.get('evaluation_criteria', []))}
"""


if __name__ == "__main__":
    optimizer = LegalPromptOptimizer()
    
    # Example usage
    contract_prompt = optimizer.generate_prompt(
        LegalTaskType.CONTRACT_REVIEW,
        PromptStrategy.ROLE_BASED,
        contract_text="Sample employment agreement with non-compete clause..."
    )
    
    print("Generated Contract Review Prompt:")
    print(contract_prompt)