from pydantic import BaseModel, ConfigDict, Field
from typing import List, Literal


class ReasoningStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_number: int
    description: str
    is_weak: bool = False
    weakness_reason: str = ""


class DecisionAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="The main answer/recommendation")
    reasoning_steps: List[ReasoningStep] = Field(description="Step-by-step reasoning chain")
    assumptions: List[str] = Field(description="Key assumptions underlying the analysis")
    confidence_score: float = Field(description="Confidence 0.0-1.0")
    uncertainty_highlights: List[str] = Field(description="Areas of genuine uncertainty")
    alternative_viewpoints: List[str] = Field(description="Alternative perspectives or opposing views")
    counter_argument: str = Field(description="Strongest argument against the main answer")
    hallucination_risk: Literal["low", "medium", "high"] = Field(description="Risk of hallucinated content")
    hallucination_reasons: List[str] = Field(description="Reasons for the hallucination risk level")
