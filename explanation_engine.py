from models import DecisionAnalysis, ReasoningStep
from typing import Tuple

# Words that signal uncertain or speculative reasoning
_WEAK_INDICATORS = [
    "might", "could", "perhaps", "maybe", "possibly", "probably",
    "seems", "appears", "likely", "unlikely", "not sure", "uncertain",
    "depends", "varies", "generally", "typically", "often", "sometimes",
    "I think", "I believe", "I feel", "I assume",
]

_MIN_STEP_LENGTH = 30


def detect_weak_steps(analysis: DecisionAnalysis) -> DecisionAnalysis:
    """
    Post-processes reasoning steps to flag any that contain uncertain language
    or lack sufficient detail, even if the model did not self-flag them.
    Returns a new DecisionAnalysis with updated step annotations.
    """
    updated_steps = []
    for step in analysis.reasoning_steps:
        if step.is_weak:
            updated_steps.append(step)
            continue

        desc_lower = step.description.lower()
        matched = [w for w in _WEAK_INDICATORS if w in desc_lower]

        if matched:
            reason = f"Contains uncertain language: {', '.join(matched[:3])}"
            updated_steps.append(
                ReasoningStep(
                    step_number=step.step_number,
                    description=step.description,
                    is_weak=True,
                    weakness_reason=reason,
                )
            )
        elif len(step.description.strip()) < _MIN_STEP_LENGTH:
            updated_steps.append(
                ReasoningStep(
                    step_number=step.step_number,
                    description=step.description,
                    is_weak=True,
                    weakness_reason="Insufficient detail in reasoning step",
                )
            )
        else:
            updated_steps.append(step)

    return DecisionAnalysis(
        answer=analysis.answer,
        reasoning_steps=updated_steps,
        assumptions=analysis.assumptions,
        confidence_score=analysis.confidence_score,
        uncertainty_highlights=analysis.uncertainty_highlights,
        alternative_viewpoints=analysis.alternative_viewpoints,
        counter_argument=analysis.counter_argument,
        hallucination_risk=analysis.hallucination_risk,
        hallucination_reasons=analysis.hallucination_reasons,
    )


def get_reasoning_quality(analysis: DecisionAnalysis) -> Tuple[float, str]:
    """
    Returns (quality_score, quality_label) based on the proportion of strong
    reasoning steps weighted against the overall confidence score.
    """
    total = len(analysis.reasoning_steps)
    if total == 0:
        return 0.0, "No Reasoning"

    weak_count = sum(1 for s in analysis.reasoning_steps if s.is_weak)
    strong_ratio = 1.0 - (weak_count / total)

    # Blend step quality (70%) with stated confidence (30%)
    score = strong_ratio * 0.7 + analysis.confidence_score * 0.3

    if score >= 0.8:
        label = "Strong"
    elif score >= 0.6:
        label = "Moderate"
    elif score >= 0.4:
        label = "Weak"
    else:
        label = "Very Weak"

    return score, label


def get_confidence_color(score: float) -> str:
    if score >= 0.7:
        return "#28a745"
    elif score >= 0.4:
        return "#fd7e14"
    return "#dc3545"


def get_hallucination_color(risk: str) -> str:
    return {"low": "#28a745", "medium": "#fd7e14", "high": "#dc3545"}.get(
        risk, "#6c757d"
    )


def get_hallucination_emoji(risk: str) -> str:
    return {"low": "✅", "medium": "⚠️", "high": "🚨"}.get(risk, "❓")
