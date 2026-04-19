from openai import OpenAI
from models import DecisionAnalysis

SYSTEM_PROMPT = """You are an Explainable AI Decision Assistant. Your role is to help users make \
informed decisions by providing transparent, well-reasoned analysis.

For every decision question you must:
1. Give a clear, direct answer
2. Show your complete chain-of-thought reasoning in numbered steps
3. Identify the key assumptions you are making
4. Honestly assess your confidence level (0.0 = completely uncertain, 1.0 = absolutely certain)
5. Highlight specific areas of genuine uncertainty
6. Present 2-3 alternative viewpoints or opposing perspectives fairly
7. Provide the single strongest counter-argument to your position
8. Assess the risk that you may be hallucinating or confabulating information

Be intellectually honest. If you do not know something with certainty, say so explicitly. \
Mark reasoning steps as weak if they rely on uncertain information, vague generalizations, \
or logical leaps without sufficient evidence."""


def analyze_decision(question: str, model: str = "gpt-4o") -> DecisionAnalysis:
    """
    Runs the core LLM pipeline: takes a decision question and returns a structured
    DecisionAnalysis using OpenAI Structured Outputs via the beta parse endpoint.
    """
    client = OpenAI()

    response = client.beta.chat.completions.parse(
        model=model,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Please analyze this decision question:\n\n{question}",
            },
        ],
        response_format=DecisionAnalysis,
    )

    result = response.choices[0].message.parsed

    if result is None:
        refusal = response.choices[0].message.refusal
        raise ValueError(
            f"Model did not return a structured analysis. "
            f"Refusal: {refusal or 'unknown'}"
        )

    # Clamp confidence to valid range in case model drifts
    result.confidence_score = max(0.0, min(1.0, result.confidence_score))
    return result
