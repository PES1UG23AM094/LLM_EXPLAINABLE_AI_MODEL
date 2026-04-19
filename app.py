import os

import streamlit as st
from dotenv import load_dotenv

from explanation_engine import (
    detect_weak_steps,
    get_confidence_color,
    get_hallucination_color,
    get_hallucination_emoji,
    get_reasoning_quality,
)
from llm_pipeline import analyze_decision
from models import DecisionAnalysis

load_dotenv()

st.set_page_config(
    page_title="ExplainAI — Decision Justification System",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
<style>
.answer-box {
    background-color: #f0f7ff;
    color: #1a1a2e !important;
    border-left: 4px solid #1a73e8;
    padding: 16px 20px;
    border-radius: 6px;
    margin: 8px 0 16px 0;
    font-size: 16px;
    line-height: 1.6;
}
.step-weak {
    background-color: #fff8e1;
    color: #1a1a2e !important;
    border-left: 3px solid #ffc107;
    padding: 10px 14px;
    margin: 6px 0;
    border-radius: 4px;
}
.step-strong {
    background-color: #f1f8f4;
    color: #1a1a2e !important;
    border-left: 3px solid #28a745;
    padding: 10px 14px;
    margin: 6px 0;
    border-radius: 4px;
}
.risk-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 13px;
}
.section-label {
    font-weight: 600;
    margin-bottom: 6px;
    font-size: 14px;
    color: #444;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
</style>
""",
    unsafe_allow_html=True,
)

MODEL_LABELS = {
    "gpt-4o": "GPT-4o (Most Capable)",
    "gpt-4o-mini": "GPT-4o Mini (Fast)",
}

EXAMPLE_QUESTIONS = [
    "Should I invest in cryptocurrency?",
    "Is it better to rent or buy a home?",
    "Should I pursue an MBA?",
    "Is remote work better than office work for productivity?",
    "Should I switch careers from engineering to product management?",
]


def render_confidence_bar(score: float) -> None:
    pct = int(score * 100)
    color = get_confidence_color(score)
    st.markdown(
        f'<div style="background: linear-gradient(to right, {color} {pct}%, '
        f'#e9ecef {pct}%); height: 22px; border-radius: 11px; margin: 4px 0;"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**{pct}%** confident")


def render_analysis(analysis: DecisionAnalysis) -> None:
    """Render all sections of a DecisionAnalysis result."""
    # --- Answer ---
    st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="answer-box">{analysis.answer}</div>',
        unsafe_allow_html=True,
    )

    # --- Metrics row ---
    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown('<div class="section-label">Confidence</div>', unsafe_allow_html=True)
        render_confidence_bar(analysis.confidence_score)

    with m2:
        risk = analysis.hallucination_risk
        color = get_hallucination_color(risk)
        emoji = get_hallucination_emoji(risk)
        st.markdown('<div class="section-label">Hallucination Risk</div>', unsafe_allow_html=True)
        st.markdown(
            f'<span class="risk-badge" style="background:{color}22; color:{color};">'
            f"{emoji} {risk.upper()}</span>",
            unsafe_allow_html=True,
        )

    with m3:
        _, quality_label = get_reasoning_quality(analysis)
        strong = sum(1 for s in analysis.reasoning_steps if not s.is_weak)
        total = len(analysis.reasoning_steps)
        st.markdown('<div class="section-label">Reasoning Quality</div>', unsafe_allow_html=True)
        st.markdown(f"**{quality_label}** ({strong}/{total} strong steps)")

    st.markdown("")

    # --- Reasoning Steps ---
    with st.expander("Reasoning Steps", expanded=True):
        for step in analysis.reasoning_steps:
            if step.is_weak:
                weakness_note = (
                    f'<br><small style="color:#856404;"><em>⚠ {step.weakness_reason}</em></small>'
                    if step.weakness_reason
                    else ""
                )
                st.markdown(
                    f'<div class="step-weak"><strong>Step {step.step_number}:</strong> '
                    f"{step.description}{weakness_note}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="step-strong"><strong>Step {step.step_number}:</strong> '
                    f"{step.description}</div>",
                    unsafe_allow_html=True,
                )

    # --- Detail columns ---
    d1, d2, d3 = st.columns(3)

    with d1:
        st.markdown('<div class="section-label">Assumptions</div>', unsafe_allow_html=True)
        for a in analysis.assumptions:
            st.markdown(f"• {a}")

    with d2:
        st.markdown('<div class="section-label">Uncertainties</div>', unsafe_allow_html=True)
        for u in analysis.uncertainty_highlights:
            st.markdown(f"⚡ {u}")

    with d3:
        st.markdown('<div class="section-label">Alternative Views</div>', unsafe_allow_html=True)
        for v in analysis.alternative_viewpoints:
            st.markdown(f"↔ {v}")

    st.markdown("")

    # --- Counter-argument ---
    st.markdown('<div class="section-label">Counter-Argument (Devil\'s Advocate)</div>', unsafe_allow_html=True)
    st.info(f"🔄 {analysis.counter_argument}")

    # --- Hallucination details ---
    if analysis.hallucination_reasons:
        with st.expander("Why this hallucination risk?"):
            for r in analysis.hallucination_reasons:
                st.markdown(f"• {r}")


def main() -> None:
    st.title("🧠 ExplainAI — Decision Justification System")
    st.markdown(
        "Ask a decision question and get a **transparent, explainable AI analysis** "
        "with reasoning steps, confidence scores, and alternative viewpoints."
    )

    if not os.getenv("OPENAI_API_KEY"):
        st.error(
            "OPENAI_API_KEY not found. "
            "Copy `.env.example` to `.env` and add your key."
        )
        st.stop()

    tab_single, tab_compare = st.tabs(["Single Analysis", "Model Comparison"])

    # ── Tab 1: Single Analysis ──────────────────────────────────────────────
    with tab_single:
        st.markdown("### Ask a Decision Question")

        example = st.selectbox(
            "Pick an example or type below:",
            ["(Type your own question)"] + EXAMPLE_QUESTIONS,
            key="ex1",
        )

        question = st.text_area(
            "Your question:",
            value="" if example == "(Type your own question)" else example,
            placeholder="e.g., Should I switch careers from engineering to product management?",
            height=100,
            key="q1",
        )

        model = st.selectbox(
            "Model:",
            list(MODEL_LABELS.keys()),
            format_func=lambda m: MODEL_LABELS[m],
            key="m1",
        )

        if st.button("Analyze Decision", type="primary", key="btn1"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner(f"Analyzing with {MODEL_LABELS[model]}…"):
                    try:
                        result = analyze_decision(question.strip(), model)
                        result = detect_weak_steps(result)
                        st.session_state["result1"] = result
                        st.session_state["q1_text"] = question.strip()
                    except Exception as exc:
                        st.error(f"Error: {exc}")

        if "result1" in st.session_state:
            st.divider()
            st.markdown(f"**Question:** *{st.session_state['q1_text']}*")
            render_analysis(st.session_state["result1"])

    # ── Tab 2: Model Comparison ─────────────────────────────────────────────
    with tab_compare:
        st.markdown("### Compare Two Models Side by Side")
        st.markdown(
            "Run the same question through two different models to see how their "
            "reasoning and confidence differ."
        )

        example2 = st.selectbox(
            "Pick an example or type below:",
            ["(Type your own question)"] + EXAMPLE_QUESTIONS,
            key="ex2",
        )

        question2 = st.text_area(
            "Your question:",
            value="" if example2 == "(Type your own question)" else example2,
            placeholder="e.g., Should I start my own business?",
            height=100,
            key="q2",
        )

        c1, c2 = st.columns(2)
        with c1:
            model_a = st.selectbox(
                "Model A:",
                list(MODEL_LABELS.keys()),
                format_func=lambda m: MODEL_LABELS[m],
                index=0,
                key="ma",
            )
        with c2:
            model_b = st.selectbox(
                "Model B:",
                list(MODEL_LABELS.keys()),
                format_func=lambda m: MODEL_LABELS[m],
                index=1,
                key="mb",
            )

        if st.button("Compare Models", type="primary", key="btn2"):
            if not question2.strip():
                st.warning("Please enter a question.")
            else:
                results, errors = {}, {}
                for m in [model_a, model_b]:
                    with st.spinner(f"Analyzing with {MODEL_LABELS[m]}…"):
                        try:
                            r = analyze_decision(question2.strip(), m)
                            results[m] = detect_weak_steps(r)
                        except Exception as exc:
                            errors[m] = str(exc)

                st.session_state["cmp_results"] = results
                st.session_state["cmp_errors"] = errors
                st.session_state["cmp_question"] = question2.strip()
                st.session_state["cmp_models"] = (model_a, model_b)

        if "cmp_results" in st.session_state:
            st.divider()
            st.markdown(f"**Question:** *{st.session_state['cmp_question']}*")

            ma, mb = st.session_state["cmp_models"]
            results = st.session_state["cmp_results"]
            errors = st.session_state.get("cmp_errors", {})

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown(f"## {MODEL_LABELS[ma].split(' (')[0]}")
                if ma in errors:
                    st.error(errors[ma])
                elif ma in results:
                    render_analysis(results[ma])

            with col_b:
                st.markdown(f"## {MODEL_LABELS[mb].split(' (')[0]}")
                if mb in errors:
                    st.error(errors[mb])
                elif mb in results:
                    render_analysis(results[mb])


if __name__ == "__main__":
    main()
