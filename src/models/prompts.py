"""Prompt templates for hypothesis generation and factual QA."""

from dataclasses import dataclass
from typing import Optional


HYPOTHESIS_SYSTEM = """You are an expert scientific researcher. Your task is to generate a novel, testable scientific hypothesis based on a research abstract.

A good hypothesis must:
1. Be grounded in the findings of the abstract
2. Propose something that is NOT already stated in the abstract
3. Be specific and empirically testable
4. Use precise scientific language
5. Be a single clear sentence"""


HYPOTHESIS_ZERO_SHOT = """Given the following research abstract, generate ONE novel scientific hypothesis that extends or builds upon the described work. The hypothesis should be specific, testable, and go beyond what is already stated.

Abstract:
{abstract}

Generate a single hypothesis sentence starting with "We hypothesize that" or "We propose that":"""


HYPOTHESIS_FEW_SHOT = """Given the following research abstract, generate ONE novel scientific hypothesis that extends or builds upon the described work. The hypothesis should be specific, testable, and go beyond what is already stated.

Here are two examples:

Example 1:
Abstract: "Transformer models trained on large text corpora demonstrate emergent reasoning abilities at scale. We find that models with over 100B parameters spontaneously develop chain-of-thought reasoning without explicit training."
Hypothesis: We hypothesize that the threshold for emergent chain-of-thought reasoning is not fixed at 100B parameters but instead depends on the diversity of the training corpus, such that models trained on scientifically diverse data will exhibit this capability at smaller scales.

Example 2:
Abstract: "CRISPR-Cas9 gene editing successfully corrected the BRCA1 mutation in 78% of human cell lines in vitro. Off-target effects were observed in 3.2% of cases."
Hypothesis: We hypothesize that the off-target editing rate of CRISPR-Cas9 can be reduced below 0.5% by engineering guide RNAs with thermodynamic mismatch tolerance profiles tuned specifically to the GC-content distribution of the target genome region.

Now generate a hypothesis for:
Abstract:
{abstract}

Generate a single hypothesis sentence:"""


HYPOTHESIS_COT = """Given the following research abstract, reason step by step and then generate a novel scientific hypothesis.

Abstract:
{abstract}

Step 1 - Key findings from abstract: First, identify the main findings.
Step 2 - Knowledge gap: What does this work leave unanswered?
Step 3 - Hypothesis: Based on steps 1-2, formulate ONE specific, testable hypothesis.

Format your response as:
FINDINGS: [key findings]
GAP: [knowledge gap identified]
HYPOTHESIS: [your hypothesis starting with "We hypothesize that"]"""


FACTUAL_QA_PROMPT = """Answer the following science question concisely and accurately.

Question: {question}

{context_block}
Provide only the answer, no explanation:"""


JUDGE_SYSTEM = """You are a scientific peer reviewer evaluating AI-generated research hypotheses. Rate each hypothesis objectively and strictly."""


JUDGE_PROMPT = """Evaluate the following AI-generated scientific hypothesis based on the given abstract.

Abstract:
{abstract}

Generated Hypothesis:
{hypothesis}

Rate the hypothesis on EACH of these dimensions (1-5 scale):
- scientific_validity: Is the hypothesis scientifically coherent and plausible? (1=nonsensical, 5=highly plausible)
- testability: Could this hypothesis be empirically tested with existing methods? (1=untestable, 5=readily testable)
- specificity: Is the hypothesis specific enough to be meaningful? (1=too vague, 5=very specific)
- novelty: Does it genuinely extend beyond what the abstract already states? (1=copied from abstract, 5=highly novel)

Respond in this exact JSON format:
{{
  "scientific_validity": <1-5>,
  "testability": <1-5>,
  "specificity": <1-5>,
  "novelty": <1-5>,
  "reasoning": "<one sentence explanation>"
}}"""


@dataclass
class PromptConfig:
    strategy: str = "few_shot"  # zero_shot | few_shot | cot
    include_system: bool = True


def build_hypothesis_prompt(abstract: str, strategy: str = "few_shot") -> list[dict]:
    if strategy == "zero_shot":
        user_content = HYPOTHESIS_ZERO_SHOT.format(abstract=abstract)
    elif strategy == "few_shot":
        user_content = HYPOTHESIS_FEW_SHOT.format(abstract=abstract)
    elif strategy == "cot":
        user_content = HYPOTHESIS_COT.format(abstract=abstract)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return [
        {"role": "system", "content": HYPOTHESIS_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def build_factual_prompt(question: str, context: Optional[str] = None) -> list[dict]:
    context_block = f"Context:\n{context}\n" if context else ""
    user_content = FACTUAL_QA_PROMPT.format(
        question=question, context_block=context_block
    )
    return [
        {"role": "system", "content": "You are a knowledgeable science assistant. Answer questions accurately and concisely."},
        {"role": "user", "content": user_content},
    ]


def build_judge_prompt(abstract: str, hypothesis: str) -> list[dict]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_PROMPT.format(abstract=abstract, hypothesis=hypothesis)},
    ]


def extract_hypothesis_from_cot(text: str) -> str:
    """Pull the hypothesis line from CoT output."""
    for line in text.split("\n"):
        if line.upper().startswith("HYPOTHESIS:"):
            return line.split(":", 1)[1].strip()
    # fallback: return last non-empty line
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines[-1] if lines else text.strip()
