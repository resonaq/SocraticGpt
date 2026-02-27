"""
Socratic Engine -- CGI Conversation Analyzer
============================================
Merges mathematical CGI metrics (cgi_semantic.py) with
conversation analysis (socratic-lens) into a single tool.

Feed it a conversation -> it measures which questions
genuinely reorganized the model's predictions.

Math tells you WHAT is transformative.
The report shows you WHY.

Usage:
    python socratic_engine.py                          # built-in demo
    python socratic_engine.py conversation.json        # your own data
"""

import sys
import json
import math
from dataclasses import dataclass, field, asdict
import unicodedata

# ---- Import CGI metrics from cgi_semantic.py ----
from cgi_semantic import (
    load_model,
    get_distribution_semantic,
    kl_divergence,
    jsd,
    spearman_rho,
    topk_jaccard,
    shannon_entropy,
    attention_pattern_comparison,
    normalized_kl,
    corpus_ngram_familiarity_semantic,
    generate_continuations_semantic,
    generation_diversity,
)


def safe_print(text):
    """Print text safely on Windows (strip non-ASCII characters)."""
    cleaned = text.encode('ascii', errors='replace').decode('ascii')
    print(cleaned)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class QuestionContext:
    """A question found in conversation, with its preceding context."""
    turn_index: int
    role: str
    question_text: str
    base_context: str


@dataclass
class QuestionMetrics:
    """CGI metrics computed for a single question."""
    turn_index: int
    role: str
    question_text: str
    KL_raw: float = 0.0
    JSD: float = 0.0
    Spearman_rho: float = 0.0
    TopK_Jaccard: float = 0.0
    Delta_H_output: float = 0.0
    Delta_Attn_H: float = 0.0
    Corpus_Familiarity: float = 0.0
    KL_normalized: float = 0.0
    Delta_Diversity: float = 0.0
    composite_score: float = 0.0
    verdict: str = ""
    continuations_before: list = field(default_factory=list)
    continuations_after: list = field(default_factory=list)


# =============================================================================
# CONVERSATION PARSING
# =============================================================================

def parse_conversation(conversation: dict) -> list[QuestionContext]:
    """
    Extract every question from a conversation.
    A question = any turn containing '?'.

    For each question, base_context = all preceding turns concatenated.
    """
    turns = conversation.get("turns", [])
    questions = []

    for i, turn in enumerate(turns):
        content = turn.get("content", "")
        if "?" not in content:
            continue

        # Build base context from all preceding turns
        base_parts = []
        for prev_turn in turns[:i]:
            role = prev_turn.get("role", "?")
            text = prev_turn.get("content", "")
            base_parts.append(f"{role}: {text}")
        base_context = " ".join(base_parts) if base_parts else ""

        questions.append(QuestionContext(
            turn_index=i,
            role=turn.get("role", "?"),
            question_text=content,
            base_context=base_context,
        ))

    return questions


# =============================================================================
# CGI METRIC COMPUTATION
# =============================================================================

def analyze_question(model, tokenizer, qctx: QuestionContext,
                     corpus_docs: list[str],
                     n_continuations: int = 10,
                     topk: int = 10) -> QuestionMetrics:
    """
    Compute full CGI metrics for a single question in context.

    P_before = model(base_context)
    P_after  = model(base_context + " " + question)
    Then: KL, Spearman, Jaccard, attention shift, diversity shift, composite.
    """
    base = qctx.base_context
    full = (base + " " + qctx.question_text).strip()

    # If base is empty (first turn is a question), use question alone
    if not base.strip():
        return QuestionMetrics(
            turn_index=qctx.turn_index,
            role=qctx.role,
            question_text=qctx.question_text,
            verdict="SKIP (no prior context)",
        )

    # --- Distributions ---
    P_before, attn_before = get_distribution_semantic(
        model, tokenizer, base, return_attn=True
    )
    P_after, attn_after = get_distribution_semantic(
        model, tokenizer, full, return_attn=True
    )

    # --- Core metrics ---
    kl_raw = kl_divergence(P_before, P_after)
    jsd_val = jsd(P_before, P_after)
    rho = spearman_rho(P_before, P_after)
    jaccard = topk_jaccard(P_before, P_after, k=topk)

    # --- Entropy shift ---
    h_before = shannon_entropy(P_before)
    h_after = shannon_entropy(P_after)
    delta_h = h_after - h_before

    # --- Attention pattern shift ---
    attn_cmp = attention_pattern_comparison(attn_before, attn_after)
    delta_attn_h = attn_cmp['delta_entropy']

    # --- Rarity control ---
    # Floor at 0.05 so novel questions in small corpora aren't zeroed out.
    # In small conversations most n-grams are "new", which would zero KL_norm.
    familiarity = corpus_ngram_familiarity_semantic(
        qctx.question_text, corpus_docs
    )
    familiarity = max(familiarity, 0.05)
    kl_norm = normalized_kl(kl_raw, familiarity)

    # --- Generation diversity ---
    cont_before = generate_continuations_semantic(
        model, tokenizer, base, n_continuations
    )
    cont_after = generate_continuations_semantic(
        model, tokenizer, full, n_continuations
    )
    div_before = generation_diversity(cont_before)
    div_after = generation_diversity(cont_after)
    delta_div = (div_after['mean_pairwise_edit_dist']
                 - div_before['mean_pairwise_edit_dist'])

    # --- Composite score ---
    rho_factor = 1.0 - rho
    jaccard_factor = 1.0 - jaccard
    div_factor = max(0.01, 1.0 + delta_div)
    composite = rho_factor * jaccard_factor * kl_norm * div_factor

    # --- Verdict ---
    verdict = classify_verdict(composite)

    return QuestionMetrics(
        turn_index=qctx.turn_index,
        role=qctx.role,
        question_text=qctx.question_text,
        KL_raw=kl_raw,
        JSD=jsd_val,
        Spearman_rho=rho,
        TopK_Jaccard=jaccard,
        Delta_H_output=delta_h,
        Delta_Attn_H=delta_attn_h,
        Corpus_Familiarity=familiarity,
        KL_normalized=kl_norm,
        Delta_Diversity=delta_div,
        composite_score=composite,
        verdict=verdict,
        continuations_before=cont_before[:3],
        continuations_after=cont_after[:3],
    )


# =============================================================================
# VERDICT CLASSIFICATION
# =============================================================================

# Thresholds derived from cgi_semantic_results.json:
#   WhatIf composite: 0.00259 -> TRANSFORMATIVE
#   Why composite:    0.00030 -> UNCERTAIN
#   When composite:   0.00000 -> MECHANICAL

THRESHOLD_TRANSFORMATIVE = 0.001
THRESHOLD_UNCERTAIN = 0.0001


def classify_verdict(composite_score: float) -> str:
    """Classify question by composite reorganization score."""
    if composite_score >= THRESHOLD_TRANSFORMATIVE:
        return "TRANSFORMATIVE"
    elif composite_score >= THRESHOLD_UNCERTAIN:
        return "UNCERTAIN"
    else:
        return "MECHANICAL"


# =============================================================================
# ANALYSIS ORCHESTRATION
# =============================================================================

def run_analysis(conversation: dict,
                 corpus_docs: list[str] | None = None,
                 n_continuations: int = 10,
                 model=None, tokenizer=None) -> list[QuestionMetrics]:
    """
    Full pipeline: parse conversation -> analyze each question -> rank by score.
    Pass model/tokenizer to avoid reloading across conversations.
    """
    # Default corpus: extract ALL turns as reference text
    if corpus_docs is None:
        corpus_docs = [
            t["content"] for t in conversation.get("turns", [])
        ]

    # Parse
    questions = parse_conversation(conversation)
    if not questions:
        print("  No questions found in conversation.")
        return []

    print(f"  Found {len(questions)} question(s). Analyzing...\n")

    # Load model if not provided
    if model is None or tokenizer is None:
        print("  Loading distilgpt2...")
        tokenizer, model = load_model()
        print("  Model loaded.\n")

    # Analyze each question
    results = []
    for i, qctx in enumerate(questions):
        q_display = qctx.question_text[:60]
        if len(qctx.question_text) > 60:
            q_display += "..."
        print(f"  [{i+1}/{len(questions)}] Turn {qctx.turn_index}: \"{q_display}\"")

        metrics = analyze_question(
            model, tokenizer, qctx, corpus_docs,
            n_continuations=n_continuations
        )
        results.append(metrics)
        print(f"         -> composite={metrics.composite_score:.6f}  "
              f"verdict={metrics.verdict}")

    # Sort by composite score descending
    results.sort(key=lambda m: m.composite_score, reverse=True)
    return results


# =============================================================================
# REPORT
# =============================================================================

def print_report(results: list[QuestionMetrics], conversation_id: str = ""):
    """Print ranked analysis report."""
    n_questions = len(results)
    n_transform = sum(1 for r in results if r.verdict == "TRANSFORMATIVE")
    n_uncertain = sum(1 for r in results if r.verdict == "UNCERTAIN")
    n_mechanical = sum(1 for r in results if r.verdict == "MECHANICAL")
    n_skip = sum(1 for r in results if r.verdict.startswith("SKIP"))

    print("\n" + "=" * 90)
    print("  SOCRATIC ENGINE -- CGI Conversation Analysis")
    print("  Model: distilgpt2 (82M params)")
    if conversation_id:
        print(f"  Conversation: {conversation_id}")
    print(f"  Questions analyzed: {n_questions}")
    print(f"  Verdicts: {n_transform} transformative, "
          f"{n_uncertain} uncertain, {n_mechanical} mechanical"
          + (f", {n_skip} skipped" if n_skip else ""))
    print("=" * 90)

    # --- Ranking table ---
    print(f"\n  {'#':<4} {'Turn':<6} {'Verdict':<16} {'Composite':>10}  Question")
    print("  " + "-" * 84)

    for i, r in enumerate(results, 1):
        q_display = r.question_text[:50]
        if len(r.question_text) > 50:
            q_display += "..."
        safe_print(f"  {i:<4} {r.turn_index:<6} {r.verdict:<16} "
                   f"{r.composite_score:>10.6f}  \"{q_display}\"")

    # --- Details for non-mechanical questions ---
    interesting = [r for r in results
                   if r.verdict in ("TRANSFORMATIVE", "UNCERTAIN")]

    if interesting:
        print(f"\n{'-' * 90}")
        print("  DETAILED METRICS")
        print(f"{'-' * 90}")

    for r in interesting:
        safe_print(f"\n  Turn {r.turn_index} [{r.verdict}]: \"{r.question_text}\"")
        print(f"    KL_raw={r.KL_raw:.4f}  KL_norm={r.KL_normalized:.4f}  "
              f"JSD={r.JSD:.4f}")
        print(f"    Spearman={r.Spearman_rho:.4f}  Jaccard={r.TopK_Jaccard:.4f}  "
              f"DeltaDiv={r.Delta_Diversity:+.4f}")
        print(f"    Familiarity={r.Corpus_Familiarity:.4f}  "
              f"DeltaAttnH={r.Delta_Attn_H:+.4f}  "
              f"DeltaOutH={r.Delta_H_output:+.4f}")
        if r.continuations_before:
            print(f"    Continuations BEFORE:")
            for c in r.continuations_before:
                safe_print(f"      \"{c[:70]}\"")
        if r.continuations_after:
            print(f"    Continuations AFTER:")
            for c in r.continuations_after:
                safe_print(f"      \"{c[:70]}\"")

    # --- Composite formula reminder ---
    print(f"\n{'-' * 90}")
    print("  Score = (1-rho) * (1-Jaccard) * KL_normalized * max(0.01, 1+DeltaDiversity)")
    print(f"  Thresholds: TRANSFORMATIVE >= {THRESHOLD_TRANSFORMATIVE}, "
          f"UNCERTAIN >= {THRESHOLD_UNCERTAIN}, MECHANICAL < {THRESHOLD_UNCERTAIN}")
    print("=" * 90)


# =============================================================================
# BUILT-IN DEMO CONVERSATIONS
# =============================================================================

DEMO_CONVERSATIONS = [
    {
        "id": "therapy_stuck",
        "description": "Therapeutic conversation about feeling stuck (CGI reference case)",
        "turns": [
            {"role": "human", "content": "I feel stuck in my career."},
            {"role": "assistant", "content": "When did this start?"},
            {"role": "human", "content": "About two years ago when I got passed over for promotion."},
            {"role": "assistant", "content": "Why do you think you feel stuck?"},
            {"role": "human", "content": "I guess I keep doing the same things expecting different results."},
            {"role": "assistant", "content": "What if stuck is protecting you from something?"},
            {"role": "human", "content": "That... I never thought about it that way."},
        ]
    },
    {
        "id": "identity_crisis",
        "description": "Deeper therapeutic dialogue with identity questioning",
        "turns": [
            {"role": "human", "content": "I have always been the good student. Now I'm graduating and I don't know what to do."},
            {"role": "assistant", "content": "What kind of jobs are you considering?"},
            {"role": "human", "content": "Finance, consulting, maybe law school. Whatever looks impressive."},
            {"role": "assistant", "content": "If you strip away grades and achievements, who is the person left underneath?"},
            {"role": "human", "content": "I... I honestly don't know. That terrifies me."},
            {"role": "assistant", "content": "What would it mean to choose something that nobody would be impressed by?"},
            {"role": "human", "content": "I think I'd feel free. But also completely lost."},
        ]
    },
    {
        "id": "relationship_conflict",
        "description": "Relationship dynamics with hidden assumptions",
        "turns": [
            {"role": "human", "content": "My mother keeps telling me how to live my life and I feel guilty when I disagree."},
            {"role": "assistant", "content": "How often do you talk to her?"},
            {"role": "human", "content": "Every day. She calls and I always pick up."},
            {"role": "assistant", "content": "Why do you believe that loving someone means obeying them?"},
            {"role": "human", "content": "I... I guess I never questioned that."},
            {"role": "assistant", "content": "What would love look like without obedience?"},
            {"role": "human", "content": "Honest. Maybe harder but more real."},
        ]
    },
]


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 90)
    print("  SOCRATIC ENGINE")
    print("  Context Grammar Induction -- Conversation Analyzer")
    print("  Math measures WHAT transforms. The report shows WHY.")
    print("=" * 90)

    # Load conversation: from file or built-in demo
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        print(f"\n  Loading conversation from: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Support single conversation or list
        if isinstance(data, list):
            conversations = data
        elif "turns" in data:
            conversations = [data]
        else:
            conversations = data.get("conversations", [data])
    else:
        print("\n  No input file provided. Running built-in demo conversations.")
        print(f"  Usage: python socratic_engine.py <conversation.json>\n")
        conversations = DEMO_CONVERSATIONS

    # Load model once for all conversations
    print("\n  Loading distilgpt2...")
    tokenizer, model = load_model()
    print("  Model loaded.\n")

    # Run analysis on each conversation
    all_results = {}
    for conv in conversations:
        conv_id = conv.get("id", "unnamed")
        desc = conv.get("description", "")
        print(f"\n{'=' * 90}")
        print(f"  Analyzing: {conv_id}")
        if desc:
            print(f"  {desc}")
        print(f"{'=' * 90}")

        results = run_analysis(conv, model=model, tokenizer=tokenizer)
        if results:
            print_report(results, conversation_id=conv_id)
            all_results[conv_id] = [asdict(r) for r in results]

    # Save results
    output_path = "socratic_engine_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
