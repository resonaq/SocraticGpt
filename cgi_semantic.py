"""
CGI Semantic-Level Reorganization Analysis
===========================================
Companion to microgpt.py char-level analysis.
Uses distilgpt2 to compute reorganization metrics at the word/subword level.

distilgpt2: 50257-token vocab, 1024-token context window, 82M params.
Unlike char-level model, the base context persists alongside the question.

Usage:
    python cgi_semantic.py
"""

import math
import json
import os
import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# ---- Configuration ----
MODEL_ID = "distilgpt2"
BASE_CONTEXT = "I feel stuck in my career."
QUESTIONS = {
    "When":   " When did this start?",
    "Why":    " Why do you think you feel stuck?",
    "WhatIf": " What if stuck is protecting you?",
}
TRAINING_DOCS = [
    "I feel stuck in my career.",
    "I feel stuck in my job.",
    "I feel stuck and frustrated.",
    "I feel stuck because nothing changes.",
    "I feel stuck and want something different.",
    "Maybe stuck is protecting you.",
    "Sometimes feeling stuck is safety.",
    "Stuck can mean waiting.",
    "Stuck may be fear of change.",
]

# ---- Model Loading ----

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, attn_implementation="eager")
    model.eval()
    return tokenizer, model

# ---- Distribution Extraction ----

def get_distribution_semantic(model, tokenizer, context_string, return_attn=False):
    inputs = tokenizer(context_string, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=return_attn)
    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    prob_list = probs.tolist()

    if return_attn:
        attn_data = []
        for layer_attn_tensor in outputs.attentions:
            if layer_attn_tensor is None:
                continue
            layer_data = []
            a = layer_attn_tensor[0]  # [n_heads, seq_len, seq_len]
            for h in range(a.shape[0]):
                layer_data.append(a[h, -1, :].tolist())
            attn_data.append(layer_data)
        return prob_list, attn_data
    return prob_list

# ---- Pure-Python Metrics (mirrored from microgpt.py) ----

def _assign_ranks(values):
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[indexed[j]] == values[indexed[j + 1]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks

def spearman_rho(p, q):
    n = len(p)
    if n < 2:
        return 1.0
    ranks_p = _assign_ranks(p)
    ranks_q = _assign_ranks(q)
    mean_rp = sum(ranks_p) / n
    mean_rq = sum(ranks_q) / n
    covariance = sum((rp - mean_rp) * (rq - mean_rq) for rp, rq in zip(ranks_p, ranks_q))
    std_p = sum((rp - mean_rp) ** 2 for rp in ranks_p) ** 0.5
    std_q = sum((rq - mean_rq) ** 2 for rq in ranks_q) ** 0.5
    if std_p < 1e-12 or std_q < 1e-12:
        return 1.0
    return covariance / (std_p * std_q)

def topk_jaccard(p, q, k=10):
    top_p = set(sorted(range(len(p)), key=lambda i: p[i], reverse=True)[:k])
    top_q = set(sorted(range(len(q)), key=lambda i: q[i], reverse=True)[:k])
    intersection = len(top_p & top_q)
    union = len(top_p | top_q)
    return intersection / union if union > 0 else 1.0

def shannon_entropy(dist, eps=1e-10):
    return -sum(max(p, eps) * math.log(max(p, eps)) for p in dist)

def kl_divergence(p, q, eps=1e-10):
    kl = 0.0
    for pi, qi in zip(p, q):
        pi = max(pi, eps)
        qi = max(qi, eps)
        kl += pi * math.log(pi / qi)
    return kl

def jsd(p, q, eps=1e-10):
    n = len(p)
    m_dist = [(p[i] + q[i]) / 2.0 for i in range(n)]
    kl_pm = sum(max(p[i], eps) * math.log(max(p[i], eps) / max(m_dist[i], eps)) for i in range(n))
    kl_qm = sum(max(q[i], eps) * math.log(max(q[i], eps) / max(m_dist[i], eps)) for i in range(n))
    return 0.5 * kl_pm + 0.5 * kl_qm

def attention_entropy_at_step(step_attn):
    entropies = []
    for layer_attn in step_attn:
        for head_weights in layer_attn:
            entropies.append(shannon_entropy(head_weights))
    return sum(entropies) / len(entropies) if entropies else 0.0

def attention_pattern_comparison(attn_before, attn_after):
    h_before = attention_entropy_at_step(attn_before)
    h_after = attention_entropy_at_step(attn_after)
    per_head_jsd = []
    for li in range(len(attn_before)):
        for hi in range(len(attn_before[li])):
            w_before = attn_before[li][hi]
            w_after = attn_after[li][hi]
            if len(w_before) == len(w_after):
                per_head_jsd.append(jsd(w_before, w_after))
            else:
                per_head_jsd.append(None)
    return {
        'entropy_before': h_before,
        'entropy_after': h_after,
        'delta_entropy': h_after - h_before,
        'per_head_jsd': per_head_jsd,
    }

def normalized_kl(kl_raw, familiarity_score, alpha=1.0):
    return kl_raw * (familiarity_score ** alpha)

# ---- Semantic-Level Rarity ----

def corpus_ngram_familiarity_semantic(question_string, training_docs, n_range=(1, 2, 3)):
    corpus_words = ' '.join(training_docs).lower().split()
    question_words = question_string.strip().lower().split()
    total = 0
    found = 0
    for n in n_range:
        corpus_ngrams = set()
        for i in range(len(corpus_words) - n + 1):
            corpus_ngrams.add(tuple(corpus_words[i:i + n]))
        for i in range(len(question_words) - n + 1):
            total += 1
            if tuple(question_words[i:i + n]) in corpus_ngrams:
                found += 1
    return found / total if total > 0 else 0.0

# ---- Generation Diversity ----

def generate_continuations_semantic(model, tokenizer, context_string,
                                     n_continuations=10, max_new_tokens=30,
                                     temperature=0.7, top_p=0.9):
    inputs = tokenizer(context_string, return_tensors="pt")
    continuations = []
    for i in range(n_continuations):
        set_seed(42 + i)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
        new_tokens = output[0][inputs['input_ids'].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        continuations.append(text)
    return continuations

def _edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]

def generation_diversity(continuations):
    n = len(continuations)
    if n == 0:
        return {'unique_ratio': 0.0, 'mean_pairwise_edit_dist': 0.0, 'first_char_entropy': 0.0}
    unique_ratio = len(set(continuations)) / n
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            max_len_pair = max(len(continuations[i]), len(continuations[j]), 1)
            distances.append(_edit_distance(continuations[i], continuations[j]) / max_len_pair)
    mean_dist = sum(distances) / len(distances) if distances else 0.0
    first_chars = [c[0] if c else '' for c in continuations]
    char_counts = {}
    for ch in first_chars:
        char_counts[ch] = char_counts.get(ch, 0) + 1
    first_char_dist = [count / n for count in char_counts.values()]
    fc_entropy = shannon_entropy(first_char_dist)
    return {
        'unique_ratio': unique_ratio,
        'mean_pairwise_edit_dist': mean_dist,
        'first_char_entropy': fc_entropy,
    }

# ---- Semantic Analysis Pipeline ----

def cgi_analysis_semantic(model, tokenizer, base_context, questions, training_docs,
                           n_continuations=10, topk=10):
    P_base, attn_base = get_distribution_semantic(model, tokenizer, base_context, return_attn=True)
    cont_base = generate_continuations_semantic(model, tokenizer, base_context, n_continuations)
    div_base = generation_diversity(cont_base)

    results = {}
    for label, question in questions.items():
        full_context = base_context + question
        P_after, attn_after = get_distribution_semantic(model, tokenizer, full_context, return_attn=True)

        kl_raw = kl_divergence(P_base, P_after)
        jsd_val = jsd(P_base, P_after)
        rho = spearman_rho(P_base, P_after)
        jaccard = topk_jaccard(P_base, P_after, k=topk)

        h_before = shannon_entropy(P_base)
        h_after = shannon_entropy(P_after)

        attn_cmp = attention_pattern_comparison(attn_base, attn_after)

        familiarity = corpus_ngram_familiarity_semantic(question, training_docs)
        kl_norm = normalized_kl(kl_raw, familiarity)

        cont_after = generate_continuations_semantic(model, tokenizer, full_context, n_continuations)
        div_after = generation_diversity(cont_after)

        results[label] = {
            'KL_raw': kl_raw, 'JSD': jsd_val, 'Spearman_rho': rho, 'TopK_Jaccard': jaccard,
            'H_output_before': h_before, 'H_output_after': h_after,
            'Delta_H_output': h_after - h_before,
            'Attn_H_before': attn_cmp['entropy_before'], 'Attn_H_after': attn_cmp['entropy_after'],
            'Delta_Attn_H': attn_cmp['delta_entropy'],
            'Corpus_Familiarity': familiarity, 'KL_normalized': kl_norm,
            'Gen_Unique_Before': div_base['unique_ratio'], 'Gen_Unique_After': div_after['unique_ratio'],
            'Gen_EditDist_Before': div_base['mean_pairwise_edit_dist'],
            'Gen_EditDist_After': div_after['mean_pairwise_edit_dist'],
            'Delta_Diversity': div_after['mean_pairwise_edit_dist'] - div_base['mean_pairwise_edit_dist'],
            'Sample_Continuations_Base': cont_base[:3],
            'Sample_Continuations_After': cont_after[:3],
        }
    return results

# ---- Reports ----

def print_semantic_report(results):
    labels = list(results.keys())

    print("\n" + "=" * 90)
    print("  CGI SEMANTIC-LEVEL ANALYSIS REPORT")
    print(f"  Model: {MODEL_ID} (50257 vocab, 1024 context, 82M params)")
    print("=" * 90)

    print("\n--- METRIC COMPARISON ---")
    col_width = 16
    header = f"  {'Metric':<32}"
    for label in labels:
        header += f" {label:>{col_width}}"
    header += f"  {'Interpretation'}"
    print(header)
    print("  " + "-" * (32 + (col_width + 1) * len(labels) + 30))

    metrics_table = [
        ('KL_raw',             'KL Divergence (raw)',      'higher = more different output'),
        ('KL_normalized',      'KL Normalized',            'high + high familiarity = reorg'),
        ('JSD',                'Jensen-Shannon Div',       'bounded [0, 0.693], symmetric'),
        ('Spearman_rho',       'Spearman Rank rho',        'lower = more rank reorganization'),
        ('TopK_Jaccard',       'Top-10 Jaccard',           'lower = top predictions changed'),
        ('Delta_Attn_H',       'Delta Attn Entropy',       'nonzero = internal change'),
        ('Delta_H_output',     'Delta Output Entropy',     'positive = more uncertain after'),
        ('Corpus_Familiarity', 'Word Familiarity',         '1=known patterns, 0=novel'),
        ('Delta_Diversity',    'Delta Gen Diversity',       'positive = more paths opened'),
    ]

    for key, display_name, interpretation in metrics_table:
        row = f"  {display_name:<32}"
        for label in labels:
            val = results[label].get(key, 0.0)
            if val is None:
                row += f" {'N/A':>{col_width}}"
            else:
                row += f" {val:>{col_width}.4f}"
        row += f"  {interpretation}"
        print(row)

    print("\n--- CGI COMPOSITE SCORES ---")
    print("  Reorganization = (1 - rho) * (1 - Jaccard) * KL_norm * max(0.01, 1 + DeltaDiv)")
    print()

    composite_scores = {}
    for label in labels:
        r = results[label]
        rho_factor = 1.0 - r['Spearman_rho']
        jaccard_factor = 1.0 - r['TopK_Jaccard']
        kl_n = r['KL_normalized']
        div_factor = max(0.01, 1.0 + r['Delta_Diversity'])
        composite = rho_factor * jaccard_factor * kl_n * div_factor
        composite_scores[label] = composite

    for label in sorted(composite_scores, key=composite_scores.get, reverse=True):
        r = results[label]
        score = composite_scores[label]
        print(f"  {label:>10}: Reorganization Score = {score:.6f}")
        print(f"             rho={r['Spearman_rho']:.3f}  "
              f"Jaccard={r['TopK_Jaccard']:.3f}  "
              f"KL_norm={r['KL_normalized']:.3f}  "
              f"DeltaDiv={r['Delta_Diversity']:+.3f}")

    print("\n--- SAMPLE CONTINUATIONS ---")
    print(f"  Base continuations:")
    for s in results[labels[0]]['Sample_Continuations_Base']:
        print(f"    '{s[:80]}'")
    for label in labels:
        print(f"  After '{label}':")
        for s in results[label]['Sample_Continuations_After']:
            print(f"    '{s[:80]}'")

    print("\n" + "=" * 90)
    return composite_scores


def print_cross_level_report(char_results, semantic_results):
    labels = list(semantic_results.keys())

    print("\n" + "=" * 100)
    print("  CROSS-LEVEL CGI COMPARISON")
    print("  Char-level: microgpt (28-char vocab, 16-token window, 4224 params)")
    print(f"  Semantic:   {MODEL_ID} (50257-token vocab, 1024-token window, 82M params)")
    print("=" * 100)

    compare_metrics = [
        ('KL_raw',           'KL Divergence (raw)'),
        ('KL_normalized',    'KL Normalized'),
        ('JSD',              'Jensen-Shannon Div'),
        ('Spearman_rho',     'Spearman Rank rho'),
        ('TopK_Jaccard',     'Top-K Jaccard'),
        ('Delta_Attn_H',     'Delta Attn Entropy'),
        ('Corpus_Familiarity','Corpus Familiarity'),
        ('Delta_Diversity',  'Delta Gen Diversity'),
    ]

    for label in labels:
        print(f"\n--- Question: {label} ---")
        print(f"  {'Metric':<28} {'Char-Level':>14} {'Semantic':>14} {'Direction':>12}")
        print("  " + "-" * 70)

        cr = char_results.get(label, {})
        sr = semantic_results[label]

        for key, display in compare_metrics:
            c_val = cr.get(key, None)
            s_val = sr.get(key, None)
            c_str = f"{c_val:.4f}" if c_val is not None else "N/A"
            s_str = f"{s_val:.4f}" if s_val is not None else "N/A"
            direction = ""
            if c_val is not None and s_val is not None:
                if abs(c_val) > 0.001 and abs(s_val) > 0.001:
                    direction = "agree" if (c_val > 0) == (s_val > 0) else "DISAGREE"
            print(f"  {display:<28} {c_str:>14} {s_str:>14} {direction:>12}")

    print("\n--- CGI HYPOTHESIS TEST ---")
    print("  If char-level KL ordering (Why > When > WhatIf) is driven by rarity,")
    print("  then semantic-level KL (where 'Why' is familiar) should show")
    print("  a DIFFERENT ordering, potentially WhatIf > Why.")
    print()

    for level_name, level_results in [("Char-level", char_results), ("Semantic", semantic_results)]:
        if not level_results:
            continue
        ranked = sorted(labels, key=lambda l: level_results.get(l, {}).get('KL_normalized', 0), reverse=True)
        print(f"  {level_name} KL_normalized ranking: {' > '.join(ranked)}")

    ranked_rho_char = sorted(labels, key=lambda l: char_results.get(l, {}).get('Spearman_rho', 1.0))
    ranked_rho_sem = sorted(labels, key=lambda l: semantic_results.get(l, {}).get('Spearman_rho', 1.0))
    print(f"  Char-level reorganization (lowest rho first): {' > '.join(ranked_rho_char)}")
    print(f"  Semantic reorganization (lowest rho first):   {' > '.join(ranked_rho_sem)}")

    print("\n" + "=" * 100)

# ---- Main ----

def main():
    print("Loading model...")
    tokenizer, model = load_model()
    print(f"Model: {MODEL_ID}, vocab_size: {tokenizer.vocab_size}")

    print("\nRunning semantic-level CGI analysis...")
    semantic_results = cgi_analysis_semantic(
        model, tokenizer, BASE_CONTEXT, QUESTIONS, TRAINING_DOCS,
        n_continuations=10, topk=10
    )

    composite = print_semantic_report(semantic_results)

    # Load char-level results if available
    char_results = {}
    char_results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cgi_char_results.json')
    if os.path.exists(char_results_path):
        with open(char_results_path) as f:
            char_results = json.load(f)
        print("\nLoaded char-level results from cgi_char_results.json")
        print_cross_level_report(char_results, semantic_results)
    else:
        print("\nNo cgi_char_results.json found. Run microgpt.py first for cross-level comparison.")

    # Save semantic results
    serializable = {}
    for label, metrics in semantic_results.items():
        serializable[label] = {k: v for k, v in metrics.items()
                               if isinstance(v, (int, float, str, bool, type(None), list))}
    with open('cgi_semantic_results.json', 'w') as f:
        json.dump(serializable, f, indent=2)
    print("\nSemantic results saved to cgi_semantic_results.json")


if __name__ == "__main__":
    main()
