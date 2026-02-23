"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.

@karpathy
"""

import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle
random.seed(42) # Let there be order among chaos

# Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [
    "I feel stuck in my career.",
    "I feel stuck in my job.",
    "I feel stuck and frustrated.",
    "I feel stuck because nothing changes.",
    "I feel stuck and want something different.",
    "Maybe stuck is protecting you.",
    "Sometimes feeling stuck is safety.",
    "Stuck can mean waiting.",
    "Stuck may be fear of change."
]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for a special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")

# Let there be Autograd to recursively apply the chain rule through a computation graph
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads') # Python optimization for memory usage

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # scalar value of this node calculated during forward pass
        self.grad = 0                   # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children       # children of this node in the computation graph
        self._local_grads = local_grads # local derivative of this node w.r.t. its children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# Initialize the parameters, to store the knowledge of the model
n_layer = 1     # depth of the transformer neural network (number of layers)
n_embd = 16     # width of the network (embedding dimension)
block_size = 16 # maximum context length of the attention window (note: the longest name is 15 characters)
n_head = 4      # number of attention heads
head_dim = n_embd // n_head # derived dimension of each head
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row] # flatten params into a single list[Value]
print(f"num params: {len(params)}")

# Define the model architecture: a function mapping tokens and parameters to logits over what comes next
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values, return_attn=False):
    tok_emb = state_dict['wte'][token_id] # token embedding
    pos_emb = state_dict['wpe'][pos_id] # position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding
    x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection

    all_attn_weights = []

    for li in range(n_layer):
        # 1) Multi-head Attention block
        layer_attn = []
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            if return_attn:
                layer_attn.append([w.data for w in attn_weights])
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        if return_attn:
            all_attn_weights.append(layer_attn)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    if return_attn:
        return logits, all_attn_weights
    return logits

# Let there be Adam, the blessed optimizer and its buffers
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params) # first moment buffer
v = [0.0] * len(params) # second moment buffer

# Repeat in sequence
num_steps = 500 # number of training steps
for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # Forward the token sequence through the model, building up the computation graph all the way to the loss
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses) # final average loss over the document sequence. May yours be low.

    # Backward the loss, calculating the gradients with respect to all model parameters
    loss.backward()

    # Adam optimizer update: update the model parameters based on the corresponding gradients
    lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

# Inference: may the model babble back to us
temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")

    
def get_distribution(context_string, return_details=False):
    tokens = [BOS] + [uchars.index(ch) for ch in context_string if ch in uchars]
    tokens = tokens[-block_size:]

    keys_kv, values_kv = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    all_step_attentions = []

    for pos_id in range(len(tokens) - 1):
        token_id = tokens[pos_id]
        if return_details:
            _, step_attn = gpt(token_id, pos_id, keys_kv, values_kv, return_attn=True)
            all_step_attentions.append(step_attn)
        else:
            _ = gpt(token_id, pos_id, keys_kv, values_kv)

    token_id = tokens[-1]
    if return_details:
        logits, final_attn = gpt(token_id, len(tokens) - 1, keys_kv, values_kv, return_attn=True)
        all_step_attentions.append(final_attn)
    else:
        logits = gpt(token_id, len(tokens) - 1, keys_kv, values_kv)

    probs = softmax(logits)
    prob_dist = [p.data for p in probs]

    if return_details:
        return prob_dist, all_step_attentions, tokens
    return prob_dist

def kl_divergence(p, q, eps=1e-10):
    kl = 0.0
    for pi, qi in zip(p, q):
        pi = max(pi, eps)
        qi = max(qi, eps)
        kl += pi * math.log(pi / qi)
    return kl

# ============================================================
# CGI REORGANIZATION METRICS
# ============================================================

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

def topk_jaccard(p, q, k=5):
    top_p = set(sorted(range(len(p)), key=lambda i: p[i], reverse=True)[:k])
    top_q = set(sorted(range(len(q)), key=lambda i: q[i], reverse=True)[:k])
    intersection = len(top_p & top_q)
    union = len(top_p | top_q)
    return intersection / union if union > 0 else 1.0

def shannon_entropy(dist, eps=1e-10):
    return -sum(max(p, eps) * math.log(max(p, eps)) for p in dist)

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

def generate_continuations(context_string, n_continuations=10, max_len=20, temperature=0.5):
    continuations = []
    for _ in range(n_continuations):
        tokens = [BOS] + [uchars.index(ch) for ch in context_string if ch in uchars]
        tokens = tokens[-block_size:]
        keys_gen = [[] for _ in range(n_layer)]
        values_gen = [[] for _ in range(n_layer)]
        for pos_id in range(len(tokens)):
            token_id = tokens[pos_id]
            logits = gpt(token_id, pos_id, keys_gen, values_gen)
        generated_chars = []
        for gen_step in range(max_len):
            probs = softmax([l / temperature for l in logits])
            probs_float = [p.data for p in probs]
            token_id = random.choices(range(vocab_size), weights=probs_float)[0]
            if token_id == BOS:
                break
            generated_chars.append(uchars[token_id])
            next_pos = min(len(tokens) + gen_step, block_size - 1)
            logits = gpt(token_id, next_pos, keys_gen, values_gen)
        continuations.append(''.join(generated_chars))
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

# ============================================================
# RARITY NORMALIZATION
# ============================================================

def corpus_ngram_familiarity(question_string, training_docs, n_range=(2, 3, 4)):
    corpus_text = ' '.join(training_docs).lower()
    question_lower = question_string.lower()
    total_ngrams = 0
    found_ngrams = 0
    for n in n_range:
        for i in range(len(question_lower) - n + 1):
            ngram = question_lower[i:i + n]
            total_ngrams += 1
            if ngram in corpus_text:
                found_ngrams += 1
    return found_ngrams / total_ngrams if total_ngrams > 0 else 0.0

def char_coverage(question_string, vocab_chars):
    if len(question_string) == 0:
        return 1.0
    vocab_set = set(vocab_chars)
    return sum(1 for ch in question_string if ch in vocab_set) / len(question_string)

def effective_context_window(context_string, vocab_chars, block_size_val):
    vocab_set = set(vocab_chars)
    filtered = [ch for ch in context_string if ch in vocab_set]
    filtered_string = ''.join(filtered)
    total_tokens = len(filtered) + 1
    window_tokens = min(total_tokens, block_size_val)
    window_chars = filtered[-(block_size_val - 1):] if total_tokens > block_size_val else filtered
    window_string = ''.join(window_chars)
    survival = window_tokens / total_tokens if total_tokens > 0 else 1.0
    return {
        'filtered_string': filtered_string,
        'token_count': total_tokens,
        'window_string': window_string,
        'context_survival_ratio': survival,
    }

def normalized_kl(kl_raw, familiarity_score, alpha=1.0):
    return kl_raw * (familiarity_score ** alpha)

# ============================================================
# UNIFIED CGI ANALYSIS PIPELINE
# ============================================================

def cgi_analysis(base_context, questions, training_docs, n_continuations=10,
                 gen_max_len=20, gen_temperature=0.5, topk=5):
    P_base, attn_base, tokens_base = get_distribution(base_context, return_details=True)
    base_window = effective_context_window(base_context, uchars, block_size)
    cont_base = generate_continuations(base_context, n_continuations, gen_max_len, gen_temperature)
    div_base = generation_diversity(cont_base)

    results = {}
    for label, question in questions.items():
        full_context = base_context + question
        P_after, attn_after, tokens_after = get_distribution(full_context, return_details=True)

        kl_raw = kl_divergence(P_base, P_after)
        jsd_val = jsd(P_base, P_after)
        rho = spearman_rho(P_base, P_after)
        jaccard = topk_jaccard(P_base, P_after, k=topk)

        h_output_before = shannon_entropy(P_base)
        h_output_after = shannon_entropy(P_after)

        attn_cmp = attention_pattern_comparison(attn_base[-1], attn_after[-1])

        familiarity = corpus_ngram_familiarity(question, training_docs)
        coverage = char_coverage(question, uchars)
        after_window = effective_context_window(full_context, uchars, block_size)
        kl_norm = normalized_kl(kl_raw, familiarity, alpha=1.0)

        cont_after = generate_continuations(full_context, n_continuations, gen_max_len, gen_temperature)
        div_after = generation_diversity(cont_after)

        results[label] = {
            'KL_raw': kl_raw, 'JSD': jsd_val, 'Spearman_rho': rho, 'TopK_Jaccard': jaccard,
            'H_output_before': h_output_before, 'H_output_after': h_output_after,
            'Delta_H_output': h_output_after - h_output_before,
            'Attn_H_before': attn_cmp['entropy_before'], 'Attn_H_after': attn_cmp['entropy_after'],
            'Delta_Attn_H': attn_cmp['delta_entropy'], 'Attn_PerHead_JSD': attn_cmp['per_head_jsd'],
            'Corpus_Familiarity': familiarity, 'Char_Coverage': coverage, 'KL_normalized': kl_norm,
            'Gen_Unique_Before': div_base['unique_ratio'], 'Gen_Unique_After': div_after['unique_ratio'],
            'Gen_EditDist_Before': div_base['mean_pairwise_edit_dist'],
            'Gen_EditDist_After': div_after['mean_pairwise_edit_dist'],
            'Delta_Diversity': div_after['mean_pairwise_edit_dist'] - div_base['mean_pairwise_edit_dist'],
            'Gen_FirstChar_H_Before': div_base['first_char_entropy'],
            'Gen_FirstChar_H_After': div_after['first_char_entropy'],
            'Window_Base': base_window['window_string'], 'Window_After': after_window['window_string'],
            'Tokens_Base': tokens_base, 'Tokens_After': tokens_after,
            'Context_Survival_Base': base_window['context_survival_ratio'],
            'Context_Survival_After': after_window['context_survival_ratio'],
            'Sample_Continuations_Base': cont_base[:3], 'Sample_Continuations_After': cont_after[:3],
        }
    return results

def print_cgi_report(results):
    labels = list(results.keys())

    print("\n" + "=" * 90)
    print("  CGI REORGANIZATION ANALYSIS REPORT")
    print("  Model: microgpt (1 layer, 16 embd, 4 heads, 4224 params, vocab=28 chars)")
    print("=" * 90)

    print("\n--- CONTEXT WINDOW DIAGNOSTICS ---")
    print(f"  Base window: '{results[labels[0]]['Window_Base']}'")
    for label in labels:
        r = results[label]
        print(f"  {label:>10} window: '{r['Window_After']}'  "
              f"(survival: {r['Context_Survival_After']:.1%}, "
              f"char coverage: {r['Char_Coverage']:.1%})")

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
        ('TopK_Jaccard',       'Top-5 Jaccard',            'lower = top predictions changed'),
        ('Delta_Attn_H',       'Delta Attn Entropy',       'nonzero = internal change'),
        ('Delta_H_output',     'Delta Output Entropy',     'positive = more uncertain after'),
        ('Corpus_Familiarity', 'Corpus Familiarity',       '1=known patterns, 0=novel'),
        ('Char_Coverage',      'Char Vocab Coverage',      '1=all chars in vocab'),
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
        print(f"  {label:>10}: Reorganization Score = {score:.4f}")
        print(f"             rho={r['Spearman_rho']:.3f}  "
              f"Jaccard={r['TopK_Jaccard']:.3f}  "
              f"KL_norm={r['KL_normalized']:.3f}  "
              f"DeltaDiv={r['Delta_Diversity']:+.3f}")
        if r['Corpus_Familiarity'] < 0.3:
            print(f"             ** LOW FAMILIARITY ({r['Corpus_Familiarity']:.2f}): "
                  f"high KL likely reflects rarity, not transformation")
        if r['Char_Coverage'] < 0.95:
            print(f"             ** CHAR GAPS ({r['Char_Coverage']:.2f}): "
                  f"some question characters stripped from input")
        if r['Context_Survival_After'] < 0.5:
            print(f"             ** TRUNCATION ({r['Context_Survival_After']:.1%} survival): "
                  f"base context absent from model window")

    print("\n--- SAMPLE CONTINUATIONS ---")
    print(f"  Base continuations:")
    for s in results[labels[0]]['Sample_Continuations_Base']:
        print(f"    '{s}'")
    for label in labels:
        print(f"  After '{label}':")
        for s in results[label]['Sample_Continuations_After']:
            print(f"    '{s}'")

    print("\n" + "=" * 90)
    return composite_scores

# ============================================================
# CGI EMPIRICAL TEST
# ============================================================

print("\n--- CGI REORGANIZATION ANALYSIS ---")

base = "I feel stuck in my career."
questions = {
    "When":   " When did this start?",
    "Why":    " Why do you think you feel stuck?",
    "WhatIf": " What if stuck is protecting you?",
}

results = cgi_analysis(base, questions, docs, n_continuations=10)
composite = print_cgi_report(results)

import json
serializable = {}
for label, metrics in results.items():
    serializable[label] = {}
    for k, v_val in metrics.items():
        if isinstance(v_val, (int, float, str, bool, type(None))):
            serializable[label][k] = v_val
        elif isinstance(v_val, list):
            serializable[label][k] = v_val

with open('cgi_char_results.json', 'w') as f:
    json.dump(serializable, f, indent=2)
print("\nResults saved to cgi_char_results.json")