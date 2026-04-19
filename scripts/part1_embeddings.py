"""Part 1 — TF-IDF, PPMI, Skip-gram (SGNS), evaluation, 4-condition comparison.

All from scratch in numpy/PyTorch. No pretrained models.
Artifacts saved under embeddings/ and figures/.
"""
from __future__ import annotations

import json, math, os, time, random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

SUB = Path(__file__).resolve().parents[1]
DATA = SUB / "data"
EMB = SUB / "embeddings"
FIG = SUB / "figures"
EMB.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

VOCAB_CAP = 10000
WIN = 5           # k = 5
EMB_DIM = 100
NEG_K = 10
EPOCHS = 5
BATCH = 1024
LR = 1e-3
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", DEVICE)

# ----------------------------------------------------------------------
# Load corpus & vocabulary
# ----------------------------------------------------------------------
articles = json.loads((DATA / "articles_cleaned.json").read_text())
raw_articles = json.loads((DATA / "articles_raw.json").read_text())
sents_per_article = json.loads((DATA / "sents_cleaned.json").read_text())
flat_tokens = [tok for art in articles for tok in art]
print(f"total tokens: {len(flat_tokens)}  unique: {len(set(flat_tokens))}")

# Vocabulary: top 10K by frequency, rest -> <UNK>
UNK = "<UNK>"
freq = Counter(flat_tokens)
vocab = [UNK] + [w for w, _ in freq.most_common(VOCAB_CAP - 1)]
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

def to_ids(tokens: list[str]) -> list[int]:
    return [word2idx.get(t, word2idx[UNK]) for t in tokens]

art_ids = [to_ids(a) for a in articles]
V = len(vocab)
print(f"|V| capped = {V}")

# Topic categories (based on keyword frequencies, used for colouring & TF-IDF top-words)
CATEGORIES = {
    "politics": ["حکومت", "وزیراعظم", "صدر", "پارلیمان", "انتخابات", "سیاسی", "اپوزیشن", "وزیر"],
    "sports": ["کرکٹ", "ٹیم", "میچ", "کھلاڑی", "اسکور", "ورلڈ", "کپ", "پاکستانی"],
    "economy": ["معیشت", "بینک", "تجارت", "قیمت", "ڈالر", "برآمدات", "سرمایہ", "بجٹ"],
    "international": ["اقوام", "متحدہ", "سفیر", "معاہدہ", "بین", "غیر", "ملکی", "بیرون"],
    "health_society": ["ہسپتال", "بیماری", "ویکسین", "تعلیم", "سیلاب", "طب", "خواتین", "بچوں"],
}

def score_article(tokens: list[str]) -> str:
    counts = Counter(tokens)
    scores = {cat: sum(counts[k] for k in kws) for cat, kws in CATEGORIES.items()}
    return max(scores, key=scores.get)

article_cats = [score_article(a) for a in articles]
print("article category distribution:", Counter(article_cats))

# ----------------------------------------------------------------------
# 1.1 TF-IDF  (term-document matrix, N documents = 78)
# ----------------------------------------------------------------------
print("\n=== 1.1 TF-IDF ===")
N_DOC = len(art_ids)
tf = np.zeros((V, N_DOC), dtype=np.float32)
for d, tokens in enumerate(art_ids):
    for t in tokens:
        tf[t, d] += 1
df = (tf > 0).sum(axis=1).astype(np.float32)
idf = np.log(N_DOC / (1.0 + df))
tfidf = tf * idf[:, None]
np.save(EMB / "tfidf_matrix.npy", tfidf)
(EMB / "word2idx.json").write_text(json.dumps(word2idx, ensure_ascii=False))
print(f"tfidf matrix shape: {tfidf.shape}")

# top-10 discriminative words per category (mean tf-idf across articles in category)
cat2docs = defaultdict(list)
for d, c in enumerate(article_cats):
    cat2docs[c].append(d)

top_words_by_cat: dict[str, list[tuple[str, float]]] = {}
for cat, docs in cat2docs.items():
    if not docs:
        continue
    mean_scores = tfidf[:, docs].mean(axis=1)
    # skip <UNK>
    mean_scores[word2idx[UNK]] = -1
    order = np.argsort(-mean_scores)[:10]
    top_words_by_cat[cat] = [(idx2word[i], float(mean_scores[i])) for i in order]
    print(f"\n[TF-IDF top-10] {cat}: {[w for w,_ in top_words_by_cat[cat]]}")
(EMB / "tfidf_top_words.json").write_text(json.dumps(top_words_by_cat, ensure_ascii=False, indent=2))

# ----------------------------------------------------------------------
# 1.2 PPMI  (word-word co-occurrence, symmetric window k=5)
# ----------------------------------------------------------------------
print("\n=== 1.2 PPMI ===")
cooc = np.zeros((V, V), dtype=np.float32)
for tokens in art_ids:
    L = len(tokens)
    for i, wi in enumerate(tokens):
        lo = max(0, i - WIN); hi = min(L, i + WIN + 1)
        for j in range(lo, hi):
            if j == i: continue
            cooc[wi, tokens[j]] += 1.0
total = cooc.sum()
row_sum = cooc.sum(axis=1, keepdims=True)
col_sum = cooc.sum(axis=0, keepdims=True)
with np.errstate(divide="ignore", invalid="ignore"):
    pmi = np.log2((cooc * total) / (row_sum * col_sum + 1e-12) + 1e-12)
    pmi[cooc == 0] = 0
ppmi = np.maximum(pmi, 0).astype(np.float32)
np.save(EMB / "ppmi_matrix.npy", ppmi)
print(f"ppmi matrix shape: {ppmi.shape}")

# t-SNE for 200 most frequent tokens
print("running t-SNE on PPMI rows...")
freq_order = [word2idx[w] for w, _ in freq.most_common(VOCAB_CAP - 1) if w in word2idx][:200]
sub = ppmi[freq_order]
tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, init="pca")
emb2d = tsne.fit_transform(sub)

def assign_cat(word: str) -> str:
    for cat, kws in CATEGORIES.items():
        if word in kws:
            return cat
    return "other"

cats_for_pts = [assign_cat(idx2word[i]) for i in freq_order]
palette = {"politics": "tab:red", "sports": "tab:green", "economy": "tab:blue",
           "international": "tab:purple", "health_society": "tab:orange", "other": "#bbbbbb"}
plt.figure(figsize=(10, 8))
for cat, colour in palette.items():
    pts = [(emb2d[i, 0], emb2d[i, 1]) for i, c in enumerate(cats_for_pts) if c == cat]
    if not pts: continue
    xs, ys = zip(*pts)
    plt.scatter(xs, ys, s=18, c=colour, label=cat, alpha=0.8 if cat != "other" else 0.3)
plt.title("t-SNE of PPMI vectors — top 200 tokens by frequency")
plt.xlabel("tsne-1"); plt.ylabel("tsne-2"); plt.legend()
plt.tight_layout(); plt.savefig(FIG / "tsne_ppmi.png", dpi=140); plt.close()
print("saved figures/tsne_ppmi.png")

# nearest neighbours by cosine on PPMI rows
def cosine_neighbors(M: np.ndarray, idx: int, topk: int = 5) -> list[tuple[str, float]]:
    v = M[idx]
    n = np.linalg.norm(M, axis=1)
    sims = M @ v / (n * np.linalg.norm(v) + 1e-12)
    sims[idx] = -1
    order = np.argsort(-sims)[:topk]
    return [(idx2word[i], float(sims[i])) for i in order]

QUERIES = ["پاکستان", "حکومت", "عدالت", "معیشت", "فوج", "صحت", "تعلیم", "آبادی", "کرکٹ", "ٹیم"]
QUERY_TRANSLIT = {
    "پاکستان": "Pakistan", "حکومت": "Hukumat", "عدالت": "Adalat",
    "معیشت": "Maeeshat", "فوج": "Fauj", "صحت": "Sehat",
    "تعلیم": "Taleem", "آبادی": "Aabadi", "کرکٹ": "Cricket", "ٹیم": "Team",
}

ppmi_nn = {}
for q in QUERIES:
    if q in word2idx:
        ppmi_nn[q] = cosine_neighbors(ppmi, word2idx[q], 5)
        print(f"PPMI-NN {q} ({QUERY_TRANSLIT.get(q,'')}):", [w for w,_ in ppmi_nn[q]])
    else:
        print(f"Query not in vocab: {q}")
(EMB / "ppmi_nearest.json").write_text(json.dumps(ppmi_nn, ensure_ascii=False, indent=2))

# ----------------------------------------------------------------------
# 2. Skip-gram Word2Vec with negative sampling
# ----------------------------------------------------------------------
print("\n=== 2. Skip-gram Word2Vec (SGNS) ===")

def build_training_pairs(articles_ids: list[list[int]], window: int):
    pairs: list[tuple[int, int]] = []
    for tokens in articles_ids:
        L = len(tokens)
        for i, c in enumerate(tokens):
            lo = max(0, i - window); hi = min(L, i + window + 1)
            for j in range(lo, hi):
                if j == i: continue
                pairs.append((c, tokens[j]))
    return np.array(pairs, dtype=np.int64)


def build_noise_distribution(counts: np.ndarray, power: float = 0.75) -> np.ndarray:
    f = counts.astype(np.float64) ** power
    return f / f.sum()


def tok_counts(ids: list[list[int]], V: int) -> np.ndarray:
    c = np.zeros(V, dtype=np.int64)
    for s in ids:
        for t in s:
            c[t] += 1
    return c


class SGNSDataset(Dataset):
    def __init__(self, pairs: np.ndarray, noise_probs: np.ndarray, K: int = 10):
        self.pairs = pairs
        self.noise = noise_probs
        self.K = K
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, i):
        c, o = self.pairs[i]
        neg = np.random.choice(len(self.noise), self.K, p=self.noise)
        return int(c), int(o), neg.astype(np.int64)


class SGNS(nn.Module):
    def __init__(self, V: int, d: int):
        super().__init__()
        self.V_emb = nn.Embedding(V, d)
        self.U_emb = nn.Embedding(V, d)
        nn.init.uniform_(self.V_emb.weight, -0.5 / d, 0.5 / d)
        nn.init.zeros_(self.U_emb.weight)

    def forward(self, c, o, neg):
        vc = self.V_emb(c)              # (B,d)
        uo = self.U_emb(o)              # (B,d)
        un = self.U_emb(neg)            # (B,K,d)
        pos = torch.log(torch.sigmoid((uo * vc).sum(-1)) + 1e-9)
        negv = torch.log(torch.sigmoid(-(un * vc.unsqueeze(1)).sum(-1)) + 1e-9).sum(-1)
        return -(pos + negv).mean()


def train_sgns(tok_ids: list[list[int]], d: int, tag: str, epochs: int = EPOCHS):
    counts = tok_counts(tok_ids, V)
    noise = build_noise_distribution(counts)
    pairs = build_training_pairs(tok_ids, WIN)
    print(f"[{tag}] pairs: {len(pairs):,}")
    ds = SGNSDataset(pairs, noise, K=NEG_K)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)
    model = SGNS(V, d).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    losses = []
    for ep in range(epochs):
        t0 = time.time(); running = 0; steps = 0
        for c, o, neg in dl:
            c = c.to(DEVICE); o = o.to(DEVICE); neg = neg.to(DEVICE)
            loss = model(c, o, neg)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item(); steps += 1
        losses.append(running / steps)
        print(f"  [{tag}] epoch {ep+1}/{epochs}  loss={losses[-1]:.4f}  ({time.time()-t0:.1f}s)")
    V_w = model.V_emb.weight.detach().cpu().numpy()
    U_w = model.U_emb.weight.detach().cpu().numpy()
    emb = 0.5 * (V_w + U_w)
    return emb, losses, V_w, U_w

# -------- C3: skip-gram on cleaned.txt, d=100 (primary) --------
emb_c3, loss_c3, V_c3, U_c3 = train_sgns(art_ids, EMB_DIM, "C3-cleaned-d100")
np.save(EMB / "embeddings_w2v.npy", emb_c3)

plt.figure()
plt.plot(range(1, len(loss_c3) + 1), loss_c3, marker="o")
plt.title("SGNS training loss (C3: cleaned, d=100)")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.grid(True)
plt.tight_layout(); plt.savefig(FIG / "w2v_loss_c3.png", dpi=140); plt.close()

# -------- Nearest neighbours & analogies (Section 2.2) --------
def cosine_sims(emb: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    return emb / n

def top_neighbors(emb_norm: np.ndarray, idx: int, k: int = 10, ban=None):
    sims = emb_norm @ emb_norm[idx]
    sims[idx] = -1
    if ban:
        for b in ban: sims[b] = -1
    order = np.argsort(-sims)[:k]
    return [(idx2word[int(i)], float(sims[int(i)])) for i in order]

emb_n = cosine_sims(emb_c3)
w2v_nn: dict[str, list[tuple[str, float]]] = {}
for q in QUERIES:
    if q in word2idx:
        nn10 = top_neighbors(emb_n, word2idx[q], 10)
        w2v_nn[q] = nn10
        print(f"W2V-NN {q}:", [w for w,_ in nn10])
(EMB / "w2v_nearest.json").write_text(json.dumps(w2v_nn, ensure_ascii=False, indent=2))

# Analogies: a : b :: c : ?
ANALOGIES = [
    ("پاکستان", "اسلام", "انڈیا", "ہندو"),            # Pakistan:Islam :: India:Hindu
    ("پاکستان", "اسلام_آباد", "انڈیا", "دلی"),       # capitals
    ("کرکٹ", "کھلاڑی", "فلم", "اداکار"),
    ("حکومت", "وزیراعظم", "فوج", "جرنیل"),
    ("عدالت", "جج", "ہسپتال", "ڈاکٹر"),
    ("تعلیم", "اسکول", "صحت", "ہسپتال"),
    ("معیشت", "بینک", "کھیل", "اسٹیڈیم"),
    ("دن", "رات", "صبح", "شام"),
    ("مرد", "عورت", "لڑکا", "لڑکی"),
    ("پاکستان", "کراچی", "چین", "بیجنگ"),
]

def analogy(a, b, c, emb_norm, topk=3):
    for w in (a, b, c):
        if w not in word2idx:
            return None, None
    va = emb_norm[word2idx[a]]; vb = emb_norm[word2idx[b]]; vc = emb_norm[word2idx[c]]
    q = vb - va + vc
    q = q / (np.linalg.norm(q) + 1e-12)
    sims = emb_norm @ q
    for w in (a, b, c):
        sims[word2idx[w]] = -1
    order = np.argsort(-sims)[:topk]
    return [idx2word[int(i)] for i in order], [float(sims[int(i)]) for i in order]

analogy_results = []
correct = 0
for a, b, c, exp in ANALOGIES:
    got, _ = analogy(a, b, c, emb_n)
    ok = got is not None and exp in got
    correct += int(bool(ok))
    analogy_results.append({"a": a, "b": b, "c": c, "expected": exp, "top3": got, "correct": bool(ok)})
    print(f"  {a} : {b} :: {c} : ?  top3={got}  expected={exp}  {'OK' if ok else ''}")
print(f"analogies correct: {correct}/{len(ANALOGIES)}")
(EMB / "analogy_results.json").write_text(json.dumps({"results": analogy_results, "correct": correct, "total": len(ANALOGIES)}, ensure_ascii=False, indent=2))

# ----------------------------------------------------------------------
# Four-condition comparison
# ----------------------------------------------------------------------
print("\n=== 2.2 Four-condition comparison ===")
# C2 — skip-gram on raw.txt (build raw vocab aligned to cleaned vocab via word2idx; OOV -> <UNK>)
raw_token_ids = [to_ids(a) for a in raw_articles]
emb_c2, loss_c2, *_ = train_sgns(raw_token_ids, EMB_DIM, "C2-raw-d100", epochs=EPOCHS)
# C4 — cleaned, d=200
emb_c4, loss_c4, *_ = train_sgns(art_ids, 200, "C4-cleaned-d200", epochs=EPOCHS)

# Build PPMI as embedding via SVD-free choice: use raw PPMI rows normalised
# (C1 — PPMI baseline)
def eval_condition(emb: np.ndarray, pairs: list[tuple[str, str]], query_pool: list[str], qlist: list[str]):
    en = cosine_sims(emb)
    # MRR on manually labelled pairs (query, gold_neighbor)
    mrr = 0; valid = 0
    for q, gold in pairs:
        if q not in word2idx or gold not in word2idx: continue
        sims = en @ en[word2idx[q]]; sims[word2idx[q]] = -1
        order = np.argsort(-sims)
        rank_list = [int(i) for i in order[:200]]
        try:
            rank = rank_list.index(word2idx[gold]) + 1
            mrr += 1.0 / rank; valid += 1
        except ValueError:
            valid += 1  # gold not in top 200 → 0 contribution
    mrr = mrr / max(valid, 1)
    # top-5 neighbours for first 5 queries
    nn = {q: [w for w,_ in top_neighbors(en, word2idx[q], 5)] for q in qlist[:5] if q in word2idx}
    return {"mrr": mrr, "neighbors": nn}

GOLD_PAIRS = [
    ("پاکستان", "اسلام_آباد"), ("پاکستان", "حکومت"), ("پاکستان", "کراچی"),
    ("حکومت", "وزیراعظم"), ("حکومت", "وزیر"), ("حکومت", "پارلیمنٹ"),
    ("عدالت", "جج"), ("عدالت", "فیصلہ"), ("عدالت", "قانون"),
    ("معیشت", "بینک"), ("معیشت", "ڈالر"), ("معیشت", "قیمت"),
    ("فوج", "جرنیل"), ("فوج", "دفاع"), ("فوج", "پاکستان"),
    ("صحت", "ہسپتال"), ("صحت", "ڈاکٹر"),
    ("تعلیم", "اسکول"), ("تعلیم", "طلبہ"),
    ("آبادی", "ملک"),
]

# C1 requires a matrix aligned to vocabulary; use PPMI rows
c1 = eval_condition(ppmi, GOLD_PAIRS, QUERIES, QUERIES)
c2 = eval_condition(emb_c2, GOLD_PAIRS, QUERIES, QUERIES)
c3 = eval_condition(emb_c3, GOLD_PAIRS, QUERIES, QUERIES)
c4 = eval_condition(emb_c4, GOLD_PAIRS, QUERIES, QUERIES)
comparison = {"C1_PPMI": c1, "C2_SGNS_raw_d100": c2, "C3_SGNS_cleaned_d100": c3, "C4_SGNS_cleaned_d200": c4}
(EMB / "four_condition_comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2))
print("\nMRR summary:")
for k, v in comparison.items():
    print(f"  {k}: MRR = {v['mrr']:.4f}")

# Also stash a copy of PPMI-as-embedding for later comparison.
print("\nPart 1 complete.")
