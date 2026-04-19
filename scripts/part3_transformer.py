"""Part 3 — Transformer encoder for 5-class topic classification (from scratch)."""
from __future__ import annotations

import json, math, time, copy
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

SUB = Path(__file__).resolve().parents[1]
DATA = SUB / "data"
EMB = SUB / "embeddings"
MOD = SUB / "models"; MOD.mkdir(exist_ok=True)
FIG = SUB / "figures"; FIG.mkdir(exist_ok=True)

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# -------- Load token sequences per article & labels --------
articles = json.loads((DATA / "articles_cleaned.json").read_text())
word2idx = json.loads((EMB / "word2idx.json").read_text())
UNK = word2idx["<UNK>"]

# Keyword-based labelling per assignment spec
CATEGORIES = [
    ("Politics",        ["حکومت","وزیراعظم","صدر","پارلیمان","انتخابات","وزیر","سیاسی"]),
    ("Sports",          ["کرکٹ","ٹیم","میچ","کھلاڑی","اسکور","کپ","ورلڈ"]),
    ("Economy",         ["معیشت","بینک","تجارت","قیمت","ڈالر","برآمدات","بجٹ"]),
    ("International",   ["اقوام","متحدہ","سفیر","معاہدہ","بیرون","غیر","بین"]),
    ("Health_Society",  ["ہسپتال","بیماری","ویکسین","تعلیم","سیلاب","طب","خواتین"]),
]
label2id = {c: i for i, (c, _) in enumerate(CATEGORIES)}

def score_article(tokens):
    c = Counter(tokens)
    scores = [sum(c[k] for k in kws) for _, kws in CATEGORIES]
    return int(np.argmax(scores))

labels = [score_article(a) for a in articles]
print("label distribution:", Counter(labels))

# Tokenise to IDs (with CLS token appended to vocab)
CLS_IDX = len(word2idx)            # new token
PAD_IDX = len(word2idx) + 1
VSIZE = len(word2idx) + 2
MAX_LEN = 256                      # includes CLS

def encode(tokens: list[str]) -> list[int]:
    ids = [word2idx.get(t, UNK) for t in tokens]
    ids = [CLS_IDX] + ids
    ids = ids[:MAX_LEN]
    if len(ids) < MAX_LEN:
        ids = ids + [PAD_IDX] * (MAX_LEN - len(ids))
    return ids

X = np.array([encode(a) for a in articles], dtype=np.int64)
Y = np.array(labels, dtype=np.int64)

# Stratified 70/15/15
idxs_by_cls = defaultdict(list)
for i, y in enumerate(Y):
    idxs_by_cls[int(y)].append(i)
rng = np.random.default_rng(SEED)
train_idx, val_idx, test_idx = [], [], []
for c, ids in idxs_by_cls.items():
    ids = list(ids); rng.shuffle(ids)
    n = len(ids); nt = max(1, int(0.7 * n)); nv = max(1, int(0.15 * n))
    train_idx += ids[:nt]; val_idx += ids[nt:nt+nv]; test_idx += ids[nt+nv:]

# Any class with too few samples may have 0 val/test; ensure at least 1 if possible
print(f"split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
class_dist = {
    "total": dict(Counter(int(y) for y in Y)),
    "train": dict(Counter(int(Y[i]) for i in train_idx)),
    "val":   dict(Counter(int(Y[i]) for i in val_idx)),
    "test":  dict(Counter(int(Y[i]) for i in test_idx)),
}
print(json.dumps(class_dist, indent=2))

# -------- Modules (from scratch) --------
class ScaledDotProductAttention(nn.Module):
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)
        return out, weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads; self.d_k = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.attn = ScaledDotProductAttention()
    def split(self, x):
        B, T, D = x.size()
        return x.view(B, T, self.h, self.d_k).transpose(1, 2)  # (B, h, T, d_k)
    def forward(self, x, mask=None):
        Q = self.split(self.Wq(x)); K = self.split(self.Wk(x)); V = self.split(self.Wv(x))
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T)
        out, weights = self.attn(Q, K, V, mask)
        B = out.size(0)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.Wo(out), weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff); self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        return self.fc2(self.drop(F.relu(self.fc1(x))))

class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)

class PreLNEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, drop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model); self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff, drop)
        self.drop = nn.Dropout(drop)
    def forward(self, x, mask=None):
        n = self.ln1(x); a, w = self.mha(n, mask)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x, w

class TransformerEncoderCls(nn.Module):
    def __init__(self, V, d_model=128, n_heads=4, d_ff=512, n_layers=4, max_len=256, n_classes=5, drop=0.1, pad_idx=None):
        super().__init__()
        self.emb = nn.Embedding(V, d_model, padding_idx=pad_idx)
        self.pe = SinusoidalPE(d_model, max_len)
        self.blocks = nn.ModuleList([PreLNEncoderBlock(d_model, n_heads, d_ff, drop) for _ in range(n_layers)])
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, n_classes))
        self.pad_idx = pad_idx
    def forward(self, x, return_attn=False):
        mask = (x == self.pad_idx)
        h = self.pe(self.emb(x))
        attn_weights = []
        for blk in self.blocks:
            h, w = blk(h, mask)
            if return_attn: attn_weights.append(w)
        h = self.ln_out(h)
        cls_rep = h[:, 0]
        logits = self.head(cls_rep)
        return (logits, attn_weights) if return_attn else logits

# -------- Data loaders --------
class ArrDataset(Dataset):
    def __init__(self, idxs):
        self.idxs = idxs
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        k = self.idxs[i]
        return X[k], Y[k]

def collate(batch):
    xs = torch.tensor(np.stack([b[0] for b in batch]))
    ys = torch.tensor(np.array([b[1] for b in batch]))
    return xs, ys

BS = 8
tr = DataLoader(ArrDataset(train_idx), batch_size=BS, shuffle=True, collate_fn=collate)
va = DataLoader(ArrDataset(val_idx), batch_size=BS, shuffle=False, collate_fn=collate)
te = DataLoader(ArrDataset(test_idx), batch_size=BS, shuffle=False, collate_fn=collate)

# -------- Training with AdamW + cosine schedule + 50 warmup --------
EPOCHS = 20
WARMUP = 50
total_steps = EPOCHS * max(1, len(tr))

model = TransformerEncoderCls(VSIZE, d_model=128, n_heads=4, d_ff=512, n_layers=4,
                               max_len=MAX_LEN, n_classes=len(CATEGORIES), drop=0.1,
                               pad_idx=PAD_IDX).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

def lr_at(step):
    if step < WARMUP:
        return step / max(1, WARMUP)
    # cosine to 0
    prog = (step - WARMUP) / max(1, total_steps - WARMUP)
    return 0.5 * (1 + math.cos(math.pi * prog))

step = 0
train_ls, val_ls, train_acc, val_acc = [], [], [], []
best_val = 0.0; best_state = None
for ep in range(EPOCHS):
    model.train(); t0 = time.time(); tl=0; ta=0; n=0
    for x, y in tr:
        x = x.to(DEVICE); y = y.to(DEVICE)
        for g in opt.param_groups: g["lr"] = 5e-4 * lr_at(step)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        tl += loss.item() * x.size(0); ta += (logits.argmax(-1) == y).sum().item(); n += x.size(0); step += 1
    model.eval(); vl=0; va_acc=0; vn=0
    with torch.no_grad():
        for x, y in va:
            x = x.to(DEVICE); y = y.to(DEVICE)
            logits = model(x); loss = F.cross_entropy(logits, y)
            vl += loss.item() * x.size(0); va_acc += (logits.argmax(-1) == y).sum().item(); vn += x.size(0)
    tr_loss = tl/max(n,1); vl_loss = vl/max(vn,1)
    tr_a = ta/max(n,1); v_a = va_acc/max(vn,1)
    train_ls.append(tr_loss); val_ls.append(vl_loss); train_acc.append(tr_a); val_acc.append(v_a)
    if v_a >= best_val:
        best_val = v_a; best_state = copy.deepcopy(model.state_dict())
    print(f"ep {ep+1:2d}  trL={tr_loss:.4f}  trA={tr_a:.3f}  valL={vl_loss:.4f}  valA={v_a:.3f}  ({time.time()-t0:.1f}s)")

model.load_state_dict(best_state); torch.save(model.state_dict(), MOD / "transformer_cls.pt")

# Training curves
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(train_ls, label="train"); ax[0].plot(val_ls, label="val")
ax[0].set_title("Transformer loss"); ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(True)
ax[1].plot(train_acc, label="train"); ax[1].plot(val_acc, label="val")
ax[1].set_title("Transformer accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(True)
plt.tight_layout(); plt.savefig(FIG / "transformer_curves.png", dpi=140); plt.close()

# -------- Evaluation --------
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
all_g, all_p, all_x = [], [], []
model.eval()
with torch.no_grad():
    for x, y in te:
        x = x.to(DEVICE); y = y.to(DEVICE)
        logits = model(x); p = logits.argmax(-1).cpu().numpy()
        all_g.extend(y.cpu().numpy().tolist()); all_p.extend(p.tolist())
        all_x.extend(x.cpu().numpy())
acc = accuracy_score(all_g, all_p)
f1 = f1_score(all_g, all_p, average="macro", zero_division=0)
cls_names = [c for c, _ in CATEGORIES]
cm = confusion_matrix(all_g, all_p, labels=list(range(len(CATEGORIES))))
print(f"Test acc={acc:.4f}  macroF1={f1:.4f}")

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.xticks(range(len(cls_names)), cls_names, rotation=30); plt.yticks(range(len(cls_names)), cls_names)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=9,
                 color="white" if cm[i, j] > cm.max()/2 else "black")
plt.title("Transformer confusion matrix"); plt.xlabel("pred"); plt.ylabel("gold")
plt.tight_layout(); plt.savefig(FIG / "transformer_confusion.png", dpi=140); plt.close()

# Attention heatmaps for 3 correctly-classified articles (2 heads from final layer)
idx2word = {v: k for k, v in word2idx.items()}
idx2word[CLS_IDX] = "[CLS]"; idx2word[PAD_IDX] = "[PAD]"
correct_samples = [i for i, (g, p) in enumerate(zip(all_g, all_p)) if g == p][:3]
plotted = 0
for n, ci in enumerate(correct_samples):
    xb = torch.tensor(all_x[ci]).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, ws = model(xb, return_attn=True)
    final = ws[-1][0]       # (h, T, T)
    # first 40 tokens for readability
    toks = [idx2word.get(int(t), "?") for t in xb[0, :40].cpu().tolist()]
    for h in range(2):
        mat = final[h, :40, :40].cpu().numpy()
        plt.figure(figsize=(9, 8))
        plt.imshow(mat, cmap="viridis")
        plt.xticks(range(40), toks, rotation=90, fontsize=6)
        plt.yticks(range(40), toks, fontsize=6)
        plt.title(f"Attention (article #{ci}, final layer, head {h})")
        plt.colorbar(); plt.tight_layout()
        plt.savefig(FIG / f"attn_a{n}_h{h}.png", dpi=140); plt.close()
        plotted += 1
print(f"saved {plotted} attention heatmaps")

(DATA / "part3_summary.json").write_text(json.dumps({
    "test_accuracy": float(acc), "macro_f1": float(f1),
    "classes": cls_names, "confusion_matrix": cm.tolist(),
    "class_distribution": class_dist,
}, ensure_ascii=False, indent=2))
print("Part 3 complete.")
