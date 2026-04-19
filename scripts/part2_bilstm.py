"""Part 2 step 4 & 5 — BiLSTM sequence labeler (POS + NER with CRF) + evaluation + ablations."""
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
MOD = SUB / "models"; MOD.mkdir(exist_ok=True)
FIG = SUB / "figures"; FIG.mkdir(exist_ok=True)
EMB = SUB / "embeddings"

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# -------- Load splits --------
def load_conll(path: Path):
    sents: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if cur: sents.append(cur); cur = []
        else:
            parts = line.split()
            if len(parts) >= 2:
                cur.append((parts[0], parts[1]))
    if cur: sents.append(cur)
    return sents

pos_train = load_conll(DATA / "pos_train.conll")
pos_val   = load_conll(DATA / "pos_val.conll")
pos_test  = load_conll(DATA / "pos_test.conll")
ner_train = load_conll(DATA / "ner_train.conll")
ner_val   = load_conll(DATA / "ner_val.conll")
ner_test  = load_conll(DATA / "ner_test.conll")

print(f"POS: train={len(pos_train)} val={len(pos_val)} test={len(pos_test)}")
print(f"NER: train={len(ner_train)} val={len(ner_val)} test={len(ner_test)}")

# -------- Vocab aligned to Part 1 (C3 embeddings) --------
word2idx = json.loads((EMB / "word2idx.json").read_text())
emb_np = np.load(EMB / "embeddings_w2v.npy")
PAD_IDX = len(word2idx)  # add PAD at end
EMB_DIM = emb_np.shape[1]
emb_full = np.vstack([emb_np, np.zeros((1, EMB_DIM), dtype=np.float32)])

def word_id(w: str) -> int:
    return word2idx.get(w, word2idx["<UNK>"])

def encode(sent_tags: list[tuple[str, str]], label2id: dict[str, int]):
    xs = [word_id(t) for t, _ in sent_tags]
    ys = [label2id[tag] for _, tag in sent_tags]
    return xs, ys

# -------- Label vocab --------
def build_labels(corpus):
    labels = sorted({t for s in corpus for _, t in s})
    return {l: i for i, l in enumerate(labels)}

pos_labels = build_labels(pos_train + pos_val + pos_test)
ner_labels = build_labels(ner_train + ner_val + ner_test)
print("POS labels:", pos_labels)
print("NER labels:", ner_labels)

PAD_TAG = 0
N_POS = len(pos_labels)
N_NER = len(ner_labels)

class SeqDataset(Dataset):
    def __init__(self, sents, label2id):
        self.data = [encode(s, label2id) for s in sents]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]

def collate(batch, pad_lbl=0):
    lens = [len(x) for x, _ in batch]
    maxlen = max(lens)
    xs = torch.full((len(batch), maxlen), PAD_IDX, dtype=torch.long)
    ys = torch.full((len(batch), maxlen), pad_lbl, dtype=torch.long)
    mask = torch.zeros((len(batch), maxlen), dtype=torch.bool)
    for i, (x, y) in enumerate(batch):
        xs[i, :len(x)] = torch.tensor(x)
        ys[i, :len(y)] = torch.tensor(y)
        mask[i, :len(x)] = 1
    return xs, ys, mask, torch.tensor(lens)

# -------- Models --------
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_labels, pad_idx,
                 emb_init=None, freeze=False, bidirectional=True,
                 layers=2, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        if emb_init is not None:
            with torch.no_grad():
                self.emb.weight.copy_(torch.tensor(emb_init))
            self.emb.weight.requires_grad = not freeze
        self.lstm = nn.LSTM(emb_dim, 128, num_layers=layers, batch_first=True,
                             bidirectional=bidirectional, dropout=dropout if layers > 1 else 0.0)
        hid = 128 * (2 if bidirectional else 1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hid, n_labels)
    def forward(self, x, lens):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(e, lens.cpu(), batch_first=True, enforce_sorted=False)
        o, _ = self.lstm(packed)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=x.size(1))
        return self.fc(self.drop(o))

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_labels, pad_idx,
                 emb_init=None, freeze=False, bidirectional=True, layers=2, dropout=0.5):
        super().__init__()
        self.tagger = BiLSTMTagger(vocab_size, emb_dim, n_labels, pad_idx,
                                    emb_init, freeze, bidirectional, layers, dropout)
        self.n_labels = n_labels
        # transitions[i, j] = score of transitioning from j -> i
        self.trans = nn.Parameter(torch.randn(n_labels, n_labels) * 0.01)
        self.start = nn.Parameter(torch.randn(n_labels) * 0.01)
        self.end = nn.Parameter(torch.randn(n_labels) * 0.01)
    def emissions(self, x, lens):
        return self.tagger(x, lens)
    def _forward_alg(self, emissions, mask):
        # emissions: (B, T, K)  mask: (B, T) bool
        B, T, K = emissions.size()
        alpha = self.start.unsqueeze(0) + emissions[:, 0]      # (B, K)
        for t in range(1, T):
            emit = emissions[:, t].unsqueeze(1)                 # (B, 1, K)
            tr = self.trans.unsqueeze(0)                        # (1, K, K)
            score = alpha.unsqueeze(2) + tr + emit              # broadcast (B, K, K)
            new_alpha = torch.logsumexp(score, dim=1)
            m = mask[:, t].unsqueeze(1)
            alpha = torch.where(m, new_alpha, alpha)
        alpha = alpha + self.end.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)
    def _score(self, emissions, tags, mask):
        B, T, _ = emissions.size()
        score = self.start[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for t in range(1, T):
            emit = emissions[:, t].gather(1, tags[:, t:t+1]).squeeze(1)
            tr = self.trans[tags[:, t], tags[:, t-1]]
            m = mask[:, t].float()
            score = score + (emit + tr) * m
        # add end score: find last valid position per batch
        last = mask.long().sum(1) - 1
        last_tags = tags.gather(1, last.unsqueeze(1)).squeeze(1)
        score = score + self.end[last_tags]
        return score
    def nll(self, x, lens, tags, mask):
        emissions = self.emissions(x, lens)
        forward = self._forward_alg(emissions, mask)
        gold = self._score(emissions, tags, mask)
        return (forward - gold).mean()
    def viterbi(self, x, lens, mask):
        emissions = self.emissions(x, lens)
        B, T, K = emissions.size()
        history = []
        alpha = self.start.unsqueeze(0) + emissions[:, 0]
        for t in range(1, T):
            emit = emissions[:, t].unsqueeze(1)
            tr = self.trans.unsqueeze(0)
            score = alpha.unsqueeze(2) + tr + emit
            best_score, best_prev = score.max(dim=1)
            m = mask[:, t].unsqueeze(1)
            alpha = torch.where(m, best_score, alpha)
            history.append(best_prev)
        alpha = alpha + self.end.unsqueeze(0)
        best_last = alpha.argmax(dim=1)
        # backtrack
        best_paths = [[int(b.item())] for b in best_last]
        for t in range(len(history) - 1, -1, -1):
            for b in range(B):
                prev = int(history[t][b, best_paths[b][-1]].item())
                if mask[b, t + 1]:
                    best_paths[b].append(prev)
        # reverse paths and pad to mask
        out = []
        for b in range(B):
            p = list(reversed(best_paths[b]))
            L = int(mask[b].sum().item())
            p = p[:L]
            if len(p) < L:
                p = p + [0] * (L - len(p))
            out.append(p)
        return out

# -------- Training loop --------
def run_tagger(train_data, val_data, test_data, n_labels, emb_init=None, freeze=False,
               epochs=25, bs=16, use_crf=False, bidirectional=True, dropout=0.5, tag="POS",
               patience=5):
    tr = DataLoader(SeqDataset(train_data, build_labels(train_data + val_data + test_data)),
                    batch_size=bs, shuffle=True,
                    collate_fn=lambda b: collate(b, PAD_TAG))
    # Rebuild datasets to share the same label2id
    lbl = build_labels(train_data + val_data + test_data)
    def ld(data, shuffle):
        return DataLoader(SeqDataset(data, lbl), batch_size=bs, shuffle=shuffle,
                          collate_fn=lambda b: collate(b, 0))
    tr = ld(train_data, True); va = ld(val_data, False); te = ld(test_data, False)
    n_labels = len(lbl)

    if use_crf:
        model = BiLSTM_CRF(PAD_IDX + 1, EMB_DIM, n_labels, PAD_IDX,
                           emb_init=emb_init, freeze=freeze, bidirectional=bidirectional,
                           dropout=dropout).to(DEVICE)
    else:
        model = BiLSTMTagger(PAD_IDX + 1, EMB_DIM, n_labels, PAD_IDX,
                             emb_init=emb_init, freeze=freeze,
                             bidirectional=bidirectional, dropout=dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def epoch_loss(model, dl, train_mode):
        total, n = 0.0, 0
        model.train(train_mode)
        for x, y, mask, lens in dl:
            x = x.to(DEVICE); y = y.to(DEVICE); mask = mask.to(DEVICE)
            if use_crf:
                loss = model.nll(x, lens, y, mask)
            else:
                logits = model(x, lens)
                loss = F.cross_entropy(logits.reshape(-1, n_labels), y.reshape(-1), ignore_index=0)
            if train_mode:
                opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * x.size(0); n += x.size(0)
        return total / max(n, 1)

    train_losses, val_losses, val_f1s = [], [], []
    best_f1, best_state, bad = -1, None, 0
    id2lbl = {v: k for k, v in lbl.items()}
    for ep in range(epochs):
        t0 = time.time()
        tl = epoch_loss(model, tr, True)
        model.eval()
        with torch.no_grad():
            vl = epoch_loss(model, va, False)
            f1 = token_f1(model, va, id2lbl, use_crf)
        train_losses.append(tl); val_losses.append(vl); val_f1s.append(f1)
        print(f"  [{tag}] ep {ep+1}  tr={tl:.4f}  val={vl:.4f}  valF1={f1:.4f}  ({time.time()-t0:.1f}s)")
        if f1 > best_f1:
            best_f1, best_state, bad = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                print("  early stop"); break
    model.load_state_dict(best_state)
    return model, {"train": train_losses, "val": val_losses, "val_f1": val_f1s, "labels": lbl}

def decode(model, dl, id2lbl, use_crf):
    golds, preds = [], []
    model.eval()
    with torch.no_grad():
        for x, y, mask, lens in dl:
            x = x.to(DEVICE); y = y.to(DEVICE); mask = mask.to(DEVICE)
            if use_crf:
                paths = model.viterbi(x, lens, mask)
                for bi, path in enumerate(paths):
                    L = int(lens[bi].item())
                    preds.append([id2lbl[t] for t in path[:L]])
                    golds.append([id2lbl[int(t.item())] for t in y[bi, :L]])
            else:
                logits = model(x, lens)
                p = logits.argmax(-1).cpu()
                for bi in range(x.size(0)):
                    L = int(lens[bi].item())
                    preds.append([id2lbl[int(t.item())] for t in p[bi, :L]])
                    golds.append([id2lbl[int(t.item())] for t in y[bi, :L]])
    return golds, preds

def token_f1(model, dl, id2lbl, use_crf):
    golds, preds = decode(model, dl, id2lbl, use_crf)
    from sklearn.metrics import f1_score
    flat_g = [t for s in golds for t in s]
    flat_p = [t for s in preds for t in s]
    if not flat_g: return 0.0
    return f1_score(flat_g, flat_p, average="macro", zero_division=0)

# -------- POS run (frozen vs fine-tuned) --------
def one_run(train_data, val_data, test_data, tag_name, use_crf,
             emb_init, freeze, epochs=25, bidirectional=True, dropout=0.5):
    m, log = run_tagger(train_data, val_data, test_data, n_labels=0,
                       emb_init=emb_init, freeze=freeze, epochs=epochs,
                       use_crf=use_crf, bidirectional=bidirectional, dropout=dropout,
                       tag=tag_name)
    lbl = log["labels"]
    id2lbl = {v: k for k, v in lbl.items()}
    te = DataLoader(SeqDataset(test_data, lbl), batch_size=16, shuffle=False,
                    collate_fn=lambda b: collate(b, 0))
    golds, preds = decode(m, te, id2lbl, use_crf)
    return m, log, golds, preds

print("\n--- POS: frozen embeddings ---")
m_pos_frozen, log_pos_frozen, g_pos_f, p_pos_f = one_run(pos_train, pos_val, pos_test,
    "POS-frozen", use_crf=False, emb_init=emb_full, freeze=True, epochs=25)
print("--- POS: fine-tuned embeddings ---")
m_pos_ft, log_pos_ft, g_pos_ft, p_pos_ft = one_run(pos_train, pos_val, pos_test,
    "POS-ft", use_crf=False, emb_init=emb_full, freeze=False, epochs=25)
torch.save(m_pos_ft.state_dict(), MOD / "bilstm_pos.pt")

print("\n--- NER: frozen, CRF ---")
m_ner_frozen, log_ner_frozen, g_ner_f, p_ner_f = one_run(ner_train, ner_val, ner_test,
    "NER-frozen-crf", use_crf=True, emb_init=emb_full, freeze=True, epochs=25)
print("--- NER: fine-tuned, CRF ---")
m_ner_ft, log_ner_ft, g_ner_ft, p_ner_ft = one_run(ner_train, ner_val, ner_test,
    "NER-ft-crf", use_crf=True, emb_init=emb_full, freeze=False, epochs=25)
torch.save(m_ner_ft.state_dict(), MOD / "bilstm_ner.pt")

print("--- NER: fine-tuned, no CRF (softmax only) ---")
m_ner_soft, log_ner_soft, g_ner_s, p_ner_s = one_run(ner_train, ner_val, ner_test,
    "NER-ft-softmax", use_crf=False, emb_init=emb_full, freeze=False, epochs=25)

# -------- Ablations on NER (primary) --------
print("\n--- Ablation A1: uni-directional LSTM ---")
_, log_a1, g_a1, p_a1 = one_run(ner_train, ner_val, ner_test, "NER-A1-uni",
    use_crf=True, emb_init=emb_full, freeze=False, bidirectional=False, epochs=20)
print("--- Ablation A2: no dropout ---")
_, log_a2, g_a2, p_a2 = one_run(ner_train, ner_val, ner_test, "NER-A2-nodrop",
    use_crf=True, emb_init=emb_full, freeze=False, dropout=0.0, epochs=20)
print("--- Ablation A3: random embedding init ---")
_, log_a3, g_a3, p_a3 = one_run(ner_train, ner_val, ner_test, "NER-A3-random",
    use_crf=True, emb_init=None, freeze=False, epochs=20)
print("--- Ablation A4: softmax instead of CRF (same as m_ner_soft) ---")
# reuse

# -------- Evaluation helpers --------
def pos_metrics(golds, preds, lbl):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    flat_g = [t for s in golds for t in s]
    flat_p = [t for s in preds for t in s]
    labels = sorted(lbl.keys())
    acc = accuracy_score(flat_g, flat_p)
    f1 = f1_score(flat_g, flat_p, average="macro", zero_division=0, labels=labels)
    cm = confusion_matrix(flat_g, flat_p, labels=labels)
    return acc, f1, cm, labels

def ner_metrics_token(golds, preds):
    from sklearn.metrics import classification_report
    try:
        from seqeval.metrics import classification_report as seq_cr, f1_score as seq_f1
        report = seq_cr(golds, preds, zero_division=0, digits=4)
        f1 = seq_f1(golds, preds, zero_division=0)
        return report, f1
    except Exception as e:
        return str(e), 0.0

pos_acc_f, pos_f1_f, pos_cm_f, pos_labels_list = pos_metrics(g_pos_f, p_pos_f, log_pos_frozen["labels"])
pos_acc_ft, pos_f1_ft, pos_cm_ft, _ = pos_metrics(g_pos_ft, p_pos_ft, log_pos_ft["labels"])
print(f"\nPOS (frozen): acc={pos_acc_f:.4f}  macroF1={pos_f1_f:.4f}")
print(f"POS (ft):     acc={pos_acc_ft:.4f}  macroF1={pos_f1_ft:.4f}")

# confusion matrix plot (fine-tuned)
plt.figure(figsize=(7, 6))
plt.imshow(pos_cm_ft, cmap="Blues")
plt.xticks(range(len(pos_labels_list)), pos_labels_list, rotation=45, ha="right")
plt.yticks(range(len(pos_labels_list)), pos_labels_list)
for i in range(pos_cm_ft.shape[0]):
    for j in range(pos_cm_ft.shape[1]):
        plt.text(j, i, int(pos_cm_ft[i, j]), ha="center", va="center",
                 color="white" if pos_cm_ft[i, j] > pos_cm_ft.max() / 2 else "black", fontsize=8)
plt.title("POS confusion matrix (fine-tuned)")
plt.xlabel("predicted"); plt.ylabel("gold")
plt.tight_layout(); plt.savefig(FIG / "pos_confusion.png", dpi=140); plt.close()

# 3 most confused tag pairs
pairs = []
for i, gi in enumerate(pos_labels_list):
    for j, gj in enumerate(pos_labels_list):
        if i != j: pairs.append(((gi, gj), int(pos_cm_ft[i, j])))
pairs.sort(key=lambda x: -x[1])
top_confused = pairs[:3]
print("top confused pairs:", top_confused)

# Confusion examples
def find_examples(golds, preds, gold_tag, pred_tag, data, k=2):
    out = []
    for (gs, ps, raw) in zip(golds, preds, data):
        for i, (g, p) in enumerate(zip(gs, ps)):
            if g == gold_tag and p == pred_tag:
                toks = [t for t, _ in raw]
                out.append({"sent": " ".join(toks), "token": toks[i], "gold": g, "pred": p})
                if len(out) >= k: return out
    return out

confusion_examples = {}
for (g, p), cnt in top_confused:
    confusion_examples[f"{g}->{p}"] = find_examples(g_pos_ft, p_pos_ft, g, p, pos_test)

# NER evaluation
ner_report_crf, ner_f1_crf = ner_metrics_token(g_ner_ft, p_ner_ft)
ner_report_soft, ner_f1_soft = ner_metrics_token(g_ner_s, p_ner_s)
ner_report_frozen, ner_f1_frozen = ner_metrics_token(g_ner_f, p_ner_f)
print("NER (crf, fine-tuned):\n", ner_report_crf)
print("NER (softmax, fine-tuned):\n", ner_report_soft)
print("NER (crf, frozen):\n", ner_report_frozen)

# Ablation table
ablations = {
    "baseline_ft_crf":   {"f1": float(ner_f1_crf)},
    "A1_uni":            {"f1": float(ner_metrics_token(g_a1, p_a1)[1])},
    "A2_nodrop":         {"f1": float(ner_metrics_token(g_a2, p_a2)[1])},
    "A3_random_emb":     {"f1": float(ner_metrics_token(g_a3, p_a3)[1])},
    "A4_softmax":        {"f1": float(ner_f1_soft)},
}
print("\nAblations:", json.dumps(ablations, indent=2))

# Error analysis — 5 FP and 5 FN on NER
def entity_spans(tag_seq):
    spans, cur = [], None
    for i, t in enumerate(tag_seq):
        if t.startswith("B-"):
            if cur: spans.append(cur)
            cur = (t[2:], i, i)
        elif t.startswith("I-") and cur and cur[0] == t[2:]:
            cur = (cur[0], cur[1], i)
        else:
            if cur: spans.append(cur); cur = None
    if cur: spans.append(cur)
    return set(spans)

fps, fns = [], []
for gs, ps, (_, raw) in zip(g_ner_ft, p_ner_ft, zip(range(len(ner_test)), ner_test)):
    tokens = [t for t, _ in raw]
    g_spans = entity_spans(gs); p_spans = entity_spans(ps)
    for sp in p_spans - g_spans:
        fps.append({"sent": " ".join(tokens), "span": " ".join(tokens[sp[1]:sp[2]+1]), "type": sp[0]})
    for sp in g_spans - p_spans:
        fns.append({"sent": " ".join(tokens), "span": " ".join(tokens[sp[1]:sp[2]+1]), "type": sp[0]})
print(f"FPs: {len(fps)}  FNs: {len(fns)}")

# -------- Training curves --------
def plot_curves(log, title, path):
    plt.figure()
    plt.plot(log["train"], label="train loss")
    plt.plot(log["val"], label="val loss")
    plt.plot(log["val_f1"], label="val F1", linestyle="--")
    plt.title(title); plt.xlabel("epoch"); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()
plot_curves(log_pos_ft, "POS BiLSTM (fine-tuned)", FIG / "pos_curves.png")
plot_curves(log_ner_ft, "NER BiLSTM-CRF (fine-tuned)", FIG / "ner_curves.png")

# Persist summary
summary = {
    "pos": {
        "frozen": {"acc": pos_acc_f, "macro_f1": pos_f1_f},
        "fine_tuned": {"acc": pos_acc_ft, "macro_f1": pos_f1_ft},
        "top_confused": [{"pair": f"{g}->{p}", "count": c} for (g, p), c in top_confused],
        "confusion_examples": confusion_examples,
    },
    "ner": {
        "fine_tuned_crf": {"seqeval_f1": float(ner_f1_crf), "report": ner_report_crf},
        "softmax":        {"seqeval_f1": float(ner_f1_soft), "report": ner_report_soft},
        "frozen_crf":     {"seqeval_f1": float(ner_f1_frozen), "report": ner_report_frozen},
        "fps_sample": fps[:5],
        "fns_sample": fns[:5],
    },
    "ablations": ablations,
}
(MOD.parent / "data" / "part2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print("Part 2 complete.")
