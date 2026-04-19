"""Split cleaned.txt and raw.txt into 78 articles and persist token-level artifacts.

cleaned.txt has no explicit article markers, while raw.txt does ("Article N"
headers). We assume cleaned.txt preserves the same paragraph order as raw.txt
after drop of article headers. We align by allocating cleaned paragraphs
proportionally to each raw article's content-line count.
"""
from __future__ import annotations
import json, re, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUB = Path(__file__).resolve().parents[1]
OUT = SUB / "data"
OUT.mkdir(parents=True, exist_ok=True)

CLEANED = ROOT / "cleaned.txt"
RAW = ROOT / "raw.txt"
META = ROOT / "Metadata.json"


def load_raw_articles(path: Path) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    articles: list[list[str]] = []
    cur: list[str] = []
    for ln in lines:
        if re.match(r"^Article\s+\d+\s*$", ln.strip()):
            if cur:
                articles.append(cur)
            cur = []
        else:
            cur.append(ln)
    if cur:
        articles.append(cur)
    return articles


def proportional_split(cleaned_lines: list[str], raw_sizes: list[int]) -> list[list[str]]:
    total_raw = sum(raw_sizes)
    total_clean = len(cleaned_lines)
    cuts = [0]
    acc = 0
    for s in raw_sizes[:-1]:
        acc += s
        cuts.append(round(acc / total_raw * total_clean))
    cuts.append(total_clean)
    out = []
    for i in range(len(raw_sizes)):
        out.append(cleaned_lines[cuts[i]:cuts[i + 1]])
    return out


def tokenize(line: str) -> list[str]:
    # cleaned.txt already has <NUM>/<UNK> markers and normalized tokens.
    return [t for t in line.strip().split() if t]


def main() -> None:
    meta = json.loads(META.read_text(encoding="utf-8"))
    raw_articles = load_raw_articles(RAW)
    assert len(raw_articles) == len(meta), (len(raw_articles), len(meta))
    raw_sizes = [sum(1 for ln in art if ln.strip()) for art in raw_articles]

    cleaned_lines = [ln for ln in CLEANED.read_text(encoding="utf-8").splitlines() if ln.strip()]
    clean_articles = proportional_split(cleaned_lines, raw_sizes)

    # Persist per-article tokens (cleaned)
    cleaned_tokens = [[tok for ln in art for tok in tokenize(ln)] for art in clean_articles]
    raw_tokens = [[tok for ln in art for tok in tokenize(ln)] for art in raw_articles]

    (OUT / "articles_cleaned.json").write_text(json.dumps(cleaned_tokens, ensure_ascii=False))
    (OUT / "articles_raw.json").write_text(json.dumps(raw_tokens, ensure_ascii=False))

    # Per-article sentence list (cleaned): split on <NUM> boundaries and common punctuation
    sents_per_article = []
    for art in clean_articles:
        sents = []
        for line in art:
            # Each non-empty cleaned line is already one coherent unit; split further on '.' or '۔'
            parts = re.split(r"[\.\u06D4\?!]+", line)
            for p in parts:
                toks = tokenize(p)
                if 3 <= len(toks) <= 80:
                    sents.append(toks)
        sents_per_article.append(sents)
    (OUT / "sents_cleaned.json").write_text(json.dumps(sents_per_article, ensure_ascii=False))

    print(f"articles cleaned: {len(cleaned_tokens)}, total tokens: {sum(map(len, cleaned_tokens))}")
    print(f"articles raw: {len(raw_tokens)}, total tokens: {sum(map(len, raw_tokens))}")
    print(f"total sentences: {sum(len(s) for s in sents_per_article)}")


if __name__ == "__main__":
    main()
