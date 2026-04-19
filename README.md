# i21-1672 — NLP Assignment 2 (Neural NLP Pipeline)

CS-4063 Natural Language Processing, FAST NUCES — Spring 2026.
Student: **i21-1672 (DS-A)**.

All code is written from scratch in PyTorch. No pretrained models, no HuggingFace, no Gensim, no `nn.Transformer` / `nn.MultiheadAttention` / `nn.TransformerEncoder`.

## Layout

```
i21-1672_Assignment2_DS-A/
├── i21-1672_Assignment2_DS-A.ipynb   # executed notebook — entry point for grading
├── report.pdf                         # 2–3 page write-up (Times New Roman 12pt, 1.5 line spacing)
├── README.md                          # this file
├── scripts/
│   ├── prep_corpus.py                 # splits cleaned.txt/raw.txt into 78 articles
│   ├── part1_embeddings.py            # TF-IDF, PPMI, Skip-gram, 4-condition comparison
│   ├── part2_annotate.py              # rule-based POS + gazetteer BIO-NER, 500 sentences
│   ├── part2_bilstm.py                # BiLSTM / BiLSTM-CRF + ablations
│   ├── part3_transformer.py           # from-scratch Transformer encoder + CLS classifier
│   ├── build_notebook.py              # rebuilds the top-level .ipynb from artefacts
│   └── build_report.py                # rebuilds report.pdf
├── embeddings/
│   ├── tfidf_matrix.npy
│   ├── ppmi_matrix.npy
│   ├── embeddings_w2v.npy             # ½(V+U) final Skip-gram embeddings (C3)
│   ├── word2idx.json
│   ├── tfidf_top_words.json
│   ├── ppmi_nearest.json
│   ├── w2v_nearest.json
│   ├── analogy_results.json
│   └── four_condition_comparison.json
├── models/
│   ├── bilstm_pos.pt
│   ├── bilstm_ner.pt
│   └── transformer_cls.pt
├── data/                              # cleaned input + derived splits
│   ├── articles_cleaned.json
│   ├── articles_raw.json
│   ├── sents_cleaned.json
│   ├── pos_train.conll / pos_val.conll / pos_test.conll
│   ├── ner_train.conll / ner_val.conll / ner_test.conll
│   ├── annotation_summary.json
│   ├── part2_summary.json
│   └── part3_summary.json
└── figures/                           # all plots referenced in the notebook & report
```

`cleaned.txt`, `raw.txt`, and `Metadata.json` are expected to sit in the repository root (one level above this folder) — they are not redistributed with the submission.

## Reproducing end-to-end

Python 3.11+ with PyTorch, numpy, scikit-learn, matplotlib, seqeval, nbformat, reportlab.

```bash
pip install torch numpy scikit-learn matplotlib seqeval nbformat nbconvert reportlab

# generate artefacts
python3 scripts/prep_corpus.py
python3 scripts/part1_embeddings.py
python3 scripts/part2_annotate.py
python3 scripts/part2_bilstm.py
python3 scripts/part3_transformer.py

# rebuild the report & notebook
python3 scripts/build_report.py
python3 scripts/build_notebook.py
jupyter nbconvert --to notebook --execute --inplace i21-1672_Assignment2_DS-A.ipynb
```

All three training parts run on CPU or Apple MPS; total wall-clock ≈ 15–20 minutes on an M-series laptop. Skip-gram training is the hot path (three separate 5-epoch runs for the four-condition comparison).

## Design notes

* **Article segmentation.** `cleaned.txt` lacks explicit article markers, so `prep_corpus.py` aligns it with `raw.txt`'s `Article N` headers by proportional line allocation. The result is 78 cleaned-token article sequences in the same order as `Metadata.json`.
* **Topic labels.** Not present in `Metadata.json`; we derive them per spec by keyword frequency over the indicative keywords listed in the assignment.
* **POS lexicon.** ≥ 200 entries per major class (NOUN / VERB / ADJ / ADV) after suffix-variant expansion, plus full closed-class lists (PRON / DET / CONJ / POST / NUM / PUNC). See `scripts/part2_annotate.py` for the dictionaries.
* **NER gazetteer.** ≥ 50 Pakistani persons, ≥ 50 locations, ≥ 30 organisations; multi-token entity phrases are matched greedily with longest-span preference.
* **Restrictions.** No `nn.Transformer`, `nn.MultiheadAttention`, `nn.TransformerEncoder`, Gensim, or HuggingFace anywhere — the whole attention stack is hand-written inside `part3_transformer.py`.

## Files produced

| Artefact                                       | Produced by                    |
| ---------------------------------------------- | ------------------------------ |
| `embeddings/tfidf_matrix.npy`                  | `part1_embeddings.py`          |
| `embeddings/ppmi_matrix.npy`                   | `part1_embeddings.py`          |
| `embeddings/embeddings_w2v.npy`                | `part1_embeddings.py`          |
| `embeddings/word2idx.json`                     | `part1_embeddings.py`          |
| `models/bilstm_pos.pt`                         | `part2_bilstm.py`              |
| `models/bilstm_ner.pt`                         | `part2_bilstm.py`              |
| `models/transformer_cls.pt`                    | `part3_transformer.py`         |
| `data/{pos,ner}_{train,val,test}.conll`        | `part2_annotate.py`            |
| `figures/*.png`                                | Parts 1–3                      |
| `report.pdf`                                   | `build_report.py`              |
| `i21-1672_Assignment2_DS-A.ipynb`              | `build_notebook.py` + execute  |
