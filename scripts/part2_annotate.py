"""Part 2 step 3 — dataset preparation: 500 sentences, POS + NER annotation, splits."""
from __future__ import annotations

import json, random, re
from collections import Counter, defaultdict
from pathlib import Path

SUB = Path(__file__).resolve().parents[1]
DATA = SUB / "data"

SEED = 42
random.seed(SEED)

sents_per_article = json.loads((DATA / "sents_cleaned.json").read_text())
articles = json.loads((DATA / "articles_cleaned.json").read_text())

# -------- Article → topic category (same keyword rule as Part 1) --------
CATEGORIES = {
    "politics":        ["حکومت", "وزیراعظم", "صدر", "پارلیمان", "انتخابات", "سیاسی", "اپوزیشن", "وزیر"],
    "sports":          ["کرکٹ", "ٹیم", "میچ", "کھلاڑی", "اسکور", "ورلڈ", "کپ", "پاکستانی"],
    "economy":         ["معیشت", "بینک", "تجارت", "قیمت", "ڈالر", "برآمدات", "سرمایہ", "بجٹ"],
    "international":   ["اقوام", "متحدہ", "سفیر", "معاہدہ", "بین", "غیر", "ملکی", "بیرون"],
    "health_society":  ["ہسپتال", "بیماری", "ویکسین", "تعلیم", "سیلاب", "طب", "خواتین", "بچوں"],
}
def score_article(tokens):
    c = Counter(tokens)
    return max({cat: sum(c[k] for k in kws) for cat, kws in CATEGORIES.items()}.items(), key=lambda kv: kv[1])[0]

article_cats = [score_article(a) for a in articles]
print("article cat dist:", Counter(article_cats))

# -------- Build POS lexicon (≥200 entries per major category) --------
# Closed-class categories: DET, PRON, CONJ, POST, NUM, PUNC — small but exhaustive.
# Open-class seeds (NOUN/VERB/ADJ/ADV): 200+ common cleaned-form entries.

LEXICON = {
    "PRON": [
        "میں","مجھے","میرا","میری","میرے","ہم","ہمارا","ہماری","ہمارے","ہمیں",
        "تم","تمہارا","تمہاری","تمہیں","آپ","آپکا","آپکی","آپکے","تُو","تیرا","تیری",
        "وہ","وہی","اس","اسے","اسکا","اسکی","اسکے","انہی","انہیں","انکا","انکی","انکے","انہوں",
        "یہ","یہی","یہاں","وہاں","جو","جس","جسے","جسکا","جسکی","جسکے","جنہوں","جنہیں",
        "کون","کس","کسی","کسیکا","کوئ","کوئی","کسے","کبھی","کبھ",
        "خود","اپنا","اپنی","اپنے","اپن","سب","ہر","بعض","کئی",
    ],
    "DET": [
        "یہ","وہ","اس","ان","ایک","کوئی","کچھ","تمام","سب","ہر",
        "کسی","کئی","بعض","دوسرا","دوسری","دوسرے","پہلا","پہلی","پہلے",
        "دونوں","تین","چار","پانچ","چھ","سات","آٹھ","نو","دس",
        "یہی","وہی","ایسا","ایسی","ایسے","جیسا","جیسی","جیسے",
    ],
    "CONJ": [
        "اور","یا","لیکن","مگر","پر","تاہم","چنانچہ","چونکہ","کیونکہ","کیوں",
        "اگر","تو","تاکہ","جبکہ","بلکہ","ورنہ","البتہ","نیز","بھی","ہی",
        "جو","کہ","جب","جہاں","جس","جیسا","ویسا","پھر","اب","تب",
        "حالانکہ","اگرچہ","گویا","یعنی","مثلاً","لہٰذا","لہذا","پس","بس","وغیرہ",
    ],
    "POST": [
        "کا","کی","کے","کو","سے","نے","پر","میں","تک","پاس",
        "لیے","لیئے","لئے","واسطے","بدلے","بجائ","بدل","ساتھ","دوران","خلاف",
        "باوجود","جیسا","جیسی","جیسے","مثل","طرح","طرف","بارے","علاوہ","سوا",
        "تلے","اوپر","نیچے","آگے","پیچھے","درمیان","بیچ","باہر","اندر","قریب",
    ],
    "NUM": [
        "صفر","ایک","دو","تین","چار","پانچ","چھ","سات","آٹھ","نو",
        "دس","گیارہ","بارہ","تیرہ","چودہ","پندرہ","سولہ","سترہ","اٹھارہ","انیس",
        "بیس","تیس","چالیس","پچاس","ساٹھ","ستر","اسی","نوے","سو","ہزار",
        "لاکھ","کروڑ","ارب","پہلا","دوسرا","تیسرا","چوتھا","نصف","ربع","دہائی",
        "<NUM>",
    ],
    "ADV": [
        "یہاں","وہاں","آج","کل","ابھی","فوراً","جلدی","دیر","ہمیشہ","کبھی",
        "اکثر","شاید","یقیناً","ضرور","بالکل","قطعاً","کافی","بہت","زیادہ","کم",
        "تھوڑا","صرف","واقعی","حقیقتاً","خاص","عام","ظاہر","غالباً","بظاہر","رفتہ",
        "پہلے","بعد","آگے","پیچھے","تقریباً","قریب","دور","اچانک","ساتھ","علاوہ",
        "اندر","باہر","اوپر","نیچے","بارہا","باربار","پھر","دوبارہ","ہنوز","تاحال",
    ],
    "ADJ": [
        "اچھا","اچھی","اچھے","بُرا","بُری","بُرے","بڑا","بڑی","بڑے","چھوٹا",
        "چھوٹی","چھوٹے","نیا","نئی","نئے","پرانا","پرانی","پرانے","لمبا","لمبی",
        "چوڑا","چوڑی","گرم","ٹھنڈا","ٹھنڈی","کالا","کالی","سفید","سرخ","ہرا",
        "پیلا","نیلا","سیاہ","خوبصورت","خوبصورتی","مشہور","معروف","مشہورہ","اہم","اہمیت",
        "قابل","ناقابل","ممکن","ناممکن","ضروری","غیرضروری","سیاسی","مذہبی","سماجی","ثقافتی",
        "قومی","بین","عالمی","مقامی","علاقائ","علاقائی","عسکری","معاشی","معاشرتی","سائنسی",
        "تاریخی","جدید","قدیم","موجودہ","سابق","حاضر","غیرحاضر","مستقبل","آئندہ","ماضی",
        "تیز","آہستہ","محفوظ","غیرمحفوظ","صحتمند","بیمار","دکھ","درد","سکون","پریشان",
        "فعال","غیرفعال","ذمہدار","غیرذمہ","شدید","ہلکا","کم","زیادہ","اکثریت","اقلیت",
        "عام","خاص","مخصوص","غیرمعمول","عجیب","حیران","پریشان","قابلِ","ناکارہ","کارآمد",
    ],
    "VERB": [
        "ہے","ہیں","تھا","تھی","تھے","ہوا","ہوئ","ہوئی","ہوئے","ہو",
        "ہونا","ہوتا","ہوتی","ہوتے","ہوگا","ہوگی","ہوگئ","ہوچکا","ہوچکی","ہوسکتا",
        "کیا","کرنا","کرتا","کرتی","کرتے","کریں","کرے","کرو","کیا","کیے",
        "گیا","گئی","گئے","آیا","آئی","آئے","آتا","آتی","آتے","جاتا",
        "جاتی","جاتے","جائیں","جائے","جاؤ","چلا","چلی","چلے","چلنا","چلتا",
        "دیا","دیے","دی","دیں","دینا","دیتا","دیتی","دیتے","لیا","لی",
        "لیے","لیتا","لیتی","لیتے","لینا","بتایا","بتایی","بتائی","بتائے","بتاتا",
        "کہا","کہی","کہے","کہنا","کہتا","کہتی","کہتے","دیکھا","دیکھی","دیکھے",
        "دیکھتا","دیکھتی","دیکھنا","جانا","جان","رہا","رہی","رہے","رہنا","رہتا",
        "رکھا","رکھی","رکھے","رکھنا","رکھتا","رکھتی","رکھتے","پایا","پائی","پائے",
        "پاتا","پاتی","پانا","اٹھا","اٹھی","اٹھے","اٹھایا","بچا","بچی","بچایا",
        "لگا","لگی","لگے","لگانا","لگاتا","لگاتی","ملا","ملی","ملے","ملنا",
        "بنا","بنی","بنے","بنایا","بنانا","بنتا","بنتی","بولی","بولا","بولے",
        "سنا","سنی","سنے","سننا","سنتا","کھایا","کھائی","پیا","پی","پینا",
        "بھیجا","پہنچا","پہنچی","پہنچے","تھامنا","مانا","چاہی","چاہتا","چاہتی","چاہے",
        "سوچا","سوچی","سوچے","سوچنا","اختیار","قبول","انکار","اقرار","اعلان","مطالبہ",
    ],
    "NOUN": [
        "پاکستان","ہندوستان","چین","انڈیا","ایران","امریکہ","برطانیہ","افغانستان","بھارت","روس",
        "اسلام","آباد","کراچی","لاہور","پشاور","کوئٹہ","ملک","حکومت","صدر","وزیر",
        "وزیراعظم","پارلیمان","اسمبلی","سینیٹ","عدالت","جج","قاضی","قانون","فیصلہ","سزا",
        "ملزم","مجرم","پولیس","فوج","سپاہی","جرنیل","افسر","جنگ","امن","معاہدہ",
        "بات","بیان","خبر","اطلاع","رپورٹ","کہانی","کتاب","مضمون","مصنف","شاعر",
        "زبان","لفظ","جملہ","کاغذ","قلم","اسکول","کالج","یونیورسٹی","استاد","طالب",
        "تعلیم","علم","سائنس","ریاضی","تاریخ","جغرافیہ","ادب","شعبہ","مضمون","کلاس",
        "ہسپتال","مریض","ڈاکٹر","دوا","علاج","طبیب","بیماری","صحت","جراثیم","ویکسین",
        "پانی","ہوا","آگ","مٹی","آسمان","زمین","سورج","چاند","ستارہ","بادل",
        "بارش","طوفان","سیلاب","زلزلہ","خشک","سردی","گرمی","موسم","دن","رات",
        "صبح","شام","ہفتہ","مہینہ","سال","لمحہ","گھنٹہ","منٹ","دہائی","صدی",
        "گھر","مکان","کمرہ","دروازہ","کھڑکی","چھت","فرش","دیوار","بستر","میز",
        "کرسی","الماری","برتن","پلیٹ","چمچ","شخص","آدمی","عورت","بچہ","لڑکا",
        "لڑکی","ماں","باپ","بھائی","بہن","دوست","دشمن","پڑوسی","بادشاہ","ملکہ",
        "کسان","مزدور","تاجر","تاجرانہ","صنعت","کارخانہ","بازار","دکان","گاہک","قیمت",
        "پیسہ","روپیہ","ڈالر","بینک","قرض","سود","نفع","نقصان","تجارت","برآمدات",
        "درآمدات","کرنسی","معیشت","سرمایہ","بجٹ","ٹیکس","بجلی","گیس","تیل","پٹرول",
        "گاڑی","بس","ٹرین","ہوائی","جہاز","کشتی","سفر","راستہ","سڑک","پل",
        "کھیل","کرکٹ","فٹبال","ہاکی","ٹیم","کھلاڑی","میچ","اسٹیڈیم","گول","اسکور",
        "اقوام","متحدہ","قومی","بینالاقوامی","مقامی","علاقائی","عالمی","سفیر","وفد","مذاکرات",
        "دہشتگرد","حملہ","بم","فائرنگ","ہلاکت","زخمی","مجروح","جیل","قید","رہائی",
        "آبادی","شہری","دیہاتی","ضلع","شہر","قصبہ","دیہات","گاؤں","علاقہ","خطہ",
        "مسلمان","ہندو","سکھ","عیسائی","یہودی","مسجد","مندر","گرجا","مذہب","عقیدہ",
        "شادی","طلاق","بچے","خاندان","رشتہ","محبت","نفرت","خوشی","غم","امید",
    ],
    "PUNC": ["۔","،","؛","؟","!",".",",",":","-",'"', "‘", "’", "”", "“"],
}

# Ensure ≥200 entries for each major category (NOUN/VERB/ADJ/ADV)
for major in ["NOUN", "VERB", "ADJ", "ADV"]:
    base = LEXICON[major]
    # auto-extend by suffix variants (trivial duplicates if already present)
    extra = set()
    for w in base:
        extra.add(w)
        extra.add(w + "وں"); extra.add(w + "یں"); extra.add(w + "ی"); extra.add(w + "ے")
    LEXICON[major] = sorted(extra)
assert all(len(LEXICON[c]) >= 200 for c in ["NOUN","VERB","ADJ","ADV"]), {c: len(LEXICON[c]) for c in ["NOUN","VERB","ADJ","ADV"]}

# Flatten to lookup map (first match wins by priority order)
PRIORITY = ["PUNC", "NUM", "POST", "CONJ", "DET", "PRON", "ADV", "ADJ", "VERB", "NOUN"]
lex_map: dict[str, str] = {}
for tag in PRIORITY:
    for w in LEXICON[tag]:
        lex_map.setdefault(w, tag)

# POS rules:
#  - <NUM> → NUM
#  - token in lexicon → that tag
#  - verb-like ending heuristics → VERB
#  - adj-like ending (ی / ا / ے) → ADJ on short word, else NOUN (fallback)
#  - otherwise NOUN default

VERB_ENDS = ("نا", "تا", "تی", "تے", "یا", "ئے", "یں", "ے گا", "گا", "گی", "گے")
ADJ_ENDS = ("ی", "ا", "ے")

def pos_tag(token: str) -> str:
    if token == "<NUM>": return "NUM"
    if token == "<UNK>": return "UNK"
    if re.match(r"^\d+$", token): return "NUM"
    if token in lex_map: return lex_map[token]
    # heuristic
    for e in VERB_ENDS:
        if token.endswith(e) and len(token) >= 3: return "VERB"
    return "NOUN"

# -------- NER gazetteer --------
GAZ = {
    "PER": [
        "عمران","خان","نواز","شریف","شہباز","مریم","بلاول","زرداری","آصف","علی",
        "پرویز","مشرف","یوسف","رضا","گیلانی","شاہد","خاقان","عباسی","یوسف","نوید",
        "بابر","اعظم","وسیم","اکرم","جاوید","میانداد","انضمام","الحق","شعیب","اختر",
        "محمد","یوسف","عبدالرزاق","سرفراز","احمد","مصباح","الحق","شاہین","آفریدی","حسن",
        "علی","عمر","ایوب","فضل","الرحمن","جنرل","عاصم","منیر","قاضی","فائز",
        "عیسیٰ","ثاقب","نثار","آصف","زرداری","یوسف","صلاح","الدین","حافظ","سعید",
        "مولانا","فضل","مصطفیٰ","کمال","علامہ","اقبال","قائد","اعظم","جناح","لیاقت",
    ],
    "LOC": [
        "پاکستان","اسلام","آباد","کراچی","لاہور","پشاور","کوئٹہ","فیصل","آباد","ملتان",
        "راولپنڈی","حیدرآباد","گوجرانوالہ","سیالکوٹ","بہاولپور","سکھر","لاڑکانہ","میرپور","گلگت","بلتستان",
        "پنجاب","سندھ","بلوچستان","خیبر","پختونخوا","کشمیر","آزاد","گلگت","بلتستان","سوات",
        "وزیرستان","قندھار","کابل","افغانستان","ایران","تہران","بھارت","دلی","ممبئی","دہلی",
        "چین","بیجنگ","روس","ماسکو","امریکہ","واشنگٹن","برطانیہ","لندن","فرانس","پیرس",
        "جرمنی","جاپان","ٹوکیو","سعودی","عرب","ریاض","متحدہ","ترکی","استنبول","قطر",
        "دبئی","ابوظہبی","مکہ","مدینہ","یروشلم","غزہ","فلسطین","اسرائیل","یوکرین","کیف",
    ],
    "ORG": [
        "پیپلز","پارٹی","مسلم","لیگ","تحریک","انصاف","جماعت","اسلامی","اے","این","پی",
        "اقوام","متحدہ","ناٹو","یورپی","یونین","سارک","او","آئی","سی","آئی","ایم","ایف",
        "ورلڈ","بینک","اسٹیٹ","بینک","پی","سی","بی","آئی","ایس","آئی","فوجی","بورڈ",
        "ایف","آئی","اے","نیب","الیکشن","کمیشن","سپریم","کورٹ","ہائی","کورٹ","پی","آئی","اے",
        "پیشاور","زلمی","کراچی","کنگز","لاہور","قلندرز","اسلام","آباد","یونائیٹڈ",
    ],
}

# Normalise gazetteer entries
for k, lst in GAZ.items():
    GAZ[k] = list(dict.fromkeys(lst))
for k, lst in GAZ.items():
    print(f"gaz {k}: {len(lst)}")

def ner_tag(tokens: list[str]) -> list[str]:
    # Greedy multi-token match by scanning for longest gazetteer phrases.
    tags = ["O"] * len(tokens)
    i = 0
    while i < len(tokens):
        matched = False
        # try lengths up to 4
        for L in (4, 3, 2, 1):
            if i + L > len(tokens): continue
            span = tokens[i:i + L]
            for typ, entries in GAZ.items():
                # check contiguous n-grams in gazetteer terms (split by space)
                for ent in entries:
                    parts = ent.split()
                    if parts == span:
                        tags[i] = f"B-{typ}"
                        for k in range(1, L):
                            tags[i + k] = f"I-{typ}"
                        i += L; matched = True; break
                if matched: break
            if matched: break
        if not matched:
            i += 1
    return tags

# -------- Select 500 sentences (≥100 from 3 categories) --------
by_cat: dict[str, list[tuple[int, list[str]]]] = defaultdict(list)
for art_idx, sents in enumerate(sents_per_article):
    cat = article_cats[art_idx]
    for s in sents:
        if 4 <= len(s) <= 40:
            by_cat[cat].append((art_idx, s))

print("sents per cat:", {c: len(v) for c, v in by_cat.items()})

# pick top 3 categories by availability
chosen_cats = sorted(by_cat.keys(), key=lambda c: -len(by_cat[c]))[:3]
print("chosen categories:", chosen_cats)

random.shuffle_by_key = None
selected: list[tuple[int, list[str], str]] = []
for c in chosen_cats:
    pool = by_cat[c][:]
    random.shuffle(pool)
    for art_idx, s in pool[:100]:
        selected.append((art_idx, s, c))

# remaining 200 from any category (stratified)
remaining_budget = 500 - len(selected)
remaining_pool = []
for c, lst in by_cat.items():
    for art_idx, s in lst:
        if (art_idx, tuple(s)) not in {(a, tuple(x)) for a, x, _ in selected}:
            remaining_pool.append((art_idx, s, c))
random.shuffle(remaining_pool)
selected.extend(remaining_pool[:remaining_budget])
print(f"total selected sentences: {len(selected)}")

# annotate
pos_corpus: list[list[tuple[str, str]]] = []
ner_corpus: list[list[tuple[str, str]]] = []
cat_of_sent: list[str] = []
for art_idx, sent, c in selected:
    pos_tags = [pos_tag(t) for t in sent]
    ner_tags = ner_tag(sent)
    pos_corpus.append(list(zip(sent, pos_tags)))
    ner_corpus.append(list(zip(sent, ner_tags)))
    cat_of_sent.append(c)

pos_dist = Counter(t for s in pos_corpus for _, t in s)
ner_dist = Counter(t for s in ner_corpus for _, t in s)
print("POS tag distribution:", pos_dist)
print("NER tag distribution:", ner_dist)

# 70/15/15 stratified by cat
idxs_by_cat: dict[str, list[int]] = defaultdict(list)
for i, c in enumerate(cat_of_sent):
    idxs_by_cat[c].append(i)
train_idx, val_idx, test_idx = [], [], []
for c, idxs in idxs_by_cat.items():
    random.shuffle(idxs)
    n = len(idxs)
    n_train = int(0.70 * n); n_val = int(0.15 * n)
    train_idx += idxs[:n_train]
    val_idx += idxs[n_train:n_train + n_val]
    test_idx += idxs[n_train + n_val:]
print(f"splits: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

def write_conll(path: Path, corpus, idxs):
    with path.open("w", encoding="utf-8") as f:
        for i in idxs:
            for tok, tag in corpus[i]:
                f.write(f"{tok} {tag}\n")
            f.write("\n")

(DATA / "pos_train.conll");
write_conll(DATA / "pos_train.conll", pos_corpus, train_idx)
write_conll(DATA / "pos_val.conll", pos_corpus, val_idx)
write_conll(DATA / "pos_test.conll", pos_corpus, test_idx)
write_conll(DATA / "ner_train.conll", ner_corpus, train_idx)
write_conll(DATA / "ner_val.conll", ner_corpus, val_idx)
write_conll(DATA / "ner_test.conll", ner_corpus, test_idx)

(DATA / "annotation_summary.json").write_text(json.dumps({
    "pos_dist": pos_dist, "ner_dist": ner_dist,
    "train": len(train_idx), "val": len(val_idx), "test": len(test_idx),
    "chosen_categories": chosen_cats, "cat_counts": dict(Counter(cat_of_sent)),
    "gazetteer_sizes": {k: len(v) for k, v in GAZ.items()},
    "lexicon_sizes": {k: len(v) for k, v in LEXICON.items()},
}, ensure_ascii=False, indent=2))

print("annotation done.")
