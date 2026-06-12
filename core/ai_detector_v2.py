"""
Enhanced AI text detection: ensemble of multiple signals.

v3 — better balanced weights and short-text handling.

Methods:
  1. RoBERTa GPT-2 output detector (primary, 70% weight)
  2. Perplexity via distilgpt2 (20% weight, only if text > 50 words)
  3. Stylometry features (10% weight, only if text > 100 words)

The key insight from v2: perplexity and stylometry add noise on short texts.
Trust RoBERTa most; use other signals as confirmatory.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Stylometry features (with safer defaults for short text)
# ---------------------------------------------------------------------------
@dataclass
class StylometryFeatures:
    avg_sentence_length: float
    sentence_length_std: float
    vocab_diversity: float
    avg_word_length: float
    punctuation_density: float
    rare_word_ratio: float
    ngram_repetition: float
    passive_voice_estimate: float
    hedging_words_ratio: float


def extract_features(text: str) -> StylometryFeatures:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if len(s) > 5]
    if not sentences:
        sentences = [text]
    sent_lengths = [len(s.split()) for s in sentences]
    avg_len = sum(sent_lengths) / len(sent_lengths)
    var_len = sum((l - avg_len) ** 2 for l in sent_lengths) / len(sent_lengths)
    std_len = math.sqrt(var_len)
    words = re.findall(r"\w+", text.lower())
    if not words:
        words = [""]
    unique = set(words)
    vocab_div = len(unique) / len(words)
    avg_word_len = sum(len(w) for w in words) / len(words)
    punct = sum(1 for c in text if c in ".,;:!?\"'()")
    punct_density = punct / max(len(text), 1)
    word_counts = Counter(words)
    rare = sum(1 for w, c in word_counts.items() if c == 1)
    rare_ratio = rare / max(len(word_counts), 1)
    ngrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    ngram_counts = Counter(ngrams)
    repeats = sum(c - 1 for c in ngram_counts.values() if c > 1)
    ngram_repetition = repeats / max(len(ngrams), 1)
    passive = sum(1 for w in words if w in {"is", "are", "was", "were", "be", "been", "being"})
    passive_estimate = passive / max(len(words), 1)
    hedges = {"might", "could", "may", "perhaps", "possibly", "arguably", "seemingly",
              "likely", "presumably", "apparently", "essentially", "fundamentally",
              "essentially", "notably", "importantly", "significantly"}
    hedge_count = sum(1 for w in words if w in hedges)
    hedge_ratio = hedge_count / max(len(words), 1)
    return StylometryFeatures(
        avg_sentence_length=avg_len, sentence_length_std=std_len,
        vocab_diversity=vocab_div, avg_word_length=avg_word_len,
        punctuation_density=punct_density, rare_word_ratio=rare_ratio,
        ngram_repetition=ngram_repetition, passive_voice_estimate=passive_estimate,
        hedging_words_ratio=hedge_ratio,
    )


def ai_score_from_features(features: StylometryFeatures) -> float:
    """Convert features to AI probability using stricter heuristics."""
    score = 0.5
    if features.sentence_length_std < 5:
        score += 0.15
    elif features.sentence_length_std > 12:
        score -= 0.15
    if features.hedging_words_ratio > 0.04:
        score += 0.20
    elif features.hedging_words_ratio < 0.005:
        score -= 0.10
    if features.ngram_repetition > 0.20:
        score += 0.10
    elif features.ngram_repetition < 0.05:
        score -= 0.10
    if features.rare_word_ratio < 0.35:
        score += 0.10
    elif features.rare_word_ratio > 0.55:
        score -= 0.10
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Perplexity (only for text > 50 words)
# ---------------------------------------------------------------------------
def compute_perplexity(text: str, model_name: str = "distilgpt2") -> float:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        global _ppl_model, _ppl_tok, _ppl_model_name
        try:
            if _ppl_model_name != model_name:
                raise AttributeError
        except NameError:
            _ppl_model = None
            _ppl_tok = None
        if _ppl_model is None:
            _ppl_tok = AutoTokenizer.from_pretrained(model_name)
            _ppl_model = AutoModelForCausalLM.from_pretrained(model_name)
            _ppl_model_name = model_name
            _ppl_model.eval()
        ids = _ppl_tok(text[:2000], return_tensors="pt",
                       truncation=True, max_length=512).input_ids
        if ids.shape[1] < 2:
            return float("nan")
        with torch.no_grad():
            outputs = _ppl_model(ids, labels=ids)
        return float(torch.exp(outputs.loss).item())
    except Exception:  # noqa: BLE001
        return float("nan")


def ai_score_from_perplexity(perplexity: float) -> float:
    if math.isnan(perplexity) or math.isinf(perplexity):
        return 0.5
    if perplexity < 25:
        return 0.95
    if perplexity < 50:
        return 0.75
    if perplexity < 100:
        return 0.45
    if perplexity < 200:
        return 0.25
    return 0.10


# ---------------------------------------------------------------------------
# Combined detector (v3 — RoBERTa-heavy)
# ---------------------------------------------------------------------------
@dataclass
class EnhancedAIDetection:
    ai_probability: float
    human_probability: float
    verdict: str
    confidence: str
    model_name: str
    signals: dict
    per_paragraph: list[dict]


def _get_roberta_score(text: str) -> tuple[float, str]:
    try:
        from .ai_detector import detect_ai_text
        r = detect_ai_text(text, per_paragraph=False)
        return r.ai_probability, r.model_name
    except Exception:  # noqa: BLE001
        return 0.5, "roberta-failed"


def detect_ai_enhanced(text: str, *, use_perplexity: bool = True,
                       use_stylometry: bool = True,
                       use_roberta: bool = True) -> EnhancedAIDetection:
    """Enhanced AI detection — RoBERTa weighted 70%, others 15% each.

    Other signals only contribute if text is long enough to be reliable.
    """
    signals = {}
    word_count = len(text.split())

    # 1. RoBERTa (always, weight 70%)
    roberta_p = 0.5
    roberta_name = "n/a"
    if use_roberta:
        roberta_p, roberta_name = _get_roberta_score(text)
        signals["roberta"] = round(roberta_p, 3)

    # 2. Perplexity (only if > 50 words, weight 15%)
    ppl_p = roberta_p  # default to roberta (no signal)
    if use_perplexity and word_count > 50:
        ppl = compute_perplexity(text)
        if not math.isnan(ppl) and not math.isinf(ppl):
            ppl_p = ai_score_from_perplexity(ppl)
            signals["perplexity"] = round(ppl, 1)
            signals["perplexity_score"] = round(ppl_p, 3)

    # 3. Stylometry (only if > 100 words, weight 15%)
    sty_p = roberta_p  # default to roberta
    if use_stylometry and word_count > 100:
        feat = extract_features(text)
        sty_p = ai_score_from_features(feat)
        signals["stylometry_score"] = round(sty_p, 3)
        signals["sentence_length_std"] = round(feat.sentence_length_std, 2)
        signals["hedging_ratio"] = round(feat.hedging_words_ratio, 4)

    # Weighted ensemble — RoBERTa heavy
    ai_p = 0.70 * roberta_p + 0.15 * ppl_p + 0.15 * sty_p
    ai_p = max(0.0, min(1.0, ai_p))
    human_p = 1.0 - ai_p

    if ai_p > 0.70:
        verdict = "AI-LIKELY"
        confidence = "high" if ai_p > 0.85 else "medium"
    elif ai_p > 0.45:
        verdict = "MIXED"
        confidence = "medium"
    else:
        verdict = "HUMAN"
        confidence = "high" if human_p > 0.85 else "medium"

    return EnhancedAIDetection(
        ai_probability=round(ai_p, 3),
        human_probability=round(human_p, 3),
        verdict=verdict, confidence=confidence,
        model_name=f"v3({roberta_name}+ppl+sty)",
        signals=signals, per_paragraph=[],
    )
