"""
AI text detection using a free HuggingFace model.

Uses "Hello-SimpleAI/chatgpt-detector-roberta" (or similar) which classifies
text as human-written vs AI-generated. This is the same category of model
that powers GPTZero / Originality.ai — open source, no API key.

Note: detection is approximate. Real GPTZero uses ensemble + perplexity
analysis + stylometry. We give it our best shot for $0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AIDetection:
    """Result of an AI text detection check."""
    ai_probability: float       # 0..1 — probability the text is AI-generated
    human_probability: float    # 0..1 — probability the text is human-written
    verdict: str                # "HUMAN" | "MIXED" | "AI-LIKELY"
    confidence: str             # "low" | "medium" | "high"
    model_name: str
    per_paragraph: list[dict]   # [{"text", "ai_prob", "verdict"}]


# Lazy-loaded detector
_detector = None
_model_name = "Hello-SimpleAI/chatgpt-detector-roberta-chinese"  # replaced if not Chinese


def _get_detector(model_name: str = "roberta-base-openai-detector"):
    """Lazy-load the AI detector. Picks a working model."""
    global _detector, _model_name
    if _detector is not None and model_name == _model_name:
        return _detector
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        # Try a few models in order of preference
        candidates = [
            "roberta-base-openai-detector",  # OpenAI detector based on RoBERTa
            "Hello-SimpleAI/chatgpt-detector-roberta",
        ]
        last_err = None
        for cand in candidates:
            try:
                tok = AutoTokenizer.from_pretrained(cand)
                mdl = AutoModelForSequenceClassification.from_pretrained(cand)
                mdl.eval()
                _detector = (tok, mdl)
                _model_name = cand
                return _detector
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        raise RuntimeError(f"no AI detector model could be loaded: {last_err}")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"AI detector setup failed: {e}")


def detect_ai_text(text: str, model_name: str = "roberta-base-openai-detector",
                   per_paragraph: bool = False) -> AIDetection:
    """Detect whether text is AI-generated.

    Returns AIDetection with overall and (optionally) per-paragraph probabilities.
    """
    import torch

    tok, mdl = _get_detector(model_name)

    def _classify(chunk: str) -> tuple[float, float]:
        """Return (ai_prob, human_prob) for a text chunk."""
        # RoBERTa can handle up to 512 tokens; truncate if longer
        ids = tok(chunk, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = mdl(**ids).logits[0]
        probs = torch.softmax(out, dim=-1).tolist()
        # The model labels depend on the model. For roberta-base-openai-detector:
        # 0 = human, 1 = machine (AI)
        if len(probs) == 2:
            human_p, ai_p = probs[0], probs[1]
        else:
            ai_p = probs[-1]
            human_p = 1 - ai_p
        return ai_p, human_p

    # Overall detection (use first 2000 chars to keep it fast)
    sample = text[:2000]
    ai_p, human_p = _classify(sample)

    if ai_p > 0.8:
        verdict = "AI-LIKELY"
        confidence = "high" if ai_p > 0.95 else "medium"
    elif ai_p > 0.5:
        verdict = "MIXED"
        confidence = "medium"
    else:
        verdict = "HUMAN"
        confidence = "high" if human_p > 0.95 else "medium"

    per_par = []
    if per_paragraph:
        for para in text.split("\n\n"):
            para = para.strip()
            if len(para) < 100:
                continue
            p_ai, p_hu = _classify(para[:1000])
            p_verdict = "AI-LIKELY" if p_ai > 0.8 else ("MIXED" if p_ai > 0.5 else "HUMAN")
            per_par.append({"text": para[:200] + "...", "ai_prob": round(p_ai, 3), "verdict": p_verdict})

    return AIDetection(
        ai_probability=round(ai_p, 3),
        human_probability=round(human_p, 3),
        verdict=verdict,
        confidence=confidence,
        model_name=_model_name,
        per_paragraph=per_par,
    )
