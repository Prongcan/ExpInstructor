import os
import json
import time
import typing as t
from dataclasses import dataclass

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from service.ChatGPT import chat_simple as chat_simple_service
from service.ChatGPT import embed_texts
from service.deepseek import chat_deepseek

_COMPARE_SYSTEM_PROMPT = """
You are a precise evaluator. You are currently dealing with the opinions of two reviewers, 'original' (gold) and 'generated' (model output).
Your task is to explain whether each point in the gold standard has been reflected in the "generated" content.

Now decide for each ORIGINAL item whether it is covered by any GENERATED item under a RELAXED criterion:
- Mark as covered if the overall meaning is similar, paraphrased, or broadly aligned (approximate semantic similarity),
  If two or more generated concerns jointly express one concern, it is also considered covered. But don't be too loose.

Thinking and reasoning Requirement (STRICT):
- Process ORIGINAL concerns sequentially, one-by-one.
- For each ORIGINAL item, carefully check all GENERATED items and determine matches.
- Perform your reasoning internally.
- Only include a concise, one-sentence justification in the "reason" field per item.

Output Policy (STRICT):
- Return ONLY a JSON object, starting with '{' and ending with '}'.
- Keys must be exactly: per_item (array), summary (object).
- Each per_item element: {"original": string, "covered": bool, "matched_indices": [int], "reason": string}.
- summary: {"covered_count": int, "total": int, "coverage_ratio": number}.
- Do NOT include code fences, markdown, comments, or extra explanatory text.
"""


def compare_coverage_via_llm(original: t.List[str], generated: t.List[str]) -> t.Tuple[dict, str]:
    data = {
        "original": original,
        "generated": generated,
        "instructions": "Mark covered if a generated item clearly expresses the same concern."
    }
    user_prompt = (
        "Original concerns (gold):\n"
        + json.dumps(original, ensure_ascii=False)
        + "\n\nGenerated concerns:\n"
        + json.dumps(generated, ensure_ascii=False)
        + "\n\nReturn JSON only."
    )
    try:
        content = chat_deepseek(_COMPARE_SYSTEM_PROMPT + "\n\n" + user_prompt)
    except Exception as e:
        return {
            "per_item": [
                {"original": o, "covered": False, "matched_indices": [], "reason": f"evaluator error: {e}"}
                for o in original
            ],
            "summary": {"covered_count": 0, "total": len(original), "coverage_ratio": 0.0}
        }, f"ERROR: {e}"

    try:
        obj = json.loads(content)
        return obj, content
    except Exception:
        fallback = {
            "per_item": [
                {"original": o, "covered": False, "matched_indices": [], "reason": "fallback-no-parse"}
                for o in original
            ],
            "summary": {"covered_count": 0, "total": len(original), "coverage_ratio": 0.0}
        }
        return fallback, content


def cosine_similarity(a: t.List[float], b: t.List[float]) -> float:
    import math
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def semantic_match_scores(original: t.List[str], generated: t.List[str], model: str = "text-embedding-3-small") -> dict:
    """return
    {
      "per_generated": [{"text": str, "best_match_index": int, "best_score": float}],
      "summary": {"avg_best": float, "max_best": float}
    }
    """
    if not generated:
        return {"per_generated": [], "summary": {"avg_best": 0.0, "max_best": 0.0}}

    emb_gen = embed_texts(generated, model=model)
    emb_ori = embed_texts(original, model=model) if original else []

    per = []
    best_scores = []
    for gi, g_vec in enumerate(emb_gen):
        if not emb_ori:
            per.append({"text": generated[gi], "best_match_index": -1, "best_score": 0.0})
            best_scores.append(0.0)
            continue
        scores = [cosine_similarity(g_vec, o_vec) for o_vec in emb_ori]
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best = scores[best_idx]
        per.append({"text": generated[gi], "best_match_index": best_idx, "best_score": best})
        best_scores.append(best)

    avg_best = sum(best_scores) / len(best_scores) if best_scores else 0.0
    max_best = max(best_scores) if best_scores else 0.0
    return {"per_generated": per, "summary": {"avg_best": avg_best, "max_best": max_best}}
