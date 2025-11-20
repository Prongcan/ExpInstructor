import os
import json
import time
import typing as t
from dataclasses import dataclass

# 确保可以导入到项目根下的 service 包
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from service.ChatGPT import chat_simple as chat_simple_service
from Evaluation_utils.test_idea import concerns
from Evaluation_utils.test_idea import raw_idea
from Evaluation_utils.eval_feasibility import semantic_match_scores, compare_coverage_via_llm
# ================== 生成 idea concerns ==================
_IDEA_SYSTEM_PROMPT = """
You are a rigorous peer-reviewer.
Task: Critically evaluate the given idea/proposal and GENERATE potential 'concerns' of feasibility
(feasibility, feasibility doubts, missing evaluations).
Do NOT extract phrases from the text verbatim; instead, propose concerns based on your assessment.
Output Policy (STRICT):
- Return ONLY a JSON array of strings, starting with '[' and ending with ']'.
- Each item must be a single-line short sentence (no line breaks).
- Do NOT include any code fences, markdown, comments, labels, or extra text.
- No leading bullets, numbering, or trailing commas inside items.
- Aim for 8-12 high-quality, non-duplicative items covering: feasibility, feasibility doubts, missing evaluations.
"""


def _fallback_parse_list(text: str) -> t.List[str]:
    lines = [ln.strip(" -*\t") for ln in text.splitlines()]
    items = [ln for ln in lines if ln]
    return items


def generate_concerns_for_idea(idea_text: str) -> t.Tuple[t.List[str], str]:
    prompt = (
        _IDEA_SYSTEM_PROMPT
        + "\n\n"
        + "Idea to review:\n\n"
        + idea_text.strip()
        + "\n\nReturn JSON array only."
    )
    try:
        content = chat_simple_service(prompt)
    except Exception as e:
        return [], f"ERROR: {e}"

    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            concerns_llm = [str(x).strip() for x in parsed if str(x).strip()]
        else:
            concerns_llm = _fallback_parse_list(content)
    except Exception:
        concerns_llm = _fallback_parse_list(content)
    return concerns_llm, content

def main() -> None:
    print("[1/3] 生成 LLM concerns ...")
    gen_concerns, raw_resp_idea = generate_concerns_for_idea(raw_idea)
    print(f"生成数量: {len(gen_concerns)}")
    print(gen_concerns)

    print("[2/3] 语义向量匹配评估 ...")
    print(json.dumps({
        "generated_concerns": gen_concerns,
        "semantic_match": semantic_match_scores(concerns, gen_concerns)
    }, ensure_ascii=False, indent=2))

    print("[3/3] 原始的concern比对 ...")
    final = compare_coverage_via_llm(concerns ,gen_concerns)
    print(final)


if __name__ == "__main__":
    main()
