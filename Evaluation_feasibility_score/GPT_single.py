#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT model generates feasibility evaluation text (concerns)
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from service.ChatGPT import chat_simple as chat_simple_service

# ================== Generate complete peer review evaluation text ==================
_IDEA_SYSTEM_PROMPT = """
You are a rigorous peer-reviewer evaluating the feasibility of an academic research idea.
Task: Critically evaluate the given idea/proposal and write a comprehensive peer review evaluation text.

Your response should:
- Be a continuous, natural text similar to a peer review comment (like "all_comments" in academic reviews)
- Discuss feasibility, implementation challenges, effectiveness, and potential issues
- Be concise and academic in tone
- Provide a balanced evaluation covering both positive aspects and concerns
- Include your assessment of:
  * How easy or difficult it is to implement the idea
  * Whether the experimental setup is feasible
  * Whether the method is likely to work effectively
  * Any resource requirements or challenges
  * Comparison with existing approaches if relevant
- Write in a natural, flowing style as if you are providing feedback to the authors
"""


def generate_feasibility_evaluation(idea_text: str) -> str:
    """
    Generate complete peer review evaluation text (similar to all_comments)
    
    Args:
        idea_text: Research idea text to evaluate
        
    Returns:
        str: Complete peer review evaluation text
    """
    prompt = (
        _IDEA_SYSTEM_PROMPT
        + "\n\n"
        + "Idea to review:\n\n"
        + idea_text.strip()
        + "\n\nWrite a comprehensive peer review evaluation text discussing the feasibility, implementation challenges, effectiveness, and any concerns about this idea. Write in a natural, flowing style as if you are providing feedback to the authors."
    )
    try:
        content = chat_simple_service(prompt)
    except Exception as e:
        return f"ERROR: {e}"

    return content


def main():
    """Test function"""
    from Evaluation_utils.test_idea import raw_idea
    
    print("[1/2] Generating feasibility evaluation text ...")
    result = generate_feasibility_evaluation(raw_idea)
    print(result)
    
    # Use eval_feasibility_score.py to score the evaluation result
    print("\n[2/2] Using eval_feasibility_score.py to evaluate score ...")
    from Evaluation_utils.eval_feasibility_score import generate_feasibility_score
    
    # Directly use evaluation text (no formatting needed)
    score_result = generate_feasibility_score(result)
    if score_result and not score_result.startswith("ERROR"):
        print(f"\nScoring result:")
        print(score_result)
    else:
        print("Scoring failed")


if __name__ == "__main__":
    main()

