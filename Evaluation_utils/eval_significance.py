import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from service.ChatGPT import chat_simple as chat_simple_service

_SCORE_SYSTEM_PROMPT = """
You are a precise scorer. I will provide you with a professional evaluation of an academic idea,
and you need to give a significance score based on this evaluation. The significance score depends on the attitude of the evaluation.
If the evaluation is positive, the significance score should be high; if the evaluation is negative, the significance score should be low.
Please note that the significance score ranges from 1 to 10, where 1 indicates the lowest significance and 10 indicates the highest significance.
Here are some examples(Some specific method or idea is replaced with Method A, Method B, Method C, etc.):

Example 1 (Low score case, score 1):
Evaluation: "The idea is simple and straightforward, but I don't think it makes sense to me, as I explained in the previous sections."
Analysis: The reviewer expresses clear skepticism and a lack of conceptual understanding or confidence in the idea. There is no indication of potential interest or novelty. The tone is dismissive and final, typical of very low excitement.
Score: 1

Example 2 (Low–medium score case, score 2):
Evaluation: "The idea has been addressed by other papers, and the difference is marginal. I don't have any new (expected) conclusion learned from this proposal, thus not excited."
Analysis: The reviewer acknowledges that the idea exists but sees little originality or learning value. Some effort is recognized, but the contribution is incremental and uninspiring.
Score: 2

Example 3 (Medium score case, score 3):
Evaluation: "Just not an interesting area in my opinion. Scope is too narrow to be impactful and existing non-prompting methods will do much better."
Analysis: The reviewer finds the idea unexciting but not completely without merit. The main reason for the moderate score is lack of broad impact rather than poor quality. There is some room for exploration, but it’s not seen as a strong direction.
Score: 3

Example 4 (Medium–high score case, score 5):
Evaluation: "The approach could be useful for certain applications, though it may not be methodologically novel. It might still help understand bias mitigation effects in large models."
Analysis: The reviewer sees limited novelty but acknowledges practical potential. There is moderate enthusiasm due to possible usefulness, though not groundbreaking.
Score: 5

Example 5 (High score case, score 6):
Evaluation: "The trained models and methodology could be useful for specific systems. While not entirely novel, it shows practical direction and interesting potential."
Analysis: The reviewer is positive and sees genuine value and possible future utility, though the contribution is not revolutionary. The tone is constructive and moderately excited.
Score: 6

Example 6 (Very high score case, score 7–8):
Evaluation: "I think XXX are quite hard to process, so if it works, the proposed method would be very useful. This is a smart way to tackle an urgent challenge."
Analysis: The reviewer shows clear enthusiasm and recognizes strong relevance and potential impact. They believe the approach addresses an important open problem, leading to high excitement.
Score: 8


"""

def generate_significance_score(evaluation_text: str) -> str:
    prompt = (
        _SCORE_SYSTEM_PROMPT
        + "\n\n"
        + "So now please start scoring formally: give me a score between 1 and 10 based on the following evaluation: \n\n"
        + evaluation_text.strip()
        + "\n\nPlease return an analysis of the evaluation and the final scoring result. "
        + "IMPORTANT: You must format your response with the score clearly marked at the end. "
        + "Use the exact format: 'Score: X' where X is an integer between 1 and 10. "
        + "For example, if your score is 7, end your response with 'Score: 7'."
    )
    try:
        content = chat_simple_service(prompt)
    except Exception as e:
        return f"ERROR: {e}"

    return content
