import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from service.ChatGPT import chat_simple as chat_simple_service

_SCORE_SYSTEM_PROMPT = """
You are a precise scorer. I will provide you with a professional evaluation of an academic idea,
and you need to give a novelty score based on this evaluation. The novelty score depends on the attitude of the evaluation.
If the evaluation is positive, the novelty score should be high; if the evaluation is negative, the novelty score should be low.
Please note that the novelty score ranges from 1 to 10, where 1 indicates the lowest novelty and 10 indicates the highest novelty.
Here are some examples(Some specific method or idea is replaced with Method A, Method B, Method C, etc.):

Example 1 (Low score case, score 1):
Evaluation: "The proposed method is not very different from Method A (Research A, 2023), Method B (Research B, 2023), and Method C (Research C, 2024). Especially, Method B also has an evaluation step in the pipeline, but on the node level."
Analysis: This evaluation clearly states that the proposed method is very similar to multiple existing works, with almost no novelty.
Score: 1

Example 2 (Low score case, score 2):
Evaluation: "Method X is a paper that already exists, focusing on a certain concept over multiple chains. It emphasizes divergent thinking rather than a linear thought structure concerning these inputs. Therefore, the proposed work does not appear to present novelty in terms of the prompt, the described structure, or the datasets on which it is tested on."
Analysis: The evaluation clearly states that related papers already exist, and the proposed work lacks novelty in multiple aspects.
Score: 2

Example 3 (Medium score case, score 6):
Evaluation: "Personally, I am not aware of similar works that describe a certain scenario where certain concepts are reversed. I do not find closely related works after a quick search using certain keywords. I'm fairly confident that the proposed approach is different from the existing works. After thinking further about the idea, I think it is similar to a certain method with some alternative approaches in the input prompt. However, I do not know similar papers off the top of my mind now."
Analysis: The evaluator acknowledges not finding similar works, but also mentions that it may be similar to some existing methods, showing a neutral attitude.
Score: 6

Example 4 (Medium-high score case, score 7):
Evaluation: "The proposed idea and framework of using a certain method with varied semantics, inspired by certain techniques from another field, which is clearly novel and makes major differences from all existing ideas. However, fundamentally, the notion of a certain approach by substituting similar concepts is very similar to existing studies, such as: (Research D, 2024). Therefore a score of 7 (between 6 and 8) is given."
Analysis: The evaluator believes some aspects are novel, but the core method is still similar to existing research, giving a medium-high score.
Score: 7

Example 5 (High score case, score 8):
Evaluation: "Combining Method A with Method B to improve a certain task for low-resource scenarios is a novel approach. While such hybrid methods have been explored in other contexts, their application to these specific forms is not widely covered, offering fresh insights and potential advancements in the field."
Analysis: The evaluator believes this is a novel method, and its application in this specific domain is new, showing a positive attitude.
Score: 8

Example 6 (High score case, score 10):
Evaluation: "While the framework of a certain method based on a certain technique is well known, the idea of trying to reach certain embedded concepts in models by bringing up pretty unrelated analogies about certain concepts in questions seems wildly novel! I would be very excited to see the results of this experiment."
Analysis: The evaluator uses strongly positive words such as "wildly novel" and "very excited", clearly expressing high recognition of the novelty.
Score: 10


"""

def generate_novelty_score(evaluation_text: str) -> str:
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
