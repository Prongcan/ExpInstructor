import os
import json
import time
import typing as t
from dataclasses import dataclass

# Ensure that the service package under the project root can be imported
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from service.ChatGPT import chat_simple as chat_simple_service
#from service.DistilBERT import build_significance_classifier, predict_significance_score
from Evaluation_utils.eval_significance import generate_significance_score
raw_idea = "Title: Divergent Thought Stream Amplification: Improving Factual Consistency and Reducing Hallucinations in Large Language Models\n\n1. Problem Statement: Large language models often struggle with maintaining factual consistency and reducing hallucinations when generating responses, especially for complex queries requiring multi-step reasoning. This issue can lead to the propagation of misinformation and reduce the reliability of AI-generated content.\n\n2. Motivation: Current approaches like chain-of-thought prompting focus on linear reasoning paths, which may not capture the full breadth of relevant information. Human cognition often involves considering multiple parallel lines of thought before converging on a solution. By mimicking this process, we can potentially improve the factual grounding and reduce hallucinations in language models. The proposed Divergent Thought Stream Amplification (DTSA) method aims to leverage the model's ability to generate multiple perspectives and critically evaluate them, leading to more robust and factually consistent outputs.\n\n3. Proposed Method: We propose Divergent Thought Stream Amplification (DTSA), a novel prompting technique that encourages the model to generate multiple parallel streams of thought before synthesizing a final response. The method consists of five main steps:\n\t(1) Present the query.\n\t(2) Instruct the model to generate three distinct thought streams, each approaching the problem from a different angle.\n\t(3) Ask the model to critically evaluate each thought stream, identifying potential factual inconsistencies or logical flaws.\n\t(4) Direct the model to synthesize the most factually consistent and logically sound elements from each stream into a final response.\n\t(5) Require the model to provide confidence scores for each fact in the final response, based on the consistency across thought streams.\n\n4. Step-by-Step Experiment Plan:\n\tStep 1: Dataset Preparation\n\t\t• TruthfulQA: A dataset designed to test the truthfulness of language models.\n\t\t• FEVER: A large-scale dataset for Fact Extraction and VERification.\n\t\t• HotpotQA: A dataset for complex multi-hop question answering, which will help evaluate the method's effectiveness on multi-step reasoning tasks.\n\tStep 2: Model Selection\n\t\t• GPT-3.5 (text-davinci-003) and GPT-4 from the OpenAI API\n\tStep 3: Baseline Implementation\n\t\t• Standard prompting: Directly asking the question without any additional instructions.\n\t\t• Chain-of-thought (CoT) prompting: Appending \"Let's approach this step-by-step:\" to the question.\n\t\t• Self-consistency method: Generate multiple CoT responses and select the most common answer.\n\tStep 4: DTSA Implementation\n\t\t• Implement the DTSA method with the specified prompt structure.\n\tStep 5: Evaluation Metrics\n\t\t• Accuracy: The percentage of correct answers.\n\t\t• Factual Consistency: Manual evaluation of a subset of responses to assess factual correctness.\n\t\t• Hallucination Rate: The percentage of responses containing information not supported by the given context or common knowledge.\n\t\t• Confidence Score Analysis: Evaluate the correlation between the model's confidence scores and actual correctness.\n\tStep 6: Experiment Execution\n\t\t• For each dataset and model combination:\n\t\t\t- Run the baseline methods and collect their outputs.\n\t\t\t- Run the DTSA method and collect its outputs.\n\t\t\t- Calculate the evaluation metrics for both baseline and DTSA outputs.\n\t\t\t- Perform statistical significance tests (e.g., paired t-test) to compare DTSA with each baseline method.\n\tStep 7: Analysis\n\t\t• Compare the performance of DTSA against baselines across different datasets and models.\n\t\t• Analyze the quality and diversity of the generated thought streams.\n\t\t• Examine the effectiveness of the critical evaluation step in identifying factual inconsistencies.\n\t\t• Assess the relationship between confidence scores and actual correctness.\n\t\t• Identify any patterns in the types of questions or topics where DTSA shows the most improvement.\n\tStep 8: Ablation Studies\n\t\t• Conduct ablation studies to understand the contribution of each component of DTSA:\n\t\t\t- Remove the multiple thought streams and use only one stream.\n\t\t\t- Remove the critical evaluation step.\n\t\t\t- Remove the confidence scoring step.\n\t\t• Compare the performance of these ablated versions with the full DTSA method.\n\n5. Test Case Examples:\n\tTest Case 1:\n\t\tBaseline Prompt Input (Standard Prompting): What is the capital of France, and in which year was the Eiffel Tower completed?\n\t\tBaseline Prompt Expected Output (Standard Prompting): The capital of France is Paris, and the Eiffel Tower was completed in 1889.\n\t\tProposed Prompt Input (DTSA): Question: What is the capital of France, and in which year was the Eiffel Tower completed?\n\t\t1) Generate three distinct thought streams to approach this question:\n\t\tStream 1:\n\t\tStream 2:\n\t\tStream 3:\n\t\t2) Critically evaluate each thought stream, identifying any potential factual inconsistencies or logical flaws:\n\t\tEvaluation of Stream 1:\n\t\tEvaluation of Stream 2:\n\t\tEvaluation of Stream 3:\n\t\t3) Synthesize the most factually consistent and logically sound elements from each stream into a final response:\n\t\tFinal Response:\n\t\t4) Provide confidence scores (0-100) for each fact in the final response, based on the consistency across thought streams:\n\t\tConfidence Scores:\n\t\tProposed Prompt Expected Output (DTSA):\n\t\t1) Generate three distinct thought streams to approach this question:\n\t\tStream 1: The capital of France is Paris, which is known as the \"City of Light.\" The Eiffel Tower, located in Paris, is an iconic landmark that was built for the 1889 World's Fair. It was completed in that same year, 1889.\n\t\tStream 2: France's capital city is Paris, situated on the Seine River. The Eiffel Tower, designed by Gustave Eiffel, was constructed between 1887 and 1889. The tower was inaugurated on March 31, 1889, marking its completion.\n\t\tStream 3: Paris is the capital and largest city of France. The Eiffel Tower, a wrought-iron lattice tower, is one of the most recognizable structures in the world. Its construction began in 1887 and was finished in 1889, coinciding with the centennial celebration of the French Revolution.\n\t\t2) Critically evaluate each thought stream, identifying any potential factual inconsistencies or logical flaws:\n\t\tEvaluation of Stream 1: This stream correctly identifies Paris as the capital of France and 1889 as the completion year of the Eiffel Tower. The information about the World's Fair adds context but is not directly relevant to the question.\n\t\tEvaluation of Stream 2: This stream also correctly states Paris as the capital and 1889 as the completion year. It provides additional details about the tower's designer and construction period, which are accurate but not asked for in the question.\n\t\tEvaluation of Stream 3: This stream corroborates the information in the previous two streams regarding Paris and the Eiffel Tower's completion year. It adds information about the French Revolution centennial, which is accurate and provides historical context.\n\t\t3) Synthesize the most factually consistent and logically sound elements from each stream into a final response:\n\t\tFinal Response: The capital of France is Paris. The Eiffel Tower, located in Paris, was completed in 1889. This iconic landmark was built for the 1889 World's Fair, which coincided with the centennial celebration of the French Revolution.\n\t\t4) Provide confidence scores (0-100) for each fact in the final response, based on the consistency across thought streams:\n\t\tConfidence Scores:\n\t\t- Paris as the capital of France: 100\n\t\t- Eiffel Tower completion in 1889: 100\n\t\t- Eiffel Tower built for 1889 World's Fair: 90\n\t\t- Coinciding with French Revolution centennial: 85\n\t\tExplanation: The DTSA method generates multiple thought streams, critically evaluates them, and synthesizes a final response with confidence scores. This approach allows for a more comprehensive and factually consistent answer compared to the standard prompting method. The confidence scores provide additional information about the reliability of each fact in the response.\n\n6. Fallback Plan: If the proposed DTSA method does not show significant improvements over the baselines, we can explore several alternatives. These include analyzing the generated thought streams to identify patterns in factual inconsistencies or logical flaws, experimenting with different numbers of thought streams to find the optimal balance between diversity of perspectives and computational efficiency, investigating the effectiveness of different prompting strategies for the critical evaluation step, exploring combinations of DTSA with other prompting techniques to create hybrid approaches, and conducting detailed error analysis to categorize the types of mistakes made by DTSA and the baselines. These analyses could inform the development of more targeted prompting strategies or post-processing techniques, providing insights into the model's reasoning process and potential areas for improvement."
 

# ================== Generate Idea Significance Evaluation ==================
_IDEA_SYSTEM_PROMPT = """
You are a professional evaluator focusing on the significance of the idea.
I will provide you with an academic idea. Your task is to evaluate only its level of significance.
Please focus exclusively on the importance, scope, and potential impact of the idea — how meaningful, influential, or valuable it would be compared to existing research or conventional approaches in the field.
Your response should:
Be concise and academic in tone.
Avoid discussing feasibility, novelty, or methodology.
Provide a clear judgment on the significance level with your serious analysis and reasoning.
"""


def generate_significance_evaluation(idea_text: str) -> str:
    prompt = (
        _IDEA_SYSTEM_PROMPT
        + "\n\n"
        + "Idea to review:\n\n"
        + idea_text.strip()
        + "\n\nReturn a text evaluating significance (which should include reasonable reasons)."
    )
    try:
        content = chat_simple_service(prompt)
    except Exception as e:
        return f"ERROR: {e}"

    return content

def main() -> None:
    print("[1/3] Generating significance evaluation...")
    result = generate_significance_evaluation(raw_idea)
    print(result)
    
    # Use GPT to evaluate the result of the significance evaluation
    print("\n[2/3] Using GPT model to evaluate score...")
    gpt_score_result = generate_significance_score(result)
    if gpt_score_result and not gpt_score_result.startswith("ERROR"):
        print(f"\nGPT score result:")
        print(gpt_score_result)
    else:
        print("GPT scoring failed")

if __name__ == "__main__":
    main()
