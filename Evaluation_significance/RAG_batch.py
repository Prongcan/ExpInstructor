import json
import os
import re
import multiprocessing as mp
from typing import Optional
from tqdm import tqdm
from functools import partial
from RAG_single import rag_pipeline_with_retrieval_system, generate_significance_score
from RAG_single import EvidenceRetrievalSystem

# Global retrieval system variable for multiprocessing
global_retrieval_system = None

def init_worker():
    """Initializes the worker process, creates the retrieval system"""
    global global_retrieval_system
    embeddings_dir = 'RAG_baseline_review_sentence'
    global_retrieval_system = EvidenceRetrievalSystem(embeddings_dir)


def extract_score_from_response(response_text: str) -> Optional[int]:
    """
    Extracts the score from the GPT response text.

    Args:
        response_text: The scoring response text returned by GPT, which should contain the "Score: X" format.

    Returns:
        int: The extracted score (1-10), or None if extraction fails.
    """
    if not response_text or response_text.startswith("ERROR"):
        return None
    
    # Method 1: Find "Score: X" format (case-insensitive)
    # Supports multiple formats: "Score: 5", "score: 5", "Score:5", "Score 5", etc.
    patterns = [
        r'Score:\s*(\d+)',  # Score: 5
        r'score:\s*(\d+)',  # score: 5
        r'Score\s*:\s*(\d+)',  # Score : 5
        r'Score\s+(\d+)',  # Score 5
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            # Ensure the score is within the valid range (1-10)
            if 1 <= score <= 10:
                return score

    # Method 2: If no clear "Score:" tag is found, try to find a single number at the end of the text
    # This is a fallback, as some responses may not be in a standard format
    lines = response_text.strip().split('\n')
    if lines:
        # Check the last few lines for numbers
        for line in reversed(lines[-3:]):  # Check the last 3 lines
            numbers = re.findall(r'\b(\d+)\b', line)
            if numbers:
                # Take the last number and check if it is within the valid range
                for num_str in reversed(numbers):
                    num = int(num_str)
                    if 1 <= num <= 10:
                        return num

    # If nothing is found, return None
    return None

def process_single_item(item_data):
    """
    Processes a single data item.
    Each process will run this function.
    """
    global global_retrieval_system
    try:
        item_id = item_data.get("id", "unknown")
        raw_idea = item_data.get("idea", "")
        golden_score = item_data.get("all_scores", "")  # Get the golden standard score

        # Check if the idea is empty
        if not raw_idea or raw_idea.strip() == "":
            return {
                "id": item_id,
                "status": "skipped",
                "golden_score": golden_score,
                "significance_evaluation": "",
                "raw_resp_idea": None,
                "significance_score": None,
                "significance_score_raw_resp": None,
                "error": None
            }

        # Use the global retrieval system to generate significance_result
        embeddings_dir = 'RAG_baseline_review_sentence'
        significance_evaluation = rag_pipeline_with_retrieval_system(raw_idea, embeddings_dir, global_retrieval_system)

        # Perform significance scoring
        significance_score_raw_resp = generate_significance_score(significance_evaluation)
        print(f"[{item_id}] Significance score calculated!")

        # Extract the score from the response
        significance_score = extract_score_from_response(significance_score_raw_resp)

        # If score extraction fails, log a warning but do not mark as an error
        if significance_score is None:
            print(f"[{item_id}] Warning: Failed to extract score from response")

        return {
            "id": item_id,
            "status": "success",
            "golden_score": golden_score,
            "significance_evaluation": significance_evaluation,
            "raw_resp_idea": significance_evaluation,  # The original response is the evaluation text
            "significance_score": significance_score,
            "significance_score_raw_resp": significance_score_raw_resp,
            "error": None
        }
        
    except Exception as e:
        return {
            "id": item_data.get("id", "unknown"),
            "status": "error",
            "reason": None,
            "gen_concerns": [],
            "coverage_result": None,
            "error": str(e)
        }


def parallel_process_items(data_items, num_processes=4):
    """
    Processes data items in parallel using multiple processes.
    """
    with mp.Pool(processes=num_processes, initializer=init_worker) as pool:
        results = list(tqdm(
            pool.imap(process_single_item, data_items),
            total=len(data_items),
            desc="Processing data items",
            unit="item"
        ))
    return results


def save_results(results, output_dir):
    """
    Saves the results to the specified directory.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the full results
    full_results_file = os.path.join(output_dir, "full_results.json")
    with open(full_results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save the successful results
    success_results = [r for r in results if r["status"] == "success"]
    success_file = os.path.join(output_dir, "success_results.json")
    with open(success_file, 'w', encoding='utf-8') as f:
        json.dump(success_results, f, ensure_ascii=False, indent=2)

    # Save the error results
    error_results = [r for r in results if r["status"] == "error"]
    error_file = os.path.join(output_dir, "error_results.json")
    with open(error_file, 'w', encoding='utf-8') as f:
        json.dump(error_results, f, ensure_ascii=False, indent=2)

    # Save the skipped results
    skipped_results = [r for r in results if r["status"] == "skipped"]
    skipped_file = os.path.join(output_dir, "skipped_results.json")
    with open(skipped_file, 'w', encoding='utf-8') as f:
        json.dump(skipped_results, f, ensure_ascii=False, indent=2)

    # Generate a statistical report
    stats = {
        "total_items": len(results),
        "successful": len(success_results),
        "errors": len(error_results),
        "skipped": len(skipped_results),
        "success_rate": len(success_results) / len(results) if results else 0
    }
    
    stats_file = os.path.join(output_dir, "processing_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    return stats


def main():
    # Input file path
    input_file = "Evaluation_significance/significance_data/Stanford_comments_with_ideas_and_scores.json"

    # Output directory
    output_dir = "Evaluation_significance/results/RAG_evidence_retrieval"

    # Number of parallel processes
    num_processes = 5

    # Read JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data_items = json.load(f)

    # Process all data items in parallel
    results = parallel_process_items(data_items, num_processes)

    # Save results
    stats = save_results(results, output_dir)
    
    # Print processing statistics
    print(f"Processing complete!")
    print(f"Total items: {stats['total_items']}")
    print(f"Successfully processed: {stats['successful']}")
    print(f"Processing errors: {stats['errors']}")
    print(f"Skipped items: {stats['skipped']}")
    print(f"Success rate: {stats['success_rate']:.2%}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
