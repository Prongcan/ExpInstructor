import json
import os
import re
import multiprocessing as mp
from typing import Optional
from tqdm import tqdm
from functools import partial
from RAG_single import rag_pipeline_with_retrieval_system, generate_novelty_score
from RAG_single import EvidenceRetrievalSystem

# Global retrieval system variable for multiprocessing
global_retrieval_system = None

def init_worker():
    """Initialize worker process, create retrieval system"""
    global global_retrieval_system
    embeddings_dir = 'RAG_baseline_review_sentence'
    global_retrieval_system = EvidenceRetrievalSystem(embeddings_dir)


def extract_score_from_response(response_text: str) -> Optional[int]:
    """
    Extract score from GPT response text
    
    Args:
        response_text: GPT returned scoring response text, should contain "Score: X" format
        
    Returns:
        int: Extracted score (1-10), returns None if extraction fails
    """
    if not response_text or response_text.startswith("ERROR"):
        return None
    
    # Method 1: Find "Score: X" format (case-insensitive)
    # Support multiple formats: "Score: 5", "score: 5", "Score:5", "Score 5", etc.
    patterns = [
        r'Score:\s*(\d+)',  # Score: 5
        r'score:\s*(\d+)',  # score: 5
        r'Score\s*:\s*(\d+)',  # Score : 5
        r'Score\s+(\d+)',  # Score 5
        r'分数[：:]\s*(\d+)',  # Chinese format: 分数: 5
        r'最终得分[：:]\s*(\d+)',  # Chinese format: 最终得分: 5
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            # Ensure score is within valid range (1-10)
            if 1 <= score <= 10:
                return score
    
    # Method 2: If no explicit "Score:" marker is found, try to find a single number at the end of the text
    # This is a fallback option, as some responses may have non-standard formats
    lines = response_text.strip().split('\n')
    if lines:
        # Check if there are numbers in the last few lines
        for line in reversed(lines[-3:]):  # Check the last 3 lines
            numbers = re.findall(r'\b(\d+)\b', line)
            if numbers:
                # Take the last number and check if it's within valid range
                for num_str in reversed(numbers):
                    num = int(num_str)
                    if 1 <= num <= 10:
                        return num
    
    # If nothing is found, return None
    return None

def process_single_item(item_data):
    """
    Function to process a single data item
    Each process will run this function
    """
    global global_retrieval_system
    try:
        item_id = item_data.get("id", "unknown")
        raw_idea = item_data.get("idea", "")
        golden_score = item_data.get("all_scores", "")  # Get golden standard score
        
        # Check if idea is empty
        if not raw_idea or raw_idea.strip() == "":
            return {
                "id": item_id,
                "status": "skipped",
                "golden_score": golden_score,
                "novelty_evaluation": "",
                "raw_resp_idea": None,
                "novelty_score": None,
                "novelty_score_raw_resp": None,
                "error": None
            }
        
        # Use global retrieval system to generate novelty_result
        embeddings_dir = 'RAG_baseline_review_sentence'
        novelty_evaluation = rag_pipeline_with_retrieval_system(raw_idea, embeddings_dir, global_retrieval_system)

        # Perform novelty scoring
        novelty_score_raw_resp = generate_novelty_score(novelty_evaluation)
        print(f"[{item_id}] Novelty scoring completed!")
        
        # Extract score from response
        novelty_score = extract_score_from_response(novelty_score_raw_resp)
        
        # If score extraction fails, log warning but don't mark as error
        if novelty_score is None:
            print(f"[{item_id}] Warning: Unable to extract score from response")

        return {
            "id": item_id,
            "status": "success",
            "golden_score": golden_score,
            "novelty_evaluation": novelty_evaluation,
            "raw_resp_idea": novelty_evaluation,  # Original response is the evaluation text
            "novelty_score": novelty_score,
            "novelty_score_raw_resp": novelty_score_raw_resp,
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
    Process data items in parallel using multiprocessing
    """
    with mp.Pool(processes=num_processes, initializer=init_worker) as pool:
        results = list(tqdm(
            pool.imap(process_single_item, data_items),
            total=len(data_items),
            desc="Processing items",
            unit="item"
        ))
    return results


def save_results(results, output_dir):
    """
    Save results to specified directory
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full results
    full_results_file = os.path.join(output_dir, "full_results.json")
    with open(full_results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Save successful results
    success_results = [r for r in results if r["status"] == "success"]
    success_file = os.path.join(output_dir, "success_results.json")
    with open(success_file, 'w', encoding='utf-8') as f:
        json.dump(success_results, f, ensure_ascii=False, indent=2)
    
    # Save error results
    error_results = [r for r in results if r["status"] == "error"]
    error_file = os.path.join(output_dir, "error_results.json")
    with open(error_file, 'w', encoding='utf-8') as f:
        json.dump(error_results, f, ensure_ascii=False, indent=2)
    
    # Save skipped results
    skipped_results = [r for r in results if r["status"] == "skipped"]
    skipped_file = os.path.join(output_dir, "skipped_results.json")
    with open(skipped_file, 'w', encoding='utf-8') as f:
        json.dump(skipped_results, f, ensure_ascii=False, indent=2)
    
    # Generate statistics report
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
    input_file = "ord_comments_with_ideas_with_scores.json"
    
    # Output directory
    output_dir = "Evaluation_novelty/results/RAG_evidence_retrieval"
    
    # Number of parallel processes
    num_processes = 5
    
    # Read JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data_items = json.load(f)
    
    # Process all data items in parallel
    results = parallel_process_items(data_items, num_processes)
    
    # Save results
    stats = save_results(results, output_dir)
    
    # Output processing statistics
    print(f"Processing completed!")
    print(f"Total items: {stats['total_items']}")
    print(f"Successfully processed: {stats['successful']}")
    print(f"Processing errors: {stats['errors']}")
    print(f"Skipped items: {stats['skipped']}")
    print(f"Success rate: {stats['success_rate']:.2%}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
