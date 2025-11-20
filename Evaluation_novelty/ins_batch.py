import json
import multiprocessing as mp
import re
from typing import Optional
from tqdm import tqdm
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from Evaluation_significance.ins_single import generate_significance_evaluation_via_agent
from Evaluation_utils.eval_significance import generate_significance_score
from ins_model import create_custom_agent

# Global agent variable for multiprocessing
global_agent = None

def init_worker():
    """Initialize worker process, create agent"""
    global global_agent
    global_agent = create_custom_agent()

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
    
    Args:
        item_data: Dictionary containing the following fields:
            - id: Item ID
            - idea: Idea content (may be empty string)
            - all_comments: Comment content (optional)
            - all_scores: Original score (optional, format like "3 5")
    
    Returns:
        Dictionary containing processing results:
            - id: Item ID
            - status: Status ("success", "error", "skipped")
            - golden_score: Golden standard score (from all_scores field, format like "3 5")
            - significance_evaluation: Significance evaluation text
            - raw_resp_idea: Original evaluation response (same as significance_evaluation)
            - significance_score: Extracted score (integer 1-10, or None)
            - significance_score_raw_resp: Original scoring response text
            - error: Error message (if any)
    """
    global global_agent
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
                "significance_evaluation": "",
                "raw_resp_idea": None,
                "significance_score": None,
                "significance_score_raw_resp": None,
                "error": None
            }
        
        # Generate significance evaluation
        significance_evaluation = generate_significance_evaluation_via_agent(global_agent, raw_idea)
        print(f"[{item_id}] Significance evaluation generated!")
        
        # Check if evaluation generation failed
        if not significance_evaluation or significance_evaluation.startswith("ERROR"):
            return {
                "id": item_id,
                "status": "error",
                "golden_score": golden_score,
                "significance_evaluation": significance_evaluation or "",
                "raw_resp_idea": significance_evaluation if significance_evaluation else None,
                "significance_score": None,
                "significance_score_raw_resp": None,
                "error": "Error generating significance evaluation"
            }
        
        # Perform significance scoring
        significance_score_raw_resp = generate_significance_score(significance_evaluation)
        print(f"[{item_id}] Significance scoring completed!")
        
        # Extract score from response
        significance_score = extract_score_from_response(significance_score_raw_resp)
        
        # If score extraction fails, log warning but don't mark as error
        if significance_score is None:
            print(f"[{item_id}] Warning: Unable to extract score from response")

        return {
            "id": item_id,
            "status": "success",
            "golden_score": golden_score,
            "significance_evaluation": significance_evaluation,
            "raw_resp_idea": significance_evaluation,  # Original response is the evaluation text
            "significance_score": significance_score,
            "significance_score_raw_resp": significance_score_raw_resp,
            "error": None
        }
        
    except Exception as e:
        return {
            "id": item_data.get("id", "unknown"),
            "status": "error",
            "golden_score": item_data.get("all_scores", ""),
            "significance_evaluation": None,
            "raw_resp_idea": None,
            "significance_score": None,
            "significance_score_raw_resp": None,
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
    input_file = "Evaluation_significance/significance_data/Stanford_comments_with_ideas_and_scores.json"
    
    # Output directory
    output_dir = "Evaluation_significance/results/instructor"
    
    # Number of parallel processes
    num_processes = 10
    
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