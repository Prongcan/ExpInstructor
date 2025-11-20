import json
import os
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
from RAG_single import rag_pipeline_with_retrieval_system, compare_coverage_via_llm
from RAG_single import EvidenceRetrievalSystem

# Global retrieval system variable for multiprocessing
global_retrieval_system = None

def init_worker():
    """Initialize worker process and create the retrieval system"""
    global global_retrieval_system
    embeddings_dir = 'RAG_baseline_review_sentence'
    global_retrieval_system = EvidenceRetrievalSystem(embeddings_dir)

def process_single_item(item_data):
    """
    Function to process a single data item.
    Each process will run this function.
    """
    global global_retrieval_system
    try:
        item_id = item_data["id"]
        raw_idea = item_data["idea"]
        concerns = item_data["concerns"]
        
        # Check if idea is empty
        if not raw_idea or raw_idea.strip() == "":
            return {
                "id": item_id,
                "status": "skipped",
                "reason": "empty_idea",
                "gen_concerns": [],
                "coverage_result": None,
                "error": None
            }
        
        # Use the global retrieval system to generate LLM concerns
        embeddings_dir = 'RAG_baseline_review_sentence'
        gen_concerns = rag_pipeline_with_retrieval_system(raw_idea, embeddings_dir, global_retrieval_system)
        print(f"Generated count: {len(gen_concerns)}")
        
        # Compare concerns coverage
        coverage_result, coverage_raw_resp = compare_coverage_via_llm(concerns, gen_concerns)
        print("Coverage computed!")

        return {
            "id": item_id,
            "status": "success",
            "reason": None,
            "gen_concerns": gen_concerns,
            "coverage_result": coverage_result,
            "coverage_raw_resp": coverage_raw_resp,
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
    Process data items in parallel using multiprocessing.
    """
    with mp.Pool(processes=num_processes, initializer=init_worker) as pool:
        results = list(tqdm(
            pool.imap(process_single_item, data_items),
            total=len(data_items),
            desc="Processing items",
            unit="items"
        ))
    return results


def save_results(results, output_dir):
    """
    Save results to the specified directory
    """
    # Ensure the output directory exists
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
    
    # Generate statistics
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
    input_file = "Evaluation_feasibility/feasibility_data/Stanford_comments_with_ideas_with_concerns.json"
    
    # Output directory
    output_dir = "Evaluation_feasibility/results/RAG_evidence_retrieval"
    
    # Number of parallel processes
    num_processes = 5
    
    # Read JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data_items = json.load(f)
    
    # Process all items in parallel
    results = parallel_process_items(data_items, num_processes)
    
    # Save results
    stats = save_results(results, output_dir)
    
    # Print processing statistics
    print(f"Processing completed!")
    print(f"Total items: {stats['total_items']}")
    print(f"Succeeded: {stats['successful']}")
    print(f"Errors: {stats['errors']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Success rate: {stats['success_rate']:.2%}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
