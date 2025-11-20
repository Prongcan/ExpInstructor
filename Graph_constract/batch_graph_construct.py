import sys
import os
import json
import re
import logging
import time
from datetime import datetime
from pathlib import Path
import argparse
from typing import Dict, List, Optional, Tuple
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add parent directory to path to import service
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service import ChatGPT

def setup_logging(log_file: str = None) -> logging.Logger:
    """Configure logging"""
    if log_file is None:
        log_file = f"batch_graph_construction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def extract_json_from_text(text: str) -> Optional[Dict]:
    """Extract JSON content from model output text"""
    # Locate content between ```json and ```
    json_pattern = r'```json\s*(.*?)\s*```'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return None
    else:
        print("No JSON block found")
        return None

def create_id_to_name_mapping(node_data: List[Dict]) -> Dict[str, str]:
    """Build an id-to-name mapping"""
    id_to_name = {}
    for node in node_data:
        if 'id' in node and 'name' in node:
            id_to_name[node['id']] = node['name']
    return id_to_name

def replace_ids_with_names(edge_data: List[Dict], id_to_name_mapping: Dict[str, str]) -> List[Dict]:
    """Replace edge ids with names and rename fields"""
    processed_edges = []
    for edge in edge_data:
        processed_edge = {}
        for key, value in edge.items():
            if key == 'source_id':
                processed_edge['source_name'] = id_to_name_mapping.get(value, value)
            elif key == 'target_id':
                processed_edge['target_name'] = id_to_name_mapping.get(value, value)
            else:
                processed_edge[key] = value
        processed_edges.append(processed_edge)
    return processed_edges

def process_single_review(paper_id: str, review_id: str, review_content: Dict, logger: logging.Logger) -> Optional[Dict]:
    """Process one review and return extracted graph data"""
    try:
        # Compose review text
        review_text = ""
        if 'strengths' in review_content:
            review_text += f"Strengths: {review_content['strengths']}\n\n"
        if 'weakness' in review_content:
            review_text += f"Weakness: {review_content['weakness']}\n\n"
        if 'suggestions' in review_content:
            review_text += f"Suggestions: {review_content['suggestions']}\n\n"
        
        if not review_text.strip():
            logger.warning(f"Paper {paper_id}, Review {review_id}: review content is empty")
            return None
        
        # Build PDF URL (assuming OpenReview format)
        pdf_url = f"https://openreview.net/pdf?id={paper_id}"
        
        logger.info(f"Processing nodes and edges for Paper {paper_id}, Review {review_id}")
        prompt_graph = f"""
        You are a professional academic evaluation experience extraction expert. 
        Please extract detailed and specific knowledge entities and experiential relationships from the following review text to construct an experiential relationship subgraph.
        Knowledge entities include but are not limited to questions, methods, concepts, theories, scenarios, and other professional terms and knowledge. Please do not use a simple word; it is better to enrich the semantic meaning of the entity based on the original evaluation by adding some adjectives. 
        Experience relations refer to, for example, "seems to be relatively good at improving... ability in xxx", "seems unable to be well achieved through... " and other similar evaluative relationship statements. Their characteristics are: they have certain positive or negative emotional evaluation information, and they have as detailed as possible semantic information of specific aspects. Nodes and edges must directly depend on the review. The original text is only for reference. Therefore, when each edge is constructed, there must be corresponding evidence.

        **Entity Naming Guidelines - Be Specific and Contextual:**

        ❌ **WRONG Examples (too generic):**
        - "LLMs" → (It doesn't have to be like this. It depends on the original text specifically.)Might be "scientific reasoning ability of current large language models"
        - "performance" → (It doesn't have to be like this. It depends on the original text specifically.)Might be "few-shot learning ability of current large language models" 
        - "dataset" → (It doesn't have to be like this. It depends on the original text specifically.)Might be "SCIBENCH scientific reasoning evaluation dataset"
        - "method" → (It doesn't have to be like this. It depends on the original text specifically.)Might be "Chain-of-Thought prompting technique"

        ✅ **CORRECT Examples (specific and contextual):**
        - "scientific reasoning ability of current large language models"
        - "few-shot learning performance on complex scientific problems"
        - "SCIBENCH scientific reasoning evaluation dataset"
        - "Chain-of-Thought (CoT) prompting technique"

        **Relationship Examples - Be Evaluative and Specific:**

        ✅ **Good Relationship Examples:(Include descriptions of both positive and negative aspects and degrees, as well as rich semantic information)**
        - "Chain-of-Thought (CoT) prompting" → "significantly improves" → "calculation skills of LLMs"
        - "SCIBENCH dataset" → "effectively differentiates" → "performance between different large language models"
        - "free-form questions" → "a certain degree prevents" → "result guessing based on multiple-choice answers"
        - "systematic zero-shot example selection" → "can not enhances" → "model's problem-solving capabilities"
        - "Wolfram Language prompts" → "deteriorates" → "few-shot learning performance of LLMs"
        - "random few-shot example selection" → "fails to fully explore" → "potential of few-shot learning"
        - "current LLMs" → "struggles with" → "complex scientific problem-solving tasks"

        ❌ **WRONG Relationship Examples (too generic):**
        - "LLMs" → "is designed to" → "performance" (too vague)
        - "Error analysis" → "requires understanding of" → "Hückel molecular orbital theory" (knowledge relation, not evaluative)

        **Attention**
        1. The evidence section must be complete and no part should be omitted. Never use "...".
        2. Node and Edge should be as specific and detailed as possible, and when combined, they should conform to the logic of the original text.

        You should check the review sentence by sentence to see if there are any evaluations from the reviewers regarding the method or professional terms, and then extract them. And speak out your thinking process aloud. 
        Your final output should be in JSON format:
        ```json
        {{
            "source_name": "node_x", 
            "target_name": "node_y", 
            "relationship": "xxx", 
            "evidence": "relevant sentence from review text(Or the rewritten text in the Node process result)"
        }}
        ```

        Example input: The proposed method cannot demonstrate its effectiveness on both image and audio datasets. And BATTLE can improve the robustness of training agents under adversarial attacks.
        Sample json output (omitting the previous thinking process):
        ```json
        [{{
            "source_name": "PIA", 
            "target_name": "some image and audio datasets", 
            "relationship": "is NOT validated on", 
            "evidence": "The proposed method cannot demonstrate its effectiveness on both image and audio datasets. PS: Based on the paper, the proposed method refer to PIA."
        }},
        {{
            "source_name": "BATTLE", 
            "target_name": "adversarial attacks of agents", 
            "relationship": "enhance the robustness of", 
            "evidence": "And BATTLE can improve the robustness of training agents under adversarial attacks."
        }}]
        ```

        Be specific and detailed !!!!!

        Now the review text is:
        {review_text}
        And the pdf url is(I'll also input the pdf to you to get the original text):
        {pdf_url}
        """
        edge_process_result = ChatGPT.chat(prompt_graph, pdf_url)
        
        # Extract edge JSON data
        edge_json_data = extract_json_from_text(edge_process_result)
        if not edge_json_data:
            logger.error(f"Paper {paper_id}, Review {review_id}: failed to parse edge JSON")
            return None
        
        logger.info(f"Paper {paper_id}, Review {review_id}: extracted {len(edge_json_data)} edges")
        
        return {
            'paper_id': paper_id,
            'review_id': review_id,
            'edges': edge_json_data,
            'edge_process_result': edge_process_result
        }
        
    except Exception as e:
        logger.error(f"Failed while processing Paper {paper_id}, Review {review_id}: {str(e)}")
        logger.error(f"Details: {traceback.format_exc()}")
        return None

def save_progress(progress_file: str, processed_papers: List[str], failed_papers: List[str], 
                 results: List[Dict], current_index: int):
    """Persist progress to disk"""
    progress_data = {
        'processed_papers': processed_papers,
        'failed_papers': failed_papers,
        'results': results,
        'current_index': current_index,
        'timestamp': datetime.now().isoformat()
    }
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, indent=2, ensure_ascii=False)

def load_progress(progress_file: str) -> Tuple[List[str], List[str], List[Dict], int]:
    """Load saved progress"""
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress_data = json.load(f)
        return (progress_data.get('processed_papers', []),
                progress_data.get('failed_papers', []),
                progress_data.get('results', []),
                progress_data.get('current_index', 0))
    return [], [], [], 0

def process_single_paper(paper: Dict, output_dir: str, i: int, total_papers: int) -> Tuple[str, Optional[List[Dict]], bool]:
    """Process a single paper (thread worker)"""
    logger = logging.getLogger(__name__)
    paper_id = paper['paper_id']
    
    try:
        logger.info(f"[{i+1}/{total_papers}] Starting paper {paper_id}")
        
        paper_results = []
        review_contents = paper.get('review_contents', [])
        
        if not review_contents:
            logger.warning(f"Paper {paper_id} has no review content")
            return (paper_id, None, False)
        
        # Process each review
        for review in review_contents:
            review_id = review['review_id']
            review_content = review['content']
            
            logger.info(f"[{i+1}/{total_papers}] Processing Paper {paper_id} Review {review_id}")
            result = process_single_review(paper_id, review_id, review_content, logger)
            
            if result:
                paper_results.append(result)
            else:
                logger.warning(f"Review {review_id} failed")
        
        if paper_results:
            # Save results for this paper
            paper_output_file = os.path.join(output_dir, f"{paper_id}_graph.json")
            with open(paper_output_file, 'w', encoding='utf-8') as f:
                json.dump(paper_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[{i+1}/{total_papers}] Paper {paper_id} finished with {len(paper_results)} reviews")
            return (paper_id, paper_results, True)
        else:
            logger.warning(f"[{i+1}/{total_papers}] All reviews failed for paper {paper_id}")
            return (paper_id, None, False)
            
    except Exception as e:
        logger.error(f"[{i+1}/{total_papers}] Exception while processing paper {paper_id}: {str(e)}")
        logger.error(f"Details: {traceback.format_exc()}")
        return (paper_id, None, False)

def batch_process_iclr_data(input_file: str, output_dir: str, start_index: int = 0, 
                           max_papers: int = None, resume: bool = False) -> None:
    """Batch process ICLR paper data"""
    logger = setup_logging()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Progress file
    progress_file = os.path.join(output_dir, 'progress.json')
    
    # Load data
    logger.info(f"Loading data file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    total_papers = len(papers)
    if max_papers:
        total_papers = min(total_papers, max_papers)
    
    logger.info(f"Total papers to process: {total_papers}")
    
    # Reload progress when resuming
    processed_papers = []
    failed_papers = []
    results = []
    current_index = start_index
    
    if resume and os.path.exists(progress_file):
        processed_papers, failed_papers, results, current_index = load_progress(progress_file)
        logger.info(f"Resumed from checkpoint: processed {len(processed_papers)}, failed {len(failed_papers)}")
    
    # Protect shared state with a thread lock
    results_lock = Lock()
    
    # Batch settings
    batch_size = 100
    max_workers = 100  # Number of parallel threads
    
    # Build remaining paper list
    remaining_papers = []
    for i in range(current_index, total_papers):
        paper = papers[i]
        paper_id = paper['paper_id']
        
        if paper_id in processed_papers or paper_id in failed_papers:
            logger.info(f"Skipping paper already handled: {paper_id}")
            continue
        
        remaining_papers.append(paper)
    
    logger.info(f"{len(remaining_papers)} papers remaining; processing in batches of {batch_size}")
    
    # Iterate through batches
    for batch_start in range(0, len(remaining_papers), batch_size):
        batch_end = min(batch_start + batch_size, len(remaining_papers))
        batch = remaining_papers[batch_start:batch_end]
        
        logger.info(f"\n=== Starting batch {batch_start//batch_size + 1}: papers {batch_start+1}-{batch_end} (total remaining {len(remaining_papers)}) ===\n")
        
        # Use ThreadPoolExecutor for this batch
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_paper = {}
            for batch_idx, paper in enumerate(batch):
                # Use relative index + batch_start for display numbering
                display_index = batch_start + batch_idx + 1
                future = executor.submit(process_single_paper, paper, output_dir, display_index, len(remaining_papers))
                future_to_paper[future] = paper['paper_id']
            
            # Collect results
            for future in as_completed(future_to_paper):
                paper_id = future_to_paper[future]
                try:
                    result_paper_id, paper_results, success = future.result()
                    
                    # Update shared state safely
                    with results_lock:
                        if success and paper_results:
                            results.extend(paper_results)
                            processed_papers.append(result_paper_id)
                        else:
                            failed_papers.append(result_paper_id)
                except Exception as e:
                    logger.error(f"Exception while processing paper {paper_id}: {str(e)}")
                    with results_lock:
                        failed_papers.append(paper_id)
        
        # Save checkpoint after each batch
        # actual_progress = original start index + relative batch_end
        actual_progress = current_index + batch_end
        save_progress(progress_file, processed_papers, failed_papers, results, actual_progress)
        logger.info(f"Processed {batch_end}/{len(remaining_papers)} papers; progress saved")
        
        # Pause between batches to avoid rate limits
        time.sleep(2)
    
    # Save final merged results
    final_output_file = os.path.join(output_dir, 'all_graphs.json')

    # 1. Load existing results if present
    if os.path.exists(final_output_file):
        with open(final_output_file, 'r', encoding='utf-8') as f:
            try:
                existing_results = json.load(f)
                # Ensure structure is a list
                if not isinstance(existing_results, list):
                    existing_results = [existing_results]
            except json.JSONDecodeError:
                existing_results = []
    else:
        existing_results = []

    # 2. Append new results
    if isinstance(results, list):
        existing_results.extend(results)
    else:
        existing_results.append(results)

    # 3. Write back to disk
    with open(final_output_file, 'w', encoding='utf-8') as f:
        json.dump(existing_results, f, indent=2, ensure_ascii=False)
    
    # Save summary stats
    stats = {
        'total_papers': total_papers,
        'processed_papers': len(processed_papers),
        'failed_papers': len(failed_papers),
        'total_reviews': len(results),
        'processed_paper_ids': processed_papers,
        'failed_paper_ids': failed_papers
    }
    
    stats_file = os.path.join(output_dir, 'processing_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info("Batch processing finished!")
    logger.info(f"Papers processed successfully: {len(processed_papers)}")
    logger.info(f"Papers failed: {len(failed_papers)}")
    logger.info(f"Total review graphs generated: {len(results)}")
    logger.info(f"Results stored in: {output_dir}")

def main():

    batch_process_iclr_data(
        input_file='data/ICLR/iclr2025_simple.json',
        output_dir='result_v2',
        start_index=9000,
        max_papers=9881,
        resume=False
    )

if __name__ == "__main__":
    main()