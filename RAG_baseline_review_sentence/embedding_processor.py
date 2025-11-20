import sys
import os
import json
import logging
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import traceback
import multiprocessing as mp
from tqdm import tqdm
from functools import partial

# Add parent directory to path to import service
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service import BGE_M3

def setup_logging(log_file: str = None) -> logging.Logger:
    """Configure logging for embedding processing"""
    if log_file is None:
        log_file = f"embedding_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def extract_evidence_sentences(graph_data: List[Dict]) -> List[Dict]:
    """Extract all evidence sentences from graph data"""
    evidence_sentences = []
    
    for item in graph_data:
        paper_id = item.get('paper_id', '')
        review_id = item.get('review_id', '')
        edges = item.get('edges', [])
        
        # Handle edges stored as dict or list
        if isinstance(edges, dict):
            # Convert dict (single edge) into a list
            edges = [edges]
        elif not isinstance(edges, list):
            # Skip unsupported edge containers
            continue
        
        for edge in edges:
            # Ensure edge is a dict
            if not isinstance(edge, dict):
                continue
                
            evidence = edge.get('evidence', '').strip()
            if evidence:
                evidence_sentences.append({
                    'paper_id': paper_id,
                    'review_id': review_id,
                    'source_name': edge.get('source_name', ''),
                    'target_name': edge.get('target_name', ''),
                    'relationship': edge.get('relationship', ''),
                    'evidence': evidence,
                    'sentence_id': f"{paper_id}_{review_id}_{len(evidence_sentences)}"
                })
    
    return evidence_sentences

def process_single_evidence_embedding_parallel(evidence_data: Dict) -> Optional[Dict]:
    """Process one evidence embedding in parallel (multiprocessing helper)"""
    try:
        evidence_text = evidence_data['evidence']
        
        if not evidence_text.strip():
            return None
        
        # Generate embedding
        embeddings = BGE_M3.embed_texts([evidence_text], model_name="BAAI/bge-m3")
        
        if not embeddings or len(embeddings) == 0:
            return None
        
        embedding_vector = embeddings[0]
        
        return {
            'sentence_id': evidence_data['sentence_id'],
            'paper_id': evidence_data['paper_id'],
            'review_id': evidence_data['review_id'],
            'source_name': evidence_data['source_name'],
            'target_name': evidence_data['target_name'],
            'relationship': evidence_data['relationship'],
            'evidence': evidence_text,
            'embedding': embedding_vector,
            'embedding_dim': len(embedding_vector)
        }
        
    except Exception as e:
        return None

def process_single_evidence_embedding(evidence_data: Dict, logger: logging.Logger) -> Optional[Dict]:
    """Process a single evidence sentence embedding"""
    try:
        evidence_text = evidence_data['evidence']
        
        if not evidence_text.strip():
            logger.warning(f"Evidence {evidence_data['sentence_id']}: empty evidence content")
            return None
        
        # Generate embedding
        logger.info(f"Processing embedding for Evidence {evidence_data['sentence_id']}")
        embeddings = BGE_M3.embed_texts([evidence_text], model_name="BAAI/bge-m3")
        
        if not embeddings or len(embeddings) == 0:
            logger.error(f"Evidence {evidence_data['sentence_id']}: embedding generation failed")
            return None
        
        embedding_vector = embeddings[0]
        
        logger.info(f"Evidence {evidence_data['sentence_id']}: embedding created, dimension {len(embedding_vector)}")
        
        return {
            'sentence_id': evidence_data['sentence_id'],
            'paper_id': evidence_data['paper_id'],
            'review_id': evidence_data['review_id'],
            'source_name': evidence_data['source_name'],
            'target_name': evidence_data['target_name'],
            'relationship': evidence_data['relationship'],
            'evidence': evidence_text,
            'embedding': embedding_vector,
            'embedding_dim': len(embedding_vector)
        }
        
    except Exception as e:
        logger.error(f"Failed to process embedding for Evidence {evidence_data['sentence_id']}: {str(e)}")
        logger.error(f"Details: {traceback.format_exc()}")
        return None

def save_progress(progress_file: str, processed_sentences: List[str], failed_sentences: List[str], 
                 results: List[Dict], current_index: int):
    """Persist progress for recovery"""
    progress_data = {
        'processed_sentences': processed_sentences,
        'failed_sentences': failed_sentences,
        'results': results,
        'current_index': current_index,
        'timestamp': datetime.now().isoformat()
    }
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, indent=2, ensure_ascii=False)

def load_progress(progress_file: str) -> Tuple[List[str], List[str], List[Dict], int]:
    """Reload saved progress if present"""
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress_data = json.load(f)
        return (progress_data.get('processed_sentences', []),
                progress_data.get('failed_sentences', []),
                progress_data.get('results', []),
                progress_data.get('current_index', 0))
    return [], [], [], 0

def batch_process_evidence_embeddings_optimized(input_dir: str, output_dir: str, start_index: int = 0, 
                                              end_index: int = None, resume: bool = False, batch_size: int = 32) -> None:
    """Optimized batch embedding pipeline using pure batching instead of multiprocessing"""
    logger = setup_logging()
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    embeddings_dir = os.path.join(output_dir, 'embeddings')
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # List all graph files
    graph_files = [f for f in os.listdir(input_dir) if f.endswith('_graph.json')]
    graph_files.sort()
    
    # Default to all files when end_index is None
    if end_index is None:
        end_index = len(graph_files)
    
    # Slice file list within the requested index range
    target_files = graph_files[start_index:end_index]
    total_files = len(target_files)
    
    logger.info(f"Processing graph files from index {start_index} to {end_index}, total {total_files}")
    
    # Build evidence dataset
    all_evidence_data = []
    for graph_file in target_files:
        file_path = os.path.join(input_dir, graph_file)
        
        try:
            # Load graph file
            with open(file_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            # Extract evidence sentences
            evidence_sentences = extract_evidence_sentences(graph_data)
            all_evidence_data.extend(evidence_sentences)
            
        except Exception as e:
            logger.error(f"Failed to process file {graph_file}: {str(e)}")
            continue
    
    logger.info(f"Total evidence sentences to process: {len(all_evidence_data)}")
    
    # Batch processing
    logger.info(f"Starting batch processing, batch size={batch_size}")
    successful_results = []
    failed_count = 0
    
    # Iterate by batch
    for i in range(0, len(all_evidence_data), batch_size):
        batch = all_evidence_data[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(all_evidence_data) + batch_size - 1) // batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch)} sentences")
        
        # Collect evidence text for the batch
        batch_texts = []
        batch_metadata = []
        for evidence_data in batch:
            evidence_text = evidence_data['evidence']
            if evidence_text.strip():
                batch_texts.append(evidence_text)
                batch_metadata.append(evidence_data)
        
        if not batch_texts:
            logger.warning(f"Batch {batch_num} has no valid evidence text")
            continue
        
        try:
            # Generate embeddings
            embeddings = BGE_M3.embed_texts(batch_texts, model_name="BAAI/bge-m3")
            
            # Store results
            for j, (embedding, metadata) in enumerate(zip(embeddings, batch_metadata)):
                result = {
                    'sentence_id': metadata['sentence_id'],
                    'paper_id': metadata['paper_id'],
                    'review_id': metadata['review_id'],
                    'source_name': metadata['source_name'],
                    'target_name': metadata['target_name'],
                    'relationship': metadata['relationship'],
                    'evidence': batch_texts[j],
                    'embedding': embedding,
                    'embedding_dim': len(embedding)
                }
                successful_results.append(result)
            
            logger.info(f"Batch {batch_num} finished with {len(embeddings)} embeddings")
            
        except Exception as e:
            logger.error(f"Batch {batch_num} failed: {str(e)}")
            failed_count += len(batch)
            continue
    
    logger.info(f"Batch workflow complete. Success: {len(successful_results)}, Failed: {failed_count}")
    
    # Group results per paper
    file_results = {}
    for result in successful_results:
        paper_id = result['paper_id']
        if paper_id not in file_results:
            file_results[paper_id] = []
        file_results[paper_id].append(result)
    
    # Write per-file embedding outputs
    for paper_id, embeddings in file_results.items():
        file_output_file = os.path.join(embeddings_dir, f"{paper_id}_embeddings.json")
        with open(file_output_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings, f, indent=2, ensure_ascii=False)
    
    # Save aggregated results
    final_output_file = os.path.join(output_dir, 'all_evidence_embeddings.json')
    with open(final_output_file, 'w', encoding='utf-8') as f:
        json.dump(successful_results, f, indent=2, ensure_ascii=False)
    
    # Persist embedding index
    embedding_index = {}
    for result in successful_results:
        sentence_id = result['sentence_id']
        embedding_index[sentence_id] = {
            'paper_id': result['paper_id'],
            'review_id': result['review_id'],
            'source_name': result['source_name'],
            'target_name': result['target_name'],
            'relationship': result['relationship'],
            'evidence': result['evidence'],
            'embedding_dim': result['embedding_dim']
        }
    
    index_file = os.path.join(output_dir, 'embedding_index.json')
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(embedding_index, f, indent=2, ensure_ascii=False)
    
    # Save stats
    stats = {
        'total_files': total_files,
        'total_sentences': len(all_evidence_data),
        'processed_sentences': len(successful_results),
        'failed_sentences': failed_count,
        'total_embeddings': len(successful_results)
    }
    
    stats_file = os.path.join(output_dir, 'embedding_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info("Optimized embedding processing complete")
    logger.info(f"Artifacts stored in: {output_dir}")

def batch_process_evidence_embeddings(input_dir: str, output_dir: str, start_index: int = 0, 
                                    end_index: int = 100, resume: bool = False) -> None:
    """Batch process evidence embeddings (legacy approach)"""
    logger = setup_logging()
    
    # Prepare output folders
    os.makedirs(output_dir, exist_ok=True)
    embeddings_dir = os.path.join(output_dir, 'embeddings')
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Progress file
    progress_file = os.path.join(output_dir, 'embedding_progress.json')
    
    # Collect all graph files
    graph_files = [f for f in os.listdir(input_dir) if f.endswith('_graph.json')]
    graph_files.sort()
    
    # Slice file list for the requested range
    target_files = graph_files[start_index:end_index]
    total_files = len(target_files)
    
    logger.info(f"Processing graph files from index {start_index} to {end_index}, total {total_files}")
    
    # Restore state when resuming
    processed_sentences = []
    failed_sentences = []
    results = []
    current_index = 0
    
    if resume and os.path.exists(progress_file):
        processed_sentences, failed_sentences, results, current_index = load_progress(progress_file)
        logger.info(f"Resumed from checkpoint: processed {len(processed_sentences)} sentences, failed {len(failed_sentences)}")
    
    # Iterate over files
    for i in range(current_index, total_files):
        graph_file = target_files[i]
        file_path = os.path.join(input_dir, graph_file)
        
        logger.info(f"Processing file {i+1}/{total_files}: {graph_file}")
        
        try:
            # Load graph JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            # Extract evidence sentences
            evidence_sentences = extract_evidence_sentences(graph_data)
            logger.info(f"Extracted {len(evidence_sentences)} evidence sentences from {graph_file}")
            
            file_results = []
            
            # Process each evidence sentence
            for evidence_data in evidence_sentences:
                sentence_id = evidence_data['sentence_id']
                
                if sentence_id in processed_sentences or sentence_id in failed_sentences:
                    logger.info(f"Skipping already processed sentence {sentence_id}")
                    continue
                
                logger.info(f"Processing Evidence {sentence_id}")
                result = process_single_evidence_embedding(evidence_data, logger)
                
                if result:
                    file_results.append(result)
                    processed_sentences.append(sentence_id)
                else:
                    failed_sentences.append(sentence_id)
                    logger.warning(f"Evidence {sentence_id} failed")
            
            if file_results:
                # Save per-file embeddings
                file_output_file = os.path.join(embeddings_dir, f"{graph_file.replace('_graph.json', '')}_embeddings.json")
                with open(file_output_file, 'w', encoding='utf-8') as f:
                    json.dump(file_results, f, indent=2, ensure_ascii=False)
                
                results.extend(file_results)
                logger.info(f"Finished embeddings for {graph_file}, total {len(file_results)} sentences")
            
        except Exception as e:
            logger.error(f"Error processing file {graph_file}: {str(e)}")
            logger.error(f"Details: {traceback.format_exc()}")
            continue
        
        # Save checkpoint every five files
        if (i + 1) % 5 == 0:
            save_progress(progress_file, processed_sentences, failed_sentences, results, i + 1)
            logger.info(f"Processed {i + 1} files, checkpoint saved")
        
        # Light delay to avoid rate limits
        time.sleep(0.1)
    
    # Save aggregated results
    final_output_file = os.path.join(output_dir, 'all_evidence_embeddings.json')
    with open(final_output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Persist embedding index for fast lookup
    embedding_index = {}
    for result in results:
        sentence_id = result['sentence_id']
        embedding_index[sentence_id] = {
            'paper_id': result['paper_id'],
            'review_id': result['review_id'],
            'source_name': result['source_name'],
            'target_name': result['target_name'],
            'relationship': result['relationship'],
            'evidence': result['evidence'],
            'embedding_dim': result['embedding_dim']
        }
    
    index_file = os.path.join(output_dir, 'embedding_index.json')
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(embedding_index, f, indent=2, ensure_ascii=False)
    
    # Write summary stats
    stats = {
        'total_files': total_files,
        'processed_sentences': len(processed_sentences),
        'failed_sentences': len(failed_sentences),
        'total_embeddings': len(results),
        'processed_sentence_ids': processed_sentences,
        'failed_sentence_ids': failed_sentences
    }
    
    stats_file = os.path.join(output_dir, 'embedding_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info("Legacy batch embedding workflow complete")
    logger.info(f"Successful sentences: {len(processed_sentences)}")
    logger.info(f"Failed sentences: {len(failed_sentences)}")
    logger.info(f"Embeddings generated: {len(results)}")
    logger.info(f"Artifacts stored in: {output_dir}")

def main():
    batch_process_evidence_embeddings_optimized(
        input_dir='result_v2',
        output_dir='RAG_baseline_review_sentence',
        start_index=0,
        end_index=None,  # Process every file
        resume=False,
        batch_size=32
    )

if __name__ == "__main__":
    main()
