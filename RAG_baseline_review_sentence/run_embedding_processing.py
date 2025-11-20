#!/usr/bin/env python3
"""
Main script for batch processing evidence sentence embeddings.
Reads graph files in result_v2, extracts evidence sentences, and generates embeddings.
"""

import sys
import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path
import argparse
from typing import Dict, List, Optional, Tuple
import traceback
import multiprocessing as mp
from tqdm import tqdm
from functools import partial

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service import BGE_M3

def setup_logging(log_file: str = None) -> logging.Logger:
    """Configure logging for the embedding pipeline"""
    if log_file is None:
        log_file = f"run_embedding_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def process_single_evidence_embedding_parallel(evidence_data: Dict) -> Optional[Dict]:
    """Process one evidence embedding in parallel (used by multiprocessing)"""
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

def extract_evidence_sentences(graph_data: List[Dict]) -> List[Dict]:
    """Extract all evidence sentences from the graph data"""
    evidence_sentences = []
    
    for item in graph_data:
        paper_id = item.get('paper_id', '')
        review_id = item.get('review_id', '')
        edges = item.get('edges', [])
        
        for edge in edges:
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
        
        logger.info(f"Evidence {evidence_data['sentence_id']}: embedding generated, dimension {len(embedding_vector)}")
        
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
    """Persist processing progress for recovery"""
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
    """Reload processing progress if available"""
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
    """Batch process evidence sentence embeddings with optional parallelism"""
    logger = setup_logging()
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    embeddings_dir = os.path.join(output_dir, 'embeddings')
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Collect all graph files
    graph_files = [f for f in os.listdir(input_dir) if f.endswith('_graph.json')]
    graph_files.sort()
    
    # Process all files if end_index not set
    if end_index is None:
        end_index = len(graph_files)
    
    # Slice by index range
    target_files = graph_files[start_index:end_index]
    total_files = len(target_files)
    
    logger.info(f"Processing graph files from index {start_index} to {end_index}, total {total_files}")
    
    # Prepare evidence data
    all_evidence_data = []
    for graph_file in target_files:
        file_path = os.path.join(input_dir, graph_file)
        
        try:
            # Load graph data
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
    logger.info(f"Starting batch embedding generation, batch size={batch_size}")
    successful_results = []
    failed_count = 0
    
    # Process per batch
    for i in range(0, len(all_evidence_data), batch_size):
        batch = all_evidence_data[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(all_evidence_data) + batch_size - 1) // batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch)} sentences")
        
        # Collect batch texts
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
            # Generate embeddings for the batch
            embeddings = BGE_M3.embed_texts(batch_texts, model_name="BAAI/bge-m3")
            
            # Record results
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
            
            logger.info(f"Batch {batch_num} processed, success count: {len(embeddings)}")
            
        except Exception as e:
            logger.error(f"Batch {batch_num} failed: {str(e)}")
            failed_count += len(batch)
            continue
    
    logger.info(f"Batch processing finished. Success: {len(successful_results)}, Failed: {failed_count}")
    
    # Group results by paper
    file_results = {}
    for result in successful_results:
        # Group by paper_id
        paper_id = result['paper_id']
        if paper_id not in file_results:
            file_results[paper_id] = []
        file_results[paper_id].append(result)
    
    # Persist embeddings per paper
    for paper_id, embeddings in file_results.items():
        file_output_file = os.path.join(embeddings_dir, f"{paper_id}_embeddings.json")
        with open(file_output_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings, f, indent=2, ensure_ascii=False)
    
    # Save aggregated results
    final_output_file = os.path.join(output_dir, 'all_evidence_embeddings.json')
    with open(final_output_file, 'w', encoding='utf-8') as f:
        json.dump(successful_results, f, indent=2, ensure_ascii=False)
    
    # Write embedding index
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
    
    # Save summary stats
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
    
    logger.info("Embedding batch processing complete")
    logger.info(f"Results stored in: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Batch process evidence sentence embeddings')
    parser.add_argument('--input', '-i', 
                       default='result_v2',
                       help='Input directory containing graph files')
    parser.add_argument('--output', '-o', 
                       default='RAG_baseline_review_sentence',
                       help='Output directory path')
    parser.add_argument('--start', '-s', type=int, default=0, help='Start index of files to process')
    parser.add_argument('--end', '-e', type=int, default=None, help='End index of files (default None for all)')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from last saved progress')
    parser.add_argument('--batch_size', '-b', type=int, default=1024, help='Batch size (default 32)')
    
    args = parser.parse_args()
    
    batch_process_evidence_embeddings_optimized(
        input_dir=args.input,
        output_dir=args.output,
        start_index=args.start,
        end_index=args.end,
        resume=args.resume,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
