#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the performance of eval_feasibility_score.py on training data
Calculate accuracy, correlation coefficients, Cohen's Kappa
Use multiprocessing for parallel processing to save time
Use all_comments field as input (complete peer review comment text)
"""

import sys
import os
import json
import re
import time
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

# Ensure modules from project root can be imported
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Evaluation_utils.eval_feasibility_score import generate_feasibility_score


def extract_score_from_response(response: str) -> Optional[float]:
    """
    Extract score from GPT response
    Based on actual output format: Analysis text followed by separate "Score: X" line
    Supports integer and decimal scores
    """
    if not response or response.startswith("ERROR"):
        return None
    
    # Clean response text
    response = response.strip()
    
    # Strategy 1: Prioritize matching "Score: X" format at the end (separate line or last few lines)
    # Check last few lines, as format is usually "Analysis: ...\n\nScore: X"
    lines = response.split('\n')
    # Check last 5 lines
    last_lines = '\n'.join(lines[-5:]) if len(lines) > 5 else response
    last_lines = last_lines.strip()
    
    # Priority matching pattern: separate line "Score: X" or "Score:X" (supports decimals)
    score_patterns = [
        r'^Score\s*:\s*(\d+(?:\.\d+)?)$',  # 单独一行 "Score: 10" 或 "Score: 4.5"
        r'\nScore\s*:\s*(\d+(?:\.\d+)?)\s*$',  # 末尾的 "Score: 10"
        r'Score\s*:\s*(\d+(?:\.\d+)?)',  # 任何位置的 "Score: X"
        r'分数\s*:\s*(\d+(?:\.\d+)?)',  # "分数: X"
        r'feasibility\s*score\s*:\s*(\d+(?:\.\d+)?)',  # "feasibility score: X"
        r'可行性分数\s*:\s*(\d+(?:\.\d+)?)',  # "可行性分数: X"
        r'评分\s*:\s*(\d+(?:\.\d+)?)',  # "评分: X"
    ]
    
    # First search in last few lines
    for pattern in score_patterns:
        match = re.search(pattern, last_lines, re.MULTILINE | re.IGNORECASE)
        if match:
            score = float(match.group(1))
            if 1.0 <= score <= 10.0:
                return score
    
    # If not found in last few lines, search in entire response
    for pattern in score_patterns:
        match = re.search(pattern, response, re.MULTILINE | re.IGNORECASE)
        if match:
            score = float(match.group(1))
            if 1.0 <= score <= 10.0:
                return score
    
    # Strategy 2: Match "X/10" format
    score_10_pattern = r'(\d+(?:\.\d+)?)\s*/\s*10'
    match = re.search(score_10_pattern, response, re.IGNORECASE)
    if match:
        score = float(match.group(1))
        if 1.0 <= score <= 10.0:
            return score
    
    # Strategy 3: Search for numbers between 1-10 at the end of response (may be formatted score)
    # Search for numbers in last few lines
    last_few_lines = '\n'.join(lines[-3:]) if len(lines) > 3 else response
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', last_few_lines)
    if numbers:
        # Take the last number (most likely to be the score)
        for num_str in reversed(numbers):
            score = float(num_str)
            if 1.0 <= score <= 10.0:
                return score
    
    # Strategy 4: If still not found, search for all numbers between 1-10 in entire response
    all_numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response)
    if all_numbers:
        # Take the last number
        for num_str in reversed(all_numbers):
            score = float(num_str)
            if 1.0 <= score <= 10.0:
                return score
    
    return None


def process_single_sample(sample: Dict) -> Tuple[str, Optional[float], Optional[float], Optional[str]]:
    """
    Process a single sample
    Returns: (sample_id, true_score, predicted_score, response_text)
    """
    sample_id = sample.get('id', 'unknown')
    all_comments = sample.get('all_comments', '')
    true_score = sample.get('average_score', None)
    
    if not all_comments or true_score is None:
        return (sample_id, true_score, None, None)
    
    try:
        # Directly use all_comments as evaluation text
        evaluation_text = all_comments.strip()
        
        # Call scoring function
        response = generate_feasibility_score(evaluation_text)
        predicted_score = extract_score_from_response(response)
        return (sample_id, true_score, predicted_score, response)
    except Exception as e:
        print(f"Error processing sample {sample_id}: {e}")
        return (sample_id, true_score, None, str(e))


def calculate_metrics(true_scores: List[float], predicted_scores: List[float]) -> Dict:
    """
    Calculate evaluation metrics
    """
    # Filter out None values
    valid_indices = [i for i, (t, p) in enumerate(zip(true_scores, predicted_scores)) 
                     if t is not None and p is not None]
    
    if len(valid_indices) == 0:
        return {
            'accuracy_exact': 0.0,
            'accuracy_within_0.5': 0.0,
            'accuracy_within_1': 0.0,
            'accuracy_within_1.5': 0.0,
            'accuracy_within_2': 0.0,
            'pearson_correlation': 0.0,
            'spearman_correlation': 0.0,
            'cohen_kappa': 0.0,
            'valid_samples': 0,
            'total_samples': len(true_scores)
        }
    
    true_valid = [true_scores[i] for i in valid_indices]
    pred_valid = [predicted_scores[i] for i in valid_indices]
    
    # Accuracy - exact match (allow 0.1 error due to float comparison)
    exact_matches = sum(1 for t, p in zip(true_valid, pred_valid) if abs(t - p) < 0.1)
    accuracy_exact = exact_matches / len(true_valid)
    
    # Accuracy - allow ±0.5 error
    within_0_5_matches = sum(1 for t, p in zip(true_valid, pred_valid) if abs(t - p) <= 0.5)
    accuracy_within_0_5 = within_0_5_matches / len(true_valid)
    
    # Accuracy - allow ±1 error
    within_1_matches = sum(1 for t, p in zip(true_valid, pred_valid) if abs(t - p) <= 1.0)
    accuracy_within_1 = within_1_matches / len(true_valid)
    
    # Accuracy - allow ±1.5 error
    within_1_5_matches = sum(1 for t, p in zip(true_valid, pred_valid) if abs(t - p) <= 1.5)
    accuracy_within_1_5 = within_1_5_matches / len(true_valid)
    
    # Accuracy - allow ±2.0 error
    within_2_matches = sum(1 for t, p in zip(true_valid, pred_valid) if abs(t - p) <= 2.0)
    accuracy_within_2 = within_2_matches / len(true_valid)
    
    # Correlation coefficients
    pearson_r, pearson_p = pearsonr(true_valid, pred_valid)
    spearman_r, spearman_p = spearmanr(true_valid, pred_valid)
    
    # Cohen's Kappa (treat scores as categories after rounding to integers)
    true_rounded = [round(t) for t in true_valid]
    pred_rounded = [round(p) for p in pred_valid]
    cohen_kappa = cohen_kappa_score(true_rounded, pred_rounded, weights='quadratic')
    
    return {
        'accuracy_exact': accuracy_exact,
        'accuracy_within_0.5': accuracy_within_0_5,
        'accuracy_within_1': accuracy_within_1,
        'accuracy_within_1.5': accuracy_within_1_5,
        'accuracy_within_2': accuracy_within_2,
        'pearson_correlation': pearson_r,
        'pearson_p_value': pearson_p,
        'spearman_correlation': spearman_r,
        'spearman_p_value': spearman_p,
        'cohen_kappa': cohen_kappa,
        'valid_samples': len(true_valid),
        'total_samples': len(true_scores),
        'mean_absolute_error': np.mean([abs(t - p) for t, p in zip(true_valid, pred_valid)]),
        'rmse': np.sqrt(np.mean([(t - p) ** 2 for t, p in zip(true_valid, pred_valid)])),
        'mean_true_score': np.mean(true_valid),
        'mean_predicted_score': np.mean(pred_valid),
        'std_true_score': np.std(true_valid),
        'std_predicted_score': np.std(pred_valid)
    }


def main():
    """Main function"""
    print("=" * 60)
    print("Starting to test the performance of eval_feasibility_score.py")
    print("Using all_comments (complete peer review comment text) as input")
    print("=" * 60)
    
    # Load data
    data_file = os.path.join(PROJECT_ROOT, 'Evaluation_feasibility_score', 
                            'Stanford_comments_with_ideas_with_scores.json')
    print(f"\nLoading data file: {data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # Filter out samples without all_comments or average_score
    valid_data = [item for item in data if item.get('all_comments') and item.get('average_score') is not None]
    print(f"Valid samples: {len(valid_data)}")
    
    if len(valid_data) == 0:
        print("Error: No valid samples!")
        return
    
    # Determine number of processes
    num_processes = min(50, cpu_count())
    print(f"Using {num_processes} processes for parallel processing")
    
    # Process in batches to avoid API rate limiting
    batch_size = num_processes * 2
    all_results = []
    
    start_time = time.time()
    
    for i in range(0, len(valid_data), batch_size):
        batch = valid_data[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(valid_data) + batch_size - 1) // batch_size
        
        print(f"\nProcessing batch {batch_num}/{total_batches} (samples {i+1}-{min(i+batch_size, len(valid_data))})")
        
        # Use multiprocessing to process batch
        with Pool(processes=num_processes) as pool:
            batch_results = pool.map(process_single_sample, batch)
        
        all_results.extend(batch_results)
        
        # Display progress
        elapsed = time.time() - start_time
        processed = i + len(batch)
        if processed > 0:
            avg_time_per_sample = elapsed / processed
            remaining_samples = len(valid_data) - processed
            estimated_remaining = avg_time_per_sample * remaining_samples
            
            print(f"  Completed: {processed}/{len(valid_data)}")
            print(f"  Elapsed time: {elapsed:.1f} seconds")
            print(f"  Estimated remaining time: {estimated_remaining:.1f} seconds")
        
        # Avoid API rate limiting, rest briefly
        if i + batch_size < len(valid_data):
            time.sleep(1)
    
    total_time = time.time() - start_time
    print(f"\nProcessing completed! Total time: {total_time:.1f} seconds")
    
    # Extract results
    sample_ids = [r[0] for r in all_results]
    true_scores = [r[1] for r in all_results]
    predicted_scores = [r[2] for r in all_results]
    responses = [r[3] for r in all_results]
    
    # Statistics
    valid_predictions = sum(1 for p in predicted_scores if p is not None)
    print(f"\nValid predictions: {valid_predictions}/{len(all_results)}")
    
    # Calculate metrics
    print("\nCalculating evaluation metrics...")
    metrics = calculate_metrics(true_scores, predicted_scores)
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Valid predictions: {metrics['valid_samples']}")
    print(f"\nAccuracy (exact match, ±0.1 error): {metrics['accuracy_exact']:.4f} ({metrics['accuracy_exact']*100:.2f}%)")
    print(f"Accuracy (±0.5 error): {metrics['accuracy_within_0.5']:.4f} ({metrics['accuracy_within_0.5']*100:.2f}%)")
    print(f"Accuracy (±1.0 error): {metrics['accuracy_within_1']:.4f} ({metrics['accuracy_within_1']*100:.2f}%)")
    print(f"Accuracy (±1.5 error): {metrics['accuracy_within_1.5']:.4f} ({metrics['accuracy_within_1.5']*100:.2f}%)")
    print(f"Accuracy (±2.0 error): {metrics['accuracy_within_2']:.4f} ({metrics['accuracy_within_2']*100:.2f}%)")
    print(f"\nPearson correlation: {metrics['pearson_correlation']:.4f} (p={metrics['pearson_p_value']:.4f})")
    print(f"Spearman correlation: {metrics['spearman_correlation']:.4f} (p={metrics['spearman_p_value']:.4f})")
    print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    print(f"\nMean Absolute Error (MAE): {metrics['mean_absolute_error']:.4f}")
    print(f"Root Mean Square Error (RMSE): {metrics['rmse']:.4f}")
    print(f"\nTrue score mean: {metrics['mean_true_score']:.4f} (std: {metrics['std_true_score']:.4f})")
    print(f"Predicted score mean: {metrics['mean_predicted_score']:.4f} (std: {metrics['std_predicted_score']:.4f})")
    
    # Save results
    results_dir = os.path.join(PROJECT_ROOT, "Evaluation_feasibility_score", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, 'test_eval_feasibility_score_result.json')
    
    # Prepare data to save
    results_data = {
        'metrics': metrics,
        'total_time_seconds': total_time,
        'samples': [
            {
                'id': sid,
                'true_score': ts,
                'predicted_score': ps,
                'error': abs(ts - ps) if ts is not None and ps is not None else None,
                'response': resp
            }
            for sid, ts, ps, resp in zip(sample_ids, true_scores, predicted_scores, responses)
        ]
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Save detailed report
    report_file = os.path.join(results_dir, 'test_eval_feasibility_score_result_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("eval_feasibility_score.py Performance Evaluation Report\n")
        f.write("Using all_comments (complete peer review comment text) as input\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total samples: {metrics['total_samples']}\n")
        f.write(f"Valid predictions: {metrics['valid_samples']}\n")
        f.write(f"Total time: {total_time:.1f} seconds\n\n")
        
        f.write("Evaluation Metrics:\n")
        f.write(f"  Accuracy (exact match, ±0.1 error): {metrics['accuracy_exact']:.4f} ({metrics['accuracy_exact']*100:.2f}%)\n")
        f.write(f"  Accuracy (±0.5 error): {metrics['accuracy_within_0.5']:.4f} ({metrics['accuracy_within_0.5']*100:.2f}%)\n")
        f.write(f"  Accuracy (±1.0 error): {metrics['accuracy_within_1']:.4f} ({metrics['accuracy_within_1']*100:.2f}%)\n")
        f.write(f"  Accuracy (±1.5 error): {metrics['accuracy_within_1.5']:.4f} ({metrics['accuracy_within_1.5']*100:.2f}%)\n")
        f.write(f"  Accuracy (±2.0 error): {metrics['accuracy_within_2']:.4f} ({metrics['accuracy_within_2']*100:.2f}%)\n")
        f.write(f"  Pearson correlation: {metrics['pearson_correlation']:.4f} (p={metrics['pearson_p_value']:.4f})\n")
        f.write(f"  Spearman correlation: {metrics['spearman_correlation']:.4f} (p={metrics['spearman_p_value']:.4f})\n")
        f.write(f"  Cohen's Kappa: {metrics['cohen_kappa']:.4f}\n")
        f.write(f"  Mean Absolute Error (MAE): {metrics['mean_absolute_error']:.4f}\n")
        f.write(f"  Root Mean Square Error (RMSE): {metrics['rmse']:.4f}\n")
        f.write(f"  True score mean: {metrics['mean_true_score']:.4f} (std: {metrics['std_true_score']:.4f})\n")
        f.write(f"  Predicted score mean: {metrics['mean_predicted_score']:.4f} (std: {metrics['std_predicted_score']:.4f})\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Detailed Prediction Results\n")
        f.write("=" * 60 + "\n\n")
        
        for i, (sid, ts, ps, resp) in enumerate(zip(sample_ids, true_scores, predicted_scores, responses)):
            f.write(f"Sample {i+1}: {sid}\n")
            f.write(f"  True score: {ts if ts is not None else 'None'}\n")
            f.write(f"  Predicted score: {ps if ps is not None else 'None'}\n")
            if ts is not None and ps is not None:
                f.write(f"  Error: {abs(ts - ps):.2f}\n")
            if resp:
                f.write(f"  Response: {resp[:200]}...\n" if len(resp) > 200 else f"  Response: {resp}\n")
            f.write("\n")
    
    print(f"Detailed report saved to: {report_file}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

