#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG model generates feasibility evaluation text (concerns)
"""

import sys
import os
import typing as t
import re
import time
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from service.ChatGPT import chat_simple
from RAG_baseline_review_sentence.retrieval_system import EvidenceRetrievalSystem


def get_keywords(prompt: str):
    """Extract retrieval keywords from user input"""
    system_prompt = (
        "You are a research idea evaluator. I will provide you with an academic idea, and you need to output 10 keywords for searching related evidence sentences.\n"
        "The output format should be a comma-separated list of keywords.\n"
        "Only return keywords, no explanations."
    )
    full_prompt = f"{system_prompt}\nAcademic Idea: {prompt}"
    keywords = chat_simple(full_prompt)
    print(f"[🔍 Keywords] {keywords}")
    return [kw.strip() for kw in keywords.split(",") if kw.strip()]


def search_evidence_with_retrieval_system(keywords, retrieval_system, top_k=5):
    """Search for relevant evidence sentences using retrieval system"""
    
    all_results = []
    
    for i, keyword in enumerate(keywords, 1):
        print(f"[🔍 Searching Evidence] Keyword {i}/{len(keywords)}: '{keyword}'")
        
        try:
            # Use retrieval system for semantic search
            results = retrieval_system.cosine_similarity_search(keyword, top_k=top_k)
            
            keyword_results = []
            for result in results:
                entry = f"📘 Paper ID: {result['paper_id']} | Review ID: {result['review_id']} | Similarity: {result['similarity']:.3f}\n"
                entry += f"{result['evidence']}\n"
                keyword_results.append(entry)
            
            all_results.extend(keyword_results)
            print(f"[📄 Found] {len(keyword_results)} evidence sentences for '{keyword}'")
            
            # Add delay to avoid API limits
            if i < len(keywords):  # Not the last keyword
                time.sleep(0.1)
                
        except Exception as e:
            print(f"[❌ Error] Failed to search '{keyword}': {e}")
            continue
    
    # Deduplicate (based on evidence_id)
    seen_evidence = set()
    unique_results = []
    for result in all_results:
        # Extract evidence_id (from Paper ID and Review ID combination)
        paper_id = result.split('|')[0].strip().replace('📘 Paper ID: ', '')
        review_id = result.split('|')[1].strip().replace('Review ID: ', '')
        evidence_id = f"{paper_id}_{review_id}"
        if evidence_id not in seen_evidence:
            seen_evidence.add(evidence_id)
            unique_results.append(result)
    
    print(f"[📊 Summary] Total: {len(all_results)} evidence sentences, Unique: {len(unique_results)} evidence sentences")
    return unique_results


def rag_pipeline_with_retrieval_system(user_query: str, embeddings_dir: str, retrieval_system=None):
    """Complete RAG pipeline using retrieval system to generate feasibility evaluation text"""
    # Step 1: Initialize retrieval system (if not provided)
    if retrieval_system is None:
        print("[🔧 Initializing] Evidence Retrieval System...")
        retrieval_system = EvidenceRetrievalSystem(embeddings_dir)
        
        # Display statistics
        stats = retrieval_system.get_statistics()
        print(f"[📊 Stats] Total evidence embeddings: {stats['total_embeddings']}, Papers: {stats['total_papers']}")
    
    # Step 2: Keyword extraction
    keywords = get_keywords(user_query)

    # Step 3: Retrieve evidence results
    max_attempts = 10
    attempt = 1
    
    while attempt <= max_attempts:
        evidence_sentences = search_evidence_with_retrieval_system(keywords, retrieval_system, top_k=5)
        print(f"[📚 Attempt {attempt}] Found {len(evidence_sentences)} evidence sentences:")
        
        if len(evidence_sentences) > 0:
            # Found results, output evidence information
            for i, evidence in enumerate(evidence_sentences, 1):
                first_line = evidence.split('\n')[0]
                print(f"  {i}. {first_line}")
            print()
            break
        else:
            # No results found, regenerate keywords
            print("  ⏳ No evidence sentences found, regenerating keywords...")
            keywords = get_keywords(user_query)
            print(f"[🔍 Regenerated Keywords] {keywords}")
            attempt += 1
            
            if attempt <= max_attempts:
                print(f"  🔄 Retrying... (attempt {attempt}/{max_attempts})")
            else:
                print("  ❌ Max attempts reached, proceeding with empty results")
                print()
            time.sleep(2)

    # Step 4: Build RAG prompt and generate complete peer review evaluation text
    evidence_text = "\n\n".join(evidence_sentences) if evidence_sentences else "No relevant evidence found."
    
    system_prompt = """
You are a rigorous peer-reviewer evaluating the feasibility of an academic research idea.
Task: Critically evaluate the given idea/proposal and write a comprehensive peer review evaluation text.

Your response should:
- Be a continuous, natural text similar to a peer review comment (like "all_comments" in academic reviews)
- Discuss feasibility, implementation challenges, effectiveness, and potential issues
- Be concise and academic in tone
- Provide a balanced evaluation covering both positive aspects and concerns
- Include your assessment of:
  * How easy or difficult it is to implement the idea
  * Whether the experimental setup is feasible
  * Whether the method is likely to work effectively
  * Any resource requirements or challenges
  * Comparison with existing approaches if relevant
- Write in a natural, flowing style as if you are providing feedback to the authors
- Use the provided evidence sentences from previous reviews as reference to inform your evaluation
"""
    
    full_prompt = (
        system_prompt
        + "\n\n"
        + "Idea to review:\n\n"
        + user_query.strip()
        + "\n\n"
        + "Relevant evidence sentences from previous reviews:\n\n"
        + evidence_text
        + "\n\n"
        + "Write a comprehensive peer review evaluation text discussing the feasibility, implementation challenges, effectiveness, and any concerns about this idea. Write in a natural, flowing style as if you are providing feedback to the authors. Use the evidence sentences above to inform your evaluation."
    )
    
    try:
        evaluation_text = chat_simple(full_prompt)
        return evaluation_text
    except Exception as e:
        return f"ERROR: {e}"


def generate_feasibility_evaluation(idea_text: str, embeddings_dir: str, retrieval_system=None) -> str:
    """
    Generate feasibility evaluation text (concerns)
    
    Args:
        idea_text: Research idea text to evaluate
        embeddings_dir: Embeddings directory path
        retrieval_system: Optional retrieval system instance
        
    Returns:
        str: Feasibility evaluation text (concerns)
    """
    return rag_pipeline_with_retrieval_system(idea_text, embeddings_dir, retrieval_system)


def main():
    """Test function"""
    from Evaluation_utils.test_idea import raw_idea
    
    embeddings_dir = 'RAG_baseline_review_sentence'
    
    print("[1/2] Generating feasibility evaluation text ...")
    result = generate_feasibility_evaluation(raw_idea, embeddings_dir)
    print(result)
    
    # Use eval_feasibility_score.py to score the evaluation result
    print("\n[2/2] Using eval_feasibility_score.py to evaluate score ...")
    from Evaluation_utils.eval_feasibility_score import generate_feasibility_score
    
    # Directly use evaluation text (no formatting needed)
    score_result = generate_feasibility_score(result)
    if score_result and not score_result.startswith("ERROR"):
        print(f"\nScoring result:")
        print(score_result)
    else:
        print("Scoring failed")


if __name__ == "__main__":
    main()

