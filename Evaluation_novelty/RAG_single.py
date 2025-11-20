# 确保可以导入到项目根下的 service 包
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
from Evaluation_utils.test_idea import raw_idea
from Evaluation_utils.eval_novelty import generate_novelty_score

def get_keywords(prompt: str):
    """Prompt1: Extract retrieval keywords from user input"""
    system_prompt = (
        "You are a research idea evaluator. I will provide you with an academic idea, and you need to output 10 keywords for searching related evidence sentences..\n"
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
                time.sleep(0.1)  # Short delay, as no external API call is needed
                
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

def _extract_first_json_array(text: str) -> t.List[str]:
    """
    Extract the first JSON array from any text and parse it as a string list.
    Fault-tolerant: returns empty list if extraction fails.
    """
    # 1) 直接找平衡的 [...]
    start = text.find("[")
    if start != -1:
        # 简单括号计数
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '[':
                depth += 1
            elif text[i] == ']':
                depth -= 1
                if depth == 0:
                    try:
                        arr = json.loads(text[start:i+1])
                        if isinstance(arr, list):
                            return [str(x).strip() for x in arr if str(x).strip()]
                    except Exception:
                        pass
                    break

    # 2) 尝试剥离```json ... ```或``` ... ```
    fence_match = re.search(r"```(?:json)?\n([\s\S]+?)```", text)
    if fence_match:
        inner = fence_match.group(1)
        return _extract_first_json_array(inner)

    # 3) 失败返回空
    return []


def rag_pipeline_with_retrieval_system(user_query: str, embeddings_dir: str, retrieval_system=None):
    """Complete RAG pipeline using retrieval system"""
    # Step 1: Initialize retrieval system (if not provided)
    if retrieval_system is None:
        print("[🔧 Initializing] Evidence Retrieval System...")
        retrieval_system = EvidenceRetrievalSystem(embeddings_dir)
        
        # Display statistics
        stats = retrieval_system.get_statistics()
        print(f"[📊 Stats] Total evidence embeddings: {stats['total_embeddings']}, Papers: {stats['total_papers']}")
    
    # Step 2: Keyword extraction
    keywords = get_keywords(user_query)

    # Step 3: Retrieve evidence results (loop until results found)
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

    # Step 4: Print retrieved results
    print(f"\n{'='*80}")
    print("Retrieved Evidence Sentences Details")
    print(f"{'='*80}")
    for i, evidence in enumerate(evidence_sentences, 1):
        print(f"\n--- Evidence {i} ---")
        print(evidence)
        print("-" * 50)
    
    # Step 4.5: Save query and retrieval results to file
    output_file = os.path.join(os.path.dirname(__file__), "evidence_retrieval_results.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Query: {user_query}\n")
        f.write(f"Keywords: {', '.join(keywords)}\n")
        f.write(f"Retrieval Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Evidence Sentences Found: {len(evidence_sentences)}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, evidence in enumerate(evidence_sentences, 1):
            f.write(f"--- Evidence {i} ---\n")
            f.write(evidence)
            f.write("\n" + "-" * 50 + "\n\n")
    
    print(f"\n[💾 Saved] Query and retrieval results saved to: {output_file}")
    
    # Step 5: Build Prompt2
    context = "\n\n".join(evidence_sentences)
    prompt2 = f"""
    You are a professional evaluator focusing on the novelty of the idea.
    I will provide you with an academic idea. Your task is to evaluate only its level of innovation.
    Please focus exclusively on the novelty and originality of the idea (analyze the problems of idea and the innovativeness of its methods) 
    (how new, unique, or creative it is compared to existing research or conventional approaches in the field).
    Your response should:
    Be concise and academic in tone.
    Avoid discussing feasibility, impact, or methodology.
    Provide a clear judgment on the innovation level with your serious analysis and reasoning.

    Below are evidence sentences retrieved from reviews based on your query:
    {context}

    Please generate concerns based on these evidence sentences and your professional knowledge for the following question:
    {user_query}

    Output Policy (STRICT):
    - Return a text evaluating innovation (which should include reasonable reasons) based on the evidence sentences.
    - Please evaluate the innovativeness of the idea clearly and emphatically.

    Return JSON array only.
    """

    # Step 6: Final answer
    final_answer = chat_simple(prompt2)
    
    return final_answer


if __name__ == "__main__":
    embeddings_dir = 'RAG_baseline_review_sentence'
    print("\n[1/2] Using RAG pipeline to generate novelty evaluation ...")
    novelty_result = rag_pipeline_with_retrieval_system(raw_idea, embeddings_dir)

    # Use GPT to score the evaluation result
    print("\n[2/2] Using GPT model to evaluate score ...")
    gpt_score_result = generate_novelty_score(novelty_result)
    if gpt_score_result and not gpt_score_result.startswith("ERROR"):
        print(f"\nGPT scoring result:")
        print(gpt_score_result)
    else:
        print("GPT scoring failed")
