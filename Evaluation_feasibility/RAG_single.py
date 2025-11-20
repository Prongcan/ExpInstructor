# Ensure the service package at the project root can be imported
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
from Evaluation_utils.eval_feasibility import semantic_match_scores, compare_coverage_via_llm
from RAG_baseline_review_sentence.retrieval_system import EvidenceRetrievalSystem
from Evaluation_utils.test_idea import concerns, raw_idea

def get_keywords(prompt: str):
    """Prompt1: Extract search keywords from user input"""
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
    """Use the retrieval system to search for related evidence sentences"""
    
    all_results = []
    
    for i, keyword in enumerate(keywords, 1):
        print(f"[🔍 Searching evidence] Keyword {i}/{len(keywords)}: '{keyword}'")
        
        try:
            # Semantic search using the retrieval system
            results = retrieval_system.cosine_similarity_search(keyword, top_k=top_k)
            
            keyword_results = []
            for result in results:
                entry = f"📘 Paper ID: {result['paper_id']} | Review ID: {result['review_id']} | Similarity: {result['similarity']:.3f}\n"
                entry += f"{result['evidence']}\n"
                keyword_results.append(entry)
            
            all_results.extend(keyword_results)
            print(f"[📄 Found] {len(keyword_results)} evidence sentences for '{keyword}'")
            
            # Add a short delay to avoid potential rate limits
            if i < len(keywords):  # not the last keyword
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


def _fallback_parse_list(text: str) -> list:
    """Fallback parsing for non-JSON responses"""
    lines = [ln.strip(" -*\t") for ln in text.splitlines()]
    items = [ln for ln in lines if ln]
    return items

def _extract_first_json_array(text: str) -> t.List[str]:
    """
    Extract the first JSON array from any text and parse it into a list of strings.
    Fault tolerance: return an empty list if extraction fails.
    """
    # 1) Directly find balanced [...]
    start = text.find("[")
    if start != -1:
        # Simple bracket counting
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

    # 2) Try to strip ```json ... ``` or ``` ... ``` blocks
    fence_match = re.search(r"```(?:json)?\n([\s\S]+?)```", text)
    if fence_match:
        inner = fence_match.group(1)
        return _extract_first_json_array(inner)

    # 3) Fail and return empty list
    return []


def rag_pipeline_with_retrieval_system(user_query: str, embeddings_dir: str, retrieval_system=None):
    """Full RAG pipeline using the retrieval system"""
    # Step 1: Initialize the retrieval system (if not provided)
    if retrieval_system is None:
        print("[🔧 Initializing] Evidence Retrieval System...")
        retrieval_system = EvidenceRetrievalSystem(embeddings_dir)
        
        # Show statistics
        stats = retrieval_system.get_statistics()
        print(f"[📊 Stats] Total evidence embeddings: {stats['total_embeddings']}, Papers: {stats['total_papers']}")
    
    # Step 2: Keyword extraction
    keywords = get_keywords(user_query)

    # Step 3: Retrieve evidence results (loop until results are found)
    max_attempts = 10
    attempt = 1
    
    while attempt <= max_attempts:
        evidence_sentences = search_evidence_with_retrieval_system(keywords, retrieval_system, top_k=5)
        print(f"[📚 Attempt {attempt}] Found {len(evidence_sentences)} evidence sentences:")
        
        if len(evidence_sentences) > 0:
            # Results found, print evidence information
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
    print("Retrieved Evidence Sentences - Details")
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
    prompt2 = (
        "You are a rigorous peer-reviewer.\n"
        "Task: Critically evaluate the given idea/proposal and GENERATE potential 'concerns'\n"
        "(risks, issues, limitations, feasibility doubts, missing evaluations, ethical/compliance risks).\n"
        "Do NOT extract phrases from the text verbatim; instead, propose concerns based on your assessment.\n"
        "Output Policy (STRICT):\n"
        "- Return ONLY a JSON array of strings, starting with '[' and ending with ']'.\n"
        "- Each item must be a single-line short sentence (no line breaks).\n"
        "- Do NOT include any code fences, markdown, comments, labels, or extra text.\n"
        "- No leading bullets, numbering, or trailing commas inside items.\n"
        "- Aim for 8-12 high-quality, non-duplicative items covering: methodology, data, feasibility, evaluation, ethics/compliance, novelty, scalability.\n\n"
        f"Below are evidence sentences retrieved from reviews based on your query:\n\n{context}\n\n"
        f"Please generate concerns based on these evidence sentences and your professional knowledge for the following question:\n{user_query}\n\n"
        f"Return JSON array only."
    )

    # Step 5: Final answer
    final_answer = chat_simple(prompt2)
    concerns_list = _extract_first_json_array(final_answer)
    
    # Step 6: Append the generated concerns to the file as well
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("Generated Concerns\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Concerns Generated: {len(concerns_list)}\n\n")
        
        for i, concern in enumerate(concerns_list, 1):
            f.write(f"{i}. {concern}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Raw LLM Response\n")
        f.write("=" * 80 + "\n")
        f.write(final_answer + "\n")
    
    print(f"[💾 Updated] Generated concerns appended to: {output_file}")
    
    return concerns_list


if __name__ == "__main__":
    print("[1/3] Generating LLM concerns (using Evidence Retrieval System) ...") 
    # Set embeddings directory
    embeddings_dir = 'RAG_baseline_review_sentence'
    
    gen_concerns = rag_pipeline_with_retrieval_system(raw_idea, embeddings_dir)
    print(f"Generated count: {len(gen_concerns)}")
    print(gen_concerns)

    print("[2/3] Semantic vector matching evaluation ...")
    print(json.dumps({
        "generated_concerns": gen_concerns,
        "semantic_match": semantic_match_scores(concerns, gen_concerns)
    }, ensure_ascii=False, indent=2))

    print("[3/3] Original concern comparison ...")
    final = compare_coverage_via_llm(concerns, gen_concerns)
    print(final)
