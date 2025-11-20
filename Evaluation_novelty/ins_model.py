import os
import sys
import json
import threading
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from Retrive_Generate.graph_retrieval_system import GraphRetrievalSystem
from service import ChatGPT

# Lazy initialization of graph retrieval system
graph_file = "result_v2/all_graphs_cleaned.json"
retrieval_system = None
_retrieval_system_lock = threading.Lock()

def get_retrieval_system():
    """Get graph retrieval system instance (lazy initialization, thread-safe)"""
    global retrieval_system
    if retrieval_system is None:
        with _retrieval_system_lock:
            # Double-check locking pattern
            if retrieval_system is None:
                print("Initializing graph retrieval system...")
                
                # Preload BGE-M3 model to ensure it's only loaded once
                try:
                    from service.BGE_M3 import preload_model
                    preload_model("BAAI/bge-m3")
                except ImportError:
                    print("BGE-M3 preload failed, will use ChatGPT embedding")
                
                # Don't build index first, only load graph data
                retrieval_system = GraphRetrievalSystem(graph_file, build_index_immediately=False)
                print("Graph data loaded, starting index construction...")
                # Delay index construction so embedding model is only loaded once
                retrieval_system.build_index()
                print("Graph retrieval system initialization completed")
    return retrieval_system

@tool
def search_similar_node_and_edge(
    node_query: str, 
    node_k: int = 5,
    edge_query: str = "",
    edge_k: int = 5
) -> str:
    """
    Comprehensive search for nodes and edges.
    All four parameters are indispensable. 
    The first two parameters respectively represent the node to be searched and the expected top node_k nodes. 
    The last two parameters indicate that for each of the selected node_k nodes, 
    all edges of each node are searched to find the top edge_k edges.
    Args:
        node_query: Knowledge entity to search for (e.g., "Large Language Models", "Graph Neural Networks")
        node_k: Number of nodes to return, default is 5
        edge_query: Predicate phrases expressing evaluations or relationships (e.g., "is better than", "is good at", "improves", "solves")
        edge_k: Number of edges per node to return, default is 5
    Returns:
        JSON string containing node-edge-node triplet information
    """
    try:
        import requests
        url = "http://127.0.0.1:9876/search_node_edge"
        payload = {
            "node_query": node_query,
            "node_k": node_k,
            "edge_query": edge_query,
            "edge_k": edge_k
        }
        response = requests.post(url, json=payload, timeout=3000)
        response.raise_for_status()
        data = response.json()
        if "result" in data:
            results = data["result"]
            # 8. Call LLM for final reranking and summarization
            prompt = f"""
            You are an innovation evaluator. Your task is to assess the novelty/innovation level of an entity based on retrieved evidence.
            The user's query intent: find evidence about entity [{node_query}] and relation [{edge_query}] to evaluate its innovation level.
            Retrieved Results: {str(results)}
            
            **Core Task: Innovation Assessment**
            This summary is specifically designed to determine the innovation level of the queried entity. 
            You need to understand the query's intent by analyzing the retrieved results to determine whether the entity [{node_query}] is primarily a **PROBLEM** or a **METHOD**, then assess innovation accordingly.
            
            **Step 1: Determine Entity Type (Problem vs Method)**
            Based on the retrieved results, analyze the context and usage patterns to determine:
            - If [{node_query}] appears more frequently as a **PROBLEM** (something to be solved, addressed, or researched) → Focus ONLY on problem innovation assessment
            - If [{node_query}] appears more frequently as a **METHOD** (a technique, approach, or solution used to solve problems) → Focus ONLY on method innovation assessment
            
            **Step 2: Innovation Assessment Based on Entity Type**
            
            **If the entity is a PROBLEM:**
            - **High Innovation**: The problem is rarely addressed, novel, or has not been effectively solved
            - **Moderate Innovation**: The problem has been partially addressed but remains challenging
            - **Low Innovation**: The problem has been solved many times, is well-researched, or is a common/standard problem
            
            **If the entity is a METHOD:**
            - **High Innovation**: The method is novel, sophisticated, and has not been widely used or applied
            - **Moderate Innovation**: The method has some unique aspects but shares similarities with existing approaches
            - **Low Innovation**: The method is common, widely used, basic/low-level (e.g., "Prompt engineering"), or is standard practice in the field
            
            **Please:**
            1. Sort the results by relevance of the query's intent.
            2. **First, determine entity type**: Analyze the retrieved results to determine if [{node_query}] is primarily a PROBLEM or a METHOD.
            3. **Then, provide a comprehensive innovation assessment summary** that includes:
            
            **Summary Structure:**
            - **Entity Type Determination**: 
              * Clearly state whether [{node_query}] is identified as a PROBLEM or a METHOD based on the retrieved results
              * Explain the reasoning for this determination (cite evidence from results)
              * If determined as PROBLEM: "Based on the results, [{node_query}] is identified as a PROBLEM. The assessment will focus solely on the problem's innovation level."
              * If determined as METHOD: "Based on the results, [{node_query}] is identified as a METHOD. The assessment will focus solely on the method's innovation level."
            
            - **Innovation Level Assessment**: Based on the entity type and retrieved results, determine the innovation level:
              * For PROBLEM: Assess whether the problem is novel/rarely addressed (high), partially addressed (moderate), or frequently solved (low)
              * For METHOD: Assess whether the method is novel/sophisticated (high), has unique aspects (moderate), or is common/basic (low)
            
            - **Evidence-Based Reasoning**: 
              * Cite specific evidence from the results to support your innovation assessment
              * For PROBLEM: Count how many times the problem has been addressed/solved (if many → low innovation)
              * For METHOD: Identify if the method is common/basic/widely used (if yes → low innovation), or assess its sophistication level
            
            - **Innovation Indicators**:
              * For PROBLEM: Problem novelty - Is this problem frequently addressed? (Many solutions → low innovation)
              * For METHOD: Method novelty and sophistication - Has this method been widely used? Is it basic/common or advanced? (Basic/common → low innovation)
            
            - **Conclusion**: Summarize what the retrieved results reveal about the innovation level of the queried entity (PROBLEM or METHOD), and explain how this relates to the overall novelty assessment.
            
            **Important Notes:**
            - Focus on innovation/novelty assessment, not just relevance
            - Be critical: If results show the problem has been solved many times or the method is common → explicitly state LOW innovation
            - If the method is basic (like "Prompt engineering") or widely used → explicitly state LOW innovation
            - Connect the evidence directly to innovation level judgment
            
            **Output Format:**
            Output a JSON object with the following structure:
            {{
              "results": [sorted and filtered results from the input, maintaining the original structure],
              "summary": "A comprehensive and detailed innovation assessment summary that includes all the elements described in the Summary Structure above. This summary should be placed at the end, after all evidence results, and should provide an overall analysis of the innovation level based on ALL the retrieved evidence."
            }}
            
            The "summary" field should be a detailed text analysis (not a JSON object) that comprehensively evaluates the innovation level based on all the evidence in the results array.
            """
            print("LLM reranking and summarizing...")
            chat_simple = ChatGPT.chat_simple(prompt)

            #print(chat_simple)
            # chat_simple = self.parse_llm_json_output(chat_simple)
            
            return chat_simple
        elif "error" in data:
            return f"Service error: {data['error']}"
        else:
            return "Unknown error"
    except Exception as e:
        return f"Service call failed: {str(e)}"

@tool
def get_original_review_text(paper_id: Annotated[str, "Paper ID"], review_id: Annotated[str, "Review ID"]) -> str:
    """
    Retrieve original review text from iclr2024_simple.json based on paper_id and review_id
    
    Args:
        paper_id: The paper ID to search for
        review_id: The review ID to search for
    
    Returns:
        Original review text including summary, strengths, weakness, and suggestions
    """
    try:
        # Load the JSON file
        with open("data/ICLR/iclr2024_simple.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Search for matching paper_id and review_id
        for entry in data:
            if entry.get("paper_id") == paper_id:
                # Find matching review_id
                for review in entry.get("review_contents", []):
                    if review.get("review_id") == review_id:
                        content = review.get("content", {})
                        
                        result = f"Found original review text for Paper ID: {paper_id}, Review ID: {review_id}\n"
                        result += "=" * 80 + "\n"
                        result += "ORIGINAL REVIEW CONTENT:\n"
                        result += "-" * 50 + "\n"
                        
                        if content.get("strengths"):
                            result += f"STRENGTHS:\n{content['strengths']}\n\n"
                        if content.get("weakness"):
                            result += f"WEAKNESS:\n{content['weakness']}\n\n"
                        if content.get("suggestions"):
                            result += f"SUGGESTIONS:\n{content['suggestions']}\n\n"
                        
                        return result
        
        return f"No review found for Paper ID: {paper_id}, Review ID: {review_id}"
        
    except Exception as e:
        return f"Error retrieving review text: {str(e)}"

def build_agent_user_prompt(research_idea: str) -> str:
    return f"""
        You are a professional evaluator focusing on the novelty of the idea.
        I will provide you with an academic idea. Your task is to evaluate only its level of innovation.
        Please focus exclusively on the novelty and originality of the idea (analyze the problems of idea and the innovativeness of its methods) 
        (how new, unique, or creative it is compared to existing research or conventional approaches in the field).
        Your response should:
        Be concise and academic in tone.
        Avoid discussing feasibility, impact, or methodology.
        Provide a clear judgment on the innovation level with your serious analysis and reasoning.

        You can use two tools to learn about relevant experiences related to innovation.

        node_query: Enter a specific knowledge entity(from problem and method of the idea) (e.g.,"LLM's hallucination", "dialect recognition", "speech processing", "the method using XXX").
        edge_query: Enter ONLY predicate phrases WITHOUT including the entity name. you can use generic relationship expressions to search for the experience of novelty:
        - For relationships: "is a method to sovle", "is solved by",  "has been researched widely", "has not been used to solve" ...(Use these more)
        - For explict novelty comment: "lacks novelty", "is a new method", "is a new problem" ...(Use these more)
        - For positive effects: "can improve", "helps"...
        - For negative effects: "has limitations", "faces challenges"...
        This will return you several pieces of evidence and a summary. 
        You need to evaluate the innovativeness of this idea based on these pieces of evidence and the summary.
        You **MUST** perform at least 20 tool calls（including at least 5 get_original_review_text calls and at least 15 search_similar_node_and_edge calls） before providing the final "concerns" answer.
        Iteration & Tool Scheduling:
        - Prefer making only 1 tool call per step; absolutely no more than 2 in any single step.
        - If multiple queries are needed, split them into multiple steps/rounds to collect evidence gradually.
        - After each tool result, briefly reflect and plan the next single tool call.
        the idea to be evaluated is: {research_idea}
    """

def create_custom_agent():
    """
    Create custom Agent
    """
    # Set OpenAI API key (please replace with your actual API key)
    os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
    
    # Initialize ChatGPT-4o model
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7
        #max_tokens=1000
    )
    
    # Define tool list
    tools = [
        search_similar_node_and_edge,
        get_original_review_text
        ]
    
    # Create Agent with enhanced parameters
    agent = create_react_agent(
        model=model, 
        tools=tools,
        prompt="""
       You are a rigorous peer-reviewer.
You are a professional evaluator focusing on the novelty of the idea.
I will provide you with an academic idea. Your task is to evaluate only its level of innovation.
Please focus exclusively on the novelty and originality of the idea — how new, unique, or creative it is compared to existing research or conventional approaches in the field.
Your response should:
Be concise and academic in tone.
Avoid discussing feasibility, impact, or methodology.
Provide a clear judgment on the innovation level with your serious analysis and reasoning.

You can use two tools to understand how existing methods address this problem and whether this problem is widely concerned,
so that you can judge the novelty of the idea. The tools are:
search_similar_node_and_edge and get_original_review_text.
1. Tool Usage Specifications (Key Supplement for Edge Query)
1.1 search_similar_node_and_edge (Search Similar Nodes and Edges)
The core function of this tool is to retrieve information related to knowledge entities (nodes) and their relationships/evaluations (edges). You should systematically explore different entities mentioned in the research idea and query them separately.

**Entity Discovery Strategy:**
- Extract ALL distinct knowledge entities(especially the entities related to the problem and method) from the research idea (e.g., for "LLM improving dialect recognition", extract "LLM", "dialect", "recognition", "speech recognition", "language processing" etc.)
- Query each entity separately with different node_query values
- Use multiple search rounds to cover different aspects and entities

**Edge Query Design Rules:**
node_query: Enter a specific knowledge entity (e.g., "Large Language Models", "Graph Neural Networks", "dialect recognition", "speech processing", "the method using XXX").
edge_query: Enter ONLY predicate phrases WITHOUT including the entity name. you can use generic relationship expressions to search for the experience of novelty:
- For relationships: "is a method to sovle", "is solved by",  "has been researched widely", "has not been used to solve" ...(Use these more)
- For explict novelty comment: "lacks novelty", "is a new method", "is a new problem" ...(Use these more)
- For positive effects: "can improve", "helps"...
- For negative effects: "has limitations", "faces challenges"...

**Examples:**
- WRONG: edge_query = "Halucination of LLM has been solved" (contains entity)
- CORRECT: node_query = "Halucination of LLM", edge_query = "has been solved"
- For "Halucination of LLM can be solved by RAG" idea, query:
  * node_query="Halucination of LLM", edge_query="can be solved by"
  * node_query="RAG", edge_query="is a method to solve"
  * node_query="RAG to solve hallucination", edge_query="lacks novelty"
1.2 get_original_review_text (Get Original Review Text)
This tool is used to retrieve full-text or key segments of academic reviews/papers. When calling it, you must ensure that the retrieved content is highly relevant to the "concerns" you want to analyze later — it should directly involve the risks, limitations, challenges, or potential problems of the research idea, rather than irrelevant background information.
2. Strict Tool Call Requirements
You MUST perform at least 15 tool calls before providing the final innovation evaluation answer. Among them, at least 5 calls must be to get_original_review_text (to ensure you have sufficient review evidence to support each concern).
Do not provide the final "concerns" answer until you have completed at least 15 tool calls.
For search_similar_node_and_edge, use different combinations of node_query and edge_query (covering both positive and negative information orientations related to the research idea) to collect comprehensive evidence — avoid repeating the same query, as this will lead to incomplete information collection.
For get_original_review_text, each call must target a review/paper that directly discusses the potential concerns, risks, or challenges of the research idea (e.g., reviews that point out the instability of a certain algorithm, the scarcity of a certain dataset, or the ethical risks of a certain application scenario).

**Search Strategy:**
- Query core problems, methods, techniques, and approaches mentioned in the idea
- Search for the promblem has been solved or not
- Look for reviews that explicitly discuss novelty or lack of novelty
- Compare the proposed approach with existing solutions to the same problem

Output Policy (STRICT):
- Return a text evaluating innovation (which should include reasonable reasons) based on the tools' results (especially the summary of the evidence).
- Please evaluate the innovativeness of the idea clearly and emphatically.

Iteration & Tool Scheduling:
- Prefer making only 1 tool call per step; absolutely no more than 2 in any single step.
- If multiple queries are needed, split them into multiple steps/rounds to collect evidence gradually.
- After each tool result, briefly reflect and plan the next single tool call.
       """, # 这个prompt是给agent的，用来指导agent的行为
        version="v2" # Use the latest version
    )
    
    return agent

def demo_basic_usage(prompt):
    """
    Demonstrate basic usage
    """
    print("=== Basic Agent Demo ===")
    
    # Create Agent
    agent = create_custom_agent()
    
    # Create user message
    input_message = HumanMessage(content=prompt)
    
    # Run Agent
    #print("User question:", input_message.content)
    #print("\nAgent response:")
    
    step_count = 0
    tool_call_count = 0
    review_text_calls = 0
    
    # Set step limit to 50 steps
    config = {"recursion_limit": 50}
    for step in agent.stream({"messages": [input_message]}, config=config, stream_mode="values"):
        step_count += 1
        print(f"=== Step {step_count} ===")
        if step["messages"]:
            last_message = step["messages"][-1]
            print(f"Message type: {type(last_message).__name__}")
            print(f"Content: {last_message.content}")
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                tool_call_count += len(last_message.tool_calls)
                print(f"Tool calls: {last_message.tool_calls}")
                for i, tool_call in enumerate(last_message.tool_calls):
                    print(f"  Tool call {i+1}: {tool_call['name']}({tool_call['args']})")
                    if tool_call['name'] == 'get_original_review_text':
                        review_text_calls += 1
                print(f"📊 Current total tool calls: {tool_call_count}")
                print(f"📊 get_original_review_text calls: {review_text_calls}")
        print("-" * 50)
    
    print(f"\n🎯 Final statistics:")
    print(f"Total steps: {step_count}")
    print(f"Total tool calls: {tool_call_count}")
    print(f"get_original_review_text calls: {review_text_calls}")
    
    if tool_call_count >= 12 and review_text_calls >= 5:
        print("✅ Successfully met requirements: at least 12 tool calls, including at least 5 get_original_review_text!")
    elif tool_call_count >= 12:
        print(f"⚠️  Met 12 tool calls requirement, but get_original_review_text calls insufficient (current: {review_text_calls})")
    elif review_text_calls >= 5:
        print(f"⚠️  Met 5 get_original_review_text calls requirement, but total tool calls insufficient (current: {tool_call_count})")
    else:
        print(f"❌ Requirements not met: need at least 12 tool calls (current: {tool_call_count}), including at least 5 get_original_review_text (current: {review_text_calls})")

def main():
    """
    Main function - run all demonstrations
    """
    print("🤖 Custom Agent Demo based on ChatGPT-4o-mini")
    print("=" * 60)
    
    # Note: Please ensure you have set the correct OpenAI API key before running
    print("⚠️  Note: Please set your OpenAI API key in the code first")
    print("   Replace 'YOUR_OPENAI_API_KEY' with your actual API key")
    print("=" * 60)
    
    try:
        
        # Basic usage demonstration

        research_idea = """
**Problem:**
Current methods for evaluating research ideas using Large Language Models (LLMs) face challenges such as instability, bias sensitivity, and an inability to capture complex semantic information effectively. This leads to poor-quality evaluations, which can undermine the development of scientific fields.

**Motivation:**
As LLMs become more integrated into research idea evaluations, there is a crucial need to improve their reliability and accuracy. Existing approaches often overlook relevant semantic details and exhibit strong biases, necessitating a new framework that can comprehensively assess the quality of research ideas.

**Method:**
The paper introduces **GraphEval**, a lightweight graph-based LLM framework designed to evaluate ideas more effectively by breaking them down into simpler "viewpoint-nodes" that capture distinct arguments or facts. This framework incorporates two key components:
1. **GraphEval-LP**: A training-free label propagation method that propagates quality labels through a network of viewpoint-nodes.
2. **GraphEval-GNN**: A Graph Neural Network approach that learns to predict quality labels based on the connections between viewpoint-nodes while also integrating novelty detection to mitigate plagiarism issues.

Extensive experiments demonstrate that GraphEval improves evaluation accuracy by at least 14% while maintaining low computational costs.
"""
        prompt = f"""
You need to analyze a research idea and provide corresponding guidance. Please make reasonable use of the tools based on the user's prompt.
You can use two tools: search_similar_node_and_edge and get_original_review_text.
1. Tool Usage Specifications (Key Supplement for Edge Query)
1.1 search_similar_node_and_edge (Search Similar Nodes and Edges)
The core function of this tool is to retrieve information related to knowledge entities (nodes) and their relationships/evaluations (edges). You should systematically explore different entities mentioned in the research idea and query them separately.

**Entity Discovery Strategy:**
- Extract ALL distinct knowledge entities from the research idea (e.g., for "LLM improving dialect recognition", extract "LLM", "dialect", "recognition", "speech recognition", "language processing" etc.)
- Query each entity separately with different node_query values
- Use multiple search rounds to cover different aspects and entities

**Edge Query Design Rules:**
node_query: Enter a specific knowledge entity (e.g., "Large Language Models", "Graph Neural Networks", "dialect recognition", "speech processing", "machine learning models").
edge_query: Enter ONLY predicate phrases WITHOUT including the entity name. Use generic relationship expressions:
- For positive effects: "can improve", "helps", "enhances", "strengthens", "boosts", "optimizes"
- For negative effects: "has limitations", "faces challenges", "has shortcomings", "struggles with", "prone to errors"
- For relationships: "is related to", "depends on", "influences", "causes"

**Examples:**
- WRONG: edge_query = "LLM improves dialect recognition" (contains entity)
- CORRECT: edge_query = "can improve recognition" (pure predicate)
- For "LLM improving dialect recognition" idea, query:
  * node_query="LLM", edge_query="can improve recognition"
  * node_query="dialect", edge_query="difficult to recognize"
  * node_query="speech recognition", edge_query="faces challenges"
1.2 get_original_review_text (Get Original Review Text)
This tool is used to retrieve full-text or key segments of academic reviews/papers. When calling it, you must ensure that the retrieved content is highly relevant to the "concerns" you want to analyze later — it should directly involve the risks, limitations, challenges, or potential problems of the research idea, rather than irrelevant background information.
2. Strict Tool Call Requirements
You MUST perform at least 15 tool calls before providing the final "concerns" answer. Among them, at least 5 calls must be to get_original_review_text (to ensure you have sufficient review evidence to support each concern).
Do not provide the final "concerns" answer until you have completed at least 15 tool calls.
For search_similar_node_and_edge, use different combinations of node_query and edge_query (covering both positive and negative information orientations related to the research idea) to collect comprehensive evidence — avoid repeating the same query, as this will lead to incomplete information collection.
For get_original_review_text, each call must target a review/paper that directly discusses the potential concerns, risks, or challenges of the research idea (e.g., reviews that point out the instability of a certain algorithm, the scarcity of a certain dataset, or the ethical risks of a certain application scenario).
3. Strategic Review Selection Standards (for get_original_review_text)
To ensure that the retrieved reviews can effectively support the subsequent "concerns" analysis, you must follow these selection principles:
Prioritize reviews that explicitly discuss the limitations, risks, or challenges of the core methods/assumptions of the research idea (e.g., if the research idea uses "few-shot learning", prioritize reviews that analyze the generalization risk of few-shot learning).
Focus on reviews that provide specific technical details to support concerns (e.g., reviews that quantify the error rate increase of a certain model in small-sample scenarios, rather than only making vague claims like "this method has limitations").
Select reviews that compare similar research and point out common concerns (e.g., reviews that mention "most existing methods in this field face the problem of data bias, and the proposed method may also inherit this risk").
Avoid retrieving reviews that only introduce background knowledge of the field or praise the advantages of related methods (these reviews cannot support "concerns" analysis).
Ensure that each retrieved review can directly correspond to at least one of the "concerns" you will list later — that is, the review content can be used as evidence to prove the existence or severity of that concern.
4. Final "Concerns" Answer Requirements
After completing at least 15 tool calls (including at least 5 get_original_review_text calls), you need to generate a "concerns" answer with a total length of at least 2000 words, presented in a point-by-point format (you must list at least 10 points, and each point is an independent concern).
Each point of concern must include the following 4 core components (to ensure depth and evidence support):
4.1 Clear Concern Definition
Clearly state what the concern is (e.g., "The research idea relies on a large amount of labeled data, which may face the risk of insufficient data availability in practical application scenarios" — avoid vague expressions like "this method may have problems").
4.2 Evidence Support from Tool Calls
Cite the evidence obtained from tool calls to prove that this concern is valid:
For evidence from search_similar_node_and_edge: Clearly state the query combination (node_query + edge_query) and the key information retrieved (e.g., "Through search_similar_node_and_edge with node_query = 'labeled data for medical image segmentation' and edge_query = 'is insufficient in practical application scenarios', it was found that 3 out of 5 retrieved studies mentioned that the labeled data required by the segmentation model is difficult to collect in primary hospitals, with an average collection cycle exceeding 6 months").
For evidence from get_original_review_text: Clearly state the core content of the retrieved review (including the review's focus on the research idea) and directly quote or paraphrase the key sentences that support the concern (e.g., "In the review 'Challenges of Medical Image Segmentation in Clinical Practice' retrieved via get_original_review_text, the authors point out that 'most state-of-the-art segmentation models require at least 1000 labeled cases to achieve stable performance, but in most non-tertiary hospitals, the number of labeled cases for rare diseases is often less than 50, leading to the failure of model deployment' — this directly supports the concern of insufficient data availability").
4.3 Potential Impact Analysis
Analyze the specific impact of this concern on the research idea if it is not addressed:
Technical impact: e.g., "If the data availability problem is not solved, the model trained in the research may have an accuracy rate lower than 60% in practical application, which is far below the clinical application standard of 85%".
Resource impact: e.g., "To collect sufficient labeled data, the research team may need to invest an additional 6-12 months and \(50,000-\)100,000 in data labeling costs, which will significantly delay the research progress and increase the budget".
Application impact: e.g., "If this concern is not resolved, the research results can only be applied in tertiary hospitals with sufficient data, and cannot be promoted to primary hospitals (which account for 70% of the market), greatly reducing the practical value of the research".
4.4 Connection to the Research Idea
Explain in detail how this concern is closely related to the core design of the research idea (avoid listing concerns that are irrelevant to the research itself):
For example, if the research idea uses "transfer learning from natural images to medical images", the concern should be connected to the core method: "The research idea adopts transfer learning from natural images (e.g., ImageNet) to medical images (e.g., chest X-rays). However, the retrieved evidence shows that natural images and medical images have very different feature distributions (e.g., the texture and color features of lung nodules in X-rays are completely different from those of objects in ImageNet). This feature distribution difference will lead to the transfer learning effect being far lower than expected, which is a concern directly related to the core method of the research".
5. Critical Citation and Expression Standards
Evidence must be traceable: For each concern, the evidence cited must clearly correspond to a specific tool call (i.e., which search_similar_node_and_edge query or which get_original_review_text retrieval it comes from) — do not use "some studies have shown" or other untraceable expressions.
Avoid redundant content: Each concern should focus on one independent problem, and there should be no overlap or repetition between different concerns (e.g., "insufficient data availability" and "high data collection costs" are two independent concerns and should be listed separately, not merged into one).
Language must be precise: Use specific data, scenarios, or technical parameters to describe concerns (e.g., "the model's inference time may exceed 5 seconds" is better than "the model's inference speed is slow"; "the error rate may increase by 15%-20% in low-light scenarios" is better than "the model performs poorly in special scenarios").
Do not ignore minor but critical concerns: In addition to major technical concerns (e.g., model accuracy), also consider other critical concerns such as ethical risks (e.g., "the research uses patient medical data, which may face privacy leakage risks"), environmental costs (e.g., "the model training requires 100 GPUs running for 30 days, resulting in high energy consumption"), and maintenance difficulties (e.g., "the model requires frequent retraining with new data, which is difficult for small medical institutions to implement").
        idea to be process： 
        {research_idea}
        
        """
        demo_basic_usage(prompt)
        
    except Exception as e:
        print(f"❌ Runtime error: {e}")
        print("Please check API key settings and network connection")


if __name__ == "__main__":
    main()