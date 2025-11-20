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

# 延迟初始化图检索系统
graph_file = "result_v2/all_graphs_cleaned.json"
retrieval_system = None
_retrieval_system_lock = threading.Lock()

def get_retrieval_system():
    """获取图检索系统实例（延迟初始化，线程安全）"""
    global retrieval_system
    if retrieval_system is None:
        with _retrieval_system_lock:
            # 双重检查锁定模式
            if retrieval_system is None:
                print("正在初始化图检索系统...")
                
                # 预加载BGE-M3模型，确保只加载一次
                try:
                    from service.BGE_M3 import preload_model
                    preload_model("BAAI/bge-m3")
                except ImportError:
                    print("BGE-M3预加载失败，将使用ChatGPT embedding")
                
                # 先不构建索引，只加载图数据
                retrieval_system = GraphRetrievalSystem(graph_file, build_index_immediately=False)
                print("图数据加载完成，开始构建索引...")
                # 延迟构建索引，这样embedding模型只会加载一次
                retrieval_system.build_index()
                print("图检索系统初始化完成")
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
            # 8. 调用 LLM 进行最终重排序和总结
            prompt = f"""
            You are a helpful assistant. 
            Sort these graph results by combined node and edge similarity and provide a summary.
            The user's request: find evidence about entity [{node_query}] and relation [{edge_query}].
            Results: {str(results)}
            
            Please:
            1. Sort the results by combined node and edge similarity (highest first)
            2. Analyze if the results contain information that closely matches the query
            3. Provide a natural language summary that analyzes:
            - Relevance assessment: How well do the results match the query? Are they highly relevant, moderately relevant, or mostly irrelevant?
            - Prior experience evaluation: Based on the quality and relevance of results, does this indicate rich prior experience, moderate experience, or limited experience in this area?
            - Overall insights: What do these results collectively suggest about the research domain, common challenges, or established patterns?
            
            Write a comprehensive summary that naturally explains what the retrieved results reveal about the query topic and the existing knowledge base.
            
            Output JSON with the same structure as the input results, but add a "summary" field at the end containing the analysis.
            """
            print("LLM重排序和总结中...")
            chat_simple = ChatGPT.chat_simple(prompt)

            #print(chat_simple)
            # chat_simple = self.parse_llm_json_output(chat_simple)
            
            return chat_simple
        elif "error" in data:
            return f"服务报错: {data['error']}"
        else:
            return "未知错误"
    except Exception as e:
        return f"调用服务失败: {str(e)}"

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
                # 查找匹配的review_id
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
        You are a rigorous peer-reviewer.
        Task: Critically evaluate the given idea/proposal and GENERATE potential 'concerns' of feasibility
        (feasibility, feasibility doubts, missing evaluations).
        Do NOT extract phrases from the text verbatim; instead, propose concerns based on your assessment.
        Output Policy (STRICT):
        - Return ONLY a JSON array of strings, starting with '[' and ending with ']'.
        - Each item must be a single-line short sentence (no line breaks).
        - Do NOT include any code fences, markdown, comments, labels, or extra text.
        - No leading bullets, numbering, or trailing commas inside items.
        - Aim for 8-12 high-quality, non-duplicative items covering: feasibility, feasibility doubts, missing evaluations.
        idea to be process： 

        You can use two tools to understand how existing methods address this problem, what are the positive and negative characteristics of the method proposed in the idea?, what are the characteristics of the dataset?: search_similar_node_and_edge and get_original_review_text.
        You **MUST** perform at least 20 tool calls（including at least 5 get_original_review_text calls and at least 15 search_similar_node_and_edge calls） before providing the final "concerns" answer.
        Iteration & Tool Scheduling:
        - Prefer making only 1 tool call per step; absolutely no more than 2 in any single step.
        - If multiple queries are needed, split them into multiple steps/rounds to collect evidence gradually.
        - After each tool result, briefly reflect and plan the next single tool call.
        {research_idea}
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
        temperature=0.9
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
Task: Critically evaluate the given idea/proposal and GENERATE potential 'concerns' of feasibility
(feasibility, feasibility doubts, missing evaluations).
Do NOT extract phrases from the text verbatim; instead, propose concerns based on your assessment.
You can use two tools to understand how existing methods address this problem, what are the positive and negative characteristics of the method proposed in the idea?, what are the characteristics of the dataset?: search_similar_node_and_edge and get_original_review_text.
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

You can use two tools to understand how existing methods address this problem, what are the positive and negative characteristics of the method proposed in the idea?, what are the characteristics of the dataset?: search_similar_node_and_edge and get_original_review_text.

Output Policy (STRICT):
- Return ONLY a JSON array of strings, starting with '[' and ending with ']'.
- Each item must be a single-line short sentence (no line breaks).
- Do NOT include any code fences, markdown, comments, labels, or extra text.
- No leading bullets, numbering, or trailing commas inside items.
- Aim for 8-12 high-quality, non-duplicative items covering: feasibility, feasibility doubts, missing evaluations.
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
    
    # 设置步数限制为50步
    config = {"recursion_limit": 50}
    for step in agent.stream({"messages": [input_message]}, config=config, stream_mode="values"):
        step_count += 1
        print(f"=== 步骤 {step_count} ===")
        if step["messages"]:
            last_message = step["messages"][-1]
            print(f"消息类型: {type(last_message).__name__}")
            print(f"内容: {last_message.content}")
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                tool_call_count += len(last_message.tool_calls)
                print(f"工具调用: {last_message.tool_calls}")
                for i, tool_call in enumerate(last_message.tool_calls):
                    print(f"  工具调用 {i+1}: {tool_call['name']}({tool_call['args']})")
                    if tool_call['name'] == 'get_original_review_text':
                        review_text_calls += 1
                print(f"📊 当前总工具调用次数: {tool_call_count}")
                print(f"📊 get_original_review_text调用次数: {review_text_calls}")
        print("-" * 50)
    
    print(f"\n🎯 最终统计:")
    print(f"总步骤数: {step_count}")
    print(f"总工具调用次数: {tool_call_count}")
    print(f"get_original_review_text调用次数: {review_text_calls}")
    
    if tool_call_count >= 12 and review_text_calls >= 5:
        print("✅ 成功完成要求: 至少12次工具调用，其中至少5次get_original_review_text!")
    elif tool_call_count >= 12:
        print(f"⚠️  达到12次工具调用要求，但get_original_review_text调用次数不足5次 (当前: {review_text_calls})")
    elif review_text_calls >= 5:
        print(f"⚠️  达到5次get_original_review_text调用要求，但总工具调用次数不足12次 (当前: {tool_call_count})")
    else:
        print(f"❌ 未达到要求: 需要至少12次工具调用(当前: {tool_call_count})，其中至少5次get_original_review_text(当前: {review_text_calls})")

def main():
    """
    Main function - run all demonstrations
    """
    print("🤖 Custom Agent Demo based on ChatGPT-4o-mini")
    print("=" * 60)
    
    # Note: Please ensure you have set the correct OpenAI API key before running
    print("⚠️  Note: Please set your OpenAI API key in the code first")
    print("   Replace 'your-openai-api-key-here' with your actual API key")
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