import os
import json
import re
import typing as t
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from ins_model import create_custom_agent, build_agent_user_prompt
from langchain_core.messages import HumanMessage
# Reuse the evaluation logic already implemented in evaluate_single
from Evaluation_utils.eval_feasibility import semantic_match_scores, compare_coverage_via_llm
from Evaluation_utils.test_idea import raw_idea, concerns

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

def generate_concerns_via_agent(agent, idea_text: str) -> t.List[str]:
    
    # Reuse the user prompt template from Generator_concern
    input_message = HumanMessage(content=build_agent_user_prompt(idea_text))

    # Run the agent once and get the final message content (long report)
    final_text = ""
    step_count = 0
    tool_call_count = 0
    tool_calls_details = []
    
    print("=" * 80)
    print("Starting Agent tool call process")
    print("=" * 80)
    
    # Set step limit to 50
    config = {"recursion_limit": 50}
    for step in agent.stream({"messages": [input_message]}, config=config, stream_mode="values"):
        step_count += 1
        print(f"\n=== Step {step_count} ===")
        
        if step.get("messages"):
            last_message = step["messages"][-1]
            print(f"Message type: {type(last_message).__name__}")
            
            # Print message content
            if last_message.content:
                print(f"Message content: {last_message.content}")
                final_text = last_message.content
            
            # Check for tool calls
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                tool_call_count += len(last_message.tool_calls)
                print(f"🔧 Number of tool calls: {len(last_message.tool_calls)}")
                
                for i, tool_call in enumerate(last_message.tool_calls):
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    tool_id = tool_call.get('id', f'tool_call_{i+1}')
                    
                    print(f"  Tool call {i+1}: {tool_name}")
                    print(f"    Tool ID: {tool_id}")
                    print(f"    Args: {json.dumps(tool_args, ensure_ascii=False, indent=4)}")
                    
                    # Record tool call details
                    tool_calls_details.append({
                        'step': step_count,
                        'tool_name': tool_name,
                        'tool_id': tool_id,
                        'args': tool_args,
                        'timestamp': f"Step {step_count}"
                    })
            
            # Check for tool results
            if hasattr(last_message, 'tool_call_id') and last_message.tool_call_id:
                print(f"🔧 Tool result (ID: {last_message.tool_call_id}):")
                if last_message.content:
                    # Limit output length to avoid overly long output
                    result_content = last_message.content
                    if len(result_content) > 1000:
                        print(f"    Result preview: {result_content[:1000]}...")
                        print(f"    [Result truncated, total length: {len(result_content)} chars]")
                    else:
                        print(f"    Full result: {result_content}")
                    
                    # Update tool call details
                    for detail in tool_calls_details:
                        if detail['tool_id'] == last_message.tool_call_id:
                            detail['result'] = result_content
                            break
        
        print("-" * 50)
    
    print("\n" + "=" * 80)
    print("Agent execution finished - Statistics")
    print("=" * 80)
    print(f"Total steps: {step_count}")
    print(f"Total tool calls: {tool_call_count}")
    
    # Print tool call statistics
    tool_stats = {}
    for detail in tool_calls_details:
        tool_name = detail['tool_name']
        if tool_name not in tool_stats:
            tool_stats[tool_name] = 0
        tool_stats[tool_name] += 1
    
    print("\nTool call statistics:")
    for tool_name, count in tool_stats.items():
        print(f"  {tool_name}: {count} times")
    
    # Print detailed records of all tool calls
    print("\n" + "=" * 80)
    print("All tool call detailed records")
    print("=" * 80)
    for i, detail in enumerate(tool_calls_details, 1):
        print(f"\nTool call #{i}:")
        print(f"  Step: {detail['timestamp']}")
        print(f"  Tool name: {detail['tool_name']}")
        print(f"  Tool ID: {detail['tool_id']}")
        print(f"  Args: {json.dumps(detail['args'], ensure_ascii=False, indent=4)}")
        if 'result' in detail:
            result = detail['result']
            if len(result) > 500:
                print(f"  Result: {result[:500]}...")
                print(f"  [Result truncated, total length: {len(result)} chars]")
            else:
                print(f"  Result: {result}")
        else:
            print(f"  Result: [No result obtained]")
    
    print("\n" + "=" * 80)
    print("Final Agent Output")
    print("=" * 80)
    print(final_text)
    
    # Convert final_text to List
    # Use the existing _extract_first_json_array function to parse JSON array
    concerns_list = _extract_first_json_array(final_text)
    
    print(f"\nParsed concerns count: {len(concerns_list)}")
    print("Parsed concerns:")
    for i, concern in enumerate(concerns_list, 1):
        print(f"  {i}. {concern}")
    
    return concerns_list


def main() -> None:
    print("[1/3] Generate concerns via Generator_v3 ...")
    agent = create_custom_agent()
    gen_concerns = generate_concerns_via_agent(agent, raw_idea)
    print(f"Generated count: {len(gen_concerns)}")

    print("[2/3] Semantic vector matching evaluation ...")
    print(json.dumps({
        "generated_concerns": gen_concerns,
        "semantic_match": semantic_match_scores(concerns, gen_concerns)
    }, ensure_ascii=False, indent=2))

    print("[3/3] Original concern comparison ...")
    final = compare_coverage_via_llm(concerns ,gen_concerns)
    print(final)


if __name__ == "__main__":
    main()
