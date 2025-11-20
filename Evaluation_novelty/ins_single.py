import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from ins_model import create_custom_agent, build_agent_user_prompt
from langchain_core.messages import HumanMessage
from Evaluation_utils.eval_novelty import generate_novelty_score
from Evaluation_utils.test_idea import raw_idea

def generate_novelty_evaluation_via_agent(agent, idea_text: str) -> str:
    """
    Generate novelty evaluation text using Agent
    
    Args:
        agent: Created Agent instance
        idea_text: Research idea text to evaluate
        
    Returns:
        str: Novelty evaluation text
    """
    # Build user prompt, directly use idea_text
    prompt = build_agent_user_prompt(idea_text)

    input_message = HumanMessage(content=prompt)

    # Run agent once, get final message content (long report)
    final_text = ""
    step_count = 0
    tool_call_count = 0
    tool_calls_details = []
    
    print("=" * 80)
    print("Starting Agent tool call process")
    print("=" * 80)
    
    # Set step limit to 50 steps
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
            
            # Check if there are tool calls
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                tool_call_count += len(last_message.tool_calls)
                print(f"🔧 Tool call count: {len(last_message.tool_calls)}")
                
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
            
            # Check if there are tool results
            if hasattr(last_message, 'tool_call_id') and last_message.tool_call_id:
                print(f"🔧 Tool result (ID: {last_message.tool_call_id}):")
                if last_message.content:
                    # Limit output length to avoid too long
                    result_content = last_message.content
                    if len(result_content) > 1000:
                        print(f"    Result preview: {result_content[:1000]}...")
                        print(f"    [Result truncated, total length: {len(result_content)} characters]")
                    else:
                        print(f"    Full result: {result_content}")
                    
                    # Update tool call details
                    for detail in tool_calls_details:
                        if detail['tool_id'] == last_message.tool_call_id:
                            detail['result'] = result_content
                            break
        
        print("-" * 50)
    
    print("\n" + "=" * 80)
    print("Agent execution completed - Statistics")
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
                print(f"  [Result truncated, total length: {len(result)} characters]")
            else:
                print(f"  Result: {result}")
        else:
            print(f"  Result: [No result obtained]")
    
    print("\n" + "=" * 80)
    print("Final Agent output")
    print("=" * 80)
    print(final_text)
    
    # Check for errors
    if not final_text or final_text.strip() == "":
        return "ERROR: Agent did not return valid result"
    
    if final_text.startswith("ERROR"):
        return final_text
    
    return final_text


def main() -> None:
    print("[1/3] Generating novelty evaluation via Agent ...")
    agent = create_custom_agent()
    result = generate_novelty_evaluation_via_agent(agent, raw_idea)
    print(result)
    
    # Use GPT to score the evaluation result
    print("\n[2/3] Using GPT model to evaluate score ...")
    gpt_score_result = generate_novelty_score(result)
    if gpt_score_result and not gpt_score_result.startswith("ERROR"):
        print(f"\nGPT scoring result:")
        print(gpt_score_result)
    else:
        print("GPT scoring failed")


if __name__ == "__main__":
    main()