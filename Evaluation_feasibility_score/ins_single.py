#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Instructor model generates feasibility evaluation text (concerns)
"""

import os
import json
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ins_model import create_custom_agent, build_agent_user_prompt
from langchain_core.messages import HumanMessage


def generate_feasibility_evaluation_via_agent(agent, idea_text: str) -> str:
    """
    Use Agent to generate complete peer review evaluation text
    
    Args:
        agent: Created Agent instance
        idea_text: Research idea text to evaluate
        
    Returns:
        str: Complete peer review evaluation text (similar to all_comments)
    """
    # Build user prompt
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
                    print(f"  {last_message.content[:200]}...")
    
    print("\n" + "=" * 80)
    print(f"Agent execution completed")
    print(f"Total steps: {step_count}")
    print(f"Total tool calls: {tool_call_count}")
    print("=" * 80)
    
    # Return final evaluation text
    if not final_text or final_text.startswith("ERROR"):
        return final_text
    
    # If returned in JSON format (shouldn't happen, but handle as fallback)
    if final_text.strip().startswith('[') and final_text.strip().endswith(']'):
        try:
            parsed = json.loads(final_text)
            if isinstance(parsed, list):
                # If it's a list, convert to text
                return ". ".join([str(item) for item in parsed])
        except:
            pass
    
    # Return text format evaluation
    return final_text


def generate_feasibility_evaluation(idea_text: str, agent=None) -> str:
    """
    Generate complete peer review evaluation text (similar to all_comments)
    
    Args:
        idea_text: Research idea text to evaluate
        agent: Optional agent instance
        
    Returns:
        str: Complete peer review evaluation text
    """
    if agent is None:
        agent = create_custom_agent()
    
    return generate_feasibility_evaluation_via_agent(agent, idea_text)


def main():
    """Test function"""
    from Evaluation_utils.test_idea import raw_idea
    
    print("[1/2] Generating feasibility evaluation text ...")
    result = generate_feasibility_evaluation(raw_idea)
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

