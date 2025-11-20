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
# 直接复用 evaluate_single 中已实现的评估逻辑
from Evaluation_utils.eval_feasibility import semantic_match_scores, compare_coverage_via_llm
from Evaluation_utils.test_idea import raw_idea, concerns

def _extract_first_json_array(text: str) -> t.List[str]:
    """
    从任意文本中提取首个 JSON 数组并解析为字符串列表。
    容错：如果提取失败，返回空列表。
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

def generate_concerns_via_agent(agent,idea_text: str) -> t.List[str]:
    
    # 完全复用 Generator_concern 的用户提示模板
    input_message = HumanMessage(content=build_agent_user_prompt(idea_text))

    # 运行一次 agent，取最终消息内容（长报告）
    final_text = ""
    step_count = 0
    tool_call_count = 0
    tool_calls_details = []
    
    print("=" * 80)
    print("开始执行 Agent 工具调用过程")
    print("=" * 80)
    
    # 设置步数限制为50步
    config = {"recursion_limit": 50}
    for step in agent.stream({"messages": [input_message]}, config=config, stream_mode="values"):
        step_count += 1
        print(f"\n=== 步骤 {step_count} ===")
        
        if step.get("messages"):
            last_message = step["messages"][-1]
            print(f"消息类型: {type(last_message).__name__}")
            
            # 打印消息内容
            if last_message.content:
                print(f"消息内容: {last_message.content}")
                final_text = last_message.content
            
            # 检查是否有工具调用
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                tool_call_count += len(last_message.tool_calls)
                print(f"🔧 工具调用数量: {len(last_message.tool_calls)}")
                
                for i, tool_call in enumerate(last_message.tool_calls):
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    tool_id = tool_call.get('id', f'tool_call_{i+1}')
                    
                    print(f"  工具调用 {i+1}: {tool_name}")
                    print(f"    工具ID: {tool_id}")
                    print(f"    参数: {json.dumps(tool_args, ensure_ascii=False, indent=4)}")
                    
                    # 记录工具调用详情
                    tool_calls_details.append({
                        'step': step_count,
                        'tool_name': tool_name,
                        'tool_id': tool_id,
                        'args': tool_args,
                        'timestamp': f"步骤{step_count}"
                    })
            
            # 检查是否有工具结果
            if hasattr(last_message, 'tool_call_id') and last_message.tool_call_id:
                print(f"🔧 工具结果 (ID: {last_message.tool_call_id}):")
                if last_message.content:
                    # 限制输出长度，避免过长
                    result_content = last_message.content
                    if len(result_content) > 1000:
                        print(f"    结果预览: {result_content[:1000]}...")
                        print(f"    [结果被截断，总长度: {len(result_content)} 字符]")
                    else:
                        print(f"    完整结果: {result_content}")
                    
                    # 更新工具调用详情
                    for detail in tool_calls_details:
                        if detail['tool_id'] == last_message.tool_call_id:
                            detail['result'] = result_content
                            break
        
        print("-" * 50)
    
    print("\n" + "=" * 80)
    print("Agent 执行完成 - 统计信息")
    print("=" * 80)
    print(f"总步骤数: {step_count}")
    print(f"总工具调用次数: {tool_call_count}")
    
    # 打印工具调用统计
    tool_stats = {}
    for detail in tool_calls_details:
        tool_name = detail['tool_name']
        if tool_name not in tool_stats:
            tool_stats[tool_name] = 0
        tool_stats[tool_name] += 1
    
    print("\n工具调用统计:")
    for tool_name, count in tool_stats.items():
        print(f"  {tool_name}: {count} 次")
    
    # 打印所有工具调用的详细记录
    print("\n" + "=" * 80)
    print("所有工具调用详细记录")
    print("=" * 80)
    for i, detail in enumerate(tool_calls_details, 1):
        print(f"\n工具调用 #{i}:")
        print(f"  步骤: {detail['timestamp']}")
        print(f"  工具名称: {detail['tool_name']}")
        print(f"  工具ID: {detail['tool_id']}")
        print(f"  参数: {json.dumps(detail['args'], ensure_ascii=False, indent=4)}")
        if 'result' in detail:
            result = detail['result']
            if len(result) > 500:
                print(f"  结果: {result[:500]}...")
                print(f"  [结果被截断，总长度: {len(result)} 字符]")
            else:
                print(f"  结果: {result}")
        else:
            print(f"  结果: [未获取到结果]")
    
    print("\n" + "=" * 80)
    print("最终 Agent 输出")
    print("=" * 80)
    print(final_text)
    
    # 将final_text转换为List
    # 使用已有的_extract_first_json_array函数来解析JSON数组
    concerns_list = _extract_first_json_array(final_text)
    
    print(f"\n解析出的 concerns 数量: {len(concerns_list)}")
    print("解析出的 concerns:")
    for i, concern in enumerate(concerns_list, 1):
        print(f"  {i}. {concern}")
    
    return concerns_list


def main() -> None:
    print("[1/3] 通过 Generator_v3 生成 concerns ...")
    agent = create_custom_agent()
    gen_concerns = generate_concerns_via_agent(agent, raw_idea)
    print(f"生成数量: {len(gen_concerns)}")

    print("[2/3] 语义向量匹配评估 ...")
    print(json.dumps({
        "generated_concerns": gen_concerns,
        "semantic_match": semantic_match_scores(concerns, gen_concerns)
    }, ensure_ascii=False, indent=2))

    print("[3/3] 原始的concern比对 ...")
    final = compare_coverage_via_llm(concerns ,gen_concerns)
    print(final)


if __name__ == "__main__":
    main()