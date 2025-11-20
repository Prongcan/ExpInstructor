import json
import os
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
from GPT_single import generate_concerns_for_idea, compare_coverage_via_llm


def process_single_item(item_data):
    """
    处理单个数据项的函数
    每个进程将运行此函数
    """
    try:
        item_id = item_data["id"]
        raw_idea = item_data["idea"]
        concerns = item_data["concerns"]
        
        # 检查idea是否为空
        if not raw_idea or raw_idea.strip() == "":
            return {
                "id": item_id,
                "status": "skipped",
                "reason": "empty_idea",
                "gen_concerns": [],
                "raw_resp_idea": "",
                "coverage_result": None,
                "error": None
            }
        
        # 生成LLM concerns
        gen_concerns, raw_resp_idea = generate_concerns_for_idea(raw_idea)
        print(f"生成数量: {len(gen_concerns)}")
        
        # 进行concern比对
        coverage_result, coverage_raw_resp = compare_coverage_via_llm(concerns, gen_concerns)
        print("coverage计算完毕！")

        return {
            "id": item_id,
            "status": "success",
            "reason": None,
            "gen_concerns": gen_concerns,
            "raw_resp_idea": raw_resp_idea,
            "coverage_result": coverage_result,
            "coverage_raw_resp": coverage_raw_resp,
            "error": None
        }
        
    except Exception as e:
        return {
            "id": item_data.get("id", "unknown"),
            "status": "error",
            "reason": None,
            "gen_concerns": [],
            "raw_resp_idea": "",
            "coverage_result": None,
            "error": str(e)
        }


def parallel_process_items(data_items, num_processes=4):
    """
    使用多进程并行处理数据项
    """
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_item, data_items),
            total=len(data_items),
            desc="处理数据项",
            unit="项"
        ))
    return results


def save_results(results, output_dir):
    """
    保存结果到指定目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存完整结果
    full_results_file = os.path.join(output_dir, "full_results.json")
    with open(full_results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 保存成功的结果
    success_results = [r for r in results if r["status"] == "success"]
    success_file = os.path.join(output_dir, "success_results.json")
    with open(success_file, 'w', encoding='utf-8') as f:
        json.dump(success_results, f, ensure_ascii=False, indent=2)
    
    # 保存错误的结果
    error_results = [r for r in results if r["status"] == "error"]
    error_file = os.path.join(output_dir, "error_results.json")
    with open(error_file, 'w', encoding='utf-8') as f:
        json.dump(error_results, f, ensure_ascii=False, indent=2)
    
    # 保存跳过的结果
    skipped_results = [r for r in results if r["status"] == "skipped"]
    skipped_file = os.path.join(output_dir, "skipped_results.json")
    with open(skipped_file, 'w', encoding='utf-8') as f:
        json.dump(skipped_results, f, ensure_ascii=False, indent=2)
    
    # 生成统计报告
    stats = {
        "total_items": len(results),
        "successful": len(success_results),
        "errors": len(error_results),
        "skipped": len(skipped_results),
        "success_rate": len(success_results) / len(results) if results else 0
    }
    
    stats_file = os.path.join(output_dir, "processing_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    return stats


def main():
    # 输入文件路径
    input_file = "Evaluation_feasibility/feasibility_data/Stanford_comments_with_ideas_with_concerns.json"
    
    # 输出目录
    output_dir = "Evaluation_feasibility/results/GPT"
    
    # 并行进程数
    num_processes = 147
    
    # 读取JSON数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data_items = json.load(f)
    
    # 并行处理所有数据项
    results = parallel_process_items(data_items, num_processes)
    
    # 保存结果
    stats = save_results(results, output_dir)
    
    # 输出处理统计信息
    print(f"处理完成！")
    print(f"总项目数: {stats['total_items']}")
    print(f"成功处理: {stats['successful']}")
    print(f"处理错误: {stats['errors']}")
    print(f"跳过项目: {stats['skipped']}")
    print(f"成功率: {stats['success_rate']:.2%}")
    print(f"结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()