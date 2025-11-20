#!/usr/bin/env python3
"""
清洗 all_graphs.json 文件
移除 source_name 或 target_name 为 None、空字符串或只有空格的边
"""

import json
import sys
from pathlib import Path

def clean_edge(edge):
    """
    检查并清理单条边
    返回 None 如果边无效，否则返回清理后的边
    """
    source = edge.get("source_name")
    target = edge.get("target_name")
    
    # 检查是否为 None
    if source is None or target is None:
        return None
    
    # 转换为字符串并去除前后空格
    source = str(source).strip()
    target = str(target).strip()
    
    # 检查是否为空字符串
    if not source or not target:
        return None
    
    # 返回清理后的边
    return {
        "source_name": source,
        "target_name": target,
        "relationship": edge.get("relationship", "").strip(),
        "evidence": edge.get("evidence", "").strip()
    }

def clean_graph_data(input_file, output_file):
    """
    清洗图数据文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    print(f"正在读取文件: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"原始数据: {len(data)} 个记录")
    
    # 统计信息
    total_edges = 0
    valid_edges = 0
    removed_edges = 0
    empty_records = 0
    
    cleaned_data = []
    
    for record_idx, record in enumerate(data):
        paper_id = record.get("paper_id", "")
        review_id = record.get("review_id", "")
        edges_raw = record.get("edges", [])
        
        # 兼容 edges 为列表或单个对象
        if isinstance(edges_raw, dict):
            edges_iter = [edges_raw]
        elif isinstance(edges_raw, list):
            edges_iter = edges_raw
        else:
            edges_iter = []
        
        # 清理边
        cleaned_edges = []
        for edge in edges_iter:
            total_edges += 1
            cleaned_edge = clean_edge(edge)
            if cleaned_edge is not None:
                valid_edges += 1
                cleaned_edges.append(cleaned_edge)
            else:
                removed_edges += 1
        
        # 只保留有有效边的记录
        if cleaned_edges:
            # 保留原始记录的所有字段
            cleaned_record = {
                "paper_id": paper_id,
                "review_id": review_id,
                "edges": cleaned_edges
            }
            # 如果原始记录有其他字段，也保留
            for key, value in record.items():
                if key not in ["edges"]:
                    cleaned_record[key] = value
            
            cleaned_data.append(cleaned_record)
        else:
            empty_records += 1
        
        # 进度显示
        if (record_idx + 1) % 1000 == 0:
            print(f"已处理 {record_idx + 1}/{len(data)} 个记录")
    
    # 保存清洗后的数据
    print(f"\n正在保存清洗后的数据到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("清洗结果统计")
    print("=" * 80)
    print(f"原始记录数: {len(data)}")
    print(f"清洗后记录数: {len(cleaned_data)}")
    print(f"移除的空记录数: {empty_records}")
    print(f"总边数: {total_edges}")
    print(f"有效边数: {valid_edges}")
    print(f"移除的无效边数: {removed_edges}")
    print(f"数据保留率: {len(cleaned_data)/len(data)*100:.2f}%")
    print(f"边保留率: {valid_edges/total_edges*100:.2f}%" if total_edges > 0 else "无边数据")
    print("=" * 80)

def main():
    input_file = "result_v2/all_graphs.json"
    output_file = "result_v2/all_graphs_cleaned.json"
    
    # 检查输入文件是否存在
    if not Path(input_file).exists():
        print(f"错误: 输入文件不存在: {input_file}")
        sys.exit(1)
    
    # 执行清洗
    try:
        clean_graph_data(input_file, output_file)
        print("\n清洗完成!")
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

