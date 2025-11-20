#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare three model result files (GPT, Instructor, RAG_evidence_retrieval)
Compute mean golden_score & novelty_score, correlation, Cohen’s kappa, hit rates, etc.
"""

import json
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def parse_golden_score(golden_score_str):
    if not golden_score_str or golden_score_str.strip() == "":
        return None
    try:
        scores = [float(x) for x in golden_score_str.strip().split()]
        return np.mean(scores) if scores else None
    except:
        return None


def load_and_process_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    result = {}
    for item in data:
        if item.get('status') != 'success':
            continue
        item_id = item.get('id')
        if not item_id:
            continue
        golden_score_mean = parse_golden_score(item.get('golden_score'))
        novelty_score = item.get('novelty_score')
        if golden_score_mean is not None and novelty_score is not None:
            result[item_id] = {
                'golden_score_mean': golden_score_mean,
                'novelty_score': novelty_score
            }
    return result


def calculate_metrics(models_data):
    """Support multiple models"""
    common_ids = set.intersection(*(set(d.keys()) for d in models_data.values()))
    if len(common_ids) == 0:
        print("Warning: No common data entries found.")
        return None, None
    print(f"Found {len(common_ids)} common data entries\n")

    raw_data = {'common_ids': list(common_ids)}
    results = {}

    # Calculate metrics for each model
    for name, data in models_data.items():
        golden = np.array([data[i]['golden_score_mean'] for i in common_ids])
        novelty = np.array([data[i]['novelty_score'] for i in common_ids])
        raw_data[f'{name}_golden'] = golden
        raw_data[f'{name}_novelty'] = novelty

        corr, p = pearsonr(golden, novelty)
        kappa = cohen_kappa_score(np.round(golden).astype(int), np.round(novelty).astype(int))
        mae = np.mean(np.abs(novelty - golden))
        rmse = np.sqrt(np.mean((novelty - golden) ** 2))
        hit1 = np.mean(np.abs(novelty - golden) <= 1)
        hit2 = np.mean(np.abs(novelty - golden) <= 2)

        results[name] = {
            'golden_mean': np.mean(golden),
            'novelty_mean': np.mean(novelty),
            'corr': corr,
            'corr_p': p,
            'kappa': kappa,
            'mae': mae,
            'rmse': rmse,
            'hit1': hit1,
            'hit2': hit2,
        }

    # Inter-model correlations
    inter_results = []
    model_names = list(models_data.keys())
    for a, b in itertools.combinations(model_names, 2):
        na = raw_data[f'{a}_novelty']
        nb = raw_data[f'{b}_novelty']
        inter_corr, _ = pearsonr(na, nb)
        inter_kappa = cohen_kappa_score(np.round(na).astype(int), np.round(nb).astype(int))
        inter_results.append({
            'pair': f"{a} vs {b}",
            'corr': inter_corr,
            'kappa': inter_kappa
        })
    results['_inter'] = inter_results
    return results, raw_data


def print_results(results):
    print("=" * 80)
    print("Multi-model statistics comparison")
    print("=" * 80)
    for name, r in results.items():
        if name == '_inter':
            continue
        print(f"\nModel: {name}")
        print(f"  Golden Score Mean: {r['golden_mean']:.4f}")
        print(f"  Novelty Score Mean: {r['novelty_mean']:.4f}")
        print(f"  Pearson Correlation: {r['corr']:.4f} (p={r['corr_p']:.4e})")
        print(f"  Cohen's Kappa: {r['kappa']:.4f}")
        print(f"  ±1 Hit Rate: {r['hit1']:.4f} ({r['hit1']*100:.2f}%)")
        print(f"  ±2 Hit Rate: {r['hit2']:.4f} ({r['hit2']*100:.2f}%)")
        print(f"  MAE: {r['mae']:.4f}")
        print(f"  RMSE: {r['rmse']:.4f}")

    print("\nInter-model comparison:")
    for pair in results['_inter']:
        print(f"  {pair['pair']}: Correlation={pair['corr']:.4f}, Kappa={pair['kappa']:.4f}")
    print("=" * 80)


def create_visualizations(results, raw_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model_names = [m for m in results.keys() if m != '_inter']

    # Radar chart
    categories = ['Correlation', 'Cohen Kappa', '±1 Hit Rate', '±2 Hit Rate']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    for name in model_names:
        vals = results[name]
        values = [
            (vals['corr'] + 1) / 2,
            (vals['kappa'] + 1) / 2,
            vals['hit1'],
            vals['hit2']
        ]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=name)
        ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Radar Chart: Model Comparison', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_radar_chart.png'), bbox_inches='tight')
    plt.close()

    # Novelty score distributions
    plt.figure(figsize=(10, 6))
    for name in model_names:
        plt.hist(raw_data[f'{name}_novelty'], bins=range(0, 12), alpha=0.5, edgecolor='black', label=name)
    plt.xlabel("Novelty Score")
    plt.ylabel("Frequency")
    plt.title("Novelty Score Distribution Across Models")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_novelty_distribution.png'), bbox_inches='tight')
    plt.close()

    # Scatter plots between models
    pairs = results['_inter']
    for pair in pairs:
        a, b = pair['pair'].split(' vs ')
        na = raw_data[f'{a}_novelty']
        nb = raw_data[f'{b}_novelty']
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(na, nb, alpha=0.6)
        z = np.polyfit(na, nb, 1)
        p = np.poly1d(z)
        ax.plot(na, p(na), 'r--', alpha=0.7, linewidth=2, label=f"r={pair['corr']:.3f}, kappa={pair['kappa']:.3f}")
        minv, maxv = min(na.min(), nb.min()), max(na.max(), nb.max())
        ax.plot([minv, maxv], [minv, maxv], 'k--', alpha=0.4)
        ax.set_xlabel(f"{a} Novelty Score")
        ax.set_ylabel(f"{b} Novelty Score")
        ax.legend()
        ax.set_title(f"{a} vs {b} Novelty Comparison")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"3_scatter_{a}_vs_{b}.png"), bbox_inches='tight')
        plt.close()

    # === Added: ±1 and ±2 hit rate comparison bar charts ===
    metrics = ['hit1', 'hit2']
    titles = ['±1 Hit Rate Comparison', '±2 Hit Rate Comparison']
    file_names = ['4_hit1_compare.png', '5_hit2_compare.png']

    for metric, title, fname in zip(metrics, titles, file_names):
        plt.figure(figsize=(7, 6))
        bars = [results[m][metric] for m in model_names]
        sns.barplot(x=model_names, y=bars, edgecolor='black', alpha=0.8)
        plt.ylim(0, 1)
        plt.ylabel("Hit Rate")
        plt.title(title)
        for i, v in enumerate(bars):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight')
        plt.close()

    print(f"Figures saved to: {output_dir}")


def main():
    model_paths = {
        "RAG_evidence_retrieval": "Evaluation_novelty/results/RAG_evidence_retrieval/success_results.json",
        "Instructor": "Evaluation_novelty/results/instructor/success_results.json",
        "GPT": "Evaluation_novelty/results/GPT/success_results.json"
    }

    output_dir = "Evaluation_novelty/results/compare_three"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    models_data = {name: load_and_process_data(path) for name, path in model_paths.items()}
    for name, data in models_data.items():
        print(f"{name} entries: {len(data)}")

    print("\nCalculating statistics...")
    results, raw_data = calculate_metrics(models_data)
    if results:
        print_results(results)

        output_file = os.path.join(output_dir, "compare_three_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_file}")

        print("\nGenerating figures...")
        create_visualizations(results, raw_data, output_dir)

        report_file = os.path.join(output_dir, "compare_three_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\nThree-Model Statistical Report\n" + "=" * 80 + "\n\n")
            for name, r in results.items():
                if name == '_inter':
                    continue
                f.write(f"Model: {name}\n")
                f.write(f"  Golden Score Mean: {r['golden_mean']:.4f}\n")
                f.write(f"  Novelty Score Mean: {r['novelty_mean']:.4f}\n")
                f.write(f"  Pearson Correlation: {r['corr']:.4f} (p={r['corr_p']:.4e})\n")
                f.write(f"  Cohen's Kappa: {r['kappa']:.4f}\n")
                f.write(f"  ±1 Hit Rate: {r['hit1']:.4f}\n")
                f.write(f"  ±2 Hit Rate: {r['hit2']:.4f}\n")
                f.write(f"  MAE: {r['mae']:.4f}\n")
                f.write(f"  RMSE: {r['rmse']:.4f}\n\n")

            f.write("Inter-model Comparison:\n")
            for p in results['_inter']:
                f.write(f"  {p['pair']}: Correlation={p['corr']:.4f}, Kappa={p['kappa']:.4f}\n")
        print(f"Text report saved to: {report_file}")
    else:
        print("Computation failed. Please check input data.")


if __name__ == "__main__":
    main()
