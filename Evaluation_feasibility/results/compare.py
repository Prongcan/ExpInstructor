#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-Model Coverage Comparison Tool
Compare concern coverage ratio and precision across three models: GPT, RAG_evidence_retrieval, and Instructor

Author: AI Assistant
Date: 2024
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse
from itertools import combinations

# Font settings
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class CoverageComparator:
    """Coverage comparator - supports three-model comparisons"""
    
    def __init__(self, gpt_file: str, rag_file: str, instructor_file: str):
        """
        Initialize comparator
        
        Args:
            gpt_file: Path to GPT results file
            rag_file: Path to RAG_evidence_retrieval results file
            instructor_file: Path to Instructor results file
        """
        self.gpt_file = gpt_file
        self.rag_file = rag_file
        self.instructor_file = instructor_file
        self.gpt_data = None
        self.rag_data = None
        self.instructor_data = None
        self.model_names = ['GPT', 'RAG', 'Instructor']
        
    def load_data(self):
        """Load data from the three JSON files"""
        print("Loading data...")
        
        with open(self.gpt_file, 'r', encoding='utf-8') as f:
            self.gpt_data = json.load(f)
            
        with open(self.rag_file, 'r', encoding='utf-8') as f:
            self.rag_data = json.load(f)
            
        with open(self.instructor_file, 'r', encoding='utf-8') as f:
            self.instructor_data = json.load(f)
            
        print(f"GPT records: {len(self.gpt_data)}")
        print(f"RAG records: {len(self.rag_data)}")
        print(f"Instructor records: {len(self.instructor_data)}")
        
    def extract_coverage_stats(self, data: List[Dict]) -> pd.DataFrame:
        """
        Extract coverage statistics from data
        
        Args:
            data: List of JSON result entries
            
        Returns:
            DataFrame containing coverage statistics
        """
        stats = []
        
        for item in data:
            if 'coverage_result' in item and 'summary' in item['coverage_result']:
                summary = item['coverage_result']['summary']
                # Compute precision: number of covered concerns / total generated concerns
                gen_concerns_count = len(item.get('gen_concerns', []))
                precision = summary['covered_count'] / gen_concerns_count if gen_concerns_count > 0 else 0
                
                stats.append({
                    'id': item['id'],
                    'covered_count': summary['covered_count'],
                    'total': summary['total'],
                    'coverage_ratio': summary['coverage_ratio'],
                    'gen_concerns_count': gen_concerns_count,
                    'precision': precision
                })
        
        return pd.DataFrame(stats)
    
    def calculate_overall_coverage(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate overall coverage statistics
        
        Returns:
            Dictionary containing overall statistics for the three models
        """
        gpt_df = self.extract_coverage_stats(self.gpt_data)
        rag_df = self.extract_coverage_stats(self.rag_data)
        instructor_df = self.extract_coverage_stats(self.instructor_data)
        
        def calculate_stats(df, model_name):
            """Compute statistics for a single model"""
            return {
                'total_covered': df['covered_count'].sum(),
                'total_items': df['total'].sum(),
                'total_gen_concerns': df['gen_concerns_count'].sum(),
                'overall_coverage_ratio': df['covered_count'].sum() / df['total'].sum() if df['total'].sum() > 0 else 0,
                'overall_precision': df['covered_count'].sum() / df['gen_concerns_count'].sum() if df['gen_concerns_count'].sum() > 0 else 0,
                'mean_coverage_ratio': df['coverage_ratio'].mean(),
                'mean_precision': df['precision'].mean(),
                'std_coverage_ratio': df['coverage_ratio'].std(),
                'std_precision': df['precision'].std(),
                'median_coverage_ratio': df['coverage_ratio'].median(),
                'median_precision': df['precision'].median(),
                'min_coverage_ratio': df['coverage_ratio'].min(),
                'max_coverage_ratio': df['coverage_ratio'].max(),
                'min_precision': df['precision'].min(),
                'max_precision': df['precision'].max()
            }
        
        return {
            'GPT': calculate_stats(gpt_df, 'GPT'),
            'RAG': calculate_stats(rag_df, 'RAG'),
            'Instructor': calculate_stats(instructor_df, 'Instructor')
        }
    
    def head_to_head_comparison(self) -> pd.DataFrame:
        """
        Head-to-head comparison for the same idea across three models
        
        Returns:
            DataFrame containing comparison results
        """
        gpt_df = self.extract_coverage_stats(self.gpt_data)
        rag_df = self.extract_coverage_stats(self.rag_data)
        instructor_df = self.extract_coverage_stats(self.instructor_data)
        
        # Rename columns to avoid conflicts
        gpt_df_renamed = gpt_df.rename(columns={
            'coverage_ratio': 'coverage_ratio_gpt',
            'precision': 'precision_gpt',
            'covered_count': 'covered_count_gpt',
            'total': 'total_gpt',
            'gen_concerns_count': 'gen_concerns_count_gpt'
        })
        rag_df_renamed = rag_df.rename(columns={
            'coverage_ratio': 'coverage_ratio_rag',
            'precision': 'precision_rag',
            'covered_count': 'covered_count_rag',
            'total': 'total_rag',
            'gen_concerns_count': 'gen_concerns_count_rag'
        })
        instructor_df_renamed = instructor_df.rename(columns={
            'coverage_ratio': 'coverage_ratio_instructor',
            'precision': 'precision_instructor',
            'covered_count': 'covered_count_instructor',
            'total': 'total_instructor',
            'gen_concerns_count': 'gen_concerns_count_instructor'
        })
        
        # Merge data on id
        merged = pd.merge(
            gpt_df_renamed, 
            rag_df_renamed, 
            on='id', 
            how='inner'
        )
        merged = pd.merge(
            merged,
            instructor_df_renamed,
            on='id',
            how='inner'
        )
        
        # Ranking for coverage ratio
        coverage_cols = ['coverage_ratio_gpt', 'coverage_ratio_rag', 'coverage_ratio_instructor']
        merged['coverage_rank_gpt'] = merged[coverage_cols].apply(lambda x: (x >= x['coverage_ratio_gpt']).sum(), axis=1)
        merged['coverage_rank_rag'] = merged[coverage_cols].apply(lambda x: (x >= x['coverage_ratio_rag']).sum(), axis=1)
        merged['coverage_rank_instructor'] = merged[coverage_cols].apply(lambda x: (x >= x['coverage_ratio_instructor']).sum(), axis=1)
        
        # Ranking for precision
        precision_cols = ['precision_gpt', 'precision_rag', 'precision_instructor']
        merged['precision_rank_gpt'] = merged[precision_cols].apply(lambda x: (x >= x['precision_gpt']).sum(), axis=1)
        merged['precision_rank_rag'] = merged[precision_cols].apply(lambda x: (x >= x['precision_rag']).sum(), axis=1)
        merged['precision_rank_instructor'] = merged[precision_cols].apply(lambda x: (x >= x['precision_instructor']).sum(), axis=1)
        
        # Determine winners (rank equals 1)
        merged['gpt_wins_coverage'] = merged['coverage_rank_gpt'] == 1
        merged['rag_wins_coverage'] = merged['coverage_rank_rag'] == 1
        merged['instructor_wins_coverage'] = merged['coverage_rank_instructor'] == 1
        
        merged['gpt_wins_precision'] = merged['precision_rank_gpt'] == 1
        merged['rag_wins_precision'] = merged['precision_rank_rag'] == 1
        merged['instructor_wins_precision'] = merged['precision_rank_instructor'] == 1
        
        return merged
    
    def generate_statistics_report(self) -> str:
        """Generate statistics report"""
        overall_stats = self.calculate_overall_coverage()
        head_to_head = self.head_to_head_comparison()
        
        report = []
        report.append("=" * 60)
        report.append("GPT vs RAG vs Instructor Coverage Comparison Report")
        report.append("=" * 60)
        
        # Overall statistics
        report.append("\n1. Overall Coverage Statistics:")
        report.append("-" * 60)
        
        for model_name, stats in overall_stats.items():
            report.append(f"\n{model_name}:")
            report.append(f"  Total covered (total_covered): {stats['total_covered']}")
            report.append(f"  Total items (total_items): {stats['total_items']}")
            report.append(f"  Total generated concerns: {stats['total_gen_concerns']}")
            report.append(f"  Overall Coverage Ratio: {stats['overall_coverage_ratio']:.4f} ({stats['overall_coverage_ratio']*100:.2f}%)")
            report.append(f"  Overall Precision: {stats['overall_precision']:.4f} ({stats['overall_precision']*100:.2f}%)")
            report.append(f"  Mean Coverage Ratio: {stats['mean_coverage_ratio']:.4f}")
            report.append(f"  Mean Precision: {stats['mean_precision']:.4f}")
            report.append(f"  Std of Coverage Ratio: {stats['std_coverage_ratio']:.4f}")
            report.append(f"  Std of Precision: {stats['std_precision']:.4f}")
            report.append(f"  Median Coverage Ratio: {stats['median_coverage_ratio']:.4f}")
            report.append(f"  Median Precision: {stats['median_precision']:.4f}")
            report.append(f"  Coverage Ratio Range: [{stats['min_coverage_ratio']:.4f}, {stats['max_coverage_ratio']:.4f}]")
            report.append(f"  Precision Range: [{stats['min_precision']:.4f}, {stats['max_precision']:.4f}]")
        
        # Head-to-head comparison
        report.append("\n2. Head-to-Head Comparison:")
        report.append("-" * 60)
        
        gpt_wins_coverage = head_to_head['gpt_wins_coverage'].sum()
        rag_wins_coverage = head_to_head['rag_wins_coverage'].sum()
        instructor_wins_coverage = head_to_head['instructor_wins_coverage'].sum()
        total_comparisons = len(head_to_head)
        
        report.append(f"Total comparisons: {total_comparisons}")
        report.append(f"\nCoverage ratio wins:")
        report.append(f"  GPT wins: {gpt_wins_coverage} ({gpt_wins_coverage/total_comparisons*100:.1f}%)")
        report.append(f"  RAG wins: {rag_wins_coverage} ({rag_wins_coverage/total_comparisons*100:.1f}%)")
        report.append(f"  Instructor wins: {instructor_wins_coverage} ({instructor_wins_coverage/total_comparisons*100:.1f}%)")
        
        # Precision head-to-head comparison
        gpt_wins_precision = head_to_head['gpt_wins_precision'].sum()
        rag_wins_precision = head_to_head['rag_wins_precision'].sum()
        instructor_wins_precision = head_to_head['instructor_wins_precision'].sum()
        
        report.append(f"\nPrecision wins:")
        report.append(f"  GPT wins: {gpt_wins_precision} ({gpt_wins_precision/total_comparisons*100:.1f}%)")
        report.append(f"  RAG wins: {rag_wins_precision} ({rag_wins_precision/total_comparisons*100:.1f}%)")
        report.append(f"  Instructor wins: {instructor_wins_precision} ({instructor_wins_precision/total_comparisons*100:.1f}%)")
        
        # Difference analysis
        report.append("\n3. Difference Analysis:")
        report.append("-" * 60)
        
        # Pairwise differences
        for model1, model2 in [('GPT', 'RAG'), ('GPT', 'Instructor'), ('RAG', 'Instructor')]:
            col1 = f'coverage_ratio_{model1.lower()}'
            col2 = f'coverage_ratio_{model2.lower()}'
            if col1 in head_to_head.columns and col2 in head_to_head.columns:
                coverage_diff = head_to_head[col1] - head_to_head[col2]
                report.append(f"\nCoverage ratio difference ({model1} - {model2}):")
                report.append(f"  Mean difference: {coverage_diff.mean():.4f}")
                report.append(f"  Std difference: {coverage_diff.std():.4f}")
            
            col1_p = f'precision_{model1.lower()}'
            col2_p = f'precision_{model2.lower()}'
            if col1_p in head_to_head.columns and col2_p in head_to_head.columns:
                precision_diff = head_to_head[col1_p] - head_to_head[col2_p]
                report.append(f"Precision difference ({model1} - {model2}):")
                report.append(f"  Mean difference: {precision_diff.mean():.4f}")
                report.append(f"  Std difference: {precision_diff.std():.4f}")
        
        # Significance tests
        report.append("\n4. Significance Tests:")
        report.append("-" * 60)
        try:
            from scipy.stats import wilcoxon, friedmanchisquare
            
            # Wilcoxon tests for each pair
            for model1, model2 in [('GPT', 'RAG'), ('GPT', 'Instructor'), ('RAG', 'Instructor')]:
                col1 = f'coverage_ratio_{model1.lower()}'
                col2 = f'coverage_ratio_{model2.lower()}'
                if col1 in head_to_head.columns and col2 in head_to_head.columns:
                    w_stat, w_p = wilcoxon(head_to_head[col1], head_to_head[col2])
                    report.append(f"\nCoverage ratio Wilcoxon test ({model1} vs {model2}):")
                    report.append(f"  Wilcoxon statistic: {w_stat:.4f}")
                    report.append(f"  p-value: {w_p:.4f}")
                    report.append(f"  Significance: {'Significant' if w_p < 0.05 else 'Not significant'} (α=0.05)")
                
                col1_p = f'precision_{model1.lower()}'
                col2_p = f'precision_{model2.lower()}'
                if col1_p in head_to_head.columns and col2_p in head_to_head.columns:
                    w_stat_p, w_p_p = wilcoxon(head_to_head[col1_p], head_to_head[col2_p])
                    report.append(f"Precision Wilcoxon test ({model1} vs {model2}):")
                    report.append(f"  Wilcoxon statistic: {w_stat_p:.4f}")
                    report.append(f"  p-value: {w_p_p:.4f}")
                    report.append(f"  Significance: {'Significant' if w_p_p < 0.05 else 'Not significant'} (α=0.05)")
            
            # Friedman test (overall comparison of three models)
            try:
                f_stat_c, f_p_c = friedmanchisquare(
                    head_to_head['coverage_ratio_gpt'],
                    head_to_head['coverage_ratio_rag'],
                    head_to_head['coverage_ratio_instructor']
                )
                report.append(f"\nCoverage ratio Friedman test (three models overall):")
                report.append(f"  Friedman statistic: {f_stat_c:.4f}")
                report.append(f"  p-value: {f_p_c:.4f}")
                report.append(f"  Significance: {'Significant' if f_p_c < 0.05 else 'Not significant'} (α=0.05)")
                
                f_stat_p, f_p_p = friedmanchisquare(
                    head_to_head['precision_gpt'],
                    head_to_head['precision_rag'],
                    head_to_head['precision_instructor']
                )
                report.append(f"Precision Friedman test (three models overall):")
                report.append(f"  Friedman statistic: {f_stat_p:.4f}")
                report.append(f"  p-value: {f_p_p:.4f}")
                report.append(f"  Significance: {'Significant' if f_p_p < 0.05 else 'Not significant'} (α=0.05)")
            except Exception as e:
                report.append(f"\nFriedman test failed: {e}")
        except ImportError:
            report.append("\nNote: scipy not installed, skipping significance tests")
        
        # Summary
        report.append("\n5. Summary:")
        report.append("-" * 60)
        
        # Sort by coverage ratio
        sorted_models_coverage = sorted(overall_stats.items(), 
                                        key=lambda x: x[1]['overall_coverage_ratio'], 
                                        reverse=True)
        report.append(f"\nCoverage ratio ranking:")
        for rank, (model_name, stats) in enumerate(sorted_models_coverage, 1):
            report.append(f"  {rank}. {model_name}: {stats['overall_coverage_ratio']*100:.2f}%")
        
        # Sort by precision
        sorted_models_precision = sorted(overall_stats.items(), 
                                        key=lambda x: x[1]['overall_precision'], 
                                        reverse=True)
        report.append(f"\nPrecision ranking:")
        for rank, (model_name, stats) in enumerate(sorted_models_precision, 1):
            report.append(f"  {rank}. {model_name}: {stats['overall_precision']*100:.2f}%")
        
        return "\n".join(report)
    
    def create_visualizations(self, output_dir: str = "coverage_comparison"):
        """Create visualization charts - supports three models"""
        Path(output_dir).mkdir(exist_ok=True)
        
        overall_stats = self.calculate_overall_coverage()
        head_to_head = self.head_to_head_comparison()
        gpt_df = self.extract_coverage_stats(self.gpt_data)
        rag_df = self.extract_coverage_stats(self.rag_data)
        instructor_df = self.extract_coverage_stats(self.instructor_data)
        
        # Define colors
        colors = ['lightcoral', 'lightgreen', 'skyblue']
        model_colors = {'GPT': 'lightcoral', 'RAG': 'lightgreen', 'Instructor': 'skyblue'}
        
        # 1. Overall coverage and precision comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        models = list(overall_stats.keys())
        
        # 1.1 Overall coverage ratio bar chart
        overall_ratios = [overall_stats[model]['overall_coverage_ratio'] for model in models]
        axes[0, 0].bar(models, overall_ratios, color=colors)
        axes[0, 0].set_title('Overall Coverage Ratio Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Coverage Ratio', fontsize=12)
        axes[0, 0].set_ylim(0, max(overall_ratios) * 1.15)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Value labels
        for i, v in enumerate(overall_ratios):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}\n({v*100:.1f}%)', ha='center', va='bottom', fontsize=11)
        
        # 1.2 Coverage ratio distribution comparison
        axes[0, 1].hist(gpt_df['coverage_ratio'], bins=20, alpha=0.6, label='GPT', color=model_colors['GPT'], edgecolor='black')
        axes[0, 1].hist(rag_df['coverage_ratio'], bins=20, alpha=0.6, label='RAG', color=model_colors['RAG'], edgecolor='black')
        axes[0, 1].hist(instructor_df['coverage_ratio'], bins=20, alpha=0.6, label='Instructor', color=model_colors['Instructor'], edgecolor='black')
        axes[0, 1].set_title('Coverage Ratio Distribution Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Coverage Ratio', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 1.3 Scatter plot matrix (GPT vs RAG)
        axes[1, 0].scatter(head_to_head['coverage_ratio_gpt'], 
                          head_to_head['coverage_ratio_rag'], 
                          alpha=0.6, color='purple', s=50, label='GPT vs RAG')
        min_val = min(head_to_head['coverage_ratio_gpt'].min(), 
                     head_to_head['coverage_ratio_rag'].min())
        max_val = max(head_to_head['coverage_ratio_gpt'].max(), 
                     head_to_head['coverage_ratio_rag'].max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, linewidth=2, label='y=x')
        axes[1, 0].set_title('GPT vs RAG Coverage Ratio', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('GPT Coverage Ratio', fontsize=12)
        axes[1, 0].set_ylabel('RAG Coverage Ratio', fontsize=12)
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(alpha=0.3)
        
        # 1.4 Winner distribution pie chart
        winner_counts = [
            head_to_head['gpt_wins_coverage'].sum(),
            head_to_head['rag_wins_coverage'].sum(),
            head_to_head['instructor_wins_coverage'].sum()
        ]
        winner_labels = ['GPT Wins', 'RAG Wins', 'Instructor Wins']
        
        axes[1, 1].pie(winner_counts, labels=winner_labels, colors=colors, autopct='%1.1f%%', 
                      startangle=90, textprops={'fontsize': 11})
        axes[1, 1].set_title('Coverage Ratio Winner Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/coverage_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Figure saved: {output_dir}/coverage_comparison.png")
        plt.close()
        
        # 2. Precision comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 2.1 Overall precision comparison
        overall_precisions = [overall_stats[model]['overall_precision'] for model in models]
        axes[0, 0].bar(models, overall_precisions, color=colors)
        axes[0, 0].set_title('Overall Precision Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Precision', fontsize=12)
        axes[0, 0].set_ylim(0, max(overall_precisions) * 1.15)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Value labels
        for i, v in enumerate(overall_precisions):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}\n({v*100:.1f}%)', ha='center', va='bottom', fontsize=11)
        
        # 2.2 Precision distribution comparison
        axes[0, 1].hist(gpt_df['precision'], bins=20, alpha=0.6, label='GPT', color=model_colors['GPT'], edgecolor='black')
        axes[0, 1].hist(rag_df['precision'], bins=20, alpha=0.6, label='RAG', color=model_colors['RAG'], edgecolor='black')
        axes[0, 1].hist(instructor_df['precision'], bins=20, alpha=0.6, label='Instructor', color=model_colors['Instructor'], edgecolor='black')
        axes[0, 1].set_title('Precision Distribution Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Precision', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 2.3 Precision scatter plot (GPT vs RAG)
        axes[1, 0].scatter(head_to_head['precision_gpt'], 
                          head_to_head['precision_rag'], 
                          alpha=0.6, color='purple', s=50, label='GPT vs RAG')
        
        # Diagonal line
        min_val_p = min(head_to_head['precision_gpt'].min(), 
                       head_to_head['precision_rag'].min())
        max_val_p = max(head_to_head['precision_gpt'].max(), 
                       head_to_head['precision_rag'].max())
        axes[1, 0].plot([min_val_p, max_val_p], [min_val_p, max_val_p], 'r--', alpha=0.5, linewidth=2, label='y=x')
        
        axes[1, 0].set_title('GPT vs RAG Precision', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('GPT Precision', fontsize=12)
        axes[1, 0].set_ylabel('RAG Precision', fontsize=12)
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(alpha=0.3)
        
        # 2.4 Precision winner distribution pie chart
        precision_winner_counts = [
            head_to_head['gpt_wins_precision'].sum(),
            head_to_head['rag_wins_precision'].sum(),
            head_to_head['instructor_wins_precision'].sum()
        ]
        
        axes[1, 1].pie(precision_winner_counts, labels=winner_labels, colors=colors, autopct='%1.1f%%',
                      startangle=90, textprops={'fontsize': 11})
        axes[1, 1].set_title('Precision Winner Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/precision_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Figure saved: {output_dir}/precision_comparison.png")
        plt.close()
        
        # 3. Box plots comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 3.1 Coverage ratio box plot
        coverage_data = [gpt_df['coverage_ratio'], rag_df['coverage_ratio'], instructor_df['coverage_ratio']]
        bp1 = axes[0].boxplot(coverage_data, labels=models, patch_artist=True, 
                             boxprops=dict(facecolor='white', alpha=0.8),
                             medianprops=dict(color='red', linewidth=2))
        colors_box = colors
        for patch, color in zip(bp1['boxes'], colors_box):
            patch.set_facecolor(color)
        
        axes[0].set_title('Coverage Ratio Box Plot Comparison', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Coverage Ratio', fontsize=12)
        axes[0].grid(axis='y', alpha=0.3)
        
        # 3.2 Precision box plot
        precision_data = [gpt_df['precision'], rag_df['precision'], instructor_df['precision']]
        bp2 = axes[1].boxplot(precision_data, labels=models, patch_artist=True,
                             boxprops=dict(facecolor='white', alpha=0.8),
                             medianprops=dict(color='red', linewidth=2))
        for patch, color in zip(bp2['boxes'], colors_box):
            patch.set_facecolor(color)
        
        axes[1].set_title('Precision Box Plot Comparison', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Precision', fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/boxplot_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Figure saved: {output_dir}/boxplot_comparison.png")
        plt.close()
        
        # 4. Detailed statistics bar charts
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 4.1 Mean comparison
        metrics = ['Coverage Ratio', 'Precision']
        gpt_means = [overall_stats['GPT']['mean_coverage_ratio'], overall_stats['GPT']['mean_precision']]
        rag_means = [overall_stats['RAG']['mean_coverage_ratio'], overall_stats['RAG']['mean_precision']]
        instructor_means = [overall_stats['Instructor']['mean_coverage_ratio'], overall_stats['Instructor']['mean_precision']]
        
        x = np.arange(len(metrics))
        width = 0.25
        
        bars1 = axes[0, 0].bar(x - width, gpt_means, width, label='GPT', color=model_colors['GPT'])
        bars2 = axes[0, 0].bar(x, rag_means, width, label='RAG', color=model_colors['RAG'])
        bars3 = axes[0, 0].bar(x + width, instructor_means, width, label='Instructor', color=model_colors['Instructor'])
        
        axes[0, 0].set_ylabel('Value', fontsize=12)
        axes[0, 0].set_title('Mean Value Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics, fontsize=11)
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                              f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4.2 Median comparison
        gpt_medians = [overall_stats['GPT']['median_coverage_ratio'], overall_stats['GPT']['median_precision']]
        rag_medians = [overall_stats['RAG']['median_coverage_ratio'], overall_stats['RAG']['median_precision']]
        instructor_medians = [overall_stats['Instructor']['median_coverage_ratio'], overall_stats['Instructor']['median_precision']]
        
        bars4 = axes[0, 1].bar(x - width, gpt_medians, width, label='GPT', color=model_colors['GPT'])
        bars5 = axes[0, 1].bar(x, rag_medians, width, label='RAG', color=model_colors['RAG'])
        bars6 = axes[0, 1].bar(x + width, instructor_medians, width, label='Instructor', color=model_colors['Instructor'])
        
        axes[0, 1].set_ylabel('Value', fontsize=12)
        axes[0, 1].set_title('Median Value Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(metrics, fontsize=11)
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        for bars in [bars4, bars5, bars6]:
            for bar in bars:
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                              f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4.3 Totals comparison
        totals_metrics = ['Total Covered', 'Total Items', 'Total Generated Concerns']
        gpt_totals = [
            overall_stats['GPT']['total_covered'],
            overall_stats['GPT']['total_items'],
            overall_stats['GPT']['total_gen_concerns']
        ]
        rag_totals = [
            overall_stats['RAG']['total_covered'],
            overall_stats['RAG']['total_items'],
            overall_stats['RAG']['total_gen_concerns']
        ]
        instructor_totals = [
            overall_stats['Instructor']['total_covered'],
            overall_stats['Instructor']['total_items'],
            overall_stats['Instructor']['total_gen_concerns']
        ]
        
        x2 = np.arange(len(totals_metrics))
        axes[1, 0].bar(x2 - width, gpt_totals, width, label='GPT', color=model_colors['GPT'])
        axes[1, 0].bar(x2, rag_totals, width, label='RAG', color=model_colors['RAG'])
        axes[1, 0].bar(x2 + width, instructor_totals, width, label='Instructor', color=model_colors['Instructor'])
        
        axes[1, 0].set_ylabel('Count', fontsize=12)
        axes[1, 0].set_title('Total Count Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(x2)
        axes[1, 0].set_xticklabels(totals_metrics, fontsize=10, rotation=15, ha='right')
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].set_yscale('log')  # Use logarithmic scale
        
        # 4.4 Difference analysis (GPT - RAG)
        coverage_diff = head_to_head['coverage_ratio_gpt'] - head_to_head['coverage_ratio_rag']
        precision_diff = head_to_head['precision_gpt'] - head_to_head['precision_rag']
        
        axes[1, 1].hist(coverage_diff, bins=30, alpha=0.7, label='Coverage Diff (GPT-RAG)', color=model_colors['GPT'], edgecolor='black')
        axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Line')
        axes[1, 1].set_title('Coverage Ratio Difference Distribution\n(GPT - RAG)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Difference', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/detailed_statistics.png', dpi=300, bbox_inches='tight')
        print(f"Figure saved: {output_dir}/detailed_statistics.png")
        plt.close()
        
        # 5. Side-by-side bar charts for coverage and precision (y-axis starts at 0.25)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        models = list(overall_stats.keys())
        colors = ['lightcoral', 'lightgreen', 'skyblue']
        
        # 5.1 Left: Coverage Ratio
        coverage_ratios = [overall_stats[model]['overall_coverage_ratio'] for model in models]
        bars1 = ax1.bar(models, coverage_ratios, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax1.set_title('Coverage Ratio', fontsize=24, fontweight='bold', pad=20)
        ax1.set_xlabel('Model', fontsize=20, fontweight='bold')
        ax1.set_ylabel('Coverage Ratio', fontsize=20, fontweight='bold')
        ax1.tick_params(axis='both', labelsize=18)
        
        # Ensure y-axis starts at 0.25 and set an appropriate upper limit
        max_coverage = max(coverage_ratios)
        ax1.set_ylim(0.25, max(max_coverage * 1.15, 0.3))  # y-axis starts at 0.25
        ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
        ax1.set_axisbelow(True)
        
        # Value labels
        for i, (bar, val) in enumerate(zip(bars1, coverage_ratios)):
            height = bar.get_height()
            label_y = height if height >= 0.25 else 0.26
            ax1.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{val:.3f}\n({val*100:.1f}%)', 
                    ha='center', va='bottom', fontsize=18, fontweight='bold')
        
        # 5.2 Right: Precision
        precisions = [overall_stats[model]['overall_precision'] for model in models]
        bars2 = ax2.bar(models, precisions, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax2.set_title('Precision', fontsize=24, fontweight='bold', pad=20)
        ax2.set_xlabel('Model', fontsize=20, fontweight='bold')
        ax2.set_ylabel('Precision', fontsize=20, fontweight='bold')
        ax2.tick_params(axis='both', labelsize=18)
        
        max_precision = max(precisions)
        ax2.set_ylim(0.25, max(max_precision * 1.15, 0.3))  # y-axis starts at 0.25
        ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
        ax2.set_axisbelow(True)
        
        for i, (bar, val) in enumerate(zip(bars2, precisions)):
            height = bar.get_height()
            label_y = height if height >= 0.25 else 0.26
            ax2.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{val:.3f}\n({val*100:.1f}%)', 
                    ha='center', va='bottom', fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/coverage_precision_bars.png', dpi=300, bbox_inches='tight')
        print(f"Figure saved: {output_dir}/coverage_precision_bars.png")
        plt.close()
    
    def save_detailed_results(self, output_dir: str = "coverage_comparison"):
        """Save detailed results to CSV and text files"""
        Path(output_dir).mkdir(exist_ok=True)
        
        overall_stats = self.calculate_overall_coverage()
        head_to_head = self.head_to_head_comparison()
        
        # Save head-to-head comparison
        head_to_head.to_csv(f'{output_dir}/head_to_head_comparison.csv', index=False, encoding='utf-8')
        
        # Save overall statistics
        overall_df = pd.DataFrame(overall_stats).T
        overall_df.to_csv(f'{output_dir}/overall_statistics.csv', encoding='utf-8')
        
        # Save report
        report = self.generate_statistics_report()
        with open(f'{output_dir}/analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\nDetailed results saved to directory: {output_dir}")
        print(f"  - head_to_head_comparison.csv: Detailed comparison results")
        print(f"  - overall_statistics.csv: Overall statistics")
        print(f"  - analysis_report.txt: Analysis report")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Compare concern coverage ratio and precision across GPT, RAG, and Instructor models')
    parser.add_argument('--gpt_file', 
                       default='Evaluation_feasibility/results/GPT/success_results.json',
                       help='Path to GPT results file')
    parser.add_argument('--rag_file', 
                       default='Evaluation_feasibility/results/RAG_evidence_retrieval/success_results.json',
                       help='Path to RAG results file')
    parser.add_argument('--instructor_file', 
                       default='Evaluation_feasibility/results/instructor/success_results.json',
                       help='Path to Instructor results file')
    parser.add_argument('--output_dir', 
                       default='Evaluation_feasibility/results/compare',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create comparator
    comparator = CoverageComparator(args.gpt_file, args.rag_file, args.instructor_file)
    
    # Load data
    comparator.load_data()
    
    # Generate report
    report = comparator.generate_statistics_report()
    print(report)
    
    # Create visualizations
    comparator.create_visualizations(args.output_dir)
    
    # Save detailed results
    comparator.save_detailed_results(args.output_dir)

if __name__ == "__main__":
    main()
