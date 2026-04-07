# -*- coding: utf-8 -*-
"""
Run RF Feature Importance Analysis for All 6 Positions
=======================================================

This script runs Random Forest feature importance analysis for all positions
and generates cross-position comparison visualizations.

Author: Football Analytics Project
Date: 2025-11-11
"""

import sys
import os

# Add project root and src to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from feature_importance.rf_feature_importance import RFFeatureImportanceAnalyzer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Position configuration 
POSITIONS = [
    'Midfielder',
    'Center Back',
    'Full Back',
    'Winger',
    'Forward',
    'Goalkeeper'
]

def run_all_positions():
    """Run RF analysis for all 6 positions."""
    print("="*70)
    print("RF FEATURE IMPORTANCE ANALYSIS - ALL 6 POSITIONS")
    print("="*70)

    results = []

    for position in POSITIONS:
        position_safe = position.lower().replace(' ', '_')

        # Paths
        clustered_data = f'data/processed/clustering/{position_safe}/{position_safe}_clustered.csv'
        f_statistics = f'data/processed/clustering/{position_safe}/{position_safe}_f_statistics.json'
        output_dir = f'data/processed/feature_importance/{position_safe}'

        print(f"\n{'='*70}")
        print(f"Processing: {position.upper()}")
        print(f"{'='*70}")

        try:
            # Run analysis
            analyzer = RFFeatureImportanceAnalyzer(
                position_name=position,
                clustered_data_path=clustered_data,
                f_statistics_path=f_statistics,
                n_estimators=100,
                random_state=42
            )

            summary = analyzer.run_full_analysis(output_dir)
            results.append(summary)

            print(f"\n[SUCCESS] {position} completed")
            print(f"  CV Accuracy: {summary['cv_accuracy']:.3f}")
            print(f"  Spearman rho: {summary['spearman_rho']:.3f}")

        except Exception as e:
            print(f"\n[ERROR] {position} failed: {str(e)}")
            continue

    print("\n" + "="*70)
    print("ALL POSITIONS COMPLETED")
    print("="*70)
    print(f"Processed: {len(results)}/6 positions")

    return results

def generate_cross_position_heatmap(results):
    """Generate cross-position feature importance heatmap."""
    print("\n" + "="*70)
    print("GENERATING CROSS-POSITION IMPORTANCE HEATMAP")
    print("="*70)

    # Collect importance data
    importance_data = {}

    for result in results:
        position = result['position']
        position_safe = position.lower().replace(' ', '_')

        # Load importance rankings
        rankings_path = f'data/processed/feature_importance/{position_safe}/{position_safe}_importance_rankings.csv'
        rankings = pd.read_csv(rankings_path)

        # Get top 10 features
        top_features = rankings.head(10)
        importance_data[position] = dict(zip(top_features['feature'],
                                            top_features['rf_importance']))

    # Create matrix (all unique features × positions)
    all_features = sorted(set(feat for pos_feats in importance_data.values()
                             for feat in pos_feats.keys()))

    matrix = []
    for feature in all_features:
        row = [importance_data[pos].get(feature, 0) for pos in [r['position'] for r in results]]
        matrix.append(row)

    matrix = np.array(matrix)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)

    sns.heatmap(matrix,
               xticklabels=[r['position'] for r in results],
               yticklabels=all_features,
               cmap='YlOrRd',
               annot=True,
               fmt='.3f',
               cbar_kws={'label': 'RF Importance'},
               linewidths=0.5,
               linecolor='gray',
               ax=ax)

    ax.set_title('Cross-Position Feature Importance Matrix\n(Random Forest)',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature (KPI)', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = 'data/processed/feature_importance/cross_position_importance_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SAVED] {output_path}")

    return output_path

def generate_summary_table(results):
    """Generate summary table of all positions."""
    print("\n" + "="*70)
    print("GENERATING SUMMARY TABLE")
    print("="*70)

    # Create summary DataFrame
    summary_data = []
    for result in results:
        summary_data.append({
            'Position': result['position'],
            'N Samples': result['n_samples'],
            'N Features': result['n_features'],
            'CV Accuracy': f"{result['cv_accuracy']:.3f}",
            'Spearman rho': f"{result['spearman_rho']:.3f}",
            'Top Feature': result['top_3_features'][0],
            'Top 3 Features': ', '.join(result['top_3_features'])
        })

    summary_df = pd.DataFrame(summary_data)

    # Save
    output_path = 'data/processed/feature_importance/rf_summary_all_positions.csv'
    summary_df.to_csv(output_path, index=False, encoding='utf-8')

    print(f"[SAVED] {output_path}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))

    return summary_df

def generate_validation_plot(results):
    """Generate validation summary plot (accuracy vs. Spearman rho)."""
    print("\n" + "="*70)
    print("GENERATING VALIDATION SUMMARY PLOT")
    print("="*70)

    positions = [r['position'] for r in results]
    accuracies = [r['cv_accuracy'] for r in results]
    rhos = [r['spearman_rho'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Plot 1: CV Accuracy
    colors1 = ['green' if acc > 0.8 else 'orange' for acc in accuracies]
    bars1 = ax1.bar(positions, accuracies, color=colors1, alpha=0.7, edgecolor='black', linewidth=1)
    ax1.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='Target (0.8)')
    ax1.set_ylabel('CV Accuracy', fontsize=11, fontweight='bold')
    ax1.set_title('Random Forest Classification Accuracy\n(5-Fold Cross-Validation)',
                 fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1.0])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 2: Spearman rho
    colors2 = ['green' if rho > 0.7 else 'orange' if rho > 0.5 else 'red' for rho in rhos]
    bars2 = ax2.bar(positions, rhos, color=colors2, alpha=0.7, edgecolor='black', linewidth=1)
    ax2.axhline(y=0.7, color='red', linestyle='--', linewidth=2, label='Target (0.7 - Strong)')
    ax2.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5, label='Moderate (0.5)')
    ax2.set_ylabel('Spearman rho', fontsize=11, fontweight='bold')
    ax2.set_title('RF Importance vs. F-Statistics Correlation\n(Validation)',
                 fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1.0])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, rho in zip(bars2, rhos):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{rho:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = 'data/processed/feature_importance/validation_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SAVED] {output_path}")

    return output_path

def main():
    """Main entry point."""
    # Run all positions
    results = run_all_positions()

    if len(results) == 0:
        print("\n[ERROR] No positions completed successfully")
        return

    # Generate cross-position visualizations
    generate_cross_position_heatmap(results)
    generate_summary_table(results)
    generate_validation_plot(results)

    # Final summary
    print("\n" + "="*70)
    print("RF ANALYSIS COMPLETE FOR ALL POSITIONS")
    print("="*70)
    print(f"Successfully processed: {len(results)}/6 positions")
    print(f"\nMean CV Accuracy: {np.mean([r['cv_accuracy'] for r in results]):.3f}")
    print(f"Mean Spearman rho: {np.mean([r['spearman_rho'] for r in results]):.3f}")

    # Check success criteria
    accuracy_pass = sum(1 for r in results if r['cv_accuracy'] > 0.8)
    rho_pass = sum(1 for r in results if r['spearman_rho'] > 0.7)

    print(f"\nSuccess Criteria:")
    print(f"  CV Accuracy > 0.8: {accuracy_pass}/{len(results)} positions")
    print(f"  Spearman rho > 0.7: {rho_pass}/{len(results)} positions")

    if accuracy_pass == len(results) and rho_pass == len(results):
        print("\n✓ ALL SUCCESS CRITERIA MET!")
    else:
        print("\n⚠ Some positions did not meet all criteria")

    print("="*70)

if __name__ == "__main__":
    main()
