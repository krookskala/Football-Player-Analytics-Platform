# -*- coding: utf-8 -*-
"""
Batch run clustering for remaining 4 positions
Optimized for speed - minimal console output
"""

import subprocess
import json
from pathlib import Path

positions_config = [
    {'name': 'Full Back', 'k_min': 2, 'k_max': 8},
    {'name': 'Winger', 'k_min': 2, 'k_max': 6},
    {'name': 'Forward', 'k_min': 2, 'k_max': 5},
    {'name': 'Goalkeeper', 'k_min': 2, 'k_max': 4}
]

print("="*70)
print("BATCH CLUSTERING FOR 4 REMAINING POSITIONS")
print("="*70)

summaries = []

for config in positions_config:
    pos_name = config['name']
    k_min = config['k_min']
    k_max = config['k_max']

    print(f"\n[{pos_name}] Starting pipeline (k={k_min}-{k_max})...")

    # Run clustering pipeline
    cmd = [
        'python', 'cluster_all_positions.py',
        '--position', pos_name,
        '--k_min', str(k_min),
        '--k_max', str(k_max)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode == 0:
            # Load summary
            pos_safe = pos_name.lower().replace(' ', '_')
            summary_path = f'data/processed/clustering/{pos_safe}/{pos_safe}_pipeline_summary.json'

            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)

            summaries.append(summary)

            print(f"[{pos_name}] SUCCESS")
            print(f"  Optimal k: {summary['optimal_k']} ({summary['confidence']})")
            print(f"  ARI: {summary['mean_ari']:.3f} ({summary['stability']})")
            print(f"  Top KPI: {summary['top_kpis'][0]}")
        else:
            print(f"[{pos_name}] FAILED")
            print(f"  Error: {result.stderr[:200]}")

    except Exception as e:
        print(f"[{pos_name}] ERROR: {str(e)}")

print("\n" + "="*70)
print("BATCH CLUSTERING COMPLETED")
print("="*70)

# Save combined summary
combined_summary_path = 'data/processed/clustering/all_positions_summary.json'
combined_summary = {
    'total_positions': len(summaries) + 2,  # +2 for Midfielder and Center Back
    'completed': summaries
}

with open(combined_summary_path, 'w', encoding='utf-8') as f:
    json.dump(combined_summary, f, indent=2, ensure_ascii=False)

print(f"\n[SUCCESS] Combined summary saved: {combined_summary_path}")
print(f"[INFO] Processed {len(summaries)}/4 positions successfully")
