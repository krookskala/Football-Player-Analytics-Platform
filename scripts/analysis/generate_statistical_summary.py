"""
Generate Statistical Summary with Confidence Intervals and Effect Sizes

This script enhances clustering reports with:
1. Cluster means in "Mean ± SD" format
2. Effect sizes (η² - eta squared) for F-statistics
3. Statistical summary tables for thesis

"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import json

# Position configuration
POSITIONS = {
    'midfielder': {
        'name': 'Midfielder',
        'kpis': ['pass_completion_pct', 'progressive_passes_per_90',
                 'ball_recoveries_per_90', 'interceptions_per_90',
                 'successful_dribbles_per_90']
    },
    'center_back': {
        'name': 'Center Back',
        'kpis': ['pressures_per_90', 'aerial_duels_won_pct',
                 'interceptions_per_90']
    },
    'full_back': {
        'name': 'Full Back',
        'kpis': ['pressures_per_90', 'interceptions_per_90']
    },
    'winger': {
        'name': 'Winger',
        'kpis': ['successful_dribbles_per_90', 'progressive_passes_per_90']
    },
    'forward': {
        'name': 'Forward',
        'kpis': ['npxg_per_90', 'successful_dribbles_per_90']
    },
    'goalkeeper': {
        'name': 'Goalkeeper',
        'kpis': ['psxg_minus_ga_per_90', 'passes_completed_pct',
                 'long_passes_completed_pct', 'sweeper_actions_per_90',
                 'aerial_duels_won_pct']
    }
}

def calculate_eta_squared(df, kpi, cluster_col='cluster_label'):
    """
    Calculate η² (eta squared) effect size for ANOVA

    η² = SS_between / SS_total

    Interpretation (Cohen, 1988):
    - 0.01: Small effect
    - 0.06: Medium effect
    - 0.14: Large effect
    """
    # Get cluster groups
    groups = [group[kpi].values for name, group in df.groupby(cluster_col)]

    # Overall mean
    grand_mean = df[kpi].mean()

    # Total sum of squares
    ss_total = np.sum((df[kpi] - grand_mean) ** 2)

    # Between-group sum of squares
    ss_between = sum([
        len(group) * (np.mean(group) - grand_mean) ** 2
        for group in groups
    ])

    # η² = SS_between / SS_total
    eta_squared = ss_between / ss_total if ss_total > 0 else 0

    return eta_squared

def interpret_eta_squared(eta2):
    """Interpret η² effect size"""
    if eta2 < 0.01:
        return "Negligible"
    elif eta2 < 0.06:
        return "Small"
    elif eta2 < 0.14:
        return "Medium"
    else:
        return "Large"

def generate_position_statistics(position_key):
    """Generate enhanced statistics for a single position"""

    pos_config = POSITIONS[position_key]
    pos_name = pos_config['name']
    kpis = pos_config['kpis']

    # Load clustered data
    data_path = Path(f'data/processed/clustering/{position_key}/{position_key}_clustered.csv')
    if not data_path.exists():
        print(f"[SKIP] {pos_name}: Data file not found")
        return None

    df = pd.read_csv(data_path)

    # Get number of clusters
    n_clusters = df['cluster_label'].nunique()

    print(f"\n{'='*60}")
    print(f"POSITION: {pos_name} (n={len(df)}, k={n_clusters})")
    print(f"{'='*60}")

    # Initialize results
    results = {
        'position': pos_name,
        'n_players': len(df),
        'n_clusters': n_clusters,
        'kpis': {}
    }

    for kpi in kpis:
        if kpi not in df.columns:
            print(f"  [SKIP] {kpi}: Column not found")
            continue

        # Calculate statistics per cluster
        cluster_stats = df.groupby('cluster_label')[kpi].agg(['mean', 'std', 'count'])

        # Calculate F-statistic
        groups = [group[kpi].values for name, group in df.groupby('cluster_label')]
        f_stat, p_value = stats.f_oneway(*groups)

        # Calculate η² effect size
        eta2 = calculate_eta_squared(df, kpi)
        eta2_interp = interpret_eta_squared(eta2)

        # Store results
        results['kpis'][kpi] = {
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'eta_squared': float(eta2),
            'eta_squared_interpretation': eta2_interp,
            'clusters': {}
        }

        print(f"\n  KPI: {kpi}")
        print(f"    F-statistic: {f_stat:.2f}, p={p_value:.4f}")
        print(f"    eta-squared = {eta2:.3f} ({eta2_interp} effect)")
        print(f"    Cluster Statistics (Mean +/- SD):")

        for cluster_id in range(n_clusters):
            if cluster_id in cluster_stats.index:
                mean = cluster_stats.loc[cluster_id, 'mean']
                std = cluster_stats.loc[cluster_id, 'std']
                n = int(cluster_stats.loc[cluster_id, 'count'])

                # Store in results
                results['kpis'][kpi]['clusters'][int(cluster_id)] = {
                    'mean': float(mean),
                    'std': float(std),
                    'n': n,
                    'formatted': f"{mean:.2f} ± {std:.2f}"
                }

                print(f"      Cluster {cluster_id}: {mean:6.2f} +/- {std:5.2f} (n={n})")

    return results

def generate_markdown_table(all_results):
    """Generate markdown table for thesis"""

    md_output = []
    md_output.append("# Statistical Summary: Cluster Means and Effect Sizes\n")
    md_output.append("**Generated**: 2025-11-23\n")
    md_output.append("**Purpose**: Enhanced statistical reporting for thesis\n")
    md_output.append("\n---\n")

    for pos_results in all_results:
        if pos_results is None:
            continue

        pos_name = pos_results['position']
        n_players = pos_results['n_players']
        n_clusters = pos_results['n_clusters']

        md_output.append(f"\n## {pos_name} (n={n_players}, k={n_clusters})\n")

        for kpi, kpi_data in pos_results['kpis'].items():
            f_stat = kpi_data['f_statistic']
            p_val = kpi_data['p_value']
            eta2 = kpi_data['eta_squared']
            eta2_interp = kpi_data['eta_squared_interpretation']

            md_output.append(f"\n### {kpi}\n")
            md_output.append(f"**F-statistic**: {f_stat:.2f}, p={p_val:.4f}  \n")
            md_output.append(f"**Effect Size (η²)**: {eta2:.3f} ({eta2_interp})  \n")
            md_output.append("\n| Cluster | Mean ± SD | n |\n")
            md_output.append("|---------|-----------|---|\n")

            for cluster_id, cluster_data in kpi_data['clusters'].items():
                formatted = cluster_data['formatted']
                n = cluster_data['n']
                md_output.append(f"| Cluster {cluster_id} | {formatted} | {n} |\n")

    return "".join(md_output)

def main():
    """Main execution"""

    print("\n" + "="*60)
    print("STATISTICAL SUMMARY GENERATOR")
    print("="*60)
    print("Calculating Mean +/- SD and eta-squared for all positions...")

    # Generate statistics for all positions
    all_results = []
    for pos_key in POSITIONS.keys():
        results = generate_position_statistics(pos_key)
        if results:
            all_results.append(results)

    # Save JSON results
    output_dir = Path('data/processed/clustering')
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / 'statistical_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] JSON saved: {json_path}")

    # Generate markdown table
    md_content = generate_markdown_table(all_results)
    md_path = Path('docs/STATISTICAL_SUMMARY.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"[OK] Markdown saved: {md_path}")

    print("\n" + "="*60)
    print("STATISTICAL SUMMARY COMPLETE")
    print("="*60)
    print(f"Total positions processed: {len(all_results)}")
    print(f"\nOutput files:")
    print(f"  1. {json_path}")
    print(f"  2. {md_path}")
    print("\nUse these enhanced statistics in your thesis!")

if __name__ == '__main__':
    main()
