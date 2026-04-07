# -*- coding: utf-8 -*-
"""
Automated K-Means Clustering Pipeline for All Positions

This script runs the complete clustering pipeline for any position:
1. Optimal k selection (4 validation metrics)
2. K-Means clustering with robustness testing (ARI)
3. Cluster profiling with F-statistics and tactical naming
4. Publication-quality visualizations

Usage:
    python cluster_all_positions.py --position "Center Back"
    python cluster_all_positions.py --position "Full Back" --k_min 2 --k_max 8
"""

import pandas as pd
import json
import sys
import os
import argparse
from pathlib import Path

# Add src to path (go up 2 levels from scripts/clustering/ to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from clustering.optimal_k_selection import OptimalKSelector
from clustering.kmeans_clustering import PositionClusterer
from clustering.cluster_profiling import ClusterProfiler
from clustering.cluster_visualization import ClusterVisualizer

# Position-specific KPI mapping (from POSITION_KPI_MAPPING.md)
POSITION_KPIS = {
    'Midfielder': [
        'pass_completion_pct',
        'progressive_passes_per_90',
        'ball_recoveries_per_90',
        'interceptions_per_90',
        'tackles_won_per_90',
        'pressures_per_90',
        'progressive_carries_per_90'
    ],
    'Center Back': [
        'interceptions_per_90',
        'blocks_per_90',
        'clearances_per_90',
        'pressures_per_90',
        'pass_completion_pct',
        'progressive_passes_per_90'
    ],
    'Full Back': [
        'progressive_passes_per_90',
        'progressive_carries_per_90',
        'xa_per_90',
        'tackles_interceptions_per_90',
        'defensive_duels_win_pct',
        'touches_final_third_per_90',
        'possession_won_per_90'
    ],
    'Winger': [
        'successful_dribbles_per_90',
        'npxg_plus_xa_per_90',
        'shot_creating_actions_per_90',
        'progressive_carries_per_90',
        'key_passes_per_90',
        'touches_penalty_area_per_90',
        'pressures_per_90'
    ],
    'Forward': [
        'npxg_per_90',
        'non_penalty_goals_per_90',
        'shots_on_target_pct',
        'conversion_rate',
        'touches_penalty_area_per_90',
        'npxg_plus_xa_per_90',
        'successful_dribbles_per_90'
    ],
    'Goalkeeper': [
        'xga_per_90',
        'gk_pass_completion',
        'cross_claiming_rate',
        'sweeper_actions_per_90',
        'progressive_passes_per_90'
    ]
}

def create_output_directories(position_safe: str):
    """Create output directory structure for position."""
    directories = [
        f'data/processed/clustering/{position_safe}',
        f'outputs/clustering/plots/{position_safe}/optimal_k',
        f'outputs/clustering/plots/{position_safe}/dimensionality_reduction',
        f'outputs/clustering/plots/{position_safe}/distributions',
        f'outputs/clustering/plots/{position_safe}/cluster_profiles'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print(f"[OK] Output directories created for {position_safe}")

def run_clustering_pipeline(position_name: str, k_min: int = 2, k_max: int = 8):
    """
    Run complete clustering pipeline for a position.

    Args:
        position_name: Position name (e.g., 'Center Back')
        k_min: Minimum k to test
        k_max: Maximum k to test
    """
    print("="*70)
    print(f"CLUSTERING PIPELINE FOR {position_name.upper()}")
    print("="*70)

    # Get position-specific KPIs
    if position_name not in POSITION_KPIS:
        print(f"ERROR: Position '{position_name}' not found in POSITION_KPIS")
        print(f"Available positions: {list(POSITION_KPIS.keys())}")
        return

    kpi_list_raw = POSITION_KPIS[position_name]
    kpi_list_scaled = [kpi + '_scaled' for kpi in kpi_list_raw]

    print(f"\nPosition: {position_name}")
    print(f"KPIs: {len(kpi_list_raw)}")
    print(f"K range: {k_min} to {k_max}")

    # Safe filename version
    position_safe = position_name.lower().replace(' ', '_')

    # Create directories
    create_output_directories(position_safe)

    # ========== 1. LOAD DATA ==========
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)

    # Load cleaned position-specific data (364 players - filtered dataset)
    # Using cleaned data for consistency between clustering and RF validation
    data_path = f'data/processed/cleaned/by_position/{position_safe}_cleaned.csv'
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}")
        return
    
    position_data = pd.read_csv(data_path)
    print(f"{position_name} players loaded: {len(position_data)}")

    if len(position_data) == 0:
        print(f"ERROR: No players found for position '{position_name}'")
        return

    # Prepare data
    player_info_cols = ['player_id', 'player_name', 'team', 'thesis_position',
                        'minutes_played', 'matches_played']

    # Filter available columns
    available_player_cols = [c for c in player_info_cols if c in position_data.columns]
    player_info = position_data[available_player_cols]
    
    # Get raw KPI features
    raw_features = position_data[kpi_list_raw].copy()
    
    # Handle NaN values - fill with column mean, then 0 for any remaining NaN
    raw_features = raw_features.fillna(raw_features.mean())
    raw_features = raw_features.fillna(0)  # For columns that were entirely NaN
    
    # Apply StandardScaler to create scaled features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(raw_features)
    scaled_features = pd.DataFrame(scaled_values, columns=[f"{c}_scaled" for c in kpi_list_raw])
    
    print(f"[OK] Data prepared and scaled")
    print(f"  Player info: {player_info.shape}")
    print(f"  Raw features: {raw_features.shape}")
    print(f"  Scaled features: {scaled_features.shape}")

    # ========== 2. OPTIMAL K SELECTION ==========
    print("\n" + "="*70)
    print("STEP 2: OPTIMAL K SELECTION")
    print("="*70)

    X = scaled_features.values
    selector = OptimalKSelector(X, k_range=(k_min, k_max))
    result = selector.find_optimal_k()

    optimal_k = result['optimal_k']
    print(f"\n[SUCCESS] Optimal k = {optimal_k}")

    # Save results
    optimal_k_path = f'data/processed/clustering/{position_safe}/{position_safe}_optimal_k_results.json'
    output_data = {
        'position': position_name,
        'n_players': len(position_data),
        'n_features': len(kpi_list_raw),
        'feature_names': kpi_list_raw,
        'k_range': [k_min, k_max],
        'optimal_k': optimal_k,
        'confidence': result['confidence'],
        'reasoning': result['reasoning'],
        'metric_recommendations': result['metric_recommendations'],
        'all_metrics': {
            str(k): {
                'wcss': result['metrics']['wcss'][i],
                'silhouette': result['metrics']['silhouette'][i],
                'davies_bouldin': result['metrics']['davies_bouldin'][i],
                'calinski_harabasz': result['metrics']['calinski_harabasz'][i]
            }
            for i, k in enumerate(result['metrics']['k_values'])
        }
    }

    with open(optimal_k_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"[SAVED] {optimal_k_path}")

    # Generate plot
    plot_path = f'outputs/clustering/plots/{position_safe}/optimal_k/{position_safe}_optimal_k_analysis.png'
    selector.plot_metrics(save_path=plot_path, position_name=position_name)

    # ========== 3. K-MEANS CLUSTERING ==========
    print("\n" + "="*70)
    print("STEP 3: K-MEANS CLUSTERING + ROBUSTNESS TESTING")
    print("="*70)

    clusterer = PositionClusterer(
        position_name=position_name,
        n_clusters=optimal_k,
        scaled_features=scaled_features,
        raw_features=raw_features,
        player_info=player_info
    )

    test_random_states = [42, 123, 456]
    clusterer.fit_with_robustness_test(test_random_states=test_random_states)

    # Save results
    output_dir = f'data/processed/clustering/{position_safe}'
    saved_paths = clusterer.save_results(output_dir)

    print(f"\n[SUCCESS] Clustering completed")
    print(f"  Mean ARI: {clusterer.robustness_scores['mean_ari']:.3f}")
    print(f"  Stability: {clusterer.robustness_scores['stability']}")

    # ========== 4. CLUSTER PROFILING ==========
    print("\n" + "="*70)
    print("STEP 4: CLUSTER PROFILING")
    print("="*70)

    clustered_data = clusterer.get_clustered_data()

    profiler = ClusterProfiler(
        clustered_data=clustered_data,
        raw_kpi_cols=kpi_list_raw,
        position_name=position_name
    )

    profiler.generate_profiles()
    profiler.calculate_f_statistics()
    profiler.calculate_z_scores()
    tactical_names = profiler.assign_tactical_names()

    # Save profiling results
    profiling_paths = profiler.save_profiles(output_dir, position_safe)

    print(f"\n[SUCCESS] Cluster profiling completed")
    for cluster_id, info in tactical_names.items():
        print(f"  Cluster {cluster_id}: {info['name']}")

    # ========== 5. VISUALIZATIONS ==========
    print("\n" + "="*70)
    print("STEP 5: VISUALIZATIONS")
    print("="*70)

    visualizer = ClusterVisualizer(
        clustered_data=clustered_data,
        scaled_features=scaled_features,
        raw_kpi_cols=kpi_list_raw,
        cluster_names=tactical_names,
        position_name=position_name
    )

    output_base_dir = f'outputs/clustering/plots/{position_safe}'
    viz_paths = visualizer.plot_all(
        output_base_dir=output_base_dir,
        z_scores=profiler.z_scores,
        f_statistics=profiler.f_statistics
    )

    # ========== 6. SUMMARY ==========
    print("\n" + "="*70)
    print(f"PIPELINE COMPLETED FOR {position_name.upper()}")
    print("="*70)
    print(f"Position: {position_name}")
    print(f"Sample Size: {len(position_data)} players")
    print(f"Features: {len(kpi_list_raw)} position-specific KPIs")
    print(f"\nOptimal k: {optimal_k} ({result['confidence'].upper()} confidence)")
    print(f"Robustness (ARI): {clusterer.robustness_scores['mean_ari']:.3f} ({clusterer.robustness_scores['stability']})")

    print(f"\nClusters Identified:")
    for cluster_id in range(optimal_k):
        cluster_players = clustered_data[clustered_data['cluster_label'] == cluster_id]
        name = tactical_names.get(cluster_id, {}).get('name', f'Cluster {cluster_id}')
        print(f"  Cluster {cluster_id}: {name} (n={len(cluster_players)})")

    print(f"\nTop 3 Discriminative KPIs:")
    for i, (kpi, f_val) in enumerate(list(profiler.f_statistics.items())[:3], 1):
        print(f"  {i}. {kpi} (F={f_val:.2f})")

    print(f"\nOutputs Saved:")
    print(f"  Data: {output_dir}")
    print(f"  Plots: {output_base_dir}")
    print(f"  Total files: ~15-20")

    print("\n" + "="*70)
    print(f"[SUCCESS] {position_name} clustering pipeline completed!")
    print("="*70)

    return {
        'position': position_name,
        'n_players': len(position_data),
        'optimal_k': optimal_k,
        'confidence': result['confidence'],
        'mean_ari': clusterer.robustness_scores['mean_ari'],
        'stability': clusterer.robustness_scores['stability'],
        'top_kpis': list(profiler.f_statistics.keys())[:3],
        'cluster_names': {k: v['name'] for k, v in tactical_names.items()}
    }

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Run K-Means clustering pipeline for a position'
    )
    parser.add_argument(
        '--position',
        type=str,
        required=True,
        choices=['Midfielder', 'Center Back', 'Full Back', 'Winger', 'Forward', 'Goalkeeper'],
        help='Position name'
    )
    parser.add_argument(
        '--k_min',
        type=int,
        default=2,
        help='Minimum k to test (default: 2)'
    )
    parser.add_argument(
        '--k_max',
        type=int,
        default=8,
        help='Maximum k to test (default: 8)'
    )

    args = parser.parse_args()

    # Adjust k_max based on position if not specified
    if args.position == 'Goalkeeper' and args.k_max > 4:
        print(f"INFO: Adjusting k_max to 4 for Goalkeeper (small sample size)")
        args.k_max = 4
    elif args.position == 'Forward' and args.k_max > 5:
        print(f"INFO: Adjusting k_max to 5 for Forward")
        args.k_max = 5
    elif args.position == 'Winger' and args.k_max > 6:
        print(f"INFO: Adjusting k_max to 6 for Winger")
        args.k_max = 6

    # Run pipeline
    summary = run_clustering_pipeline(
        position_name=args.position,
        k_min=args.k_min,
        k_max=args.k_max
    )

    if summary:
        # Save summary
        position_safe = args.position.lower().replace(' ', '_')
        summary_path = f'data/processed/clustering/{position_safe}/{position_safe}_pipeline_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n[SAVED] Pipeline summary: {summary_path}")

if __name__ == "__main__":
    # If no args, run interactively
    if len(sys.argv) == 1:
        print("\nAvailable positions:")
        for i, pos in enumerate(POSITION_KPIS.keys(), 1):
            print(f"  {i}. {pos}")

        choice = input("\nSelect position (1-6): ")
        positions = list(POSITION_KPIS.keys())

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(positions):
                position = positions[idx]

                # Auto-adjust k_max
                k_max_map = {
                    'Goalkeeper': 4,
                    'Forward': 5,
                    'Winger': 6
                }
                k_max = k_max_map.get(position, 8)

                run_clustering_pipeline(position, k_min=2, k_max=k_max)
            else:
                print("Invalid choice")
        except ValueError:
            print("Invalid input")
    else:
        main()
