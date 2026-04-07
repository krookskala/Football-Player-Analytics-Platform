"""
Persistent Cache Manager
-------------------------
Level 2 caching with joblib.Memory (disk-based).

Survives Streamlit restarts, providing:
- Precomputed PCA coordinates
- Radar chart normalization parameters
- Correlation matrices

Usage:
    from cache_manager import compute_pca_coordinates

    pca_df = compute_pca_coordinates('midfielder')  # Cached to disk
"""

from joblib import Memory
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import config

# ============================================================================
# INITIALIZE DISK CACHE
# ============================================================================

# Ensure cache directory exists
config.CACHE_PATH.mkdir(parents=True, exist_ok=True)

# Initialize joblib Memory
disk_cache = Memory(
    location=config.CACHE_CONFIG['disk_location'],
    verbose=config.CACHE_CONFIG['disk_verbose']
)

import logging
logger = logging.getLogger(__name__)
logger.debug(f"Disk cache initialized at: {config.CACHE_PATH}")

# ============================================================================
# CACHED COMPUTATIONS
# ============================================================================

@disk_cache.cache
def compute_pca_coordinates(position: str) -> pd.DataFrame:
    """
    Compute PCA coordinates (heavy operation).

    Cached to disk - survives restarts.

    Priority:
    1. Check if precomputed file exists ({position}_pca_coords.csv)
    2. If yes, load and return
    3. If no, compute from scratch (fallback)

    Args:
        position: Position key (e.g., 'midfielder')

    Returns:
        DataFrame with columns: player_id, player_name, pca_x, pca_y,
                                cluster_label, explained_variance_ratio_1,
                                explained_variance_ratio_2

    Performance:
        - Precomputed file: ~0.05 seconds
        - Runtime computation: ~0.8 seconds
        - Disk cache hit: ~0.1 seconds
    """
    # Check for precomputed file first
    precomputed_path = config.get_clustering_path(position, 'pca_coords')

    if precomputed_path.exists():
        print(f"[OK] Loading precomputed PCA coords: {position}")
        return pd.read_csv(precomputed_path)

    # Fallback: Compute from scratch
    print(f"[!] Precomputed PCA not found for {position}, computing...")

    from sklearn.decomposition import PCA

    # Load clustered data
    clustered_path = config.get_clustering_path(position, 'clustered')
    data = pd.read_csv(clustered_path)

    # Get scaled KPI columns (suffix _scaled, not prefix scaled_)
    scaled_cols = [col for col in data.columns if col.endswith('_scaled')]

    if len(scaled_cols) == 0:
        raise ValueError(f"No scaled KPI columns found for {position}")

    X_scaled = data[scaled_cols].values

    # Fit PCA
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    # Create output dataframe
    pca_df = pd.DataFrame({
        'player_id': data['player_id'],
        'player_name': data['player_name'],
        'pca_x': coords[:, 0],
        'pca_y': coords[:, 1],
        'cluster_label': data['cluster_label'],
        'explained_variance_ratio_1': pca.explained_variance_ratio_[0],
        'explained_variance_ratio_2': pca.explained_variance_ratio_[1]
    })

    print(f"  PC1 explained variance: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"  PC2 explained variance: {pca.explained_variance_ratio_[1]:.2%}")

    return pca_df


@disk_cache.cache
def compute_radar_normalization(position: str) -> Dict[str, Tuple[float, float]]:
    """
    Compute min/max for each KPI (for radar normalization).

    Cached to disk.

    Args:
        position: Position key

    Returns:
        Dict mapping KPI name to (min, max) tuple

    Example:
        {
            'ball_recoveries_per_90': (3.2, 12.8),
            'progressive_passes_per_90': (1.5, 9.3),
            ...
        }

    Performance:
        - Runtime computation: ~0.05 seconds
        - Disk cache hit: ~0.02 seconds
    """
    # Load clustered data
    clustered_path = config.get_clustering_path(position, 'clustered')
    data = pd.read_csv(clustered_path)

    # Get KPIs for this position
    kpis = config.get_position_kpis(position)

    # Compute min/max for each KPI
    norm_params = {}
    for kpi in kpis:
        if kpi in data.columns:
            norm_params[kpi] = (float(data[kpi].min()), float(data[kpi].max()))
        else:
            print(f"[!] Warning: KPI '{kpi}' not found in {position} data")

    return norm_params


@disk_cache.cache
def compute_correlation_matrix(position: str) -> pd.DataFrame:
    """
    Compute RF importance vs F-stat correlation.

    Heavy computation - cache to disk.

    Args:
        position: Position key

    Returns:
        DataFrame with columns: kpi, rf_importance, f_statistic,
                                rf_rank, f_rank

    Performance:
        - Runtime computation: ~0.2 seconds
        - Disk cache hit: ~0.05 seconds
    """
    import json

    # Load RF results
    rf_path = config.get_feature_importance_path(position, 'rf_results')
    with open(rf_path, 'r') as f:
        rf_results = json.load(f)

    # Load F-statistics
    f_stats_path = config.get_clustering_path(position, 'f_statistics')
    with open(f_stats_path, 'r') as f:
        f_stats = json.load(f)

    # Extract importance scores
    rf_importance = rf_results.get('feature_importance', {})

    # Create dataframe
    data = []
    for kpi in rf_importance.keys():
        rf_score = rf_importance[kpi]
        f_score = f_stats.get(kpi, {}).get('f_statistic', np.nan)

        data.append({
            'kpi': kpi,
            'rf_importance': rf_score,
            'f_statistic': f_score
        })

    df = pd.DataFrame(data)

    # Add ranks
    df['rf_rank'] = df['rf_importance'].rank(ascending=False)
    df['f_rank'] = df['f_statistic'].rank(ascending=False)

    # Sort by RF importance
    df = df.sort_values('rf_importance', ascending=False).reset_index(drop=True)

    return df


@disk_cache.cache
def compute_cluster_statistics(position: str) -> Dict:
    """
    Compute aggregated cluster statistics.

    Cached to disk.

    Args:
        position: Position key

    Returns:
        Dict with cluster statistics

    Performance:
        - Runtime computation: ~0.1 seconds
        - Disk cache hit: ~0.03 seconds
    """
    # Load clustered data
    clustered_path = config.get_clustering_path(position, 'clustered')
    data = pd.read_csv(clustered_path)

    # Load cluster profiles
    profiles_path = config.get_clustering_path(position, 'profiles')
    profiles = pd.read_csv(profiles_path)

    # Get KPIs
    kpis = config.get_position_kpis(position)

    # Compute statistics per cluster
    stats = {}
    for cluster_id in data['cluster_label'].unique():
        cluster_data = data[data['cluster_label'] == cluster_id]

        stats[int(cluster_id)] = {
            'n_players': len(cluster_data),
            'percentage': len(cluster_data) / len(data) * 100,
            'mean_distance_to_centroid': cluster_data.get('distance_to_centroid', pd.Series([np.nan])).mean(),
            'kpi_means': {kpi: float(cluster_data[kpi].mean()) for kpi in kpis if kpi in cluster_data.columns}
        }

    return stats


# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

def clear_cache():
    """
    Clear all disk cache.

    Warning: This will force recomputation on next access.
    """
    disk_cache.clear()
    print("[CACHE] Disk cache cleared")


def get_cache_info():
    """
    Get information about disk cache.

    Returns:
        Dict with cache statistics
    """
    cache_dir = Path(config.CACHE_CONFIG['disk_location'])

    if not cache_dir.exists():
        return {'exists': False}

    # Count files
    n_files = len(list(cache_dir.rglob('*')))

    # Calculate total size
    total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
    total_size_mb = total_size / (1024 * 1024)

    return {
        'exists': True,
        'location': str(cache_dir),
        'n_files': n_files,
        'total_size_mb': round(total_size_mb, 2)
    }


# ============================================================================
# PRECOMPUTATION CHECK
# ============================================================================

def check_precomputed_files():
    """
    Check if precomputed PCA files exist for all positions.

    Returns:
        Dict mapping position to bool (file exists)
    """
    results = {}

    for position in config.POSITION_KEYS:
        pca_path = config.get_clustering_path(position, 'pca_coords')
        results[position] = pca_path.exists()

    return results


def print_precomputed_status():
    """
    Print status of precomputed files.
    """
    results = check_precomputed_files()

    print("\n" + "="*60)
    print("PRECOMPUTED PCA FILES STATUS")
    print("="*60)

    for position, exists in results.items():
        status = "[OK]" if exists else "[NO]"
        print(f"{status} {config.POSITIONS[position]['name']:15} | {position:12} | Exists: {exists}")

    n_exists = sum(results.values())
    print(f"\nTotal: {n_exists}/{len(results)} positions have precomputed PCA files")

    if n_exists < len(results):
        print("\n[!] Warning: Some positions missing precomputed PCA files.")
        print("   Run 'precompute_pca_for_dashboard.py' to generate them.")
    else:
        print("\n[OK] All positions have precomputed PCA files!")

    print("="*60 + "\n")


# ============================================================================
# INITIALIZATION
# ============================================================================

if __name__ == '__main__':
    # Test cache manager
    print("[TEST] Testing cache manager...")

    # Print cache info
    cache_info = get_cache_info()
    print(f"\nCache info: {cache_info}")

    # Check precomputed files
    print_precomputed_status()

    # Test computation (will use cache if available)
    print("\n[TEST] Testing PCA computation (midfielder)...")
    pca_df = compute_pca_coordinates('midfielder')
    print(f"[OK] PCA computed: {len(pca_df)} players")

    print("\n[TEST] Testing radar normalization (midfielder)...")
    norm_params = compute_radar_normalization('midfielder')
    print(f"[OK] Normalization params: {len(norm_params)} KPIs")

    print("\n[OK] Cache manager tests passed!")
