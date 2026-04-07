"""
Data Loader Module
------------------
Centralized data loading with dual-layer caching:
- Level 1: Streamlit RAM cache (@st.cache_data)
- Level 2: Disk cache (joblib.Memory via cache_manager)

Performance:
- First load: ~0.5 seconds (read from disk + cache)
- Subsequent loads: ~0.01 seconds (RAM cache hit)
- After restart: ~0.1 seconds (disk cache hit)

Usage:
    from data_loader import load_clustered_data, load_pca_coordinates

    # Load data for a position
    df = load_clustered_data('midfielder')  # Cached
    pca_df = load_pca_coordinates('midfielder')  # Precomputed + cached
"""

import streamlit as st
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import config
from cache_manager import (
    compute_pca_coordinates,
    compute_radar_normalization,
    compute_correlation_matrix,
    compute_cluster_statistics
)

# ============================================================================
# CORE DATA LOADERS (Level 1: Streamlit Cache)
# ============================================================================

@st.cache_data(ttl=config.CACHE_CONFIG['ram_ttl'])
def load_clustered_data(position: str) -> pd.DataFrame:
    """
    Load clustered player data for a position.

    Args:
        position: Position key (e.g., 'midfielder')

    Returns:
        DataFrame with player data + cluster assignments

    Columns:
        - player_id, player_name, team, thesis_position
        - minutes_played, matches_played
        - KPI columns (raw values)
        - KPI_scaled columns (standardized)
        - cluster_label, silhouette_score, distance_to_centroid

    Performance:
        - First load: ~50ms
        - Cache hit: ~1ms
    """
    path = config.get_clustering_path(position, 'clustered')

    if not path.exists():
        raise FileNotFoundError(f"Clustered data not found: {path}")

    df = pd.read_csv(path)

    # Handle column naming: 'cluster' -> 'cluster_label'
    if 'cluster' in df.columns and 'cluster_label' not in df.columns:
        df = df.rename(columns={'cluster': 'cluster_label'})

    # Validate essential columns
    required_cols = ['player_id', 'player_name', 'cluster_label']
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in {position}: {missing}")

    return df


@st.cache_data(ttl=config.CACHE_CONFIG['ram_ttl'])
def load_cluster_profiles(position: str) -> pd.DataFrame:
    """
    Load cluster profile statistics (mean, std, median per cluster).

    Args:
        position: Position key

    Returns:
        DataFrame with cluster statistics

    Columns:
        - cluster_id, n_players
        - {kpi}_mean, {kpi}_std, {kpi}_median for each KPI

    Performance:
        - First load: ~30ms
        - Cache hit: ~1ms
    """
    path = config.get_clustering_path(position, 'profiles')

    if not path.exists():
        raise FileNotFoundError(f"Cluster profiles not found: {path}")

    df = pd.read_csv(path)
    return df


@st.cache_data(ttl=config.CACHE_CONFIG['ram_ttl'])
def load_tactical_names(position: str) -> Dict:
    """
    Load tactical cluster names and metadata.

    Args:
        position: Position key

    Returns:
        Dict mapping cluster_id to metadata:
        {
            0: {
                'name': 'Balanced Midfielder',
                'justification': '...',
                'top_3_kpis': [...],
                'intensity_score': 1.2,
                'progression_score': 0.5
            },
            ...
        }

    Performance:
        - First load: ~20ms
        - Cache hit: ~1ms
    """
    path = config.get_clustering_path(position, 'tactical_names')

    if not path.exists():
        raise FileNotFoundError(f"Tactical names not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert string keys to int
    return {int(k): v for k, v in data.items()}


@st.cache_data(ttl=config.CACHE_CONFIG['ram_ttl'])
def load_z_scores(position: str) -> pd.DataFrame:
    """
    Load z-score normalized cluster profiles.

    Args:
        position: Position key

    Returns:
        DataFrame with z-scores for each cluster and KPI

    Performance:
        - First load: ~30ms
        - Cache hit: ~1ms
    """
    path = config.get_clustering_path(position, 'z_scores')

    if not path.exists():
        raise FileNotFoundError(f"Z-scores not found: {path}")

    df = pd.read_csv(path)
    return df


@st.cache_data(ttl=config.CACHE_CONFIG['ram_ttl'])
def load_f_statistics(position: str) -> Dict:
    """
    Load F-statistics for discriminative KPIs.

    Args:
        position: Position key

    Returns:
        Dict mapping KPI name to F-statistic value:
        {
            'ball_recoveries_per_90': 69.13,
            ...
        }

    Performance:
        - First load: ~20ms
        - Cache hit: ~1ms
    """
    path = config.get_clustering_path(position, 'f_statistics')

    if not path.exists():
        raise FileNotFoundError(f"F-statistics not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


# ============================================================================
# PCA DATA (Level 2: Disk Cache via cache_manager)
# ============================================================================

@st.cache_data(ttl=config.CACHE_CONFIG['ram_ttl'])
def load_pca_coordinates(position: str) -> pd.DataFrame:
    """
    Load PCA coordinates (precomputed or computed on-the-fly).

    Uses dual-layer caching:
    1. Streamlit RAM cache (this function)
    2. Disk cache (cache_manager.compute_pca_coordinates)
    3. Precomputed file (if exists)

    Args:
        position: Position key

    Returns:
        DataFrame with columns:
        - player_id, player_name, pca_x, pca_y, cluster_label
        - explained_variance_ratio_1, explained_variance_ratio_2

    Performance:
        - Precomputed file: ~50ms first load, ~1ms cached
        - Runtime computation: ~800ms first load, ~1ms cached
        - After restart: ~100ms (disk cache hit)
    """
    return compute_pca_coordinates(position)


@st.cache_data(ttl=config.CACHE_CONFIG['ram_ttl'])
def load_radar_norm_params(position: str) -> Dict[str, Tuple[float, float]]:
    """
    Load radar chart normalization parameters (min, max for each KPI).

    Uses dual-layer caching.

    Args:
        position: Position key

    Returns:
        Dict mapping KPI to (min, max) tuple

    Performance:
        - First load: ~50ms
        - Cache hit: ~1ms
        - After restart: ~20ms (disk cache hit)
    """
    return compute_radar_normalization(position)

@st.cache_data(ttl=config.CACHE_CONFIG['ram_ttl'])
def compute_percentiles(position: str) -> pd.DataFrame:
    """
    Compute percentile ranks for all players in a position.
    
    Args:
        position: Position key
        
    Returns:
        DataFrame with same index as clustered data, but values are percentiles (0-100).
        Columns: {kpi}_percentile for each numeric KPI.
        
    Performance:
        - First load: ~50ms
        - Cache hit: ~1ms
    """
    # Load raw data
    df = load_clustered_data(position)
    
    # Get numeric columns (KPIs)
    kpis = get_position_kpis(position)
    
    # Calculate percentiles
    percentiles = pd.DataFrame(index=df.index)
    percentiles['player_name'] = df['player_name']
    percentiles['player_id'] = df['player_id']
    
    for kpi in kpis:
        if kpi in df.columns:
            # Rank pct=True gives 0.0 to 1.0
            # We want integer 0-100
            # method='min' ensures ties get same lower rank, 'average' is better for percentiles
            percentiles[kpi] = df[kpi].rank(pct=True, method='average') * 100
            
    return percentiles
# ============================================================================
# FEATURE IMPORTANCE DATA
# ============================================================================

@st.cache_data(ttl=config.CACHE_CONFIG['ram_ttl'])
def load_rf_results(position: str) -> Dict:
    """
    Load Random Forest feature importance results.

    Args:
        position: Position key

    Returns:
        Dict with:
        {
            'position': 'midfielder',
            'cv_accuracy': 0.876,
            'cv_std': 0.081,
            'spearman_rho': 0.750,
            'validation_status': 'VALIDATED',
            'feature_importance': {
                'ball_recoveries_per_90': 0.315,
                ...
            }
        }

    Performance:
        - First load: ~20ms
        - Cache hit: ~1ms
    """
    path = config.get_feature_importance_path(position, 'rf_results')

    if not path.exists():
        raise FileNotFoundError(f"RF results not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Load importance rankings for feature_importance dict
    rankings_path = config.get_feature_importance_path(position, 'importance_rankings')
    feature_importance = {}
    if rankings_path.exists():
        df = pd.read_csv(rankings_path)
        if 'feature' in df.columns and 'rf_importance' in df.columns:
             feature_importance = dict(zip(df['feature'], df['rf_importance']))

    # Map nested structure to flat structure
    performance = data.get('performance', {})
    validation = data.get('validation', {})
    
    # Map status
    status_map = {
        'STRONG': 'VALIDATED',
        'MODERATE': 'WEAK',
        'WEAK': 'WEAK',
        'POOR': 'FAILED'
    }
    raw_status = validation.get('interpretation', 'UNKNOWN')
    status = status_map.get(raw_status, raw_status)

    return {
        'position': data.get('position', position),
        'cv_accuracy': performance.get('cv_mean_accuracy', 0),
        'cv_std': performance.get('cv_std_accuracy', 0),
        'spearman_rho': validation.get('spearman_rho', 0),
        'validation_status': status,
        'feature_importance': feature_importance
    }


@st.cache_data(ttl=config.CACHE_CONFIG['ram_ttl'])
def load_importance_rankings(position: str) -> pd.DataFrame:
    """
    Load feature importance rankings (RF vs F-statistics comparison).

    Args:
        position: Position key

    Returns:
        DataFrame with columns:
        - kpi, rf_importance, f_statistic, rf_rank, f_rank

    Performance:
        - First load: ~30ms
        - Cache hit: ~1ms
    """
    path = config.get_feature_importance_path(position, 'importance_rankings')

    if not path.exists():
        # Fallback: compute from cache_manager
        return compute_correlation_matrix(position)

    df = pd.read_csv(path)
    return df


# ============================================================================
# SUMMARY / AGGREGATED DATA
# ============================================================================

@st.cache_data(ttl=config.CACHE_CONFIG['ram_ttl'])
def load_all_positions_summary() -> Dict:
    """
    Load summary statistics for all 6 positions from JSON file.

    Returns:
        Dict mapping position to summary:
        {
            'midfielder': {
                'position_name': 'Midfielder',
                'n_players': 83,
                'n_clusters': 2,
                'mean_silhouette': 0.523,
                'rf_accuracy': 0.876,
                'spearman_rho': 0.750,
                'validation_status': 'VALIDATED'
            },
            ...
        }

    Performance:
        - First load: ~50ms
        - Cache hit: ~1ms
    """
    summary = {}
    
    # Try to load from all_positions_summary.json first
    summary_json_path = config.CLUSTERING_PATH / 'all_positions_summary.json'
    
    if summary_json_path.exists():
        with open(summary_json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Convert JSON structure to expected format
        for pos_data in json_data.get('positions', []):
            position_key = pos_data.get('position_key')
            if position_key:
                # Load RF results for validation status
                try:
                    rf_results = load_rf_results(position_key)
                except FileNotFoundError:
                    rf_results = None
                
                summary[position_key] = {
                    'position_name': pos_data.get('position', config.POSITIONS.get(position_key, {}).get('name', position_key)),
                    'n_players': pos_data.get('n_players', 0),
                    'n_clusters': pos_data.get('n_clusters', 2),
                    'mean_silhouette': pos_data.get('silhouette'),
                    'rf_accuracy': rf_results.get('cv_accuracy') if rf_results else None,
                    'spearman_rho': rf_results.get('spearman_rho') if rf_results else None,
                    'validation_status': rf_results.get('validation_status') if rf_results else 'N/A'
                }
        
        return summary
    
    # Fallback: Load from individual position data
    for position in config.POSITION_KEYS:
        try:
            clustered = load_clustered_data(position)
            
            try:
                rf_results = load_rf_results(position)
            except FileNotFoundError:
                rf_results = None

            summary[position] = {
                'position_name': config.POSITIONS[position]['name'],
                'n_players': len(clustered),
                'n_clusters': config.POSITIONS[position]['n_clusters'],
                'mean_silhouette': float(clustered['silhouette_score'].mean()) if 'silhouette_score' in clustered.columns else None,
                'rf_accuracy': rf_results.get('cv_accuracy') if rf_results else None,
                'spearman_rho': rf_results.get('spearman_rho') if rf_results else None,
                'validation_status': rf_results.get('validation_status') if rf_results else 'N/A'
            }

        except Exception as e:
            print(f"[!] Error loading summary for {position}: {e}")
            summary[position] = None

    return summary


@st.cache_data(ttl=config.CACHE_CONFIG['ram_ttl'])
def get_cluster_statistics(position: str) -> Dict:
    """
    Get aggregated cluster statistics via cache_manager.

    Uses disk cache.

    Args:
        position: Position key

    Returns:
        Dict with cluster stats per cluster_id

    Performance:
        - First load: ~100ms
        - Cache hit: ~1ms
        - After restart: ~30ms (disk cache hit)
    """
    return compute_cluster_statistics(position)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_position_player_names(position: str) -> List[str]:
    """
    Get sorted list of player names for a position.

    Args:
        position: Position key

    Returns:
        Sorted list of player names

    Performance:
        - Uses cached load_clustered_data
        - ~1ms (cache hit)
    """
    df = load_clustered_data(position)
    return sorted(df['player_name'].unique().tolist())


def get_player_data(position: str, player_name: str) -> pd.Series:
    """
    Get data for a specific player.

    Args:
        position: Position key
        player_name: Player name (exact match)

    Returns:
        Series with player data

    Raises:
        ValueError if player not found

    Performance:
        - ~1ms (uses cached data)
    """
    df = load_clustered_data(position)
    player_df = df[df['player_name'] == player_name]

    if player_df.empty:
        raise ValueError(f"Player '{player_name}' not found in {position}")

    return player_df.iloc[0]


def get_cluster_players(position: str, cluster_id: int) -> pd.DataFrame:
    """
    Get all players in a specific cluster.

    Args:
        position: Position key
        cluster_id: Cluster ID (0, 1, 2, ...)

    Returns:
        DataFrame with players in the cluster

    Performance:
        - ~1ms (uses cached data)
    """
    df = load_clustered_data(position)
    return df[df['cluster_label'] == cluster_id].copy()


def get_position_kpis(position: str, scaled: bool = False) -> List[str]:
    """
    Get list of KPI column names for a position.

    Args:
        position: Position key
        scaled: If True, return scaled KPI columns

    Returns:
        List of KPI column names

    Performance:
        - ~1ms (uses config)
    """
    if not scaled:
        return config.get_position_kpis(position)
    else:
        # Return scaled versions
        base_kpis = config.get_position_kpis(position)
        return [f"{kpi}_scaled" for kpi in base_kpis]


# ============================================================================
# VALIDATION / HEALTH CHECK
# ============================================================================

def check_data_availability(position: str) -> Dict[str, bool]:
    """
    Check which data files are available for a position.

    Args:
        position: Position key

    Returns:
        Dict mapping data type to availability:
        {
            'clustered': True,
            'profiles': True,
            'tactical_names': True,
            'pca_coords': True,
            'rf_results': False,
            ...
        }
    """
    checks = {}

    file_types = [
        'clustered', 'profiles', 'tactical_names', 'z_scores',
        'f_statistics', 'pca_coords'
    ]

    for file_type in file_types:
        try:
            path = config.get_clustering_path(position, file_type)
            checks[file_type] = path.exists()
        except:
            checks[file_type] = False

    # Check RF results
    try:
        path = config.get_feature_importance_path(position, 'rf_results')
        checks['rf_results'] = path.exists()
    except:
        checks['rf_results'] = False

    return checks


def validate_all_positions() -> Dict[str, Dict[str, bool]]:
    """
    Validate data availability for all positions.

    Returns:
        Dict mapping position to availability dict
    """
    results = {}

    for position in config.POSITION_KEYS:
        results[position] = check_data_availability(position)

    return results


def print_data_status():
    """
    Print data availability status for all positions.

    Useful for debugging.
    """
    print("\n" + "="*60)
    print("DATA AVAILABILITY STATUS")
    print("="*60)

    validation = validate_all_positions()

    for position in config.POSITION_KEYS:
        print(f"\n{config.POSITIONS[position]['name']} ({position}):")
        checks = validation[position]

        for data_type, available in checks.items():
            status = "[OK]" if available else "[X]"
            print(f"  {status} {data_type:20}")

    print("\n" + "="*60 + "\n")


# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

def clear_streamlit_cache():
    """
    Clear Streamlit RAM cache.

    Warning: Forces reload from disk on next access.
    """
    st.cache_data.clear()
    print("[CACHE] Streamlit RAM cache cleared")


def get_cache_statistics() -> Dict:
    """
    Get cache statistics.

    Returns:
        Dict with cache info
    """
    from cache_manager import get_cache_info

    return {
        'streamlit_cache': 'Active (ttl=1h)',
        'disk_cache': get_cache_info()
    }


# ============================================================================
# INITIALIZATION
# ============================================================================

if __name__ == '__main__':
    # Test data loader
    print("[TEST] Testing data_loader.py...")

    # Check data availability
    print_data_status()

    # Test loading for midfielder
    print("\n[TEST] Loading data for midfielder...")

    try:
        clustered = load_clustered_data('midfielder')
        print(f"[OK] Clustered data: {len(clustered)} players")

        pca_coords = load_pca_coordinates('midfielder')
        print(f"[OK] PCA coordinates: {len(pca_coords)} players")

        tactical_names = load_tactical_names('midfielder')
        print(f"[OK] Tactical names: {len(tactical_names)} clusters")

        rf_results = load_rf_results('midfielder')
        print(f"[OK] RF results: accuracy={rf_results['cv_accuracy']:.3f}")

        print("\n[OK] All data loaders working!")

    except Exception as e:
        print(f"[X] Error: {e}")

    # Print cache stats
    cache_stats = get_cache_statistics()
    print(f"\n[CACHE] Stats: {cache_stats}")
