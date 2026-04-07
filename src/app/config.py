"""
Dashboard Configuration
-----------------------
Centralized configuration for Streamlit dashboard.

Contains:
- Position metadata
- File paths
- Theme configurations
- Demo scenarios
- Animation settings
- Cache settings
"""

from pathlib import Path
import plotly.express as px

# ============================================================================
# PATHS
# ============================================================================

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'processed'
CLUSTERING_PATH = DATA_PATH / 'clustering'
FEATURE_IMPORTANCE_PATH = DATA_PATH / 'feature_importance'
KPI_PATH = DATA_PATH / 'player_kpis_by_position'

# Cache path
CACHE_PATH = Path(__file__).parent / 'cache'

# ============================================================================
# POSITIONS METADATA
# ============================================================================

POSITIONS = {
    'midfielder': {
        'name': 'Midfielder',
        'icon': '⚙️',
        'n_clusters': 2,
        'n_players': 83,
        'kpis': [
            'pass_completion_pct',
            'progressive_passes_per_90',
            'ball_recoveries_per_90',
            'interceptions_per_90',
            'tackles_won_per_90',
            'pressures_per_90',
            'progressive_carries_per_90'
        ]
    },
    'center_back': {
        'name': 'Center Back',
        'icon': '🛡️',
        'n_clusters': 2,
        'n_players': 74,
        'kpis': [
            'pressures_per_90',
            'interceptions_per_90',
            'blocks_per_90',
            'clearances_per_90',
            'pass_completion_pct',
            'progressive_passes_per_90'
        ]
    },
    'full_back': {
        'name': 'Full Back',
        'icon': '🏃',
        'n_clusters': 4,
        'n_players': 69,
        'kpis': [
            'progressive_passes_per_90',
            'progressive_carries_per_90',
            'xa_per_90',
            'tackles_interceptions_per_90',
            'defensive_duels_win_pct',
            'touches_final_third_per_90',
            'possession_won_per_90'
        ]
    },
    'winger': {
        'name': 'Winger',
        'icon': '⚡',
        'n_clusters': 2,
        'n_players': 55,
        'kpis': [
            'successful_dribbles_per_90',
            'npxg_plus_xa_per_90',
            'shot_creating_actions_per_90',
            'progressive_carries_per_90',
            'key_passes_per_90',
            'touches_penalty_area_per_90',
            'pressures_per_90'
        ]
    },
    'forward': {
        'name': 'Forward',
        'icon': '⚽',
        'n_clusters': 2,
        'n_players': 45,
        'kpis': [
            'non_penalty_goals_per_90',
            'npxg_per_90',
            'npxg_plus_xa_per_90',
            'shots_on_target_pct',
            'conversion_rate',
            'touches_penalty_area_per_90',
            'successful_dribbles_per_90'
        ]
    },
    'goalkeeper': {
        'name': 'Goalkeeper',
        'icon': '🧤',
        'n_clusters': 4,
        'n_players': 32,
        'kpis': [
            'xga_per_90',
            'gk_pass_completion',
            'cross_claiming_rate',
            'sweeper_actions_per_90',
            'progressive_passes_per_90'
        ]
    }
}

# Position keys for easy iteration
POSITION_KEYS = list(POSITIONS.keys())

# ============================================================================
# KPI CATEGORIES & COLORS
# ============================================================================

CATEGORY_COLORS = {
    'Attacking': '#FF5252',   # Red
    'Defending': '#448AFF',   # Blue
    'Possession': '#69F0AE',  # Green
    'Physical': '#FFD740',    # Amber
    'Goalkeeping': '#E040FB'  # Purple
}

# Map KPIs to categories for Pizza Chart
KPI_CATEGORIES = {
    # Midfielder
    'ball_recoveries_per_90': 'Defending',
    'progressive_passes_per_90': 'Possession',
    'pressures_per_90': 'Defending',
    'pass_completion_pct': 'Possession',
    'progressive_carries_per_90': 'Possession',
    'tackles_won_per_90': 'Defending',
    'aerial_duels_win_pct': 'Physical',
    'interceptions_per_90': 'Defending',
    'key_passes_per_90': 'Attacking',

    # Center Back
    'interceptions_per_90': 'Defending',
    'aerial_duels_win_pct': 'Physical',
    'blocks_per_90': 'Defending',
    'clearances_per_90': 'Defending',

    # Full Back
    'xa_per_90': 'Attacking',
    'defensive_duels_win_pct': 'Defending',
    'touches_final_third_per_90': 'Attacking',
    'possession_won_per_90': 'Defending',
    'tackles_interceptions_per_90': 'Defending',
    'accurate_crosses_per_90': 'Attacking',

    # Winger
    'shot_creating_actions_per_90': 'Attacking',
    'successful_dribbles_per_90': 'Attacking',
    'touches_penalty_area_per_90': 'Attacking',
    'npxg_per_90': 'Attacking',

    # Forward
    'non_penalty_goals_per_90': 'Attacking',
    'npxg_per_90': 'Attacking',
    'shots_on_target_pct': 'Attacking',
    'conversion_rate': 'Attacking',
    'npxg_plus_xa_per_90': 'Attacking',

    # Goalkeeper
    'xga_per_90': 'Goalkeeping',
    'gk_pass_completion': 'Possession',
    'cross_claiming_rate': 'Goalkeeping',
    'sweeper_actions_per_90': 'Goalkeeping',
}

# Human-readable KPI names
KPI_READABLE_NAMES = {
    # Attacking
    'non_penalty_goals_per_90': 'Non-Penalty Goals / 90',
    'npxg_per_90': 'npxG / 90',
    'npxg_plus_xa_per_90': 'npxG + xA / 90',
    'shots_on_target_pct': 'Shots on Target %',
    'conversion_rate': 'Conversion Rate %',
    'shot_creating_actions_per_90': 'Shot-Creating Actions / 90',
    'touches_penalty_area_per_90': 'Touches in Penalty Area / 90',
    'successful_dribbles_per_90': 'Successful Dribbles / 90',
    'xa_per_90': 'xA (Expected Assists) / 90',
    'key_passes_per_90': 'Key Passes / 90',
    'touches_final_third_per_90': 'Touches in Final Third / 90',
    'accurate_crosses_per_90': 'Accurate Crosses / 90',

    # Possession
    'pass_completion_pct': 'Pass Completion %',
    'progressive_passes_per_90': 'Progressive Passes / 90',
    'progressive_carries_per_90': 'Progressive Carries / 90',
    'gk_pass_completion': 'GK Pass Completion %',

    # Defending
    'tackles_interceptions_per_90': 'Tackles + Interceptions / 90',
    'tackles_won_per_90': 'Tackles Won / 90',
    'ball_recoveries_per_90': 'Ball Recoveries / 90',
    'pressures_per_90': 'Pressures / 90',
    'interceptions_per_90': 'Interceptions / 90',
    'blocks_per_90': 'Blocks / 90',
    'clearances_per_90': 'Clearances / 90',
    'defensive_duels_win_pct': 'Defensive Duels Won %',
    'possession_won_per_90': 'Possessions Won / 90',

    # Physical
    'aerial_duels_win_pct': 'Aerial Duels Won %',

    # Goalkeeping
    'xga_per_90': 'xGA (Expected Goals Against) / 90',
    'cross_claiming_rate': 'Cross Claiming Rate %',
    'sweeper_actions_per_90': 'Sweeper Actions / 90',
}

# ============================================================================
# THEMES
# ============================================================================

THEMES = {
    'light': {
        'name': 'Light Mode',
        'background': '#FFFFFF',
        'text': '#000000',
        'accent': '#1f77b4',
        'success': '#28A745',
        'warning': '#FFC107',
        'error': '#DC3545',
        'grid_color': '#E0E0E0',
        'cluster_colors': px.colors.qualitative.Set2
    },
    'dark': {
        'name': 'Dark Mode',
        'background': '#0E1117',
        'text': '#FAFAFA',
        'accent': '#00D4FF',
        'success': '#4CAF50',
        'warning': '#FF9800',
        'error': '#F44336',
        'grid_color': '#2E2E2E',
        'cluster_colors': px.colors.qualitative.Dark2
    }
}

# ============================================================================
# CACHE CONFIGURATION
# ============================================================================

CACHE_CONFIG = {
    # Streamlit cache TTL (seconds)
    'ram_ttl': 3600,  # 1 hour

    # joblib Memory settings
    'disk_location': str(CACHE_PATH),
    'disk_verbose': 0,

    # Precompute settings
    'precompute_pca': True,  # Use precomputed PCA coordinates

    # Cache subdirectories
    'pca_cache': CACHE_PATH / 'pca_cache',
    'radar_norm_cache': CACHE_PATH / 'radar_norm_cache',
    'correlation_cache': CACHE_PATH / 'correlation_cache'
}

# ============================================================================
# ANIMATION CONFIGURATION
# ============================================================================

ANIMATION_CONFIG = {
    'transition_duration': 300,  # milliseconds
    'ease': 'cubic-in-out',
    'frame_duration': 50,  # milliseconds per frame
    'enable_animations': True  # Global toggle
}

# ============================================================================
# DEMO SCENARIOS (Thesis Defense)
# ============================================================================

DEMO_SCENARIOS = {
    'forward_comparison_case_study': {
        'name': 'Case Study 1: Forward Comparison',
        'description': 'Elite forward comparison (Messi vs Giroud)',
        'position': 'forward',
        'players': ['Lionel Andrés Messi Cuccittini', 'Olivier Giroud'],
        'kpis': [
            'non_penalty_goals_per_90',
            'npxg_per_90',
            'successful_dribbles_per_90',
            'shots_on_target_pct',
            'npxg_plus_xa_per_90',
            'touches_penalty_area_per_90'
        ],
        'page': 'Scouting_Tool'
    },
    'clustering_visualization_case_study': {
        'name': 'Case Study 2: Clustering Visualization',
        'description': 'Midfielder cluster separation analysis',
        'position': 'midfielder',
        'view': 'pca_scatter',
        'highlight_cluster': 2,  # Offensive Midfielder
        'page': 'K_Means_Clustering_Analysis'
    },
    'model_validation_case_study': {
        'name': 'Case Study 3: Model Validation',
        'description': 'Random Forest performance metrics',
        'position': 'midfielder',
        'view': 'correlation_scatter',
        'page': 'Random_Forest_Validation'
    }
}

# ============================================================================
# UI LABELS
# ============================================================================

UI_LABELS = {
    # Navigation
    'home': 'Home',
    'player_comparison': 'Player Comparison',
    'cluster_profiles': 'Cluster Profiles',
    'feature_importance': 'Feature Importance',

    # Common
    'position_selector': 'Select Position',
    'player_selector': 'Select Player',
    'select_players': 'Select at least 2 players',
    'loading': 'Loading data...',
    'generating_chart': 'Generating chart...',

    # Cluster
    'cluster_info': 'Cluster Information',
    'cluster_name': 'Cluster Name',
    'player_count': 'Player Count',
    'top_kpis': 'Top KPIs',

    # Export
    'export_png': 'Download PNG',
    'export_csv': 'Download CSV',

    # Demo
    'demo_mode': 'Demo Mode',
    'thesis_defense': 'Thesis Defense',

    # Messages
    'no_data': 'No data found',
    'select_min_players': 'Select at least {min} players',
    'file_not_found': 'Data file missing',
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_clustering_path(position: str, file_type: str) -> Path:
    """
    Get path to clustering output file.

    Args:
        position: Position key (e.g., 'midfielder')
        file_type: File type (e.g., 'clustered', 'profiles', 'tactical_names')

    Returns:
        Path object

    Examples:
        get_clustering_path('midfielder', 'clustered')
        -> .../clustering/midfielder/midfielder_clustered.csv
    """
    extensions = {
        'clustered': '.csv',
        'profiles': '_cluster_profiles.csv',
        'tactical_names': '_tactical_names.json',
        'z_scores': '_z_scores.csv',
        'f_statistics': '_f_statistics.json',
        'pca_coords': '_pca_coords.csv',  # NEW
        'robustness': '_robustness.json',
        'optimal_k': '_optimal_k_results.json',
        'model': '_kmeans_model.pkl'
    }

    if file_type == 'clustered':
        filename = f"{position}_clustered.csv"
    else:
        filename = f"{position}{extensions[file_type]}"

    return CLUSTERING_PATH / position / filename


def get_feature_importance_path(position: str, file_type: str) -> Path:
    """
    Get path to feature importance output file.

    Args:
        position: Position key
        file_type: 'rf_results', 'importance_rankings', 'barplot', 'scatter'

    Returns:
        Path object
    """
    extensions = {
        'rf_results': '_rf_results.json',
        'importance_rankings': '_importance_rankings.csv',
        'barplot': '_importance_barplot.json',
        'scatter': '_correlation_scatter.json'
    }

    filename = f"{position}{extensions[file_type]}"
    return FEATURE_IMPORTANCE_PATH / position / filename


def get_position_kpis(position: str) -> list:
    """
    Get list of KPIs for a position.

    Args:
        position: Position key

    Returns:
        List of KPI keys
    """
    return POSITIONS[position]['kpis']


def get_kpi_path(position: str) -> Path:
    """
    Get path to KPI file for a position.

    Args:
        position: Position key

    Returns:
        Path object
    """
    return KPI_PATH / f"{position}_kpis.csv"


# ============================================================================
# UI SETTINGS
# ============================================================================

UI_SETTINGS = {
    'page_title': 'Football Player Analytics Platform',
    'page_icon': '⚽',
    'page_layout': 'wide',
    'sidebar_state': 'expanded',
    'default_position': 'midfielder'
}
