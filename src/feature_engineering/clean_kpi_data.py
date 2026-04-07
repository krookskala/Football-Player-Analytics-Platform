"""
Data Cleaning Script for KPI Data
==================================
Cleans and prepares KPI data for ML modeling.

Steps:
1. Filter by minimum minutes played (90 minutes)
2. Position-specific missing value imputation (median)
3. Feature scaling (StandardScaler)
4. Save cleaned data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Position-specific KPI columns (exclude accurate_crosses_per_90 - 100% missing)
POSITION_KPIS = {
    'Goalkeeper': [
        'save_percentage', 'xga_per_90', 'cross_claiming_rate',
        'sweeper_actions_per_90', 'gk_pass_completion', 'progressive_passes_per_90'
    ],
    'Center Back': [
        'interceptions_per_90', 'aerial_duels_win_pct', 'blocks_per_90',
        'clearances_per_90', 'pressures_per_90', 'pass_completion_pct',
        'progressive_passes_per_90'
    ],
    'Full Back': [
        'progressive_carries_per_90', 'xa_per_90', 'progressive_passes_per_90',
        'tackles_interceptions_per_90', 'touches_final_third_per_90',
        'defensive_duels_win_pct', 'possession_won_per_90'
    ],
    'Midfielder': [
        'pass_completion_pct', 'progressive_passes_per_90', 'ball_recoveries_per_90',
        'interceptions_per_90', 'tackles_won_per_90', 'pressures_per_90',
        'aerial_duels_win_pct', 'progressive_carries_per_90'
    ],
    'Winger': [
        'successful_dribbles_per_90', 'npxg_per_90', 'xa_per_90',
        'progressive_carries_per_90', 'key_passes_per_90',
        'touches_penalty_area_per_90', 'pressures_per_90',
        'shot_creating_actions_per_90', 'npxg_plus_xa_per_90'
    ],
    'Forward': [
        'npxg_per_90', 'non_penalty_goals_per_90', 'shots_on_target_pct',
        'conversion_rate', 'touches_penalty_area_per_90', 'xa_per_90',
        'successful_dribbles_per_90', 'aerial_duels_win_pct', 'npxg_plus_xa_per_90'
    ]
}


def load_data():
    """Load raw KPI data."""
    print("=" * 80)
    print("DATA CLEANING - FIFA WORLD CUP 2022 KPIs")
    print("=" * 80)

    print("\n[1/6] Loading data...")
    data_path = os.path.join('data', 'processed', 'player_kpis_all.csv')
    df = pd.read_csv(data_path)

    print(f"  [OK] Loaded {len(df)} players")
    print(f"  Positions: {df['thesis_position'].value_counts().to_dict()}")

    return df


def filter_by_minutes(df, min_minutes=90):
    """Filter players by minimum minutes played."""
    print(f"\n[2/6] Filtering by minimum minutes ({min_minutes} min)...")

    initial_count = len(df)
    df_filtered = df[df['minutes_played'] >= min_minutes].copy()
    final_count = len(df_filtered)
    removed = initial_count - final_count

    print(f"  [OK] Kept {final_count}/{initial_count} players ({final_count/initial_count*100:.1f}%)")
    print(f"  Removed {removed} players with <{min_minutes} minutes")

    print(f"\n  Players by position (after filter):")
    for pos in ['Goalkeeper', 'Center Back', 'Full Back', 'Midfielder', 'Winger', 'Forward']:
        count = len(df_filtered[df_filtered['thesis_position'] == pos])
        print(f"    {pos:15s}: {count:3d} players")

    return df_filtered


def impute_missing_values(df):
    """Impute missing values using position-specific median."""
    print("\n[3/6] Imputing missing values (position-specific median)...")

    df_imputed = df.copy()
    imputation_stats = {}

    for position, kpi_cols in POSITION_KPIS.items():
        pos_mask = df_imputed['thesis_position'] == position
        pos_df = df_imputed[pos_mask]

        if len(pos_df) == 0:
            continue

        missing_before = pos_df[kpi_cols].isna().sum().sum()

        # Impute with median for each KPI
        for col in kpi_cols:
            if col in df_imputed.columns:
                median_val = pos_df[col].median()
                df_imputed.loc[pos_mask, col] = pos_df[col].fillna(median_val)

        missing_after = df_imputed[pos_mask][kpi_cols].isna().sum().sum()

        imputation_stats[position] = {
            'before': missing_before,
            'after': missing_after,
            'filled': missing_before - missing_after
        }

    print(f"  [OK] Missing values imputed")
    print(f"\n  Imputation summary:")
    for pos, stats in imputation_stats.items():
        print(f"    {pos:15s}: {stats['filled']:4d} values filled ({stats['before']} -> {stats['after']})")

    return df_imputed


def scale_features(df):
    """Scale features using StandardScaler (position-specific)."""
    print("\n[4/6] Scaling features (StandardScaler, position-specific)...")

    df_scaled = df.copy()
    scalers = {}

    for position, kpi_cols in POSITION_KPIS.items():
        pos_mask = df_scaled['thesis_position'] == position
        pos_df = df_scaled[pos_mask]

        if len(pos_df) == 0:
            continue

        # Filter to only KPIs that exist
        existing_kpis = [col for col in kpi_cols if col in df_scaled.columns]

        if len(existing_kpis) == 0:
            continue

        # Scale
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(pos_df[existing_kpis])

        # Create new column names with _scaled suffix
        scaled_cols = [f"{col}_scaled" for col in existing_kpis]

        # Add scaled columns
        df_scaled.loc[pos_mask, scaled_cols] = scaled_values

        scalers[position] = {
            'scaler': scaler,
            'feature_names': existing_kpis
        }

        print(f"    {position:15s}: {len(existing_kpis)} features scaled")

    return df_scaled, scalers


def save_cleaned_data(df, scalers):
    """Save cleaned data and scalers."""
    print("\n[5/6] Saving cleaned data...")

    # Create output directory
    output_dir = os.path.join('data', 'processed', 'cleaned')
    os.makedirs(output_dir, exist_ok=True)

    # Save complete cleaned data
    all_data_path = os.path.join(output_dir, 'player_kpis_cleaned.csv')
    df.to_csv(all_data_path, index=False)
    print(f"  [OK] Saved complete cleaned data: {all_data_path}")
    print(f"       {len(df)} players × {len(df.columns)} columns")

    # Save position-specific cleaned data
    pos_dir = os.path.join(output_dir, 'by_position')
    os.makedirs(pos_dir, exist_ok=True)

    for position in POSITION_KPIS.keys():
        pos_df = df[df['thesis_position'] == position]
        if len(pos_df) > 0:
            pos_name = position.lower().replace(' ', '_')
            pos_path = os.path.join(pos_dir, f'{pos_name}_cleaned.csv')
            pos_df.to_csv(pos_path, index=False)
            print(f"  [OK] Saved {position:15s}: {pos_path} ({len(pos_df)} players)")

    # Save scaler info
    import pickle
    scaler_path = os.path.join(output_dir, 'scalers.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"  [OK] Saved scalers: {scaler_path}")


def generate_summary_stats(df_original, df_cleaned):
    """Generate summary statistics."""
    print("\n[6/6] Summary statistics...")

    print(f"\n  Data cleaning summary:")
    print(f"    Original players: {len(df_original)}")
    print(f"    Cleaned players:  {len(df_cleaned)}")
    print(f"    Retention rate:   {len(df_cleaned)/len(df_original)*100:.1f}%")

    print(f"\n  Players by position (cleaned):")
    for pos in ['Goalkeeper', 'Center Back', 'Full Back', 'Midfielder', 'Winger', 'Forward']:
        count_original = len(df_original[df_original['thesis_position'] == pos])
        count_cleaned = len(df_cleaned[df_cleaned['thesis_position'] == pos])
        retention = count_cleaned / count_original * 100 if count_original > 0 else 0
        print(f"    {pos:15s}: {count_cleaned:3d}/{count_original:3d} ({retention:5.1f}%)")

    # Count scaled features
    scaled_cols = [col for col in df_cleaned.columns if col.endswith('_scaled')]
    print(f"\n  Total scaled features: {len(scaled_cols)}")

    print("\n" + "=" * 80)
    print("DATA CLEANING COMPLETED!")
    print("=" * 80)


def main():
    """Main execution."""

    # Load data
    df = load_data()

    # Filter by minutes
    df_filtered = filter_by_minutes(df, min_minutes=90)

    # Impute missing values
    df_imputed = impute_missing_values(df_filtered)

    # Scale features
    df_scaled, scalers = scale_features(df_imputed)

    # Save cleaned data
    save_cleaned_data(df_scaled, scalers)

    # Generate summary
    generate_summary_stats(df, df_scaled)

    print(f"\nCleaned data ready for ML modeling!")
    print(f"Location: data/processed/cleaned/")


if __name__ == "__main__":
    main()
