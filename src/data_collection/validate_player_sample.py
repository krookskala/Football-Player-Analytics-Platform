"""
Player Sample Validation Script
================================

This script validates the collected player data and generates
a comprehensive report on the sample quality.

Validations:
1. Player count per position
2. Minutes distribution analysis
3. Team distribution
4. Top players by minutes
5. Data quality checks

Author: Football Analytics Thesis Project
Date: 2025
"""

import sys
import io
from pathlib import Path

import pandas as pd
import numpy as np

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.position_mapping import get_position_statistics

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

print("="*70)
print("PLAYER SAMPLE VALIDATION REPORT")
print("="*70)


def load_player_data():
    """Load player minutes data."""
    player_file = PROCESSED_DATA_DIR / "wc2022_player_minutes.csv"

    if not player_file.exists():
        print(f"\nError: Player minutes file not found at {player_file}")
        print("Please run collect_wc2022_events.py first.")
        sys.exit(1)

    df = pd.read_csv(player_file)
    print(f"\n[OK] Loaded data for {len(df)} players from {player_file}")

    return df


def validate_position_counts(df):
    """Validate player counts per position."""
    print("\n" + "="*70)
    print("1. POSITION DISTRIBUTION")
    print("="*70)

    position_counts = df['thesis_position'].value_counts().sort_index()
    expected_stats = get_position_statistics()

    print(f"\n{'Position':<15} {'Actual':<10} {'Expected':<12} {'Difference':<12} {'Status'}")
    print("-" * 70)

    for position in ['Center Back', 'Full Back', 'Winger', 'Forward']:
        actual = position_counts.get(position, 0)
        expected = expected_stats.get(position, 0)
        difference = actual - expected
        percentage = (actual / expected * 100) if expected > 0 else 0

        if percentage >= 70:
            status = "[OK] Good"
        elif percentage >= 50:
            status = "! Warning"
        else:
            status = "[X] Low"

        print(f"{position:<15} {actual:<10} {expected:<12} {difference:+11} {status}")

    total_actual = len(df)
    total_expected = expected_stats.get('Total', 460)
    print("-" * 70)
    print(f"{'TOTAL':<15} {total_actual:<10} {total_expected:<12} {total_actual - total_expected:+11}")
    print(f"\nSample coverage: {total_actual / total_expected * 100:.1f}% of expected")


def analyze_minutes_distribution(df):
    """Analyze minutes played distribution."""
    print("\n" + "="*70)
    print("2. MINUTES PLAYED DISTRIBUTION")
    print("="*70)

    print("\nOverall statistics:")
    print(f"  Mean:     {df['minutes_played'].mean():.1f} minutes")
    print(f"  Median:   {df['minutes_played'].median():.1f} minutes")
    print(f"  Std Dev:  {df['minutes_played'].std():.1f} minutes")
    print(f"  Min:      {df['minutes_played'].min():.1f} minutes")
    print(f"  Max:      {df['minutes_played'].max():.1f} minutes")

    print("\nPercentiles:")
    percentiles = [25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = df['minutes_played'].quantile(p / 100)
        print(f"  {p:2d}th: {value:.1f} minutes")

    print("\nMinutes ranges:")
    bins = [0, 90, 180, 270, 360, 450, 1000]
    labels = ['<90', '90-180', '180-270', '270-360', '360-450', '450+']
    df['minutes_range'] = pd.cut(df['minutes_played'], bins=bins, labels=labels)

    range_counts = df['minutes_range'].value_counts().sort_index()
    for range_label, count in range_counts.items():
        percentage = (count / len(df)) * 100
        bar = '#' * int(percentage / 2)
        print(f"  {range_label:10s}: {count:3d} players ({percentage:5.1f}%) {bar}")

    # Per position
    print("\nMean minutes by position:")
    for position in ['Center Back', 'Full Back', 'Winger', 'Forward']:
        pos_df = df[df['thesis_position'] == position]
        if len(pos_df) > 0:
            mean_min = pos_df['minutes_played'].mean()
            print(f"  {position:<10}: {mean_min:.1f} minutes")


def analyze_match_distribution(df):
    """Analyze matches played distribution."""
    print("\n" + "="*70)
    print("3. MATCHES PLAYED DISTRIBUTION")
    print("="*70)

    print("\nOverall statistics:")
    print(f"  Mean:   {df['matches_played'].mean():.1f} matches")
    print(f"  Median: {df['matches_played'].median():.1f} matches")
    print(f"  Min:    {df['matches_played'].min()} matches")
    print(f"  Max:    {df['matches_played'].max()} matches")

    print("\nMatches distribution:")
    matches_dist = df['matches_played'].value_counts().sort_index()
    for matches, count in matches_dist.items():
        percentage = (count / len(df)) * 100
        bar = '#' * int(percentage / 2)
        print(f"  {matches:2d} matches: {count:3d} players ({percentage:5.1f}%) {bar}")


def analyze_team_distribution(df):
    """Analyze team distribution."""
    print("\n" + "="*70)
    print("4. TEAM DISTRIBUTION")
    print("="*70)

    team_counts = df['team'].value_counts()

    print(f"\nTotal teams: {len(team_counts)}")
    print(f"\nTop 10 teams by player count:")

    for i, (team, count) in enumerate(team_counts.head(10).items(), 1):
        percentage = (count / len(df)) * 100
        print(f"  {i:2d}. {team:<25s}: {count:3d} players ({percentage:5.1f}%)")

    print("\nTeams with fewest players:")
    for team, count in team_counts.tail(5).items():
        print(f"  - {team:<25s}: {count} players")


def show_top_players(df):
    """Show top players by minutes played."""
    print("\n" + "="*70)
    print("5. TOP PLAYERS BY POSITION")
    print("="*70)

    for position in ['Center Back', 'Full Back', 'Winger', 'Forward']:
        pos_df = df[df['thesis_position'] == position].copy()

        if len(pos_df) == 0:
            continue

        pos_df = pos_df.sort_values('minutes_played', ascending=False)

        print(f"\n{position} - Top 5 by minutes:")
        print(f"  {'Rank':<6} {'Player':<25} {'Team':<20} {'Minutes':<10} {'Matches'}")
        print("  " + "-" * 70)

        for i, row in enumerate(pos_df.head(5).itertuples(), 1):
            print(f"  {i:<6} {row.player_name:<25.25} {row.team:<20.20} "
                  f"{row.minutes_played:>8.1f} {row.matches_played:>7}")


def data_quality_checks(df):
    """Perform data quality checks."""
    print("\n" + "="*70)
    print("6. DATA QUALITY CHECKS")
    print("="*70)

    # Check for missing values
    print("\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  [OK] No missing values found")
    else:
        for col, count in missing[missing > 0].items():
            percentage = (count / len(df)) * 100
            print(f"  - {col}: {count} ({percentage:.1f}%)")

    # Check for duplicates
    print("\nDuplicate players:")
    duplicates = df[df.duplicated(subset=['player_id'], keep=False)]
    if len(duplicates) == 0:
        print("  [OK] No duplicate player IDs found")
    else:
        print(f"  Warning: {len(duplicates)} duplicate entries found")
        print(duplicates[['player_id', 'player_name', 'team', 'minutes_played']])

    # Check for invalid minutes
    print("\nMinutes validation:")
    max_possible_minutes = 7 * 120  # 7 matches × 120 minutes (with extra time)
    invalid_minutes = df[df['minutes_played'] > max_possible_minutes]
    if len(invalid_minutes) == 0:
        print(f"  [OK] All minutes values are valid (<{max_possible_minutes})")
    else:
        print(f"  Warning: {len(invalid_minutes)} players with invalid minutes")

    # Check position mapping
    print("\nPosition mapping:")
    unmapped = df[df['thesis_position'].isnull()]
    if len(unmapped) == 0:
        print("  [OK] All positions successfully mapped")
    else:
        print(f"  Warning: {len(unmapped)} players with unmapped positions")


def save_validated_list(df):
    """Save validated player list."""
    output_file = PROCESSED_DATA_DIR / "wc2022_validated_players.csv"

    # Sort by position and minutes
    df_sorted = df.sort_values(['thesis_position', 'minutes_played'],
                                ascending=[True, False])

    df_sorted.to_csv(output_file, index=False, encoding='utf-8')

    file_size_kb = output_file.stat().st_size / 1024

    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    print(f"\n[OK] Validated player list saved to:")
    print(f"  {output_file}")
    print(f"  File size: {file_size_kb:.1f} KB")
    print(f"  Total players: {len(df_sorted)}")


def main():
    """Main execution function."""

    # Load data
    df = load_player_data()

    # Run validations
    validate_position_counts(df)
    analyze_minutes_distribution(df)
    analyze_match_distribution(df)
    analyze_team_distribution(df)
    show_top_players(df)
    data_quality_checks(df)

    # Save validated list
    save_validated_list(df)

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
