"""
FIFA World Cup 2022 Event Data Collection Script
=================================================

This script collects all event data from FIFA World Cup 2022 matches
and prepares it for KPI calculation.

Tasks:
1. Collect match metadata for all 64 WC 2022 matches
2. Collect event data for each match
3. Filter events for selected positions (Goalkeeper, Center Back, Full Back, Midfielder, Winger, Forward)
4. Calculate minutes played per player
5. Filter players with minimum 45 minutes
6. Save data to data/raw/ and data/processed/ folders

Expected outputs:
- data/raw/wc2022_events.parquet (~40-60 MB)
- data/raw/wc2022_matches.csv (~5 KB)
- data/processed/wc2022_player_minutes.csv (~20 KB)
"""

import sys
import os
import time
from pathlib import Path

import pandas as pd
import numpy as np
from statsbombpy import sb
from tqdm import tqdm

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.position_mapping import map_position, is_selected_position

# Configuration
COMPETITION_ID = 43  # FIFA World Cup
SEASON_ID = 106  # 2022
MINIMUM_MINUTES = 45  # Minimum minutes played to include player
API_DELAY = 0.3  # Delay between API calls (seconds) to avoid rate limiting
MAX_RETRIES = 3  # Maximum retries for API calls

# Output paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("FIFA WORLD CUP 2022 - EVENT DATA COLLECTION")
print("="*70)
print(f"\nProject root: {PROJECT_ROOT}")
print(f"Raw data directory: {RAW_DATA_DIR}")
print(f"Processed data directory: {PROCESSED_DATA_DIR}")
print(f"\nConfiguration:")
print(f"  Competition ID: {COMPETITION_ID}")
print(f"  Season ID: {SEASON_ID}")
print(f"  Minimum minutes: {MINIMUM_MINUTES}")
print(f"  API delay: {API_DELAY}s")
print("="*70)


def collect_matches():
    """
    Collect all match metadata for FIFA World Cup 2022.

    Returns:
        DataFrame with match information
    """
    print("\n[1/5] Collecting match data...")

    try:
        matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
        print(f"  [OK] Found {len(matches)} matches")

        # Save match metadata
        matches_file = RAW_DATA_DIR / "wc2022_matches.csv"
        matches.to_csv(matches_file, index=False, encoding='utf-8')
        print(f"  [OK] Saved match metadata to {matches_file}")

        return matches

    except Exception as e:
        print(f"  [ERROR] Error collecting matches: {e}")
        raise


def collect_events(match_ids):
    """
    Collect event data for all matches with retry logic.

    Args:
        match_ids: List of match IDs to collect

    Returns:
        DataFrame with all events
    """
    print(f"\n[2/5] Collecting event data for {len(match_ids)} matches...")
    print("  (This will take approximately 5-10 minutes)")

    all_events = []
    failed_matches = []

    # Progress bar
    for match_id in tqdm(match_ids, desc="  Matches processed"):
        success = False

        # Retry logic
        for attempt in range(MAX_RETRIES):
            try:
                events = sb.events(match_id=match_id)

                if events is not None and len(events) > 0:
                    # Add match_id column
                    events['match_id'] = match_id
                    all_events.append(events)
                    success = True
                    break
                else:
                    print(f"\n  Warning: No events for match {match_id}")
                    break

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                else:
                    print(f"\n  [ERROR] Failed to collect match {match_id} after {MAX_RETRIES} attempts: {e}")
                    failed_matches.append(match_id)

        if success:
            # Delay to avoid rate limiting
            time.sleep(API_DELAY)

    if failed_matches:
        print(f"\n  Warning: {len(failed_matches)} matches failed: {failed_matches}")

    if not all_events:
        raise ValueError("No events collected!")

    # Combine all events
    print("\n  Combining all events...")
    combined_events = pd.concat(all_events, ignore_index=True)

    print(f"  [OK] Collected {len(combined_events):,} total events")
    print(f"  [OK] Unique event types: {combined_events['type'].nunique()}")

    return combined_events


def filter_selected_positions(events_df):
    """
    Filter events to include only selected positions.

    Args:
        events_df: DataFrame with all events

    Returns:
        Filtered DataFrame
    """
    print("\n[3/5] Filtering events for selected positions...")

    original_count = len(events_df)

    # Filter by position
    events_df = events_df[events_df['position'].apply(is_selected_position)].copy()

    filtered_count = len(events_df)
    percentage = (filtered_count / original_count) * 100

    print(f"  [OK] Kept {filtered_count:,} events ({percentage:.1f}%) for selected positions")
    print(f"  [OK] Removed {original_count - filtered_count:,} events from excluded positions")

    # Add mapped position column
    events_df['thesis_position'] = events_df['position'].apply(map_position)

    # Position distribution
    print("\n  Position distribution:")
    position_counts = events_df['thesis_position'].value_counts()
    for pos, count in position_counts.items():
        print(f"    {pos}: {count:,} events")

    return events_df


def calculate_player_minutes(events_df):
    """
    Calculate minutes played for each player.

    Args:
        events_df: DataFrame with filtered events

    Returns:
        DataFrame with player minutes
    """
    print("\n[4/5] Calculating minutes played per player...")

    player_minutes_list = []

    # Group by player and match
    grouped = events_df.groupby(['player_id', 'player', 'match_id'])

    print(f"  Processing {len(grouped)} unique player-match combinations...")

    for (player_id, player_name, match_id), group in tqdm(grouped, desc="  Calculating minutes"):
        # Get position (most frequent in this match)
        position = group['position'].mode()[0] if len(group['position'].mode()) > 0 else group['position'].iloc[0]
        thesis_position = map_position(position)

        # Get team
        team = group['team'].iloc[0] if 'team' in group.columns else None

        # Calculate minutes from timestamps
        # Convert timestamp to minutes
        if 'timestamp' in group.columns:
            # Parse timestamp (format: 00:00:00.000)
            group['timestamp_seconds'] = pd.to_timedelta(group['timestamp']).dt.total_seconds()

            min_time = group['timestamp_seconds'].min()
            max_time = group['timestamp_seconds'].max()

            minutes = (max_time - min_time) / 60.0
        else:
            # Fallback: use minute column
            min_minute = group['minute'].min()
            max_minute = group['minute'].max()
            minutes = max_minute - min_minute + 1

        player_minutes_list.append({
            'player_id': player_id,
            'player_name': player_name,
            'statsbomb_position': position,
            'thesis_position': thesis_position,
            'team': team,
            'match_id': match_id,
            'minutes_played': minutes
        })

    player_minutes_df = pd.DataFrame(player_minutes_list)

    # Aggregate across all matches
    player_total_minutes = player_minutes_df.groupby([
        'player_id', 'player_name', 'statsbomb_position', 'thesis_position', 'team'
    ]).agg({
        'minutes_played': 'sum',
        'match_id': 'count'  # Number of matches played
    }).reset_index()

    player_total_minutes.rename(columns={'match_id': 'matches_played'}, inplace=True)

    print(f"\n  [OK] Calculated minutes for {len(player_total_minutes)} unique players")

    # Statistics
    print(f"\n  Minutes played statistics:")
    print(f"    Mean: {player_total_minutes['minutes_played'].mean():.1f} minutes")
    print(f"    Median: {player_total_minutes['minutes_played'].median():.1f} minutes")
    print(f"    Min: {player_total_minutes['minutes_played'].min():.1f} minutes")
    print(f"    Max: {player_total_minutes['minutes_played'].max():.1f} minutes")

    # Filter by minimum minutes
    print(f"\n  Filtering players with at least {MINIMUM_MINUTES} minutes...")
    original_player_count = len(player_total_minutes)
    player_total_minutes = player_total_minutes[
        player_total_minutes['minutes_played'] >= MINIMUM_MINUTES
    ].copy()

    filtered_player_count = len(player_total_minutes)
    removed = original_player_count - filtered_player_count

    print(f"  [OK] Kept {filtered_player_count} players")
    print(f"  [OK] Removed {removed} players with <{MINIMUM_MINUTES} minutes")

    # Position distribution after filtering
    print(f"\n  Final position distribution:")
    position_counts = player_total_minutes['thesis_position'].value_counts()
    for pos, count in position_counts.items():
        print(f"    {pos}: {count} players")

    # Save player minutes
    player_minutes_file = PROCESSED_DATA_DIR / "wc2022_player_minutes.csv"
    player_total_minutes.to_csv(player_minutes_file, index=False, encoding='utf-8')
    print(f"\n  [OK] Saved player minutes to {player_minutes_file}")

    return player_total_minutes


def save_events(events_df):
    """
    Save events data in Parquet format.

    Args:
        events_df: DataFrame with filtered events
    """
    print("\n[5/5] Saving event data...")

    # Save as Parquet (compressed, efficient)
    events_file = RAW_DATA_DIR / "wc2022_events.parquet"

    try:
        events_df.to_parquet(events_file, index=False, compression='snappy')

        # Get file size
        file_size_mb = events_file.stat().st_size / (1024 * 1024)

        print(f"  [OK] Saved {len(events_df):,} events to {events_file}")
        print(f"  [OK] File size: {file_size_mb:.1f} MB")

    except Exception as e:
        print(f"  [ERROR] Error saving Parquet file: {e}")
        print(f"  Falling back to CSV...")

        # Fallback to CSV
        events_file_csv = RAW_DATA_DIR / "wc2022_events.csv"
        events_df.to_csv(events_file_csv, index=False, encoding='utf-8')

        file_size_mb = events_file_csv.stat().st_size / (1024 * 1024)
        print(f"  [OK] Saved to CSV: {events_file_csv}")
        print(f"  [OK] File size: {file_size_mb:.1f} MB")


def main():
    """
    Main execution function.
    """
    start_time = time.time()

    try:
        # Step 1: Collect matches
        matches = collect_matches()
        match_ids = matches['match_id'].tolist()

        # Step 2: Collect events
        events = collect_events(match_ids)

        # Step 3: Filter positions
        events = filter_selected_positions(events)

        # Step 4: Calculate player minutes
        player_minutes = calculate_player_minutes(events)

        # Step 5: Save events
        save_events(events)

        # Summary
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)

        print("\n" + "="*70)
        print("DATA COLLECTION COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nSummary:")
        print(f"  • Matches collected: {len(matches)}")
        print(f"  • Total events: {len(events):,}")
        print(f"  • Players (45+ min): {len(player_minutes)}")
        print(f"  • Execution time: {minutes}m {seconds}s")
        print(f"\nOutput files:")
        print(f"  • {RAW_DATA_DIR / 'wc2022_matches.csv'}")
        print(f"  • {RAW_DATA_DIR / 'wc2022_events.parquet'}")
        print(f"  • {PROCESSED_DATA_DIR / 'wc2022_player_minutes.csv'}")
        print("\n" + "="*70)

        return True

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR: Data collection failed!")
        print(f"{'='*70}")
        print(f"\nError details: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
