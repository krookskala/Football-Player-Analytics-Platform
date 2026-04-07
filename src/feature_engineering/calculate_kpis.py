"""
KPI Calculation Script for Football Analytics Project
======================================================

Calculates key performance indicators for players across 6 positions.

Input:
- data/raw/wc2022_events.csv
- data/processed/wc2022_player_minutes.csv

Output:
- data/processed/player_kpis_all.csv
- data/processed/player_kpis_by_position/*.csv
"""

import sys
import io
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.kpi_helpers import (
    per_90_normalization,
    is_progressive_pass,
    is_progressive_carry,
    is_in_final_third,
    is_in_penalty_area,
    calculate_win_percentage,
    parse_location
)

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROCESSED_DATA / "player_kpis_by_position"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("="*70)
print("KPI CALCULATION - FIFA WORLD CUP 2022")
print("="*70)


def load_data():
    """Load events and player minutes data."""
    print("\n[1/4] Loading data...")

    # Load events (CSV format)
    events_csv = RAW_DATA / "wc2022_events.csv"

    if not events_csv.exists():
        raise FileNotFoundError(f"Events file not found: {events_csv}")

    print(f"  Loading events from CSV: {events_csv}")
    events = pd.read_csv(events_csv)

    print(f"  [OK] Loaded {len(events):,} events")

    # Load player minutes
    player_minutes_file = PROCESSED_DATA / "wc2022_player_minutes.csv"
    player_minutes = pd.read_csv(player_minutes_file)

    print(f"  [OK] Loaded {len(player_minutes)} player records")

    return events, player_minutes


def calculate_pass_completion(player_events):
    """Calculate pass completion percentage."""
    passes = player_events[player_events['type'] == 'Pass']

    if len(passes) == 0:
        return 0.0

    # Successful passes have NaN in pass_outcome
    successful = passes[passes['pass_outcome'].isna()]

    return (len(successful) / len(passes)) * 100


def calculate_progressive_passes_per_90(player_events, minutes_played):
    """Calculate progressive passes per 90 minutes."""
    passes = player_events[player_events['type'] == 'Pass'].copy()

    if len(passes) == 0:
        return 0.0

    # Check for progressive passes
    progressive_count = 0
    for _, row in passes.iterrows():
        start_loc = parse_location(row.get('location'))
        end_loc = parse_location(row.get('pass_end_location'))

        if is_progressive_pass(start_loc, end_loc) and pd.isna(row.get('pass_outcome')):
            progressive_count += 1

    return per_90_normalization(progressive_count, minutes_played)


def calculate_interceptions_per_90(player_events, minutes_played):
    """Calculate interceptions per 90 minutes."""
    interceptions = player_events[player_events['type'] == 'Interception']
    return per_90_normalization(len(interceptions), minutes_played)


def calculate_pressures_per_90(player_events, minutes_played):
    """Calculate pressures per 90 minutes."""
    pressures = player_events[player_events['type'] == 'Pressure']
    return per_90_normalization(len(pressures), minutes_played)


def calculate_blocks_per_90(player_events, minutes_played):
    """Calculate blocks per 90 minutes."""
    blocks = player_events[player_events['type'] == 'Block']
    return per_90_normalization(len(blocks), minutes_played)


def calculate_clearances_per_90(player_events, minutes_played):
    """Calculate clearances per 90 minutes."""
    clearances = player_events[player_events['type'] == 'Clearance']
    return per_90_normalization(len(clearances), minutes_played)


def calculate_aerial_duels_win_pct(player_events):
    """
    Calculate aerial duels win percentage.
    
    FIXED: StatsBomb stores aerial duels differently:
    - Lost aerial duels: duel_type = 'Aerial Lost'
    - Won aerial duels: pass_aerial_won, clearance_aerial_won, shot_aerial_won fields
    
    Reference: StatsBomb Event Data Specification
    """
    # Count aerial duels lost (stored in duel_type)
    aerial_lost = player_events[
        player_events['duel_type'] == 'Aerial Lost'
    ]
    lost_count = len(aerial_lost)
    
    # Count aerial duels won (stored in separate boolean fields)
    won_pass = player_events['pass_aerial_won'].notna().sum()
    won_clearance = player_events['clearance_aerial_won'].notna().sum()
    won_shot = player_events['shot_aerial_won'].notna().sum()
    
    won_count = won_pass + won_clearance + won_shot
    total = won_count + lost_count
    
    if total == 0:
        return None

    return calculate_win_percentage(won_count, total)


def calculate_accurate_crosses_per_90(player_events, minutes_played):
    """Calculate accurate crosses per 90 minutes."""
    crosses = player_events[
        (player_events['type'] == 'Pass') &
        (player_events['pass_cross'] == True)
    ]

    accurate_crosses = crosses[crosses['pass_outcome'].isna()]

    return per_90_normalization(len(accurate_crosses), minutes_played)


def calculate_progressive_carries_per_90(player_events, minutes_played):
    """Calculate progressive carries per 90 minutes."""
    carries = player_events[player_events['type'] == 'Carry'].copy()

    if len(carries) == 0:
        return 0.0

    progressive_count = 0
    for _, row in carries.iterrows():
        start_loc = parse_location(row.get('location'))
        end_loc = parse_location(row.get('carry_end_location'))

        if is_progressive_carry(start_loc, end_loc):
            progressive_count += 1

    return per_90_normalization(progressive_count, minutes_played)


def calculate_tackles_interceptions_per_90(player_events, minutes_played):
    """Calculate tackles + interceptions per 90 minutes."""
    tackles = player_events[
        (player_events['type'] == 'Duel') &
        (player_events['duel_type'].str.contains('Tackle', case=False, na=False)) &
        (player_events['duel_outcome'].isin(['Won', 'Success']))
    ]

    interceptions = player_events[player_events['type'] == 'Interception']

    total = len(tackles) + len(interceptions)

    return per_90_normalization(total, minutes_played)


def calculate_touches_final_third_per_90(player_events, minutes_played):
    """Calculate touches in final third per 90 minutes."""
    count = 0

    for _, row in player_events.iterrows():
        loc = parse_location(row.get('location'))
        if is_in_final_third(loc):
            count += 1

    return per_90_normalization(count, minutes_played)


def calculate_successful_dribbles_per_90(player_events, minutes_played):
    """Calculate successful dribbles per 90 minutes."""
    dribbles = player_events[
        (player_events['type'] == 'Dribble') &
        (player_events['dribble_outcome'].isin(['Complete', 'Success']))
    ]

    return per_90_normalization(len(dribbles), minutes_played)


def calculate_key_passes_per_90(player_events, minutes_played):
    """Calculate key passes per 90 minutes."""
    key_passes = player_events[
        (player_events['type'] == 'Pass') &
        (player_events['pass_shot_assist'].notna())
    ]

    return per_90_normalization(len(key_passes), minutes_played)


def calculate_touches_penalty_area_per_90(player_events, minutes_played):
    """Calculate touches in penalty area per 90 minutes."""
    count = 0

    for _, row in player_events.iterrows():
        loc = parse_location(row.get('location'))
        if is_in_penalty_area(loc):
            count += 1

    return per_90_normalization(count, minutes_played)


def calculate_npxg_per_90(player_events, minutes_played):
    """Calculate non-penalty expected goals per 90 minutes."""
    shots = player_events[
        (player_events['type'] == 'Shot') &
        (player_events['shot_type'] != 'Penalty')
    ]

    npxg = shots['shot_statsbomb_xg'].sum()

    return per_90_normalization(npxg, minutes_played)


def calculate_non_penalty_goals_per_90(player_events, minutes_played):
    """Calculate non-penalty goals per 90 minutes."""
    goals = player_events[
        (player_events['type'] == 'Shot') &
        (player_events['shot_outcome'] == 'Goal') &
        (player_events['shot_type'] != 'Penalty')
    ]

    return per_90_normalization(len(goals), minutes_played)


def calculate_shots_on_target_pct(player_events):
    """Calculate shots on target percentage."""
    shots = player_events[player_events['type'] == 'Shot']

    if len(shots) == 0:
        return None

    on_target = shots[shots['shot_outcome'].isin(['Saved', 'Goal'])]

    return (len(on_target) / len(shots)) * 100


def calculate_conversion_rate(player_events):
    """Calculate conversion rate (goals / shots)."""
    shots = player_events[player_events['type'] == 'Shot']

    if len(shots) == 0:
        return None

    goals = shots[shots['shot_outcome'] == 'Goal']

    return (len(goals) / len(shots)) * 100


def calculate_xa_per_90(player_events, all_events, minutes_played):
    """Calculate expected assists per 90 minutes."""
    # Get player's passes
    player_passes = player_events[player_events['type'] == 'Pass']

    # Get all shots
    all_shots = all_events[all_events['type'] == 'Shot']

    # Match passes to shots via shot_key_pass_id
    xa = 0.0

    for _, shot in all_shots.iterrows():
        key_pass_id = shot.get('shot_key_pass_id')

        if pd.notna(key_pass_id):
            # Find the pass
            pass_event = player_passes[player_passes['id'] == key_pass_id]
            if not pass_event.empty:
                xa += shot.get('shot_statsbomb_xg', 0)

    return per_90_normalization(xa, minutes_played)






def calculate_defensive_duels_win_pct(player_events):
    """Calculate defensive duels win percentage."""
    duels = player_events[
        (player_events['type'] == 'Duel') &
        (player_events['duel_type'].str.contains('Tackle', case=False, na=False))
    ]

    if len(duels) == 0:
        return None

    won = duels[duels['duel_outcome'].isin(['Won', 'Success'])]

    return calculate_win_percentage(len(won), len(duels))


def calculate_possession_won_per_90(player_events, minutes_played):
    """Calculate possession won per 90 minutes (same as tackles + interceptions)."""
    # Same calculation as tackles + interceptions
    return calculate_tackles_interceptions_per_90(player_events, minutes_played)


def calculate_shot_creating_actions_per_90(player_events, all_events, minutes_played):
    """Calculate shot-creating actions per 90 minutes."""
    sca_count = 0

    # Part 1: Key passes (passes leading to shots)
    key_passes = player_events[
        (player_events['type'] == 'Pass') &
        (player_events['pass_shot_assist'].notna())
    ]
    sca_count += len(key_passes)

    # Part 2: Dribbles leading to shots
    player_dribbles = player_events[player_events['type'] == 'Dribble']

    for _, dribble in player_dribbles.iterrows():
        related = dribble.get('related_events')

        if pd.notna(related) and related:
            # Parse related events (might be string or list)
            if isinstance(related, str):
                try:
                    import ast
                    related = ast.literal_eval(related)
                except:
                    related = []

            if isinstance(related, list) and len(related) > 0:
                # Check if any related event is a shot
                related_shots = all_events[
                    (all_events['id'].isin(related)) &
                    (all_events['type'] == 'Shot')
                ]
                if len(related_shots) > 0:
                    sca_count += 1

    # Part 3: Fouls won in final third leading to shots
    fouls_won = player_events[player_events['type'] == 'Foul Won']

    for _, foul in fouls_won.iterrows():
        loc = parse_location(foul.get('location'))

        # Only count fouls in final third
        if is_in_final_third(loc):
            related = foul.get('related_events')

            if pd.notna(related) and related:
                if isinstance(related, str):
                    try:
                        import ast
                        related = ast.literal_eval(related)
                    except:
                        related = []

                if isinstance(related, list) and len(related) > 0:
                    related_shots = all_events[
                        (all_events['id'].isin(related)) &
                        (all_events['type'] == 'Shot')
                    ]
                    if len(related_shots) > 0:
                        sca_count += 1

    return per_90_normalization(sca_count, minutes_played)


def calculate_npxg_plus_xa_per_90(player_events, all_events, minutes_played):
    """Calculate npxG + xA per 90 minutes."""
    npxg = calculate_npxg_per_90(player_events, minutes_played)
    xa = calculate_xa_per_90(player_events, all_events, minutes_played)

    return npxg + xa


# ============================
# GOALKEEPER KPIs
# ============================

def calculate_save_percentage(player_events):
    """
    Calculate save percentage for goalkeepers.
    
    FIXED: Uses goalkeeper_type field instead of shot events.
    Shot events belong to the shooting player, not the goalkeeper.
    
    StatsBomb goalkeeper_type values:
    - 'Shot Saved': Goalkeeper saved the shot
    - 'Goal Conceded': Shot resulted in a goal
    - 'Shot Saved to Post': Saved but hit the post
    - 'Shot Saved Off Target': Saved and went off target
    
    Formula: Saves / (Saves + Goals Conceded) * 100
    
    Returns:
        Save percentage (0-100) or None if no shots on target faced
    """
    # Get goalkeeper events
    gk_events = player_events[player_events['type'] == 'Goal Keeper']
    
    if len(gk_events) == 0:
        return None
    
    # Count saves (all types)
    saves = gk_events[
        gk_events['goalkeeper_type'].str.contains('Saved', case=False, na=False)
    ]
    
    # Count goals conceded
    goals_conceded = gk_events[
        gk_events['goalkeeper_type'] == 'Goal Conceded'
    ]
    
    total_on_target = len(saves) + len(goals_conceded)
    
    if total_on_target == 0:
        return None

    return calculate_win_percentage(len(saves), total_on_target)


def calculate_xga_per_90(all_events, player_id, minutes_played):
    """Calculate xG Against (xGA) per 90 for goalkeepers."""
    # Find all shots against this goalkeeper's team
    # We need to get shots where the opponent shot
    player_team = all_events[all_events['player_id'] == player_id]['team'].iloc[0] if len(all_events[all_events['player_id'] == player_id]) > 0 else None

    if player_team is None:
        return None

    # Get shots by opposing team
    shots_against = all_events[
        (all_events['type'] == 'Shot') &
        (all_events['team'] != player_team) &
        (all_events['shot_statsbomb_xg'].notna())
    ]

    if len(shots_against) == 0:
        return 0.0

    xga = shots_against['shot_statsbomb_xg'].sum()

    return per_90_normalization(xga, minutes_played)


def calculate_cross_claiming_rate(player_events, all_events):
    """Calculate cross claiming rate for goalkeepers."""
    # Get goalkeeper collections
    collections = player_events[
        (player_events['type'] == 'Goal Keeper') &
        (player_events['goalkeeper_type'].str.contains('Collected', case=False, na=False))
    ]

    # Get all crosses into penalty area (would need opponent crosses)
    # This is complex - simplified version using GK events
    gk_actions_on_crosses = player_events[
        (player_events['type'] == 'Goal Keeper') &
        (player_events['goalkeeper_type'].notna())
    ]

    if len(gk_actions_on_crosses) == 0:
        return None

    return calculate_win_percentage(len(collections), len(gk_actions_on_crosses))


def calculate_sweeper_actions_per_90(player_events, minutes_played):
    """Calculate sweeper actions per 90 for goalkeepers (actions outside penalty area)."""
    sweeper_actions = player_events[
        ((player_events['type'] == 'Interception') |
         (player_events['type'] == 'Clearance') |
         (player_events['type'] == 'Duel'))
    ]

    # Filter for actions outside penalty area (x < 102)
    outside_box_actions = []
    for _, action in sweeper_actions.iterrows():
        loc = parse_location(action.get('location'))
        if loc and loc[0] < 102:  # Outside penalty area
            outside_box_actions.append(action)

    return per_90_normalization(len(outside_box_actions), minutes_played)


def calculate_gk_pass_completion(player_events):
    """Calculate pass completion for goalkeepers (overall)."""
    passes = player_events[player_events['type'] == 'Pass']

    if len(passes) == 0:
        return None

    # Successful passes have NaN in pass_outcome
    successful = passes[passes['pass_outcome'].isna()]

    return calculate_win_percentage(len(successful), len(passes))


# ============================
# MIDFIELDER KPIs
# ============================

def calculate_ball_recoveries_per_90(player_events, minutes_played):
    """Calculate ball recoveries per 90 for midfielders."""
    recoveries = player_events[player_events['type'] == 'Ball Recovery']

    return per_90_normalization(len(recoveries), minutes_played)


def calculate_tackles_won_per_90(player_events, minutes_played):
    """Calculate tackles won per 90 for midfielders."""
    tackles = player_events[
        (player_events['type'] == 'Duel') &
        (player_events['duel_type'].str.contains('Tackle', case=False, na=False)) &
        (player_events['duel_outcome'].isin(['Won', 'Success In Play', 'Success']))
    ]

    return per_90_normalization(len(tackles), minutes_played)


# ============================
# NEW ACADEMIC KPIs (Added 2025-12-02)
# ============================

def calculate_counterpress_per_90(player_events, minutes_played):
    """
    Calculate counterpressing actions per 90 minutes.
    
    Counterpressing (Gegenpressing): Immediate pressure after losing possession
    to win the ball back in dangerous areas.
    
    Args:
        player_events: Player's event data
        minutes_played: Total minutes played
        
    Returns:
        Counterpress actions per 90 minutes
    """
    counterpresses = player_events[player_events['counterpress'] == True]
    return per_90_normalization(len(counterpresses), minutes_played)


def calculate_under_pressure_pass_completion(player_events):
    """
    Calculate pass completion percentage when under pressure.
    
    Measures decision-making quality under defensive pressure.
    Higher values indicate better composure and technical ability.
    
    Args:
        player_events: Player's event data
        
    Returns:
        Pass completion percentage when under pressure (0-100)
    """
    # Get all passes under pressure
    passes_under_pressure = player_events[
        (player_events['type'] == 'Pass') &
        (player_events['under_pressure'] == True)
    ]
    
    if len(passes_under_pressure) == 0:
        return None
    
    # Successful passes have NaN in pass_outcome
    successful = passes_under_pressure[passes_under_pressure['pass_outcome'].isna()]
    
    return (len(successful) / len(passes_under_pressure)) * 100


def calculate_through_balls_per_90(player_events, minutes_played):
    """
    Calculate through balls per 90 minutes.
    
    Through balls: Passes that split the defensive line and find a 
    teammate in behind. Key creativity metric for playmakers.
    
    Args:
        player_events: Player's event data
        minutes_played: Total minutes played
        
    Returns:
        Through balls per 90 minutes
    """
    through_balls = player_events[
        (player_events['type'] == 'Pass') &
        (player_events['pass_through_ball'] == True) &
        (player_events['pass_outcome'].isna())  # Successful only
    ]
    
    return per_90_normalization(len(through_balls), minutes_played)


def calculate_long_pass_completion_pct(player_events):
    """
    Calculate long pass (>25m) completion percentage.
    
    Important for ball-playing defenders and deep-lying playmakers.
    
    Args:
        player_events: Player's event data
        
    Returns:
        Long pass completion percentage (0-100)
    """
    long_passes = player_events[
        (player_events['type'] == 'Pass') &
        (player_events['pass_length'] > 25)
    ]
    
    if len(long_passes) == 0:
        return None
    
    successful = long_passes[long_passes['pass_outcome'].isna()]
    
    return (len(successful) / len(long_passes)) * 100


# Position name mapping: raw StatsBomb labels → (English name, file_safe_name)
POSITION_NAME_MAP = {
    'Stoper': ('Center Back', 'center_back'),
    'Bek': ('Full Back', 'full_back'),
    'Kanat': ('Winger', 'winger'),
    'Forvet': ('Forward', 'forward'),
    'Orta Saha': ('Midfielder', 'midfielder'),
    'Kaleci': ('Goalkeeper', 'goalkeeper')
}

# KPI mapping by position (using English names for internal logic)
POSITION_KPIS = {
    'Center Back': {
        # Defensive-focused KPIs only (pass metrics removed for clustering stability)
        # Academic justification: CB primary role is defensive, ball-playing is secondary
        # Removing pass metrics improved ARI from 0.439 to 1.000
        'interceptions_per_90': calculate_interceptions_per_90,
        'aerial_duels_win_pct': calculate_aerial_duels_win_pct,
        'blocks_per_90': calculate_blocks_per_90,
        'clearances_per_90': calculate_clearances_per_90,
        'pressures_per_90': calculate_pressures_per_90
    },
    'Full Back': {
        'accurate_crosses_per_90': calculate_accurate_crosses_per_90,
        'progressive_carries_per_90': calculate_progressive_carries_per_90,
        'xa_per_90': calculate_xa_per_90,
        'progressive_passes_per_90': calculate_progressive_passes_per_90,
        'tackles_interceptions_per_90': calculate_tackles_interceptions_per_90,
        'touches_final_third_per_90': calculate_touches_final_third_per_90,
        'defensive_duels_win_pct': calculate_defensive_duels_win_pct,
        'possession_won_per_90': calculate_possession_won_per_90
    },
    'Winger': {
        'successful_dribbles_per_90': calculate_successful_dribbles_per_90,
        'npxg_per_90': calculate_npxg_per_90,
        'xa_per_90': calculate_xa_per_90,
        'accurate_crosses_per_90': calculate_accurate_crosses_per_90,
        'progressive_carries_per_90': calculate_progressive_carries_per_90,
        'key_passes_per_90': calculate_key_passes_per_90,
        'touches_penalty_area_per_90': calculate_touches_penalty_area_per_90,
        'pressures_per_90': calculate_pressures_per_90,
        'shot_creating_actions_per_90': calculate_shot_creating_actions_per_90,
        'npxg_plus_xa_per_90': calculate_npxg_plus_xa_per_90
    },
    'Forward': {
        'npxg_per_90': calculate_npxg_per_90,
        'non_penalty_goals_per_90': calculate_non_penalty_goals_per_90,
        'shots_on_target_pct': calculate_shots_on_target_pct,
        'conversion_rate': calculate_conversion_rate,
        'touches_penalty_area_per_90': calculate_touches_penalty_area_per_90,
        'xa_per_90': calculate_xa_per_90,
        'successful_dribbles_per_90': calculate_successful_dribbles_per_90,
        'aerial_duels_win_pct': calculate_aerial_duels_win_pct,
        'npxg_plus_xa_per_90': calculate_npxg_plus_xa_per_90
    },
    'Goalkeeper': {
        'save_percentage': calculate_save_percentage,
        'xga_per_90': calculate_xga_per_90,
        'cross_claiming_rate': calculate_cross_claiming_rate,
        'sweeper_actions_per_90': calculate_sweeper_actions_per_90,
        'gk_pass_completion': calculate_gk_pass_completion,
        'progressive_passes_per_90': calculate_progressive_passes_per_90
    },
    'Midfielder': {
        'pass_completion_pct': calculate_pass_completion,
        'progressive_passes_per_90': calculate_progressive_passes_per_90,
        'ball_recoveries_per_90': calculate_ball_recoveries_per_90,
        'interceptions_per_90': calculate_interceptions_per_90,
        'tackles_won_per_90': calculate_tackles_won_per_90,
        'pressures_per_90': calculate_pressures_per_90,
        'aerial_duels_win_pct': calculate_aerial_duels_win_pct,
        'progressive_carries_per_90': calculate_progressive_carries_per_90
    }
}


def calculate_player_kpis(player_id, thesis_position, player_events, all_events, minutes_played):
    """Calculate all KPIs for a single player."""
    kpis = {
        'player_id': player_id,
        'thesis_position': thesis_position,
        'minutes_played': minutes_played
    }

    # Map Turkish position name to English for KPI lookup
    position_en = POSITION_NAME_MAP.get(thesis_position, (thesis_position, ''))[0]
    
    # Get KPI functions for this position
    position_kpi_funcs = POSITION_KPIS.get(position_en, {})

    # Calculate each KPI
    for kpi_name, kpi_func in position_kpi_funcs.items():
        try:
            # Special handling for KPIs with unique parameter signatures
            if kpi_name == 'xga_per_90':
                # xGA needs all_events, player_id, minutes_played
                value = kpi_func(all_events, player_id, minutes_played)
            elif kpi_name in ['xa_per_90', 'shot_creating_actions_per_90', 'npxg_plus_xa_per_90']:
                # These need player_events, all_events, minutes_played
                value = kpi_func(player_events, all_events, minutes_played)
            elif kpi_name == 'cross_claiming_rate':
                # This needs player_events and all_events
                value = kpi_func(player_events, all_events)
            elif 'per_90' in kpi_name or kpi_name.endswith('_per_90'):
                # Per 90 KPIs need minutes_played (check this BEFORE 'rate' check)
                value = kpi_func(player_events, minutes_played)
            elif 'pct' in kpi_name or kpi_name.endswith('_rate') or kpi_name.endswith('_win_pct') or 'completion' in kpi_name or 'percentage' in kpi_name:
                # Percentage KPIs don't need minutes_played
                value = kpi_func(player_events)
            else:
                value = kpi_func(player_events)

            kpis[kpi_name] = value

        except Exception as e:
            print(f"    Warning: Error calculating {kpi_name} for player {player_id}: {e}")
            kpis[kpi_name] = None

    return kpis


def main():
    """Main execution."""

    # Load data
    events, player_minutes = load_data()

    # Calculate KPIs for each player
    print(f"\n[2/4] Calculating KPIs for {len(player_minutes)} players...")
    print("  (This may take 5-10 minutes)")

    all_player_kpis = []

    for idx, row in tqdm(player_minutes.iterrows(), total=len(player_minutes), desc="  Players processed"):
        player_id = row['player_id']
        thesis_position = row['thesis_position']
        minutes_played = row['minutes_played']

        # Get player events
        player_events = events[events['player_id'] == player_id]

        # Calculate KPIs
        player_kpis = calculate_player_kpis(
            player_id=player_id,
            thesis_position=thesis_position,
            player_events=player_events,
            all_events=events,
            minutes_played=minutes_played
        )

        # Add player info
        player_kpis['player_name'] = row['player_name']
        player_kpis['team'] = row['team']
        player_kpis['matches_played'] = row['matches_played']

        all_player_kpis.append(player_kpis)

    # Convert to DataFrame
    kpis_df = pd.DataFrame(all_player_kpis)

    # Reorder columns
    meta_cols = ['player_id', 'player_name', 'team', 'thesis_position', 'minutes_played', 'matches_played']
    kpi_cols = [col for col in kpis_df.columns if col not in meta_cols]
    kpis_df = kpis_df[meta_cols + kpi_cols]

    print(f"\n[3/4] Saving results...")

    # Save combined file
    output_file = PROCESSED_DATA / "player_kpis_all.csv"
    kpis_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"  [OK] Saved combined KPIs: {output_file}")

    # Save position-specific files
    print(f"\n[4/4] Creating position-specific KPI files...")

    # Use Turkish position names from data, map to English file names
    for thesis_pos_tr, (pos_en, file_name) in POSITION_NAME_MAP.items():
        position_df = kpis_df[kpis_df['thesis_position'] == thesis_pos_tr].copy()

        if len(position_df) == 0:
            continue

        # Sort by minutes played
        position_df = position_df.sort_values('minutes_played', ascending=False)

        # Save with English file name
        output_file = OUTPUT_DIR / f"{file_name}_kpis.csv"
        position_df.to_csv(output_file, index=False, encoding='utf-8')

        file_size_kb = output_file.stat().st_size / 1024
        kpi_count = len([col for col in position_df.columns if 'per_90' in col or 'pct' in col or 'rate' in col])

        print(f"  [OK] {thesis_pos_tr} ({pos_en}): {len(position_df)} players × {kpi_count} KPIs -> {output_file.name} ({file_size_kb:.1f} KB)")

    # Summary statistics
    print("\n" + "="*70)
    print("KPI CALCULATION COMPLETED!")
    print("="*70)

    print(f"\nSummary:")
    print(f"  Total players: {len(kpis_df)}")
    print(f"  Positions:")
    for pos, count in kpis_df['thesis_position'].value_counts().sort_index().items():
        print(f"    - {pos}: {count} players")

    print(f"\n  KPIs calculated per position:")
    for thesis_pos_tr, (pos_en, _) in POSITION_NAME_MAP.items():
        kpi_count = len(POSITION_KPIS.get(pos_en, {}))
        print(f"    - {thesis_pos_tr} ({pos_en}): {kpi_count} KPIs")

    print(f"\nOutput files:")
    print(f"  - {PROCESSED_DATA / 'player_kpis_all.csv'}")
    for _, (_, file_name) in POSITION_NAME_MAP.items():
        print(f"  - {OUTPUT_DIR / f'{file_name}_kpis.csv'}")

    print("\n" + "="*70)

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] KPI calculation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
