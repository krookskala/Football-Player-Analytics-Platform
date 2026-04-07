"""
KPI Helper Functions for Football Analytics Project
====================================================

Shared utility functions for KPI calculations.

"""

import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Optional


def per_90_normalization(count: Union[int, float], minutes_played: float) -> float:
    """
    Normalize a count to per 90 minutes.

    Args:
        count: Number of events/actions
        minutes_played: Total minutes played

    Returns:
        Normalized value per 90 minutes

    Examples:
        >>> per_90_normalization(10, 180)
        5.0
        >>> per_90_normalization(3, 45)
        6.0
    """
    if minutes_played == 0 or pd.isna(minutes_played):
        return 0.0

    return (count / minutes_played) * 90


def is_progressive_pass(start_loc: Union[List, np.ndarray, None],
                       end_loc: Union[List, np.ndarray, None],
                       min_distance: float = 10.0) -> bool:
    """
    Check if a pass is progressive (moves ball 10m+ toward opponent goal).

    Args:
        start_loc: Starting location [x, y]
        end_loc: Ending location [x, y]
        min_distance: Minimum forward distance in meters (default: 10)

    Returns:
        True if pass is progressive, False otherwise

    Examples:
        >>> is_progressive_pass([30, 40], [45, 42])
        True
        >>> is_progressive_pass([30, 40], [35, 38])
        False
    """
    if start_loc is None or end_loc is None:
        return False

    if not isinstance(start_loc, (list, np.ndarray)) or not isinstance(end_loc, (list, np.ndarray)):
        return False

    if len(start_loc) < 2 or len(end_loc) < 2:
        return False

    # Calculate forward progress (x-axis)
    forward_progress = end_loc[0] - start_loc[0]

    return forward_progress >= min_distance


def is_progressive_carry(start_loc: Union[List, np.ndarray, None],
                        end_loc: Union[List, np.ndarray, None],
                        min_distance: float = 10.0) -> bool:
    """
    Check if a carry is progressive (moves ball 10m+ toward opponent goal).

    Args:
        start_loc: Starting location [x, y]
        end_loc: Ending location [x, y]
        min_distance: Minimum forward distance in meters (default: 10)

    Returns:
        True if carry is progressive, False otherwise
    """
    # Same logic as progressive pass
    return is_progressive_pass(start_loc, end_loc, min_distance)


def is_in_final_third(location: Union[List, np.ndarray, None]) -> bool:
    """
    Check if location is in the final third of the pitch.

    Args:
        location: Location [x, y] on 120m × 80m pitch

    Returns:
        True if in final third (x > 80), False otherwise

    Examples:
        >>> is_in_final_third([90, 40])
        True
        >>> is_in_final_third([50, 40])
        False
    """
    if location is None:
        return False

    if not isinstance(location, (list, np.ndarray)):
        return False

    if len(location) < 1:
        return False

    return location[0] > 80


def is_in_penalty_area(location: Union[List, np.ndarray, None]) -> bool:
    """
    Check if location is in the penalty area.

    Args:
        location: Location [x, y] on 120m × 80m pitch

    Returns:
        True if in penalty area, False otherwise

    Examples:
        >>> is_in_penalty_area([110, 40])
        True
        >>> is_in_penalty_area([90, 40])
        False

    Notes:
        Penalty area dimensions:
        - X: 102-120m (18m from goal line)
        - Y: 18-62m (44m width, centered at 40m)
    """
    if location is None:
        return False

    if not isinstance(location, (list, np.ndarray)):
        return False

    if len(location) < 2:
        return False

    x, y = location[0], location[1]

    # Check if in penalty area
    return (102 <= x <= 120) and (18 <= y <= 62)


def calculate_win_percentage(won_count: int, total_count: int) -> Optional[float]:
    """
    Calculate win percentage with zero handling.

    Args:
        won_count: Number of won events
        total_count: Total number of events

    Returns:
        Win percentage (0-100) or None if no events

    Examples:
        >>> calculate_win_percentage(7, 10)
        70.0
        >>> calculate_win_percentage(0, 0)
        None
    """
    if total_count == 0:
        return None

    return (won_count / total_count) * 100


def parse_location(location_value: Union[str, List, np.ndarray, None]) -> Optional[List]:
    """
    Parse location field which may be string, list, or array.

    Args:
        location_value: Location value in various formats

    Returns:
        List [x, y] or None

    Examples:
        >>> parse_location([30, 40])
        [30, 40]
        >>> parse_location("[30, 40]")
        [30, 40]
        >>> parse_location(None)
        None
    """
    if pd.isna(location_value):
        return None

    if isinstance(location_value, (list, np.ndarray)):
        return list(location_value)

    if isinstance(location_value, str):
        try:
            return ast.literal_eval(location_value)
        except:
            return None

    return None


def parse_related_events(related_value: Union[str, List, None]) -> List:
    """
    Parse related_events field which may be string or list.

    Args:
        related_value: Related events in various formats

    Returns:
        List of event IDs

    Examples:
        >>> parse_related_events([123, 456])
        [123, 456]
        >>> parse_related_events("[123, 456]")
        [123, 456]
        >>> parse_related_events(None)
        []
    """
    if pd.isna(related_value):
        return []

    if isinstance(related_value, list):
        return related_value

    if isinstance(related_value, str):
        try:
            import ast
            parsed = ast.literal_eval(related_value)
            return parsed if isinstance(parsed, list) else []
        except:
            return []

    return []


def safe_percentage(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely calculate percentage with zero handling.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if denominator is zero (default: 0.0)

    Returns:
        Percentage value or default

    Examples:
        >>> safe_percentage(3, 10)
        30.0
        >>> safe_percentage(5, 0)
        0.0
        >>> safe_percentage(5, 0, default=None)
        None
    """
    if denominator == 0:
        return default

    return (numerator / denominator) * 100


# Field dimension constants
PITCH_LENGTH = 120  # meters
PITCH_WIDTH = 80    # meters
FINAL_THIRD_START = 80  # x-coordinate
PENALTY_AREA_START_X = 102  # x-coordinate
PENALTY_AREA_MIN_Y = 18  # y-coordinate
PENALTY_AREA_MAX_Y = 62  # y-coordinate


if __name__ == "__main__":
    # Test functions
    print("="*70)
    print("KPI HELPER FUNCTIONS - TEST")
    print("="*70)

    # Test per 90 normalization
    print("\n1. Per 90 Normalization:")
    print(f"   10 events in 180 min = {per_90_normalization(10, 180):.2f} per 90")
    print(f"   3 events in 45 min = {per_90_normalization(3, 45):.2f} per 90")

    # Test progressive pass
    print("\n2. Progressive Pass Detection:")
    print(f"   [30,40] -> [45,42] = {is_progressive_pass([30, 40], [45, 42])}")
    print(f"   [30,40] -> [35,38] = {is_progressive_pass([30, 40], [35, 38])}")

    # Test location checks
    print("\n3. Location Checks:")
    print(f"   [90,40] in final third = {is_in_final_third([90, 40])}")
    print(f"   [50,40] in final third = {is_in_final_third([50, 40])}")
    print(f"   [110,40] in penalty area = {is_in_penalty_area([110, 40])}")
    print(f"   [90,40] in penalty area = {is_in_penalty_area([90, 40])}")

    # Test win percentage
    print("\n4. Win Percentage:")
    print(f"   7/10 wins = {calculate_win_percentage(7, 10):.1f}%")
    print(f"   0/0 wins = {calculate_win_percentage(0, 0)}")

    print("\n" + "="*70)
    print("All helper functions ready!")
    print("="*70)
