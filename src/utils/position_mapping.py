"""
Position Mapping Utility for Football Analytics Project
========================================================

This module provides utilities to map StatsBomb's 24 detailed positions
to the 6 strategic position groups used in this thesis:
- Goalkeeper
- Center Back
- Full Back
- Midfielder
- Winger
- Forward
"""

from typing import Optional, List, Dict

# Position mapping from StatsBomb to thesis categories
POSITION_MAPPING = {
    # CENTER BACK - 108 players
    "Right Center Back": "Center Back",
    "Left Center Back": "Center Back",
    "Center Back": "Center Back",

    # FULL BACK - 120 players
    "Right Back": "Full Back",
    "Left Back": "Full Back",
    "Right Wing Back": "Full Back",
    "Left Wing Back": "Full Back",

    # WINGER - 138 players
    "Right Wing": "Winger",
    "Left Wing": "Winger",
    "Left Midfield": "Winger",
    "Right Midfield": "Winger",
    "Left Attacking Midfield": "Winger",
    "Right Attacking Midfield": "Winger",

    # FORWARD - 94 players
    "Center Forward": "Forward",
    "Left Center Forward": "Forward",
    "Right Center Forward": "Forward",

    # GOALKEEPER - 41 players
    "Goalkeeper": "Goalkeeper",

    # MIDFIELDER - 177 players
    "Defensive Midfield": "Midfielder",
    "Center Defensive Midfield": "Midfielder",
    "Right Defensive Midfield": "Midfielder",
    "Left Defensive Midfield": "Midfielder",
    "Central Midfield": "Midfielder",
    "Center Midfield": "Midfielder",
    "Right Center Midfield": "Midfielder",
    "Left Center Midfield": "Midfielder",
    "Center Attacking Midfield": "Midfielder",

    # EXCLUDED POSITIONS
    "Secondary Striker": None,
}

# Reverse mapping: thesis position -> list of StatsBomb positions
REVERSE_MAPPING = {
    "Goalkeeper": ["Goalkeeper"],
    "Center Back": ["Right Center Back", "Left Center Back", "Center Back"],
    "Full Back": ["Right Back", "Left Back", "Right Wing Back", "Left Wing Back"],
    "Midfielder": ["Defensive Midfield", "Center Defensive Midfield", "Right Defensive Midfield",
                  "Left Defensive Midfield", "Central Midfield", "Center Midfield",
                  "Right Center Midfield", "Left Center Midfield", "Center Attacking Midfield"],
    "Winger": ["Right Wing", "Left Wing", "Left Midfield", "Right Midfield",
              "Left Attacking Midfield", "Right Attacking Midfield"],
    "Forward": ["Center Forward", "Left Center Forward", "Right Center Forward"]
}

# Selected positions for this thesis
SELECTED_POSITIONS = ["Goalkeeper", "Center Back", "Full Back", "Midfielder", "Winger", "Forward"]


def map_position(statsbomb_position: str) -> Optional[str]:
    """
    Map a StatsBomb position to thesis position category.

    Args:
        statsbomb_position: Position name from StatsBomb data

    Returns:
        Thesis position category (Center Back, Full Back, Winger, Forward) or None if not selected

    Examples:
        >>> map_position("Right Center Back")
        'Center Back'
        >>> map_position("Left Wing")
        'Winger'
        >>> map_position("Goalkeeper")
        'Goalkeeper'
    """
    return POSITION_MAPPING.get(statsbomb_position)


def is_selected_position(statsbomb_position: str) -> bool:
    """
    Check if a StatsBomb position is in the selected thesis positions.

    Args:
        statsbomb_position: Position name from StatsBomb data

    Returns:
        True if position is selected for thesis, False otherwise

    Examples:
        >>> is_selected_position("Right Center Back")
        True
        >>> is_selected_position("Goalkeeper")
        True
    """
    mapped = map_position(statsbomb_position)
    return mapped is not None


def get_selected_positions() -> List[str]:
    """
    Get list of selected thesis position categories.

    Returns:
        List of position categories: ['Goalkeeper', 'Center Back', 'Full Back', 'Midfielder', 'Winger', 'Forward']
    """
    return SELECTED_POSITIONS.copy()


def get_statsbomb_positions_for_category(category: str) -> List[str]:
    """
    Get list of StatsBomb positions that map to a thesis category.

    Args:
        category: Thesis position category (Center Back, Full Back, Winger, or Forward)

    Returns:
        List of StatsBomb position names

    Examples:
        >>> get_statsbomb_positions_for_category("Center Back")
        ['Right Center Back', 'Left Center Back', 'Center Back']
    """
    return REVERSE_MAPPING.get(category, [])


def get_position_statistics() -> Dict[str, int]:
    """
    Get expected player counts per position category (from WC 2022 analysis).

    Returns:
        Dictionary mapping position category to expected player count
    """
    return {
        "Goalkeeper": 41,
        "Center Back": 108,
        "Full Back": 120,
        "Midfielder": 177,
        "Winger": 138,
        "Forward": 94,
        "Total": 678
    }


def validate_position_mapping(statsbomb_position: str) -> tuple[bool, str]:
    """
    Validate a StatsBomb position and provide detailed feedback.

    Args:
        statsbomb_position: Position name from StatsBomb data

    Returns:
        Tuple of (is_valid, message)

    Examples:
        >>> validate_position_mapping("Right Center Back")
        (True, 'Valid position: maps to Center Back')
        >>> validate_position_mapping("Unknown Position")
        (False, 'Unknown position: not in StatsBomb position list')
    """
    if statsbomb_position not in POSITION_MAPPING:
        return False, f"Unknown position: '{statsbomb_position}' not in StatsBomb position list"

    mapped = map_position(statsbomb_position)
    if mapped is None:
        return False, f"Excluded position: '{statsbomb_position}' is not selected for this thesis"

    return True, f"Valid position: '{statsbomb_position}' maps to {mapped}"


def get_all_statsbomb_positions() -> List[str]:
    """
    Get complete list of all StatsBomb positions (selected and excluded).

    Returns:
        List of all StatsBomb position names
    """
    return list(POSITION_MAPPING.keys())


if __name__ == "__main__":
    # Test the mapping functions
    print("=" * 60)
    print("POSITION MAPPING UTILITY - TEST")
    print("=" * 60)

    print("\n1. Selected Thesis Positions:")
    print(f"   {get_selected_positions()}")

    print("\n2. Expected Player Counts (WC 2022):")
    for pos, count in get_position_statistics().items():
        print(f"   {pos}: {count}")

    print("\n3. Sample Mappings:")
    test_positions = [
        "Right Center Back",
        "Left Back",
        "Right Wing",
        "Center Forward",
        "Goalkeeper",
        "Defensive Midfield"
    ]

    for pos in test_positions:
        mapped = map_position(pos)
        is_valid, message = validate_position_mapping(pos)
        status = "[OK]" if is_valid else "[X]"
        arrow = "->"
        print(f"   {status} {pos:25s} {arrow} {mapped or 'EXCLUDED'}")

    print("\n4. Reverse Mapping (Center Back):")
    stoper_positions = get_statsbomb_positions_for_category("Center Back")
    print(f"   {stoper_positions}")

    print("\n5. Total Mapped Positions:")
    selected_count = sum(1 for v in POSITION_MAPPING.values() if v is not None)
    excluded_count = sum(1 for v in POSITION_MAPPING.values() if v is None)
    print(f"   Selected: {selected_count}")
    print(f"   Excluded: {excluded_count}")
    print(f"   Total: {len(POSITION_MAPPING)}")

    print("\n" + "=" * 60)
    print("Position mapping utility is ready!")
    print("=" * 60)
