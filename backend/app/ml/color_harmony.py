"""
CIELAB color harmony and seasonal undertone matcher.
"""

import numpy as np
from typing import Dict, List, Tuple

SEASONAL_PALETTES = {
    "warm_autumn": ["#8B4513", "#D2691E", "#CD853F", "#556B2F", "#DAA520"],
    "warm_spring": ["#FF7F50", "#FFA07A", "#FFD700", "#98FB98", "#40E0D0"],
    "cool_winter": ["#000080", "#4B0082", "#C0C0C0", "#E0FFFF", "#800020"],
    "cool_summer": ["#708090", "#B0C4DE", "#DDA0DD", "#BC8F8F", "#E6E6FA"],
}

def hex_to_rgb(hex_code: str) -> Tuple[int, int, int]:
    hex_code = hex_code.lstrip("#")
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def delta_e_cie76(lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]) -> float:
    """Compute CIE76 color difference."""
    return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2))))

def find_harmonious_colors(undertone: str) -> List[str]:
    """Return complementary palette based on undertone."""
    if "warm" in undertone.lower():
        return SEASONAL_PALETTES["warm_autumn"]
    return SEASONAL_PALETTES["cool_winter"]
