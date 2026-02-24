"""Tests for color harmony algorithms."""
from backend.app.ml.color_harmony import delta_e_cie76, find_harmonious_colors

def test_delta_e_identical_colors():
    c1 = (50.0, 10.0, 20.0)
    assert delta_e_cie76(c1, c1) == 0.0

def test_palette_selection_warm():
    colors = find_harmonious_colors("warm")
    assert len(colors) == 5
    assert "#8B4513" in colors
