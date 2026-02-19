from matplotlib import font_manager

from zensvi.visualization.font_property import _get_font_properties


def test_returns_three_font_properties():
    prop_title, prop, prop_legend = _get_font_properties(20)
    assert isinstance(prop_title, font_manager.FontProperties)
    assert isinstance(prop, font_manager.FontProperties)
    assert isinstance(prop_legend, font_manager.FontProperties)


def test_title_font_is_bold():
    prop_title, _, _ = _get_font_properties(20)
    assert prop_title.get_weight() == "bold"


def test_font_sizes_decrease():
    prop_title, prop, prop_legend = _get_font_properties(30)
    assert prop_title.get_size() == 30
    assert prop.get_size() == 20
    assert prop_legend.get_size() == 15


def test_small_font_size_clamps_to_one():
    _, prop, prop_legend = _get_font_properties(5)
    assert prop.get_size() >= 1
    assert prop_legend.get_size() >= 1
