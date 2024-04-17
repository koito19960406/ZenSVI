from matplotlib import font_manager


def _get_font_properties(font_size):
    prop_title = font_manager.FontProperties(
        family="Arial Narrow", weight="bold", size=font_size
    )  # Specify your font properties
    prop = font_manager.FontProperties(
        family="Arial Narrow", size=max(1, font_size - 10)
    )  # Specify your font properties
    prop_legend = font_manager.FontProperties(
        family="Arial Narrow", size=max(1, font_size - 15)
    )  # Specify your font properties
    return prop_title, prop, prop_legend
