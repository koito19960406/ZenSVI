from pathlib import Path

import pytest

from zensvi import visualization


@pytest.fixture
def output(base_output_dir, ensure_dir):
    output_dir = base_output_dir / "visualization"
    ensure_dir(output_dir)
    return output_dir


def test_plot_map(output, input_dir):
    path_pid = str(input_dir / "visualization/gsv_pids.csv")
    dir_input = str(input_dir / "visualization/cityscapes_semantic_summary")
    csv_file_pattern = "pixel_ratios.csv"
    variable_name_list = ["sky", None]
    plot_type_list = ["point", "line", "hexagon"]

    for variable in variable_name_list:
        for plot_type in plot_type_list:
            path_output = str(output / f"plot_map_{variable}_{plot_type}.png")
            fig, ax = visualization.plot_map(
                path_pid,
                dir_input=dir_input,
                csv_file_pattern=csv_file_pattern,
                variable_name=variable,
                plot_type=plot_type,
                path_output=path_output,
                resolution=14,
                cmap="viridis",
                legend=True,
                title=f"{plot_type.title()} Map",
                legend_title=("Count" if variable is None else f"{variable.title()} View Factor"),
                dark_mode=False,
            )
            output_path = Path(path_output)
            assert output_path.exists() and output_path.stat().st_size > 0


def test_plot_map_edge_color(output, input_dir):
    path_pid = str(input_dir / "visualization/gsv_pids.csv")
    dir_input = str(input_dir / "visualization/cityscapes_semantic_summary")
    csv_file_pattern = "pixel_ratios.csv"
    variable_name = "sky"
    plot_type = "hexagon"
    path_output = str(output / "plot_map_edge_color.png")
    fig, ax = visualization.plot_map(
        path_pid,
        dir_input=dir_input,
        csv_file_pattern=csv_file_pattern,
        variable_name=variable_name,
        plot_type=plot_type,
        path_output=path_output,
        edgecolor="black",
        resolution=14,
        cmap="viridis",
        legend=True,
        title="Test Plot Map",
        legend_title="Test Legend Title",
        dark_mode=False,
    )
    output_path = Path(path_output)
    assert output_path.exists() and output_path.stat().st_size > 0


def test_plot_map_batch(output, input_dir):
    path_pid = str(input_dir / "visualization/gsv_pids.csv")
    dir_input = str(input_dir / "visualization/cityscapes_semantic_summary")
    variable_name = "sky"
    plot_type = "hexagon"
    path_output = str(output / "plot_map_batch.png")
    fig, ax = visualization.plot_map(
        path_pid,
        pid_column="panoid",
        dir_input=dir_input,
        variable_name=variable_name,
        plot_type=plot_type,
        path_output=path_output,
        resolution=14,
        cmap="viridis",
        legend=True,
        title="Test Plot Map",
        legend_title="Test Legend Title",
        dark_mode=False,
    )
    output_path = Path(path_output)
    assert output_path.exists() and output_path.stat().st_size > 0


def test_plot_image(output, input_dir):
    dir_image_input = str(input_dir / "visualization/images/batch_1")
    dir_csv_input = str(input_dir / "visualization/cityscapes_semantic_summary")
    csv_file_pattern = "pixel_ratios.csv"
    path_output = str(output / "plot_image.png")
    fig, ax = visualization.plot_image(
        dir_image_input,
        2,
        1,
        dir_csv_input=dir_csv_input,
        csv_file_pattern=csv_file_pattern,
        sort_by="random",
        title="Image Grid",
        path_output=path_output,
        dark_mode=False,
        random_seed=123,
    )
    output_path = Path(path_output)
    assert output_path.exists() and output_path.stat().st_size > 0


def test_plot_image_image_file_pattern(output, input_dir):
    dir_image_input = str(input_dir / "visualization/mapillary_panoptic")
    image_file_pattern = "*_blend.png"
    path_output = str(output / "plot_image_image_file_pattern.png")
    fig, ax = visualization.plot_image(
        dir_image_input,
        2,
        2,
        subplot_width=2,
        subplot_height=1,
        image_file_pattern=image_file_pattern,
        title="Test Plot Image",
        path_output=path_output,
        dark_mode=False,
        random_seed=123,
    )
    output_path = Path(path_output)
    assert output_path.exists() and output_path.stat().st_size > 0


def test_plot_image_sort_by(output, input_dir):
    dir_image_input = str(input_dir / "visualization/mapillary_panoptic")
    image_file_pattern = "*_blend.png"
    path_output = str(output / "plot_image_sort_by.png")
    fig, ax = visualization.plot_image(
        dir_image_input,
        2,
        2,
        subplot_width=2,
        subplot_height=1,
        dir_csv_input=str(input_dir / "visualization/cityscapes_semantic_summary"),
        image_file_pattern=image_file_pattern,
        csv_file_pattern="pixel_ratios.csv",
        sort_by="sky",
        title="Image Grid Sorted by Sky View Factor",
        path_output=path_output,
        dark_mode=False,
        random_seed=123,
    )
    output_path = Path(path_output)
    assert output_path.exists() and output_path.stat().st_size > 0


def test_plot_image_batch(output, input_dir):
    dir_image_input = str(input_dir / "visualization/images")
    path_output = str(output / "plot_image_batch.png")
    fig, ax = visualization.plot_image(
        dir_image_input,
        2,
        2,
        dir_csv_input=str(input_dir / "visualization/cityscapes_semantic_summary"),
        csv_file_pattern="pixel_ratios.csv",
        sort_by="vegetation",
        subplot_width=2,
        subplot_height=1,
        title="Image Grid",
        path_output=path_output,
        dark_mode=False,
        random_seed=123,
    )
    output_path = Path(path_output)
    assert output_path.exists() and output_path.stat().st_size > 0


def test_plot_kde(output, input_dir):
    path_input = str(input_dir / "visualization/cityscapes_semantic_summary/batch_1/pixel_ratios.csv")
    columns = ["sky", "building", "vegetation", "road", "sidewalk"]
    path_output = str(output / "plot_kde.png")
    fig, ax = visualization.plot_kde(
        path_input,
        columns,
        path_output=path_output,
        palette="twilight",
        legend=True,
        title="KDE Plot",
        legend_title="View Factor",
        dark_mode=False,
        font_size=30,
        clip=(0, 1),
    )
    output_path = Path(path_output)
    assert output_path.exists() and output_path.stat().st_size > 0


def test_plot_kde_long(output, input_dir):
    path_input = str(input_dir / "visualization/classification/places365/long/summary/results.csv")
    columns = [
        "residential_neighborhood",
        "highway",
        "field_road",
        "hospital",
        "building_facade",
    ]
    path_output = str(output / "plot_kde_long.png")
    fig, ax = visualization.plot_kde(
        path_input,
        columns,
        path_output=path_output,
        palette="twilight",
        legend=True,
        title="KDE Plot",
        legend_title="Scene Category",
        dark_mode=False,
        font_size=30,
        clip=(0, 1),
    )
    output_path = Path(path_output)
    assert output_path.exists() and output_path.stat().st_size > 0


def test_plot_kde_batch(output, input_dir):
    dir_input = str(input_dir / "visualization/cityscapes_semantic_summary")
    columns = ["sky", "building", "vegetation", "road", "sidewalk"]
    path_output = str(output / "plot_kde_batch.png")
    fig, ax = visualization.plot_kde(
        dir_input,
        columns,
        path_output=path_output,
        palette="twilight",
        legend=True,
        title="KDE Plot",
        legend_title="View Factor",
        dark_mode=False,
        font_size=30,
        clip=(0, 1),
    )
    output_path = Path(path_output)
    assert output_path.exists() and output_path.stat().st_size > 0


def test_map_year(output, input_dir):
    path_output = str(output / "plot_map_year.png")
    visualization.plot_map(
        str(input_dir / "visualization/gsv_pids.csv"),
        pid_column="panoid",
        plot_type="point",
        variable_name="year",
        path_output=path_output,
        title="Mapillary images in the study area",
        legend_title="Year of collection",
        markersize=1,
    )
    output_path = Path(path_output)
    assert output_path.exists() and output_path.stat().st_size > 0


def test_plot_hist(output, input_dir):
    dir_input = str(input_dir / "visualization/cityscapes_semantic_summary")
    columns = ["sky", "building", "vegetation", "road", "sidewalk"]
    path_output = str(output / "plot_hist.png")
    fig, ax = visualization.plot_hist(
        dir_input,
        columns,
        path_output=path_output,
        palette="twilight",
        legend=True,
        title="Histogram",
        legend_title="View Factor",
        dark_mode=False,
        font_size=30,
    )
    output_path = Path(path_output)
    assert output_path.exists() and output_path.stat().st_size > 0
