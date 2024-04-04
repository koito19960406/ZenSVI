from zensvi import visualization
import unittest
from pathlib import Path


class TestVisualization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    # skip for now
    # @unittest.skip("skip for now")
    def test_plot_map(self):
        path_pid = "tests/data/input/visualization/gsv_pids.csv"
        dir_input = "tests/data/input/visualization/cityscapes_semantic_summary"
        csv_file_pattern = "pixel_ratios.csv"
        variable_name_list = ["sky", None]
        plot_type_list = ["point", "line", "hexagon"]
        for variable in variable_name_list:
            for plot_type in plot_type_list:
                
                path_output = f"tests/data/output/visualization/plot_map_{variable}_{plot_type}.png"
                # create the directory if it does not exist
                Path(path_output).parent.mkdir(parents=True, exist_ok=True)
                fig, ax = visualization.plot_map(
                    path_pid,
                    dir_input=dir_input,
                    csv_file_pattern=csv_file_pattern,
                    variable_name=variable,
                    plot_type=plot_type,
                    path_output=path_output,
                    resolution=13,
                    cmap="viridis",
                    legend=True,
                    title=f"{plot_type.title()} Map",
                    legend_title="Count" if variable is None else f"{variable.title()} View Factor",
                    dark_mode=False,
                )

    def test_plot_map_edge_color(self):
        path_pid = "tests/data/input/visualization/gsv_pids.csv"
        dir_input = "tests/data/input/visualization/cityscapes_semantic_summary"
        csv_file_pattern = "pixel_ratios.csv"
        variable_name = "sky"
        plot_type = "hexagon"
        path_output = f"tests/data/output/visualization/plot_map_edge_color.png"
        # create the directory if it does not exist
        Path(path_output).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = visualization.plot_map(
            path_pid,
            dir_input=dir_input,
            csv_file_pattern=csv_file_pattern,
            variable_name=variable_name,
            plot_type=plot_type,
            path_output=path_output,
            edgecolor="black",
            resolution=13,
            cmap="viridis",
            legend=True,
            title="Test Plot Map",
            legend_title="Test Legend Title",
            dark_mode=False,
        )

    # @unittest.skip("skip for now")
    def test_plot_image(self):
        dir_image_input = "tests/data/input/visualization/images"
        dir_csv_input = "tests/data/input/visualization"
        csv_file_pattern = "pixel_ratios.csv"
        path_output = "tests/data/output/visualization/plot_image.png"
        # create the directory if it does not exist
        Path(path_output).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = visualization.plot_image(
            dir_image_input,
            4,
            5,
            dir_csv_input=dir_csv_input,
            csv_file_pattern=csv_file_pattern,
            sort_by="random",
            title="Image Grid",
            path_output=path_output,
            dark_mode=False,
            random_seed=123,
        )

    # @unittest.skip("skip for now")
    def test_plot_image_image_file_pattern(self):
        dir_image_input = "tests/data/input/visualization/mapillary_panoptic"
        image_file_pattern = "*_blend.png"
        path_output = (
            "tests/data/output/visualization/plot_image_image_file_pattern.png"
        )
        # create the directory if it does not exist
        Path(path_output).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = visualization.plot_image(
            dir_image_input,
            4,
            5,
            subplot_width=2,
            subplot_height=1,
            image_file_pattern=image_file_pattern,
            title="Test Plot Image",
            path_output=path_output,
            dark_mode=False,
            random_seed=123,
        )

    # @unittest.skip("skip for now")
    def test_plot_image_sort_by(self):
        dir_image_input = "tests/data/input/visualization/mapillary_panoptic"
        image_file_pattern = "*_blend.png"
        path_output = "tests/data/output/visualization/plot_image_sort_by.png"
        # create the directory if it does not exist
        Path(path_output).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = visualization.plot_image(
            dir_image_input,
            4,
            5,
            subplot_width=2,
            subplot_height=1,
            dir_csv_input="tests/data/input/visualization/mapillary_panoptic_summary",
            image_file_pattern=image_file_pattern,
            csv_file_pattern="pixel_ratios.csv",
            sort_by="Sky",
            title="Image Grid Sorted by Sky View Factor",
            path_output=path_output,
            dark_mode=False,
            random_seed=123,
        )


if __name__ == "__main__":
    unittest.main()
