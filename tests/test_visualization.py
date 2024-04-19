from zensvi import visualization
import unittest
from pathlib import Path
import pandas as pd

# import shutil


class TestVisualization(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.output = Path("tests/data/output/visualization")
        self.output.mkdir(parents=True, exist_ok=True)
        pass

    # def tearDown(self):
    #     # remove output directory
    #     shutil.rmtree(self.output, ignore_errors=True)

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
                path_output = str(self.output / f"plot_map_{variable}_{plot_type}.png")
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
                    legend_title=(
                        "Count"
                        if variable is None
                        else f"{variable.title()} View Factor"
                    ),
                    dark_mode=False,
                )
                # assert True if path_output exists and size is not zero
                self.assertTrue(
                    Path(path_output).exists() and Path(path_output).stat().st_size > 0
                )

    def test_plot_map_edge_color(self):
        path_pid = "tests/data/input/visualization/gsv_pids.csv"
        dir_input = "tests/data/input/visualization/cityscapes_semantic_summary"
        csv_file_pattern = "pixel_ratios.csv"
        variable_name = "sky"
        plot_type = "hexagon"
        path_output = str(self.output / "plot_map_edge_color.png")
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
        # assert True if path_output exists and size is not zero
        self.assertTrue(
            Path(path_output).exists() and Path(path_output).stat().st_size > 0
        )

    def test_plot_map_batch(self):
        path_pid = "tests/data/input/visualization/mly_pids.csv"
        dir_input = "tests/data/input/visualization/batch_segmentation"
        variable_name = "sky"
        plot_type = "hexagon"
        path_output = str(self.output / "plot_map_batch.png")
        fig, ax = visualization.plot_map(
            path_pid,
            pid_column="id",
            dir_input=dir_input,
            variable_name=variable_name,
            plot_type=plot_type,
            path_output=path_output,
            resolution=10,
            cmap="viridis",
            legend=True,
            title="Test Plot Map",
            legend_title="Test Legend Title",
            dark_mode=False,
        )
        # assert True if path_output exists and size is not zero
        self.assertTrue(
            Path(path_output).exists() and Path(path_output).stat().st_size > 0
        )

    # @unittest.skip("skip for now")
    def test_plot_image(self):
        dir_image_input = "tests/data/input/visualization/images"
        dir_csv_input = "tests/data/input/visualization"
        csv_file_pattern = "pixel_ratios.csv"
        path_output = str(self.output / "plot_image.png")
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
        # assert True if path_output exists and size is not zero
        self.assertTrue(
            Path(path_output).exists() and Path(path_output).stat().st_size > 0
        )

    # @unittest.skip("skip for now")
    def test_plot_image_image_file_pattern(self):
        dir_image_input = "tests/data/input/visualization/mapillary_panoptic"
        image_file_pattern = "*_blend.png"
        path_output = str(self.output / "plot_image_image_file_pattern.png")
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
        # assert True if path_output exists and size is not zero
        self.assertTrue(
            Path(path_output).exists() and Path(path_output).stat().st_size > 0
        )

    # @unittest.skip("skip for now")
    def test_plot_image_sort_by(self):
        dir_image_input = "tests/data/input/visualization/mapillary_panoptic"
        image_file_pattern = "*_blend.png"
        path_output = str(self.output / "plot_image_sort_by.png")
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
        # assert True if path_output exists and size is not zero
        self.assertTrue(
            Path(path_output).exists() and Path(path_output).stat().st_size > 0
        )

    def test_plot_image_batch(self):
        dir_image_input = "tests/data/input/visualization/batch_images"
        path_output = str(self.output / "plot_image_batch.png")
        fig, ax = visualization.plot_image(
            dir_image_input,
            4,
            5,
            dir_csv_input="tests/data/input/visualization/batch_segmentation",
            csv_file_pattern="pixel_ratios.csv",
            sort_by="vegetation",
            subplot_width=2,
            subplot_height=1,
            title="Image Grid",
            path_output=path_output,
            dark_mode=False,
            random_seed=123,
        )
        # assert True if path_output exists and size is not zero
        self.assertTrue(
            Path(path_output).exists() and Path(path_output).stat().st_size > 0
        )

    def test_plot_kde(self):
        path_input = "tests/data/input/visualization/cityscapes_semantic_summary/pixel_ratios.csv"
        columns = ["sky", "building", "vegetation", "road", "sidewalk"]
        path_output = str(self.output / "plot_kde.png")
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
        # assert True if path_output exists and size is not zero
        self.assertTrue(
            Path(path_output).exists() and Path(path_output).stat().st_size > 0
        )

    def test_plot_kde_long(self):
        path_input = "tests/data/input/visualization/classification/places365/long/summary/results.csv"
        columns = [
            "residential_neighborhood",
            "highway",
            "field_road",
            "hospital",
            "building_facade",
        ]
        path_output = str(self.output / "plot_kde_long.png")
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
        # assert True if path_output exists and size is not zero
        self.assertTrue(
            Path(path_output).exists() and Path(path_output).stat().st_size > 0
        )

    def test_plot_kde_batch(self):
        dir_input = "tests/data/input/visualization/batch_segmentation"
        columns = ["sky", "building", "vegetation", "road", "sidewalk"]
        path_output = str(self.output / "plot_kde_batch.png")
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
        # assert True if path_output exists and size is not zero
        self.assertTrue(
            Path(path_output).exists() and Path(path_output).stat().st_size > 0
        )

    def test_map_year(self):
        # load mly_pids and calculate the year from captured_at and save to csv with id, year, lat, lon
        mly_pids = pd.read_csv("tests/data/input/visualization/mly_pids.csv")
        # Convert 'captured_at' from milliseconds to datetime, then extract the year
        mly_pids["year"] = pd.to_datetime(
            mly_pids["captured_at"], unit="ms"
        ).dt.year.astype(int)
        mly_pids = mly_pids[["id", "year", "lat", "lon"]]
        mly_pids.to_csv(str(self.output / "mly_pids_year.csv"), index=False)
        path_output = str(self.output / "plot_map_year.png")
        # plot map of mly_pids with plot_map
        visualization.plot_map(
            str(self.output / "mly_pids_year.csv"),
            pid_column="id",
            plot_type="point",
            variable_name="year",
            path_output=path_output,
            title="Mapillary images in the study area",
            legend_title="Year of collection",
            markersize=1,
        )
        # assert True if path_output exists and size is not zero
        self.assertTrue(
            Path(path_output).exists() and Path(path_output).stat().st_size > 0
        )

    def test_plot_hist(self):
        dir_input = "tests/data/input/visualization/cityscapes_semantic_summary"
        columns = ["sky", "building", "vegetation", "road", "sidewalk"]
        path_output = str(self.output / "plot_hist.png")
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
        # assert True if path_output exists and size is not zero
        self.assertTrue(
            Path(path_output).exists() and Path(path_output).stat().st_size > 0
        )


if __name__ == "__main__":
    unittest.main()
