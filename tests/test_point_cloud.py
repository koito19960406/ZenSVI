import unittest
from zensvi.transform import PointCloudProcessor
import pandas as pd
from pathlib import Path
from test_base import TestBase


class TestPointCloudProcessor(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Define the data directory relative to the script file
        data_dir = cls.input_dir / 'images'

        # Ensure the directories exist
        assert data_dir.exists(), f"Data directory {data_dir} does not exist"

        # Construct full paths to the folders
        image_folder = data_dir / 'color'
        depth_folder = data_dir / 'depth'

        # Load the CSV data
        cls.data = pd.read_csv(cls.input_dir / 'point_cloud_test_df.csv')

        # Initialize the processor
        cls.processor = PointCloudProcessor(
            image_folder=str(image_folder),
            depth_folder=str(depth_folder),
            log_path=cls.base_output_dir / 'point_cloud_processor.log'
        )

    def test_process_multiple_images(self):
        # Generate point clouds from the data without saving
        point_clouds = self.processor.process_multiple_images(self.data)
        self.assertEqual(len(point_clouds), len(self.data))

        # Test saving point clouds in PCD format
        output_dir = self.base_output_dir / 'pcd_files'
        output_dir.mkdir(parents=True, exist_ok=True)
        self.processor.process_multiple_images(self.data, output_dir=output_dir, save_format='pcd')

        # Verify that PCD files were saved
        for image_id in self.data['id']:
            output_file = output_dir / f"{image_id}.pcd"
            self.assertTrue(output_file.exists(), f"PCD file {output_file} was not saved")

    def test_transform_point_cloud(self):
        # Assuming that we already have some point clouds
        point_clouds = self.processor.process_multiple_images(self.data)
        transformed_clouds = []
        for i, pcd in enumerate(point_clouds):
            origin_x = self.data.at[i, 'lon']
            origin_y = self.data.at[i, 'lat']
            angle = self.data.at[i, 'heading']
            box_extent = [4, 4, 4]  # Example box dimensions
            box_center = [origin_x, origin_y, 0]  # Example box center
            transformed_pcd = self.processor.transform_point_cloud(pcd, origin_x, origin_y, angle, box_extent, box_center)
            transformed_clouds.append(transformed_pcd)
        self.assertEqual(len(transformed_clouds), len(point_clouds))
        # Additional assertions to check the properties of transformed_clouds can be added here

    def test_save_point_cloud_formats(self):
        # Generate a point cloud
        point_clouds = self.processor.process_multiple_images(self.data)
        output_dir = self.base_output_dir

        # Test saving in NumPy format
        npz_path = output_dir / 'point_cloud.npz'
        self.processor.save_point_cloud_numpy(point_clouds[0], npz_path)
        self.assertTrue(npz_path.exists(), f"NumPy file {npz_path} was not saved")

        # Test saving in CSV format
        csv_path = output_dir / 'point_cloud.csv'
        self.processor.save_point_cloud_csv(point_clouds[0], csv_path)
        self.assertTrue(csv_path.exists(), f"CSV file {csv_path} was not saved")

    def test_visualize_point_cloud(self):
        # Testing visualization can be tricky as it doesn't return a value
        # This test can ensure that no exceptions are raised during visualization
        point_clouds = self.processor.process_multiple_images(self.data)
        try:
            self.processor.visualize_point_cloud(point_clouds[0])
            self.assertTrue(True)  # Pass the test if no exceptions
        except Exception as e:
            self.fail(f"Visualize point cloud raised an exception {e}")

if __name__ == '__main__':
    unittest.main()
