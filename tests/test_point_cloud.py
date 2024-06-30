import unittest
from zensvi.transform import PointCloudProcessor
import pandas as pd
from pathlib import Path


class TestPointCloudProcessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Get the directory of the current script
        base_dir = Path(__file__).resolve().parent

        # Define the data directory relative to the script file
        data_dir = base_dir / 'data' / 'input' / 'images'

        # Ensure the directories exist
        assert data_dir.exists(), f"Data directory {data_dir} does not exist"

        # Construct full paths to the folders
        image_folder = data_dir / 'color'
        depth_folder = data_dir / 'depth'

        # Load the CSV data
        cls.data = pd.read_csv(base_dir / 'data' / 'input' / 'point_cloud_test_df.csv')

        # Initialize the processor
        cls.processor = PointCloudProcessor(
            image_folder=str(image_folder),
            depth_folder=str(depth_folder),
            log_path= base_dir / 'data' / 'output' / 'point_cloud_processor.log'
        )

    def test_process_multiple_images(self):
        # Generate point clouds from the data
        point_clouds = self.processor.process_multiple_images(self.data)
        self.assertEqual(len(point_clouds), len(self.data))


    def test_transform_point_cloud(self):
        # Assuming that we already have some point clouds
        point_clouds = self.processor.process_multiple_images(self.data)
        transformed_clouds = []
        for i, pcd in enumerate(point_clouds):
            origin_x = self.data.at[i, 'x_proj'] 
            origin_y = self.data.at[i, 'y_proj'] 
            angle = self.data.at[i, 'heading']
            box_extent = [4, 4, 4]  # Example box dimensions
            box_center = [origin_x, origin_y, 0]  # Example box center
            transformed_pcd = self.processor.transform_point_cloud(pcd, origin_x, origin_y, angle, box_extent, box_center)
            transformed_clouds.append(transformed_pcd)
        self.assertEqual(len(transformed_clouds), len(point_clouds))
        # Additional assertions to check the properties of transformed_clouds can be added here

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