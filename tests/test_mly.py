import json
import unittest
import os
import shutil
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
from zensvi.download.mapillary import interface
from zensvi.download import MLYDownloader


class TestMapillary(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.mly_api_key = os.getenv("MLY_API_KEY")
        interface.set_access_token(self.mly_api_key)
        self.mly_output = "tests/data/output/mly_output/"
        os.makedirs(self.mly_output, exist_ok=True)
        self.mly_input_multipolygon = "tests/data/input/test_multipolygon.geojson"
        self.mly_input_polygon = "tests/data/input/test_polygon.geojson"
        self.mly_output_json = Path(self.mly_output) / "mly_output.json"
        self.mly_svi_output = Path(self.mly_output) / "mly_svi"
        self.mly_svi_output_multipolygon = Path(self.mly_output) / "mly_svi_multipolygon"
        self.mly_svi_output_polygon = Path(self.mly_output) / "mly_svi_polygon"
        self.mly_log = Path(self.mly_output) / "log.log"
        pass
    
    # @classmethod   
    # def tearDown(self):
    #     # remove output directory
    #     shutil.rmtree(self.mly_output, ignore_errors=True)

    # def test_interface(self):
    #     # Skip test if the output file already exists
    #     if os.path.exists(self.mly_output_json):
    #         self.skipTest("Result exists")
    #     # read geojson as dict
    #     with open(self.mly_input_polygon) as f:
    #         geojson = json.load(f)
    #     output = interface.images_in_geojson(geojson)
    #     # assert True if output is not empty
    #     self.assertTrue(len(output.to_dict()) > 0)

    # def test_downloader(self):
    #     # Skip test if the output file already exists
    #     if os.path.exists(os.path.join(self.mly_svi_output, "mly_svi")):
    #         self.skipTest("Result exists")
    #     # download images
    #     mly_downloader = MLYDownloader(self.mly_api_key)
    #     mly_downloader.download_svi(self.mly_svi_output, input_shp_file=self.mly_input_polygon)

    #     # assert True if there are files in the output directory
    #     self.assertTrue(len(os.listdir(self.mly_svi_output)) > 0)

    # def test_downloader_metadata_only(self):
    #     # Skip test if the output file already exists
    #     if os.path.exists(os.path.join(self.mly_svi_output, "mly_pids.csv")):
    #         self.skipTest("Result exists")
    #     # download metadata only
    #     mly_downloader = MLYDownloader(
    #         self.mly_api_key,
    #         log_path="tests/data/output/mly_svi/log.log",
    #         max_workers=200,
    #     )
    #     # mly_downloader.download_svi(self.mly_svi_output, input_place_name="Singapore", metadata_only=True)
    #     mly_downloader.download_svi(
    #         self.mly_svi_output, input_shp_file=self.mly_input_polygon, metadata_only=True
    #     )
    #     # assert True if mly_pids.csv is not empty
    #     self.assertTrue(
    #         os.path.getsize(os.path.join(self.mly_svi_output, "mly_pids.csv")) > 0
    #     )

    # # test multipolygon
    # def test_downloader_multipolygon(self):
    #     # # Skip test if the output file already exists
    #     # if os.path.exists(os.path.join(self.mly_svi_output_multipolygon, "mly_svi")):
    #     #     self.skipTest("Result exists")
    #     # download images
    #     mly_downloader = MLYDownloader(self.mly_api_key, max_workers=200)
    #     mly_downloader.download_svi(
    #         self.mly_svi_output_multipolygon, input_shp_file=self.mly_input_multipolygon
    #     )
    #     # assert True if there are files in the output directory
    #     self.assertTrue(len(os.listdir(self.mly_svi_output_multipolygon)) > 0)

    # # test polygon
    # def test_downloader_polygon(self):
    #     # # Skip test if the output file already exists
    #     # if os.path.exists(os.path.join(self.mly_svi_output_polygon, "mly_svi")):
    #     #     self.skipTest("Result exists")
    #     # download images
    #     mly_downloader = MLYDownloader(self.mly_api_key, max_workers=200)
    #     mly_downloader.download_svi(
    #         self.mly_svi_output_polygon, input_shp_file=self.mly_input_polygon
    #     )
    #     # assert True if there are files in the output directory
    #     self.assertTrue(len(os.listdir(self.mly_svi_output_polygon)) > 0)

    def test_downloader_single_field(self):
        # Set up a new output directory for this test
        single_field_output = Path(self.mly_output) / "mly_svi_single_field"
        
        # Download images with only the 'captured_at' field
        mly_downloader = MLYDownloader(self.mly_api_key, max_workers=300)
        mly_downloader.download_svi(
            single_field_output,
            input_shp_file=self.mly_input_polygon,
            additional_fields=["all"]
        )
        
        # Check if the output directory is created and contains files
        self.assertTrue(single_field_output.exists())
        self.assertTrue(len(list(single_field_output.glob('**/*'))) > 0)
        
        # Check if the mly_pids.csv file exists and contains all the expected columns
        pids_file = single_field_output / "pids_urls.csv"
        self.assertTrue(pids_file.exists())
        
        df = pd.read_csv(pids_file)
        
        # Check for all fields from Entities.get_image_fields()
        expected_columns = [
            'altitude', 'atomic_scale', 'camera_parameters', 'camera_type',
            'captured_at', 'compass_angle', 'computed_altitude',
            'computed_compass_angle', 'computed_geometry', 'computed_rotation',
            'exif_orientation', 'geometry', 'height', 'url', 'merge_cc', 'mesh',
            'quality_score', 'sequence', 'sfm_cluster', 'width'
        ]
        
        for column in expected_columns:
            self.assertIn(column, df.columns)
        
        # Optionally, you can check the content of some image files to ensure they're valid
        image_files = list(single_field_output.glob('**/*.jpg'))
        if image_files:
            with Image.open(image_files[0]) as img:
                self.assertTrue(img.format in ('JPEG', 'PNG'))

    # # test with kwargs for mly
    # def test_downloader_kwargs(self):
    #     # Skip test if the output file already exists
    #     if os.path.exists(os.path.join(self.mly_svi_output, "mly_svi")):
    #         self.skipTest("Result exists")
    #     # download images
    #     mly_downloader = MLYDownloader(self.mly_api_key, log_path=str(self.mly_log), max_workers=200)
    #     kwarg = {
    #         "image_type": "flat",  # The tile image_type to be obtained, either as 'flat', 'pano' (panoramic), or 'all'.
    #         "min_captured_at": 1484549945000,  # The min date. Format from 'YYYY', to 'YYYY-MM-DDTHH:MM:SS'
    #         "max_captured_at": 1642935417694,  # The max date. Format from 'YYYY', to 'YYYY-MM-DDTHH:MM:SS'
    #         "organization_id": [1805883732926354],  # The organization id, ID of the organization this image (or sets of images) belong to. It can be absent. Thus, default is -1 (None)
    #         "compass_angle": (0,180)
    #     }
    #     mly_downloader.download_svi(self.mly_svi_output, input_shp_file=self.mly_input_polygon, **kwarg)
    #     # assert True if there are files in the output directory
    #     self.assertTrue(len(os.listdir(self.mly_svi_output)) > 0)

    # def test_adjust_image_orientation(self):
    #     pids_urls_path = Path(self.mly_output) / "mly_svi_single_field" / "pids_urls.csv"
    #     batch_dirs = [
    #         Path(self.mly_output) / "mly_svi_single_field" / "mly_svi" / "batch_1",
    #         Path(self.mly_output) / "mly_svi_single_field" / "mly_svi" / "batch_2"
    #     ]
    #     df = pd.read_csv(pids_urls_path)
        
    #     def get_camera_matrix_and_dist_coeffs(camera_parameters, camera_type, image_shape):
    #         params = json.loads(camera_parameters)
    #         h, w = image_shape[:2]
            
    #         focal_length, k1, k2 = params
    #         cx, cy = w / 2, h / 2  # Assume principal point is at the center
            
    #         camera_matrix = np.array([[focal_length, 0, cx],
    #                                   [0, focal_length, cy],
    #                                   [0, 0, 1]], dtype=np.float64)
            
    #         if camera_type in ["perspective", "fisheye"]:
    #             dist_coeffs = np.array([k1, k2, 0, 0], dtype=np.float64)
    #         elif camera_type in ["equirectangular", "spherical"]:
    #             dist_coeffs = np.zeros(4, dtype=np.float64)
    #         else:
    #             raise ValueError(f"Unsupported camera type: {camera_type}")
            
    #         return camera_matrix, dist_coeffs

    #     def apply_rotation_to_image(image, rotation_vector, camera_matrix, dist_coeffs, camera_type):
    #         rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    #         h, w = image.shape[:2]

    #         if camera_type == "perspective":
    #             new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    #             undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    #             full_transform = new_camera_matrix @ rotation_matrix @ np.linalg.inv(camera_matrix)
    #             rotated_image = cv2.warpPerspective(undistorted, full_transform, (w, h))
            
    #         elif camera_type == "fisheye":
    #             new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(camera_matrix, dist_coeffs, (w, h), np.eye(3))
    #             map1, map2 = cv2.fisheye.initUndistortRectifyMap(camera_matrix, dist_coeffs, np.eye(3), new_camera_matrix, (w, h), cv2.CV_16SC2)
    #             undistorted = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    #             full_transform = new_camera_matrix @ rotation_matrix @ np.linalg.inv(camera_matrix)
    #             rotated_image = cv2.warpPerspective(undistorted, full_transform, (w, h))
            
    #         elif camera_type in ["equirectangular", "spherical"]:
    #             # For equirectangular images, we need to use a different approach
    #             # This is a simple rotation that might not be perfect for all use cases
    #             rotated_image = cv2.warpAffine(image, rotation_matrix[:2, :2], (w, h))
            
    #         else:
    #             raise ValueError(f"Unsupported camera type: {camera_type}")

    #         return rotated_image

    #     for _, row in df.iterrows():
    #         image_id = row['id']
    #         computed_rotation = row['computed_rotation']
    #         camera_parameters = row['camera_parameters']
    #         camera_type = row['camera_type']
            
    #         rotation_vector = np.array(json.loads(computed_rotation), dtype=np.float64)
            
    #         image_path = None
    #         for batch_dir in batch_dirs:
    #             temp_path = batch_dir / f"{image_id}.png"
    #             if temp_path.exists():
    #                 image_path = temp_path
    #                 break
            
    #         if image_path is not None:
    #             image = cv2.imread(str(image_path))
                
    #             camera_matrix, dist_coeffs = get_camera_matrix_and_dist_coeffs(camera_parameters, camera_type, image.shape)
                
    #             rotated_image = apply_rotation_to_image(image, rotation_vector, camera_matrix, dist_coeffs, camera_type)
                
    #             output_path = image_path.with_name(f"{image_id}_rotated.png")
    #             cv2.imwrite(str(output_path), rotated_image)
            
    #             assert output_path.exists()

    #     rotated_images = []
    #     for batch_dir in batch_dirs:
    #         rotated_images.extend(list(batch_dir.glob("*_rotated.png")))
    #     assert len(rotated_images) > 0, "No images were rotated"

if __name__ == "__main__":
    unittest.main()
