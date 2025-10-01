"""Test for VGGT batch processing fix."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from zensvi.transform.image_to_pointcloud_vggt import VGGTProcessor


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_image_files(temp_output_dir):
    """Create sample image files for testing."""
    image_files = []
    for i in range(3):
        image_file = temp_output_dir / f"test_image_{i}.jpg"
        image_file.touch()  # Create empty file
        image_files.append(image_file)
    return image_files


def test_batch_processing_generates_separate_files(temp_output_dir, sample_image_files):
    """Test that batch processing generates separate point cloud files for each image."""
    
    # Mock the heavy dependencies
    with patch('zensvi.transform.image_to_pointcloud_vggt.torch'), \
         patch('zensvi.transform.image_to_pointcloud_vggt.o3d') as mock_o3d:
        
        # Mock the VGGT model initialization to avoid loading the actual model
        with patch.object(VGGTProcessor, '__init__', return_value=None):
            processor = VGGTProcessor.__new__(VGGTProcessor)
            processor.device = "cpu"
            processor.dtype = "float32"
            
            # Mock the process_images method
            mock_predictions = {"test": "data"}
            processor.process_images = Mock(return_value=mock_predictions)
            
            # Mock the generate_point_cloud method
            import numpy as np
            mock_points = np.array([[1, 2, 3], [4, 5, 6]])
            mock_colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
            mock_conf = np.array([0.9, 0.8])
            mock_cam_to_world = np.array([[1, 0, 0], [0, 1, 0]])
            processor.generate_point_cloud = Mock(return_value=(mock_points, mock_colors, mock_conf, mock_cam_to_world))
            
            # Mock open3d point cloud creation and saving
            mock_pcd = Mock()
            mock_o3d.geometry.PointCloud.return_value = mock_pcd
            mock_o3d.utility.Vector3dVector = Mock()
            mock_o3d.io.write_point_cloud = Mock()
            
            # Prepare test data
            image_paths = [str(f) for f in sample_image_files]
            original_sizes = [(100, 100)] * len(sample_image_files)
            
            # Call the method under test
            processor._process_batch_to_pointcloud(
                sample_image_files, 
                image_paths, 
                original_sizes, 
                temp_output_dir
            )
            
            # Verify that process_images was called once for each image (not once for the batch)
            assert processor.process_images.call_count == len(sample_image_files)
            
            # Verify that each call was made with a single image
            for i, call in enumerate(processor.process_images.call_args_list):
                args, kwargs = call
                assert len(args[0]) == 1  # Each call should have exactly one image path
                assert args[0][0] == image_paths[i]
            
            # Verify that write_point_cloud was called once for each image
            assert mock_o3d.io.write_point_cloud.call_count == len(sample_image_files)
            
            # Verify that each point cloud file has the correct name
            for i, call in enumerate(mock_o3d.io.write_point_cloud.call_args_list):
                args, kwargs = call
                output_path = args[0]
                expected_filename = f"test_image_{i}.ply"
                assert expected_filename in output_path


def test_batch_processing_with_single_image(temp_output_dir):
    """Test that batch processing works correctly with a single image."""
    
    # Create a single test image file
    image_file = temp_output_dir / "single_test.jpg"
    image_file.touch()
    
    with patch('zensvi.transform.image_to_pointcloud_vggt.torch'), \
         patch('zensvi.transform.image_to_pointcloud_vggt.o3d') as mock_o3d:
        
        with patch.object(VGGTProcessor, '__init__', return_value=None):
            processor = VGGTProcessor.__new__(VGGTProcessor)
            processor.device = "cpu"
            processor.dtype = "float32"
            
            # Mock the methods
            import numpy as np
            processor.process_images = Mock(return_value={"test": "data"})
            processor.generate_point_cloud = Mock(return_value=(np.array([[1, 2, 3]]), np.array([[255, 0, 0]], dtype=np.uint8), np.array([0.9]), np.array([[1, 0, 0]])))
            
            mock_pcd = Mock()
            mock_o3d.geometry.PointCloud.return_value = mock_pcd
            mock_o3d.utility.Vector3dVector = Mock()
            mock_o3d.io.write_point_cloud = Mock()
            
            # Call the method
            processor._process_batch_to_pointcloud(
                [image_file], 
                [str(image_file)], 
                [(100, 100)], 
                temp_output_dir
            )
            
            # Verify single call for single image
            assert processor.process_images.call_count == 1
            assert mock_o3d.io.write_point_cloud.call_count == 1
            
            # Verify correct filename
            call_args = mock_o3d.io.write_point_cloud.call_args[0]
            assert "single_test.ply" in call_args[0]

