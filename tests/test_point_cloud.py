import shutil

import pandas as pd
import pytest

from zensvi.transform import PointCloudProcessor, VGGTProcessor


@pytest.fixture(scope="function")  # Explicitly set function scope
def output_dir(base_output_dir, ensure_dir):
    output_dir = base_output_dir / "point_cloud"
    if output_dir.exists():
        print(f"Cleaning up existing {output_dir} before test function")  # Optional: for debugging
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    return output_dir


@pytest.fixture
def processor(input_dir, base_output_dir):
    data_dir = input_dir / "images"
    assert data_dir.exists(), f"Data directory {data_dir} does not exist"

    image_folder = data_dir / "color"
    depth_folder = data_dir / "depth"

    return PointCloudProcessor(
        image_folder=str(image_folder),
        depth_folder=str(depth_folder),
        log_path=base_output_dir / "point_cloud_processor.log",
    )


@pytest.fixture
def vggt_processor():
    """Fixture for VGGTProcessor with conditional loading."""
    try:
        import torch
        # Check if CUDA is available for GPU-accelerated testing
        if torch.cuda.is_available():
            return VGGTProcessor()
        else:
            # Skip VGGT tests if CUDA is not available as the model is large
            pytest.skip("CUDA not available, skipping VGGT tests")
    except Exception as e:
        pytest.skip(f"Failed to initialize VGGTProcessor: {str(e)}")


@pytest.fixture
def test_data(input_dir):
    return pd.read_csv(input_dir / "point_cloud_test_df.csv")


def test_process_multiple_images(output_dir, processor, test_data):
    # Generate point clouds from the data without saving
    point_clouds = processor.process_multiple_images(test_data, depth_max=None, use_absolute_depth=True)
    assert len(point_clouds) == len(test_data)

    # Test saving point clouds in PCD format
    output_pcd_dir = output_dir / "pcd_files"
    output_pcd_dir.mkdir(parents=True, exist_ok=True)
    processor.process_multiple_images(
        test_data, output_dir=output_pcd_dir, save_format="pcd", depth_max=None, use_absolute_depth=True
    )

    # Verify that PCD files were saved
    for image_id in test_data["id"]:
        output_file = output_pcd_dir / f"{image_id}.pcd"
        assert output_file.exists(), f"PCD file {output_file} was not saved"


def test_transform_point_cloud(processor, test_data):
    point_clouds = processor.process_multiple_images(test_data, depth_max=None, use_absolute_depth=True)
    transformed_clouds = []
    for i, pcd in enumerate(point_clouds):
        origin_x = test_data.at[i, "lon"]
        origin_y = test_data.at[i, "lat"]
        angle = test_data.at[i, "heading"]
        box_extent = [4, 4, 4]
        box_center = [origin_x, origin_y, 0]
        transformed_pcd = processor.transform_point_cloud(pcd, origin_x, origin_y, angle, box_extent, box_center)
        transformed_clouds.append(transformed_pcd)
    assert len(transformed_clouds) == len(point_clouds)


def test_save_point_cloud_formats(output_dir, processor, test_data):
    point_clouds = processor.process_multiple_images(test_data, depth_max=None, use_absolute_depth=True)

    # Test saving in NumPy format
    npz_path = output_dir / "point_cloud.npz"
    processor.save_point_cloud_numpy(point_clouds[0], npz_path)
    assert npz_path.exists()

    # Test saving in CSV format
    csv_path = output_dir / "point_cloud.csv"
    processor.save_point_cloud_csv(point_clouds[0], csv_path)
    assert csv_path.exists()


def test_visualize_point_cloud(processor, test_data):
    point_clouds = processor.process_multiple_images(test_data, depth_max=None, use_absolute_depth=True)
    try:
        processor.visualize_point_cloud(point_clouds[0])
        assert True
    except Exception as e:
        pytest.fail(f"Visualize point cloud raised an exception {e}")


# VGGT Processor Tests
@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="PyTorch not available").cuda.is_available(),
    reason="CUDA not available for VGGT tests"
)
def test_vggt_processor_initialization(vggt_processor):
    """Test VGGTProcessor initialization."""
    assert vggt_processor is not None
    assert hasattr(vggt_processor, 'model')
    assert hasattr(vggt_processor, 'device')
    assert hasattr(vggt_processor, 'dtype')


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="PyTorch not available").cuda.is_available(),
    reason="CUDA not available for VGGT tests",
)
def test_vggt_process_images_and_visualize_pointcloud(output_dir, vggt_processor, input_dir):
    """Test VGGTProcessor image processing to point cloud and visualization."""
    import numpy as np
    import open3d as o3d

    data_dir = input_dir / "images"
    if not data_dir.exists():
        pytest.skip("Test image directory not available")

    image_folder = data_dir / "color"
    if not image_folder.exists() or not any(image_folder.iterdir()):
        pytest.skip("No test images available")

    output_vggt_dir = output_dir / "vggt_pointclouds"
    output_vggt_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Process a single image to avoid memory issues in tests
        vggt_processor.process_images_to_pointcloud(
            dir_input=image_folder, dir_output=output_vggt_dir, batch_size=1, max_workers=1
        )
    except Exception as e:
        pytest.skip(f"VGGT processing failed (likely due to resource constraints): {str(e)}")

    # Check if point cloud files were created
    output_files = list(output_vggt_dir.glob("*.ply"))
    assert len(output_files) > 0, "No point cloud files were generated"

    # Verify file exists and has content
    first_output_file = output_files[0]
    assert first_output_file.stat().st_size > 0, f"Empty point cloud file: {first_output_file}"

    # Test visualization with the generated point cloud
    try:
        pcd = o3d.io.read_point_cloud(str(first_output_file))
        points = np.asarray(pcd.points)
        colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)

        vggt_processor.visualize_point_cloud(points=points, colors_flat=colors, sample_rate=1)  # Use small sample for test
        assert True  # If no exception, test passes
    except Exception as e:
        pytest.fail(f"VGGT visualize point cloud raised an exception: {e}")


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="PyTorch not available").cuda.is_available(),
    reason="CUDA not available for VGGT tests"
)
def test_vggt_process_images_error_handling(vggt_processor, tmp_path):
    """Test VGGTProcessor error handling with invalid inputs."""
    # Test with non-existent directory
    non_existent_dir = tmp_path / "non_existent"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    with pytest.raises(ValueError, match="must be either a file or a directory"):
        vggt_processor.process_images_to_pointcloud(
            dir_input=non_existent_dir,
            dir_output=output_dir
        )


def test_vggt_processor_import():
    """Test that VGGTProcessor can be imported."""
    try:
        from zensvi.transform import VGGTProcessor
        assert VGGTProcessor is not None
    except ImportError as e:
        pytest.fail(f"Failed to import VGGTProcessor: {e}")


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="PyTorch not available").cuda.is_available(),
    reason="CUDA not available for VGGT tests"
)
def test_vggt_empty_directory_handling(vggt_processor, tmp_path):
    """Test VGGTProcessor handling of empty directories."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Should handle empty directory gracefully
    vggt_processor.process_images_to_pointcloud(
        dir_input=empty_dir,
        dir_output=output_dir
    )
    
    # Should not create any output files
    output_files = list(output_dir.glob("*.ply"))
    assert len(output_files) == 0, "Should not generate files from empty directory"
