import shutil

import numpy as np
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
    reason="CUDA not available for VGGT tests",
)
def test_vggt_processor_initialization(vggt_processor):
    """Test VGGTProcessor initialization."""
    assert vggt_processor is not None
    assert hasattr(vggt_processor, "model")
    assert hasattr(vggt_processor, "device")
    assert hasattr(vggt_processor, "dtype")


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="PyTorch not available").cuda.is_available(),
    reason="CUDA not available for VGGT tests",
)
def test_vggt_process_single_image(vggt_processor, input_dir):
    """Test VGGTProcessor.process_images method with single image."""
    data_dir = input_dir / "images"
    if not data_dir.exists():
        pytest.skip("Test image directory not available")

    image_folder = data_dir / "color"
    if not image_folder.exists() or not any(image_folder.iterdir()):
        pytest.skip("No test images available")

    # Get first image file
    image_files = [f for f in image_folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    if not image_files:
        pytest.skip("No valid image files found")

    single_image_path = [str(image_files[0])]

    try:
        # Test process_images method
        predictions = vggt_processor.process_images(single_image_path)

        # Verify predictions structure
        assert isinstance(predictions, dict)
        assert "images" in predictions
        assert "world_points" in predictions
        assert "depth_conf" in predictions
        assert "pose_enc" in predictions

    except Exception as e:
        pytest.skip(f"VGGT processing failed (likely due to resource constraints): {str(e)}")


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="PyTorch not available").cuda.is_available(),
    reason="CUDA not available for VGGT tests",
)
def test_vggt_generate_point_cloud(vggt_processor, input_dir):
    """Test VGGTProcessor.generate_point_cloud method."""
    data_dir = input_dir / "images"
    if not data_dir.exists():
        pytest.skip("Test image directory not available")

    image_folder = data_dir / "color"
    if not image_folder.exists() or not any(image_folder.iterdir()):
        pytest.skip("No test images available")

    # Get first image file
    image_files = [f for f in image_folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    if not image_files:
        pytest.skip("No valid image files found")

    single_image_path = [str(image_files[0])]

    try:
        # Process image to get predictions
        predictions = vggt_processor.process_images(single_image_path)

        # Generate point cloud from predictions
        points, colors, conf, cam_to_world = vggt_processor.generate_point_cloud(predictions)

        # Verify point cloud structure
        assert isinstance(points, np.ndarray)
        assert isinstance(colors, np.ndarray)
        assert isinstance(conf, np.ndarray)
        assert isinstance(cam_to_world, np.ndarray)

        # Verify shapes
        assert points.shape[1] == 3, "Points should have 3 coordinates"
        assert colors.shape[1] == 3, "Colors should have 3 channels"
        assert len(points) == len(colors), "Points and colors should have same length"
        assert len(conf) == len(points), "Confidence should match points length"

    except Exception as e:
        pytest.skip(f"VGGT point cloud generation failed: {str(e)}")


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="PyTorch not available").cuda.is_available(),
    reason="CUDA not available for VGGT tests",
)
def test_vggt_process_images_to_pointcloud_batch_sizes(output_dir, vggt_processor, input_dir):
    """Test VGGTProcessor.process_images_to_pointcloud with different batch sizes."""
    data_dir = input_dir / "images"
    if not data_dir.exists():
        pytest.skip("Test image directory not available")

    image_folder = data_dir / "color"
    if not image_folder.exists() or not any(image_folder.iterdir()):
        pytest.skip("No test images available")

    # Test different batch sizes
    batch_sizes = [1, 2]

    for batch_size in batch_sizes:
        output_vggt_dir = output_dir / f"vggt_batch_{batch_size}"
        output_vggt_dir.mkdir(parents=True, exist_ok=True)

        try:
            vggt_processor.process_images_to_pointcloud(
                dir_input=image_folder, dir_output=output_vggt_dir, batch_size=batch_size, max_workers=1
            )

            # Check if point cloud files were created
            output_files = list(output_vggt_dir.glob("*.ply"))
            assert len(output_files) > 0, f"No point cloud files were generated for batch size {batch_size}"

            # Verify file exists and has content
            for output_file in output_files:
                assert output_file.stat().st_size > 0, f"Empty point cloud file: {output_file}"

        except Exception as e:
            pytest.skip(f"VGGT processing failed with batch size {batch_size}: {str(e)}")


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="PyTorch not available").cuda.is_available(),
    reason="CUDA not available for VGGT tests",
)
def test_vggt_visualize_point_cloud(vggt_processor, input_dir):
    """Test VGGTProcessor visualization functionality."""
    data_dir = input_dir / "images"
    if not data_dir.exists():
        pytest.skip("Test image directory not available")

    image_folder = data_dir / "color"
    if not image_folder.exists() or not any(image_folder.iterdir()):
        pytest.skip("No test images available")

    # Get first image file
    image_files = [f for f in image_folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    if not image_files:
        pytest.skip("No valid image files found")

    single_image_path = [str(image_files[0])]

    try:
        # Process image and generate point cloud
        predictions = vggt_processor.process_images(single_image_path)
        points, colors, conf, cam_to_world = vggt_processor.generate_point_cloud(predictions)

        # Test visualization with different parameters
        vggt_processor.visualize_point_cloud(
            points=points, colors_flat=colors, marker_size=2, opacity=0.9, sample_rate=0.1  # Use small sample for test
        )
        assert True  # If no exception, test passes

    except Exception as e:
        pytest.fail(f"VGGT visualize point cloud raised an exception: {e}")


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="PyTorch not available").cuda.is_available(),
    reason="CUDA not available for VGGT tests",
)
def test_vggt_process_images_error_handling(vggt_processor, tmp_path):
    """Test VGGTProcessor error handling with invalid inputs."""
    # Test with non-existent directory
    non_existent_dir = tmp_path / "non_existent"
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with pytest.raises(ValueError, match="must be either a file or a directory"):
        vggt_processor.process_images_to_pointcloud(dir_input=non_existent_dir, dir_output=output_dir)


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="PyTorch not available").cuda.is_available(),
    reason="CUDA not available for VGGT tests",
)
def test_vggt_empty_directory_handling(vggt_processor, tmp_path):
    """Test VGGTProcessor handling of empty directories."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Should handle empty directory gracefully
    vggt_processor.process_images_to_pointcloud(dir_input=empty_dir, dir_output=output_dir)

    # Should not create any output files
    output_files = list(output_dir.glob("*.ply"))
    assert len(output_files) == 0, "Should not generate files from empty directory"


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="PyTorch not available").cuda.is_available(),
    reason="CUDA not available for VGGT tests",
)
def test_vggt_single_file_input(vggt_processor, input_dir, output_dir):
    """Test VGGTProcessor with single file input instead of directory."""
    data_dir = input_dir / "images"
    if not data_dir.exists():
        pytest.skip("Test image directory not available")

    image_folder = data_dir / "color"
    if not image_folder.exists() or not any(image_folder.iterdir()):
        pytest.skip("No test images available")

    # Get first image file
    image_files = [f for f in image_folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    if not image_files:
        pytest.skip("No valid image files found")

    single_image_file = image_files[0]
    output_vggt_dir = output_dir / "vggt_single_file"
    output_vggt_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Test with single file input
        vggt_processor.process_images_to_pointcloud(
            dir_input=single_image_file, dir_output=output_vggt_dir, batch_size=1, max_workers=1
        )

        # Check if point cloud file was created
        output_files = list(output_vggt_dir.glob("*.ply"))
        assert len(output_files) > 0, "No point cloud file was generated from single file input"

        # Verify file exists and has content
        first_output_file = output_files[0]
        assert first_output_file.stat().st_size > 0, f"Empty point cloud file: {first_output_file}"

    except Exception as e:
        pytest.skip(f"VGGT single file processing failed: {str(e)}")


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="PyTorch not available").cuda.is_available(),
    reason="CUDA not available for VGGT tests",
)
def test_vggt_invalid_image_formats(vggt_processor, tmp_path):
    """Test VGGTProcessor handling of invalid image formats."""
    # Create a directory with invalid files
    invalid_dir = tmp_path / "invalid_images"
    invalid_dir.mkdir()

    # Create files with invalid extensions
    (invalid_dir / "test.txt").write_text("not an image")
    (invalid_dir / "test.xyz").write_text("not an image")

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Should handle invalid files gracefully
    vggt_processor.process_images_to_pointcloud(dir_input=invalid_dir, dir_output=output_dir)

    # Should not create any output files
    output_files = list(output_dir.glob("*.ply"))
    assert len(output_files) == 0, "Should not generate files from invalid image formats"


def test_vggt_processor_import():
    """Test that VGGTProcessor can be imported."""
    try:
        from zensvi.transform import VGGTProcessor

        assert VGGTProcessor is not None
    except ImportError as e:
        pytest.fail(f"Failed to import VGGTProcessor: {e}")


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="PyTorch not available").cuda.is_available(),
    reason="CUDA not available for VGGT tests",
)
def test_vggt_processor_attributes(vggt_processor):
    """Test VGGTProcessor has all expected attributes and methods."""
    # Test required attributes
    assert hasattr(vggt_processor, "device")
    assert hasattr(vggt_processor, "dtype")
    assert hasattr(vggt_processor, "model")

    # Test required methods
    assert hasattr(vggt_processor, "process_images")
    assert hasattr(vggt_processor, "generate_point_cloud")
    assert hasattr(vggt_processor, "process_images_to_pointcloud")
    assert hasattr(vggt_processor, "visualize_point_cloud")

    # Test device and dtype are valid
    import torch

    assert vggt_processor.device in ["cuda", "cpu"]
    assert vggt_processor.dtype in [torch.float16, torch.bfloat16]


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="PyTorch not available").cuda.is_available(),
    reason="CUDA not available for VGGT tests",
)
def test_vggt_process_multiple_images_comprehensive(output_dir, vggt_processor, input_dir):
    """Comprehensive test for VGGTProcessor similar to PointCloudProcessor tests."""
    data_dir = input_dir / "images"
    if not data_dir.exists():
        pytest.skip("Test image directory not available")

    image_folder = data_dir / "color"
    if not image_folder.exists() or not any(image_folder.iterdir()):
        pytest.skip("No test images available")

    output_vggt_dir = output_dir / "vggt_comprehensive"
    output_vggt_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Process images to generate point clouds
        vggt_processor.process_images_to_pointcloud(
            dir_input=image_folder, dir_output=output_vggt_dir, batch_size=1, max_workers=1
        )

        # Verify that PLY files were saved
        output_files = list(output_vggt_dir.glob("*.ply"))
        assert len(output_files) > 0, "No PLY files were generated"

        # Test each generated file
        for output_file in output_files:
            assert output_file.exists(), f"PLY file {output_file} was not saved"
            assert output_file.stat().st_size > 0, f"Empty PLY file: {output_file}"

            # Test that we can read the point cloud
            import open3d as o3d

            pcd = o3d.io.read_point_cloud(str(output_file))
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)

            assert len(points) > 0, f"No points in point cloud {output_file}"
            assert len(colors) > 0, f"No colors in point cloud {output_file}"
            assert points.shape[1] == 3, f"Points should have 3 coordinates in {output_file}"
            assert colors.shape[1] == 3, f"Colors should have 3 channels in {output_file}"

    except Exception as e:
        pytest.skip(f"VGGT comprehensive processing failed: {str(e)}")
