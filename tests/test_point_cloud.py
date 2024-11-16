import shutil

import pandas as pd
import pytest

from zensvi.transform import PointCloudProcessor


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
def test_data(input_dir):
    return pd.read_csv(input_dir / "point_cloud_test_df.csv")


def test_process_multiple_images(output_dir, processor, test_data):
    # Generate point clouds from the data without saving
    point_clouds = processor.process_multiple_images(test_data)
    assert len(point_clouds) == len(test_data)

    # Test saving point clouds in PCD format
    output_pcd_dir = output_dir / "pcd_files"
    output_pcd_dir.mkdir(parents=True, exist_ok=True)
    processor.process_multiple_images(test_data, output_dir=output_pcd_dir, save_format="pcd")

    # Verify that PCD files were saved
    for image_id in test_data["id"]:
        output_file = output_pcd_dir / f"{image_id}.pcd"
        assert output_file.exists(), f"PCD file {output_file} was not saved"


def test_transform_point_cloud(processor, test_data):
    point_clouds = processor.process_multiple_images(test_data)
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
    point_clouds = processor.process_multiple_images(test_data)

    # Test saving in NumPy format
    npz_path = output_dir / "point_cloud.npz"
    processor.save_point_cloud_numpy(point_clouds[0], npz_path)
    assert npz_path.exists()

    # Test saving in CSV format
    csv_path = output_dir / "point_cloud.csv"
    processor.save_point_cloud_csv(point_clouds[0], csv_path)
    assert csv_path.exists()


def test_visualize_point_cloud(processor, test_data):
    point_clouds = processor.process_multiple_images(test_data)
    try:
        processor.visualize_point_cloud(point_clouds[0])
        assert True
    except Exception as e:
        pytest.fail(f"Visualize point cloud raised an exception {e}")
