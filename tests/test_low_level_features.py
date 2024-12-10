import shutil

import pandas as pd
import pytest

from zensvi.cv import get_low_level_features


@pytest.fixture(scope="function")  # Explicitly set function scope
def output_dir(base_output_dir, ensure_dir):
    output_dir = base_output_dir / "low_level_features"
    if output_dir.exists():
        print(f"Cleaning up existing {output_dir} before test function")  # Optional: for debugging
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    return output_dir


def test_low_level(output_dir, input_dir):
    image_input = str(input_dir / "images")
    image_output = output_dir / "images"
    summary_output = output_dir / "summary"
    get_low_level_features(
        image_input,
        dir_image_output=image_output,
        dir_summary_output=summary_output,
        save_format="json csv",
        csv_format="wide",
    )
    files_in_image_output = [f for f in image_output.rglob("*") if f.is_file()]
    df = pd.read_csv(summary_output / "low_level_features.csv")
    assert len(files_in_image_output) > 0 and len(df) > 0
