import pandas as pd
import pytest

from zensvi.cv import get_low_level_features


@pytest.fixture
def output(base_output_dir, ensure_dir):
    output_dir = base_output_dir / "low_level"
    ensure_dir(output_dir)
    return output_dir


def test_low_level(output, input_dir):
    image_input = str(input_dir / "images")
    image_output = output / "images"
    summary_output = output / "summary"
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
