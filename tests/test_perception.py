#!/usr/bin/env python3
from pathlib import Path

import pytest

from zensvi.cv import ClassifierPerception, ClassifierPerceptionViT


@pytest.fixture
def output(base_output_dir, ensure_dir):
    output_dir = base_output_dir / "classification/perception"
    ensure_dir(output_dir)
    return output_dir


def test_classify_directory(output, input_dir, all_devices):
    classifier = ClassifierPerception(perception_study="more boring", device=all_devices)
    image_input = str(input_dir / "images")
    dir_summary_output = str(output / f"{all_devices}/directory/summary")
    classifier.classify(
        image_input,
        dir_summary_output=dir_summary_output,
        batch_size=3,
    )
    assert len(list(Path(dir_summary_output).iterdir())) > 0


def test_classify_single_image(output, input_dir, all_devices):
    classifier = ClassifierPerception(perception_study="more boring", device=all_devices)
    image_input = str(input_dir / "images/test1.jpg")
    dir_summary_output = str(output / f"{all_devices}/single/summary")
    classifier.classify(
        image_input,
        dir_summary_output=dir_summary_output,
    )
    assert len(list(Path(dir_summary_output).iterdir())) > 0


def test_classify_directory_vit(output, input_dir, all_devices):
    classifier = ClassifierPerceptionViT(perception_study="more boring", device=all_devices)
    image_input = str(input_dir / "images")
    dir_summary_output = str(output / f"{all_devices}/directory_vit/summary")
    classifier.classify(
        image_input,
        dir_summary_output=dir_summary_output,
        batch_size=3,
    )
    assert len(list(Path(dir_summary_output).iterdir())) > 0
