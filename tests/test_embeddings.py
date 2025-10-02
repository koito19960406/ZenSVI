import shutil
from collections import namedtuple

import pytest
import torch

from zensvi.cv import Embeddings


@pytest.fixture(scope="function")  # Explicitly set function scope
def output_dir(base_output_dir, ensure_dir):
    output_dir = base_output_dir / "embeddings"
    if output_dir.exists():
        print(f"Cleaning up existing {output_dir} before test function")  # Optional: for debugging
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    return output_dir


@pytest.fixture(params=[True, False])
def cuda(request):
    if request.param and not torch.cuda.is_available():
        pytest.skip("CUDA device not available")
    return request.param


def test_embeddings(output_dir, input_dir, cuda):
    embedding = Embeddings(model_name="alexnet", cuda=cuda)
    image_output = output_dir / f"cuda_{cuda}/images"

    embs = embedding.generate_embedding(
        str(input_dir / "images"),
        image_output,
        batch_size=1000,
    )

    assert embs is not None


def test_search_similar_images(output_dir, input_dir, cuda):
    embedding = Embeddings(model_name="alexnet", cuda=cuda)
    image_output = output_dir / f"cuda_{cuda}/images"

    embs = embedding.generate_embedding(
        str(input_dir / "images"),
        image_output,
        batch_size=1000,
    )

    assert embs is not None

    image_path = list((input_dir / "images").glob("*"))[0]
    image_base_name = image_path.stem

    similar_images = embedding.search_similar_images(
        image_base_name,
        image_output,
        5,
    )
    assert similar_images is not None


def test_all_models(output_dir, input_dir, cuda):
    _Model = namedtuple("Model", ["name", "layer", "layer_output_size"])
    models_dict = {
        "resnet18": _Model("resnet18", "avgpool", 512),
        "alexnet": _Model("alexnet", "classifier", 4096),
        "vgg": _Model("vgg11", "classifier", 4096),
        "densenet": _Model("densenet", "classifier", 1024),
        "efficientnet_b0": _Model("efficientnet_b0", "avgpool", 1280),
        "efficientnet_b1": _Model("efficientnet_b1", "avgpool", 1280),
        "efficientnet_b2": _Model("efficientnet_b2", "avgpool", 1408),
        "efficientnet_b3": _Model("efficientnet_b3", "avgpool", 1536),
        "efficientnet_b4": _Model("efficientnet_b4", "avgpool", 1792),
        "efficientnet_b5": _Model("efficientnet_b5", "avgpool", 2048),
        "efficientnet_b6": _Model("efficientnet_b6", "avgpool", 2304),
        "efficientnet_b7": _Model("efficientnet_b7", "avgpool", 2560),
    }
    for model_name in models_dict.keys():
        embedding = Embeddings(model_name=model_name, cuda=cuda)
        image_output = output_dir / f"cuda_{cuda}/{model_name}"

        embs = embedding.generate_embedding(
            str(input_dir / "images"),
            image_output,
            batch_size=10,
        )

        assert embs is not None


def test_verbosity_levels(output_dir, input_dir):
    """Test that verbosity levels work correctly in the Embeddings class."""

    # Test default verbosity (should be 1)
    default_embeddings = Embeddings(model_name="alexnet", cuda=False)
    assert default_embeddings.verbosity == 1

    # Test explicitly setting verbosity to 0 (no progress bars)
    silent_embeddings = Embeddings(model_name="alexnet", cuda=False, verbosity=0)
    assert silent_embeddings.verbosity == 0

    # Test explicitly setting verbosity to 2 (all progress bars)
    verbose_embeddings = Embeddings(model_name="alexnet", cuda=False, verbosity=2)
    assert verbose_embeddings.verbosity == 2

    # Test the generate_embedding method with different verbosity levels
    image_output = output_dir / "verbosity_test"

    # Run with verbosity=0
    result = silent_embeddings.generate_embedding(str(input_dir / "images"), str(image_output / "silent"), batch_size=2)
    assert result is True

    # Run with verbosity set in method call (overriding instance verbosity)
    result = default_embeddings.generate_embedding(
        str(input_dir / "images"), str(image_output / "explicit_verbosity"), batch_size=2, verbosity=0
    )
    assert result is True
