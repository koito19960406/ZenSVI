from collections import namedtuple

import pytest
import torch

from zensvi.cv import Embeddings


@pytest.fixture
def output(base_output_dir, ensure_dir):
    output_dir = base_output_dir / "embeddings"
    ensure_dir(output_dir)
    return output_dir


@pytest.fixture(params=[True, False])
def cuda(request):
    if request.param and not torch.cuda.is_available():
        pytest.skip("CUDA device not available")
    return request.param


def test_embeddings(output, input_dir, cuda):
    embedding = Embeddings(model_name="alexnet", cuda=cuda)
    image_output = output / f"cuda_{cuda}/images"

    embs = embedding.generate_embedding(
        str(input_dir / "images"),
        image_output,
        batch_size=1000,
    )

    assert embs is not None


def test_search_similar_images(output, input_dir, cuda):
    embedding = Embeddings(model_name="alexnet", cuda=cuda)
    image_output = output / f"cuda_{cuda}/images"

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


def test_all_models(output, input_dir, cuda):
    _Model = namedtuple("Model", ["name", "layer", "layer_output_size"])
    models_dict = {
        "resnet-18": _Model("resnet18", "avgpool", 512),
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
        image_output = output / f"cuda_{cuda}/{model_name}"

        embs = embedding.generate_embedding(
            str(input_dir / "images"),
            image_output,
            batch_size=10,
        )

        assert embs is not None
