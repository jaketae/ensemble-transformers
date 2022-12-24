import pytest

from ensemble_transformers import EnsembleModelForSequenceClassification
from ensemble_transformers.base import EnsembleBaseModel


def test_modality():
    with pytest.raises(ValueError):
        EnsembleModelForSequenceClassification.from_multiple_pretrained(
            "hf-internal-testing/tiny-random-distilbert",
            "hf-internal-testing/tiny-random-vit",
        )


def test_to_multiple():
    ensemble = EnsembleModelForSequenceClassification.from_multiple_pretrained(
        "hf-internal-testing/tiny-bert",
        "hf-internal-testing/tiny-random-distilbert",
    )
    with pytest.raises(ValueError):
        ensemble.to_multiple(["cpu"])


def test_base_init():
    with pytest.raises(RuntimeError):
        EnsembleBaseModel.from_multiple_pretrained("hf-internal-testing/tiny-bert")
