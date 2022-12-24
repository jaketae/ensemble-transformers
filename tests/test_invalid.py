import pytest

from ensemble_transformers import EnsembleModelForSequenceClassification
from ensemble_transformers.base import EnsembleBaseModel


def test_modality():
    pass


def test_to_multiple():
    ensemble = EnsembleModelForSequenceClassification.from_multiple_pretrained(
        "distilroberta-base", "bert-base-uncased"
    )
    with pytest.raises(ValueError):
        ensemble.to_multiple(["cpu"])


def test_base_init():
    with pytest.raises(ValueError):
        EnsembleBaseModel.from_multiple_pretrained("bert-base-uncased")
