import torch

from ensemble_transformers import EnsembleModelForSequenceClassification


def test_ensemble():
    model_names = ["hf-internal-testing/tiny-bert", "hf-internal-testing/tiny-albert", "hf-internal-testing/tiny-electra", "hf-internal-testing/tiny-xlm-roberta"]
    ensemble = EnsembleModelForSequenceClassification.from_multiple_pretrained(*model_names)
    batch = ["This is a test sentence", "This is another test sentence."]
    with torch.no_grad():
        output = ensemble(batch)
    assert len(output.logits) == len(model_names)
    shape = output.logits[0].shape
    output.stack(ensemble.config.weights, "cpu")
    assert output.logits.shape == shape
