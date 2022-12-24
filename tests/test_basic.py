import torch

from ensemble_transformers import EnsembleModelForSequenceClassification


class TestTextEnsemble:
    @classmethod
    def setup_class(cls):
        cls.model_names = ["distilroberta-base", "xlnet-base-cased", "bert-base-uncased"]
        cls.ensemble = EnsembleModelForSequenceClassification.from_multiple_pretrained(*cls.model_names)

    def test_ensemble(self):
        batch = ["This is a test sentence", "This is another test sentence."]
        with torch.no_grad():
            output = self.ensemble(batch)
        assert len(output.logits) == len(self.model_names)
        output.stack(self.ensemble.config.weights, "cpu")
        assert len(output.logits) == 1
