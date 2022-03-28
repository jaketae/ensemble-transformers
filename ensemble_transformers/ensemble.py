import torch
from transformers import AutoModelForSequenceClassification

from ensemble_transformers.base import EnsembleBaseModel, EnsembleConfig


class EnsembleModelForSequenceClassification(EnsembleBaseModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        for model_name in config.model_names:
            self.models.append(AutoModelForSequenceClassification.from_pretrained(model_name, *args, **kwargs))

    @classmethod
    def from_pretrained(cls, model_names, *args, **kwargs):
        config = EnsembleConfig(model_names=model_names, modality="text")
        return cls(config, *args, **kwargs)

    def forward(
        self,
        text,
        main_device="cpu",
        return_all_outputs=False,
        tokenizer_kwargs={"return_tensors": "pt", "padding": True},
    ):
        outputs = []
        for i, (model, tokenizer) in enumerate(zip(self.models, self.preprocessors)):
            inputs = tokenizer(text, **tokenizer_kwargs).to(self.devices[i])
            output = model(**inputs)
            outputs.append(output)
        if return_all_outputs:
            return outputs
        return torch.stack([output.logits.to(main_device) for output in outputs]).mean(dim=0)
