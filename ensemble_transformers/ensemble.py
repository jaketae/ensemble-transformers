import torch

from ensemble_transformers.base import EnsembleBaseModel


class EnsembleModelForSequenceClassification(EnsembleBaseModel):
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
