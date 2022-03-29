import torch

from ensemble_transformers.base import EnsembleBaseModel


class EnsembleModelForSequenceClassification(EnsembleBaseModel):
    def forward(
        self,
        text,
        main_device="cpu",
        return_all_outputs=False,
        preprocessor_kwargs={"return_tensors": "pt", "padding": True},
    ):
        outputs = []
        for i, (model, preprocessor) in enumerate(zip(self.models, self.preprocessors)):
            inputs = preprocessor(text, **preprocessor_kwargs).to(self.devices[i])
            output = model(**inputs)
            outputs.append(output)
        if return_all_outputs:
            return outputs
        return torch.stack([output.logits.to(main_device) for output in outputs]).mean(dim=0)


class EnsembleModelForImageClassification(EnsembleBaseModel):
    def forward(
        self,
        images,
        main_device="cpu",
        return_all_outputs=False,
        preprocessor_kwargs={"return_tensors": "pt"},
    ):
        outputs = []
        for i, (model, preprocessor) in enumerate(zip(self.models, self.preprocessors)):
            inputs = preprocessor(images, **preprocessor_kwargs).to(self.devices[i])
            output = model(**inputs)
            outputs.append(output)
        if return_all_outputs:
            return outputs
        return torch.stack([output.logits.to(main_device) for output in outputs]).mean(dim=0)


class EnsembleModelForAudioClassification(EnsembleBaseModel):
    def forward(
        self,
        audio,
        main_device="cpu",
        return_all_outputs=False,
        preprocessor_kwargs={"return_tensors": "pt", "sampling_rate": None, "padding": "longest"},
    ):
        outputs = []
        for i, (model, preprocessor) in enumerate(zip(self.models, self.preprocessors)):
            inputs = preprocessor(audio, **preprocessor_kwargs).to(self.devices[i])
            output = model(**inputs)
            outputs.append(output)
        if return_all_outputs:
            return outputs
        return torch.stack([output.logits.to(main_device) for output in outputs]).mean(dim=0)
