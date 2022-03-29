from typing import List, Union

import numpy as np
import torch
from PIL.Image import Image

from ensemble_transformers.base import EnsembleBaseModel


class EnsembleModelForSequenceClassification(EnsembleBaseModel):
    def forward(
        self,
        text: List[str],
        main_device: Union[str, torch.device] = "cpu",
        return_all_outputs: bool = False,
        preprocessor_kwargs: dict = {"return_tensors": "pt", "padding": True},
    ):
        outputs = []
        for i, (model, preprocessor) in enumerate(zip(self.models, self.preprocessors)):
            inputs = preprocessor(text, **preprocessor_kwargs).to(self.devices[i])
            output = model(**inputs)
            outputs.append(output)
        if return_all_outputs:
            return outputs
        return torch.stack(
            [weight * output.logits.to(main_device) for weight, output in zip(self.config.weights, outputs)]
        ).sum(dim=0)


class EnsembleModelForImageClassification(EnsembleBaseModel):
    def forward(
        self,
        images: List[Image],
        main_device: Union[str, torch.device] = "cpu",
        return_all_outputs: bool = False,
        preprocessor_kwargs: dict = {"return_tensors": "pt"},
    ):
        outputs = []
        for i, (model, preprocessor) in enumerate(zip(self.models, self.preprocessors)):
            inputs = preprocessor(images, **preprocessor_kwargs).to(self.devices[i])
            output = model(**inputs)
            outputs.append(output)
        if return_all_outputs:
            return outputs
        return torch.stack(
            [weight * output.logits.to(main_device) for weight, output in zip(self.config.weights, outputs)]
        ).sum(dim=0)


class EnsembleModelForAudioClassification(EnsembleBaseModel):
    def forward(
        self,
        audio: np.ndarray,
        main_device: Union[str, torch.device] = "cpu",
        return_all_outputs: bool = False,
        preprocessor_kwargs: dict = {"return_tensors": "pt", "sampling_rate": None, "padding": "longest"},
    ):
        outputs = []
        for i, (model, preprocessor) in enumerate(zip(self.models, self.preprocessors)):
            inputs = preprocessor(audio, **preprocessor_kwargs).to(self.devices[i])
            output = model(**inputs)
            outputs.append(output)
        if return_all_outputs:
            return outputs
        return torch.stack(
            [weight * output.logits.to(main_device) for weight, output in zip(self.config.weights, outputs)]
        ).sum(dim=0)
