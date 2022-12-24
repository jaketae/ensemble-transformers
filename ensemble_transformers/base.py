from typing import List, Optional, Union

import torch
from torch import nn
from transformers import PreTrainedModel

from .config import EnsembleConfig
from .output import EnsembleModelOutput


class EnsembleBaseModel(PreTrainedModel):
    config_class = EnsembleConfig

    def __init__(self, config: EnsembleConfig, *args, **kwargs):
        super().__init__(config)
        self.num_models = len(config.model_names)
        self.devices = ["cpu" for _ in range(self.num_models)]
        self.preprocessors = []
        self.models = nn.ModuleList()
        for model_name in config.model_names:
            self.preprocessors.append(config.preprocessor_class.from_pretrained(model_name))
            self.models.append(config.auto_class.from_pretrained(model_name, *args, **kwargs))

    def to(self, device: Union[str, torch.device]) -> None:
        super().to(device)
        self.devices = [device for _ in range(self.num_models)]

    def to_multiple(self, devices: List[Union[str, torch.device]]) -> None:
        if len(devices) != self.num_models:
            raise ValueError(f"Expected {self.num_models} devices, but got {len(devices)} instead.")
        for i, (model, device) in enumerate(zip(self.models, devices)):
            model.to(device)
            self.devices[i] = device

    @classmethod
    def from_multiple_pretrained(
        cls, *model_names: str, weights: Optional[List[float]] = None, **kwargs
    ) -> PreTrainedModel:
        class_name = cls.__name__
        if "For" not in class_name:
            raise RuntimeError(
                "`EnsembleBaseModel` is not designed to be instantiated using `from_multiple_pretrained(model_names)`."
            )
        _, suffix = class_name.split("For")
        auto_class = f"AutoModelFor{suffix}"
        config = EnsembleConfig(auto_class, model_names, weights=weights)
        return cls(config, **kwargs)

    def forward(
        self, inputs, preprocessor_kwargs: dict, mean_pool: bool, main_device: Union[str, torch.device]
    ) -> EnsembleModelOutput:
        outputs = []
        for i, (model, preprocessor) in enumerate(zip(self.models, self.preprocessors)):
            preprocessed = preprocessor(inputs, **preprocessor_kwargs).to(self.devices[i])
            output = model(**preprocessed)
            outputs.append(output)
        ensemble_output = EnsembleModelOutput(outputs)
        if not mean_pool:
            return ensemble_output
        ensemble_output.stack(self.config.weights, main_device)
        return ensemble_output
