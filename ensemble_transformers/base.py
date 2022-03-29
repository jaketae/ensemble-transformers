from abc import abstractmethod
from typing import List, Union

import torch
from torch import nn
from transformers import PreTrainedModel

from .config import EnsembleConfig


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
        for i, (model, device) in enumerate(zip(self.models, devices)):
            model.to(device)
            self.devices[i] = device

    @classmethod
    def from_multiple_pretrained(cls, *model_names: str, **kwargs) -> PreTrainedModel:
        class_name = cls.__name__
        if "For" not in class_name:
            raise RuntimeError(
                "`EnsembleBaseModel` is not designed to be instantiated using `from_multiple_pretrained(model_names)`."
            )
        _, suffix = class_name.split("For")
        auto_class = f"AutoModelFor{suffix}"
        config = EnsembleConfig(auto_class, model_names)
        return cls(config, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
