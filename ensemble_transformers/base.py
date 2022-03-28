from abc import abstractmethod

from torch import nn
from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedModel


MODALITY2AUTOCLASS = {
    "text": AutoTokenizer,
    "vision": AutoFeatureExtractor,
    "audio": AutoProcessor,
}


class EnsembleConfig(PretrainedConfig):
    def __init__(self, model_names, modality, *args, **kwargs):
        self.model_names = model_names
        self.modality = modality
        super().__init__(*args, **kwargs)


class EnsembleBaseModel(PreTrainedModel):
    config_class = EnsembleConfig

    def __init__(self, config):
        super().__init__(config)
        self.devices = ["cpu" for _ in range(len(config.model_names))]
        self.preprocessors = []
        self.models = nn.ModuleList()
        preprocessor_class = MODALITY2AUTOCLASS[config.modality]
        for model_name in config.model_names:
            self.preprocessors.append(preprocessor_class.from_pretrained(model_name))

    def to(self, device):
        super().to(device)
        self.devices = [device for _ in range(self.num_models)]

    def to_multiple(self, devices):
        for i, (model, device) in enumerate(zip(self.models, devices)):
            model.to(device)
            self.devices[i] = device

    @property
    def num_models(self):
        return len(self.devices)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
