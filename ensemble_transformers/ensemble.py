from abc import abstractmethod

import torch
from torch import nn

from transformers import AutoModelForSequenceClassification, AutoTokenizer


class EnsembleBaseModel(nn.Module):
    def __init__(self, model_names):
        super().__init__()
        self.devices = ["cpu" for _ in range(len(model_names))]
        self.tokenizers = []
        self.models = nn.ModuleList()
        for model_name in model_names:
            self.tokenizers.append(AutoTokenizer.from_pretrained(model_name))

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


class EnsembleModelForSequenceClassification(EnsembleBaseModel):
    def __init__(self, model_names, *args, **kwargs):
        super().__init__(model_names)
        for model_name in model_names:
            self.models.append(
                AutoModelForSequenceClassification.from_pretrained(model_name, *args, **kwargs)
            )

    def forward(
        self, 
        text,
        main_device="cpu", 
        return_all_outputs=False, 
        tokenizer_kwargs={"return_tensors": "pt", "padding": True}
    ):
        outputs = []
        for i, (model, tokenizer) in enumerate(zip(self.models, self.tokenizers)):
            inputs = tokenizer(text, **tokenizer_kwargs).to(self.devices[i])
            output = model(**inputs)
            outputs.append(output)
        if return_all_outputs:
            return outputs
        return torch.stack([output.logits.to(main_device) for output in outputs]).mean(dim=0)
