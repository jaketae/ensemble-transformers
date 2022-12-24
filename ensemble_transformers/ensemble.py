from typing import List, Union

import numpy as np
import torch
from PIL.Image import Image

from ensemble_transformers.base import EnsembleBaseModel


class EnsembleModelForSequenceClassification(EnsembleBaseModel):
    def forward(
        self,
        text: List[str],
        preprocessor_kwargs: dict = {"return_tensors": "pt", "padding": True},
        return_all_outputs: bool = True,
        main_device: Union[str, torch.device] = "cpu",
    ):
        return super().forward(text, preprocessor_kwargs, return_all_outputs, main_device)


class EnsembleModelForImageClassification(EnsembleBaseModel):
    def forward(
        self,
        images: List[Image],
        preprocessor_kwargs: dict = {"return_tensors": "pt"},
        return_all_outputs: bool = True,
        main_device: Union[str, torch.device] = "cpu",
    ):
        return super().forward(images, preprocessor_kwargs, return_all_outputs, main_device)


class EnsembleModelForAudioClassification(EnsembleBaseModel):
    def forward(
        self,
        audio: np.ndarray,
        preprocessor_kwargs: dict = {"return_tensors": "pt", "sampling_rate": None, "padding": "longest"},
        return_all_outputs: bool = True,
        main_device: Union[str, torch.device] = "cpu",
    ):
        return super().forward(audio, preprocessor_kwargs, return_all_outputs, main_device)
