import warnings
from typing import List, Optional

import transformers
from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig


def detect_preprocessor_from_model_name(model_name: str) -> str:
    for preprocessor_class in [AutoFeatureExtractor, AutoProcessor, AutoTokenizer]:
        try:
            _ = preprocessor_class.from_pretrained(model_name)
            return preprocessor_class
        except (OSError, KeyError, ValueError):
            continue
    raise ValueError(
        "Unable to auto-detect preprocessor class. Please consider opening an issue at https://github.com/jaketae/ensemble-transformers/issues."
    )


def check_modalities(model_names: List[str]) -> set:
    return set([detect_preprocessor_from_model_name(model_name) for model_name in model_names])


class EnsembleConfig(PretrainedConfig):
    def __init__(
        self,
        auto_class: str,
        model_names: List[str],
        weights: Optional[List[float]] = None,
        *args,
        **kwargs,
    ) -> PretrainedConfig:
        num_models = len(model_names)
        if num_models == 1:
            warnings.warn(
                "Initializing ensemble with one model. "
                "If this is intended, consider using Hugging Face transformers as-is."
            )
        try:
            self.auto_class = getattr(transformers, auto_class)
        except AttributeError:
            raise ImportError(f"Failed to import `{auto_class}` from Hugging Face transformers.")
        preprocessor_classes = check_modalities(model_names)
        if len(preprocessor_classes) > 1:
            raise ValueError("Cannot ensemble models of different modalities.")
        if weights is not None:
            if len(weights) != num_models:
                raise ValueError(
                    f"Expected `weights` to contain {num_models} elements, "
                    f"but got {len(weights)} elements instead."
                )
            weight_sum = sum(weights)
            if weight_sum != 1:
                warnings.warn("Normalizing `weights` to sum to 1.")
                self.weights = [weight / weight_sum for weight in weights]
            else:
                self.weights = weights
        else:
            self.weights = [1 / num_models for _ in range(num_models)]
        self.model_names = model_names
        self.preprocessor_class = preprocessor_classes.pop()
        super().__init__(*args, **kwargs)
