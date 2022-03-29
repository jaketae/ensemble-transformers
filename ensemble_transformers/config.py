import warnings
from typing import List

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
    def __init__(self, auto_class: str, model_names: List[str], *args, **kwargs) -> PretrainedConfig:
        if len(model_names) == 1:
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
        self.model_names = model_names
        self.preprocessor_class = preprocessor_classes.pop()
        super().__init__(*args, **kwargs)
