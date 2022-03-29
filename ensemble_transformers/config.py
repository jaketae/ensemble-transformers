import warnings
from typing import List

import transformers
from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig


def detect_preprocessor_from_model_name(model_name: str) -> str:
    for preprocessor_class in [AutoTokenizer, AutoFeatureExtractor, AutoProcessor]:
        try:
            _ = preprocessor_class.from_pretrained(model_name)
            return preprocessor_class
        except KeyError:
            continue
    raise ValueError(
        "Unable to auto-detect modality. Please consider opening an issue at https://github.com/jaketae/ensemble-transformers/issues."
    )


def detect_preprocessor_from_auto_class(auto_class: str) -> str:
    auto_class = auto_class.lower()
    for vision_keyword in ["image", "vision", "object", "segmentation"]:
        if vision_keyword in auto_class:
            return AutoFeatureExtractor
    for audio_keyword in ["audio", "ctc", "speech"]:
        if audio_keyword in auto_class:
            return AutoProcessor
    return AutoTokenizer


def check_modalities(model_names):
    return set([detect_preprocessor_from_model_name(model_name) for model_name in model_names])


class EnsembleConfig(PretrainedConfig):
    def __init__(self, auto_class: str, model_names: List[str], *args, **kwargs):
        if len(model_names) == 1:
            warnings.warn(
                "Initializing ensemble with one model. "
                "If this is intended, consider using Hugging Face transformers as-is."
            )
        try:
            self.auto_class = getattr(transformers, auto_class)
        except AttributeError:
            raise ImportError(f"Failed to import `{auto_class}` from `transformers`.")
        preprocessor_classes = check_modalities(model_names)
        if len(preprocessor_classes) > 1:
            raise ValueError("Cannot ensemble models of different modalities.")
        self.model_names = model_names
        self.preprocessor_class = preprocessor_classes.pop()
        preprocessor_class_from_auto_class = detect_preprocessor_from_auto_class(auto_class)
        if self.preprocessor_class != preprocessor_class_from_auto_class:
            raise ValueError(
                f"Expected `auto_class` and `model_names` to point to the same modality, "
                f"but got {preprocessor_class_from_auto_class.__name__} from `auto_class` "
                f"and {self.preprocessor_class.__name__} from `model_names`."
            )
        super().__init__(*args, **kwargs)
