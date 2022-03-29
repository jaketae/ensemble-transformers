# Ensemble Transformers

Ensembling Hugging Face Transformers made easy!

## Why Ensemble Transformers?

Ensembling is a simple yet powerful way of combining predictions from different models to increase performance. Since multiple models are used to derive a prediction, ensembling offers a way of decreasing variance and increasing robustness. Ensemble Transformers  provides an intuitive interface for ensembling pretrained models available in Hugging Face [`transformers`](https://huggingface.co/docs/transformers/index).

## Installation

Ensemble Transformers is available on [PyPI](https://pypi.org/project/ensemble-transformers/) and can easily be installed with the `pip` package manager.

```
pip install ensemble-transformers
```

## Getting Started

### Declaring an Ensemble

To declare an ensemble, first create a configuration object specifying the Hugging Face transformers auto class, as well as the list of models to use to create the ensemble. 

```python
from ensemble_transformers import EnsembleConfig, EnsembleModelForSequenceClassification

config = EnsembleConfig(
    "AutoModelForSequenceClassification", 
    model_names=["bert-base-uncased", "distilroberta-base", "xlnet-base-cased"]
)
```

The ensemble model can then be declared via 

```python
ensemble = EnsembleModelForSequenceClassification(config)
```

### `from_multiple_pretrained`

A more convenient way of declaring an ensemble is via `from_multiple_pretrained`, a method similar to `from_pretrained` in Hugging Face transformers. For instance, to perform text classification, we can use the `EnsembleModelForSequenceClassification` class.

```python
from ensemble_transformers import EnsembleModelForSequenceClassification

ensemble = EnsembleModelForSequenceClassification.from_multiple_pretrained(
    "bert-base-uncased", "distilroberta-base", "xlnet-base-cased"
)
```

Unlike Hugging Face transformers, which requires users to explicitly declare and initialize a preprocessor (e.g. `tokenizer`, `feature_extractor`, or `processor`) separate from the model, Ensemble Transformers automatically detects the preprocessor class and holds it within the `EnsembleModelForX` class as an internal attribute. Therefore, you do not have to declare a preprocessor yourself; Ensemble Transformers will do it for you.

In the example below, we see that the `ensemble` object correctly holds 3 tokenizers for each model.

```python
>>> len(ensemble.preprocessors)
3
>>> ensemble.preprocessors
[PreTrainedTokenizerFast(name_or_path='bert-base-uncased', vocab_size=30522, model_max_len=512, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}), PreTrainedTokenizerFast(name_or_path='distilroberta-base', vocab_size=50265, model_max_len=512, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'sep_token': '</s>', 'pad_token': '<pad>', 'cls_token': '<s>', 'mask_token': AddedToken("<mask>", rstrip=False, lstrip=True, single_word=False, normalized=False)}), PreTrainedTokenizerFast(name_or_path='xlnet-base-cased', vocab_size=32000, model_max_len=1000000000000000019884624838656, is_fast=True, padding_side='left', truncation_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'sep_token': '<sep>', 'pad_token': '<pad>', 'cls_token': '<cls>', 'mask_token': AddedToken("<mask>", rstrip=False, lstrip=True, single_word=False, normalized=False), 'additional_special_tokens': ['<eop>', '<eod>']})]
```

### Note on Heterogenous Modality

For the majority of use cases, it does not make sense to ensemble models from different modalities, e.g., a language model and an image model. As mentioned, Ensemble Transformers will auto-detect the modality of each model and prevent unintended mixing of models.

```python
>>> from ensemble_transformers import EnsembleConfig
>>> config = EnsembleConfig("AutoModelForSequenceClassification", model_names=["bert-base-uncased", "google/vit-base-patch16-224-in21k"])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/Users/jaketae/Documents/Dev/github/ensemble-transformers/ensemble_transformers/config.py", line 37, in __init__
    raise ValueError("Cannot ensemble models of different modalities.")
ValueError: Cannot ensemble models of different modalities.
```

### Forward Propagation

To run forward propagation, simply pass a batch of raw input to the ensemble. In the case of language models, this is just a batch of text.

```python
>>> batch = ["This is a test sentence", "This is another test sentence."]
>>> ensemble(batch)
tensor([[ 0.2858, -0.0892],
        [ 0.2437, -0.0338]], grad_fn=<MeanBackward1>)
```

By default, the ensemble returns the mean logits, that is, logits from each model is averaged to produce a final pooled output. If you want to obtain the output from each model, you can set `return_all_outputs=True` in the forward call.

```python
>>> ensemble(batch, return_all_outputs=True)
[SequenceClassifierOutput(loss=None, logits=tensor([[ 0.1681, -0.3470],
        [ 0.1573, -0.1571]], grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None), SequenceClassifierOutput(loss=None, logits=tensor([[ 0.1388, -0.0711],
        [ 0.1429, -0.0841]], grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None), XLNetForSequenceClassificationOutput(loss=None, logits=tensor([[0.5506, 0.1506],
        [0.4308, 0.1397]], grad_fn=<AddmmBackward0>), mems=(tensor([[[ 0.0344,  0.0202,  0.0261,  ..., -0.0175, -0.0343,  0.0252],
         [-0.0281, -0.0198, -0.0387,  ..., -0.0420, -0.0160, -0.0253]],
       ...,
        [[ 0.2468, -0.4007, -1.0839,  ..., -0.2943, -0.3944,  0.0605],
         [ 0.1970,  0.2106, -0.1448,  ..., -0.6331, -0.0655,  0.7427]]])), hidden_states=None, attentions=None)]
```

In the example above, we obtain a list containing three `SequenceClassifierOutput`s, one for each model.

Preprocessors accept a number of optional arguments. For instance, for simple batching, `padding=True` is used. Moreover, PyTorch models require `return_tensors="pt"`. Ensemble Transformers already ships with minimal, sensible defaults so that it works out-of-the-box. However, for more custom behavior, you can modify the `preprocessor_kwargs` argument. The example below demonstrates how to use TensorFlow language models without padding.

```python
ensemble(batch, preprocessor_kwargs={"return_tensors": "tf", "padding": False})
```

## Contributing

This repository is under active development. Any and all issues and pull requests are welcome. If you would prefer, feel free to reach out to me at jaesungtae@gmail.com.

## License

Released under the MIT License.