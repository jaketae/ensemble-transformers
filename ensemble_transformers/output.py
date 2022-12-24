from typing import List, Union

import torch


class EnsembleModelOutput:
    def __init__(self, outputs):
        self.outputs = outputs
        self.common_keys = None
        for output in outputs:
            if self.common_keys is None:
                self.common_keys = output.keys()
            else:
                self.common_keys &= output.keys()
        for common_key in self.common_keys:
            setattr(self, common_key, [getattr(output, common_key) for output in outputs])

    def stack(self, weights: List[float], main_device: Union[str, torch.device]) -> None:
        for common_key in self.common_keys:
            setattr(
                self,
                common_key,
                torch.stack(
                    [
                        weight * getattr(output, common_key).to(main_device)
                        for weight, output in zip(weights, self.outputs)
                    ]
                ).sum(dim=0),
            )

    def __repr__(self) -> str:
        result = ["EnsembleModelOutput("]
        for common_key in self.common_keys:
            value = getattr(self, common_key)
            result.append(f"\t{common_key}: {repr(value)},")
        result.append(")")
        return "\n".join(result)
