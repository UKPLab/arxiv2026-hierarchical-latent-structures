import torch
from typing import Any


class NNSightModelWrapper:
    def __init__(self, model, nnsight_model):
        self.model = model
        self.nnsight_model = nnsight_model

    def save_mlp_activation(self, layer_idx: int, position: int = -1) -> Any:
        mlp_output = self.nnsight_model.model.layers[layer_idx].mlp.output[0]

        if position == -1:
            activation = mlp_output[:, -1, :]
        else:
            activation = mlp_output[:, position, :]

        return activation.save()


def wrap_model_for_nnsight(model, nnsight_model) -> NNSightModelWrapper:
    return NNSightModelWrapper(model, nnsight_model)
