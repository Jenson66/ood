from typing import NamedTuple

import torch
import torch.nn as nn

from ood.utils.model import load_model, Model
from ood.utils.dataset import load_data


def benchmark(model: Model, data: torch.utils.data.DataLoader) -> dict:
    """Benchmark the current model based on the data provided

    Parameters
    ----------
    model: utils.Model
        The model to benchmark
    data: torch.utils.data.DataLoader
        The validation data
    """
    criterion = nn.CrossEntropyLoss()
    data_count = 0
    acc = 0
    loss = 0
    for x, y in iter(data):
        x = x.to(model.device)
        y = y.to(model.device)
        output = model(x)
        loss += criterion(output, y)
        acc += (output.argmax(axis=1).cpu() == y).sum()
        data_count += len(y)
    return {"loss": loss / len(data), "accuracy": acc / data_count}


class InstanceData(NamedTuple):
    """Data that is stored by each instance of a node in the system"""
    model: Model
    dataset: torch.utils.data.DataLoader
    total_nodes: int
    other_params: dict
