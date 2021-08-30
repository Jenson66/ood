#!/usr/bin/env python

import argparse
import os
import pickle

import numpy as np
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_openml

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
import skorch

import ood

AGG_ALG = "foolsgold"
DEVICE = "cpu"


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.ModuleList([
            nn.Linear(784, 5000),
            nn.Sigmoid(),
            nn.Linear(5000, 500),
            nn.Sigmoid(),
            nn.Linear(500, 10),
            nn.Softmax(dim=1)
        ]).eval()

    def forward(self, x):
        for feature in self.features:
            x = feature(x)
        return x

class MNISTData(torch.utils.data.Dataset):
    def __init__(self, train, classes=None):
        ds = fetch_openml("MNIST_784")
        ids, ide = (0, 60_000) if train else (60_001, 70_000)
        self.data = ds['data'].to_numpy().astype(np.float32)[ids:ide]
        self.targets = ds['target'].to_numpy().astype(np.int64)[ids:ide]
        if classes:
            idx = self.targets == classes[0]
            self.data = self.data[idx]
            self.targets = self.targets[idx]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return (self.data[i], self.targets[i])

def load_data(batch_size, train, classes=None):
    return torch.utils.data.DataLoader(MNISTData(train, classes), batch_size=batch_size, shuffle=True, pin_memory=True)


def init_server(num_endpoints, fn, backend='gloo'):
    # Setup server
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=0, world_size=num_endpoints + 1)
    params = [{
        "rounds": 10000,
        "lr": 0.01
    }]
    dist.broadcast_object_list(params, 0)
    sys_data = params[0]
    data = load_data(10_000, False)
    X, y = next(iter(data))
    net = skorch.NeuralNetClassifier(
        MLP,
        lr=sys_data['lr'],
        iterator_train__shuffle=True,
        warm_start=True
    )
    net.initialize()
    agg_data = ood.gen_agg_data(AGG_ALG, net, num_endpoints, DEVICE)

    # Run learning
    for r in range(sys_data["rounds"]):
        fn(net, num_endpoints, kappa=1, histories=agg_data['histories'], device=DEVICE)
        loss = metrics.log_loss(y, net.predict_proba(X))
        accuracy = metrics.accuracy_score(y, net.predict(X))
        print(f"\rRound {r + 1}/{sys_data['rounds']}, loss: {loss}, acc: {accuracy}", end='')
    print()


def init_endpoint(e_id, num_endpoints, fn, data, backend='gloo'):
    # Setup endpoint
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=e_id + 1, world_size=num_endpoints + 1)
    params = [None]
    dist.broadcast_object_list(params, 0)
    sys_data = params[0]
    net = skorch.NeuralNetClassifier(
        MLP,
        lr=sys_data['lr'],
        iterator_train__shuffle=True,
        batch_size=data.batch_size,
        warm_start=True
    )
    net.initialize()

    # Run learning
    for _ in range(sys_data["rounds"]):
        fn(net, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform federated learning")
    parser.add_argument("--endpoints", dest="endpoints", metavar="N", type=int, default=1, help="Number of endpoints/users (default: 1)")
    args = parser.parse_args()

    srv_alg, end_alg = ood.load_algorithm_pair(AGG_ALG)
    processes = [mp.Process(target=init_server, args=(args.endpoints, srv_alg))]
    mp.set_start_method("spawn")
    batch_sizes = [128 for _ in range(args.endpoints)]
    print(f"Starting federated learning training for {args.endpoints} endpoints...")
    for e in range(args.endpoints):
        data = load_data(batch_sizes[e], True, classes=[e])
        processes.append(mp.Process(target=init_endpoint, args=(e, args.endpoints, end_alg, data)))
    for p in processes:
        p.start()
    for p in processes:
        p.join()