#!/usr/bin/env python

import argparse
import os
import pickle

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import ood


def save_checkpoint(filename, sys_data, instance_data, data, num_endpoints, r, stats=None):
    if not os.path.exists(filename[:filename.rfind('/')]):
        os.makedirs(filename[:filename.rfind('/')])
    torch.save(
        {
            "sys_data": sys_data,
            "round": r,
            "model_state_dict": instance_data.model.state_dict(),
            "optimizer_state_dict": instance_data.model.optimizer.state_dict(),
            "num_endpoints": num_endpoints,
            "data": data,
            "stats": stats,
            "other_params": instance_data.other_params
        },
        filename
    )


def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    sys_data = checkpoint['sys_data']
    data = checkpoint['data']
    sys_data['num_in'] = data['x_dim']
    sys_data['num_out'] = data['y_dim']
    instance_data = ood.utils.InstanceData(
        ood.utils.load_model(sys_data['architecture'], sys_data['lr'], sys_data['device'], sys_data),
        data['dataloader'],
        checkpoint['num_endpoints'],
        checkpoint['other_params']
    )
    instance_data.model.load_state_dict(checkpoint['model_state_dict'])
    instance_data.model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return sys_data, instance_data, data, checkpoint['round'], checkpoint["stats"]


def init_server(num_endpoints, fn, batch_size, checkpoint_round=1, checkpoint_fn=None, backend='gloo'):
    # Setup server
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=0, world_size=num_endpoints + 1)
    if checkpoint_fn is None:
        params = [{
            "data": "mnist",
            "rounds": 100,
            "architecture": "softmax",
            "device": "cpu",
            "lr": 0.01,
            "params_mul": 10,
            "checkpoint": checkpoint_round,
            "delay": -1
        }]
        dist.broadcast_object_list(params, 0)
        sys_data = params[0]
        data = ood.utils.load_data(sys_data['data'], batch_size, train=False)
        sys_data['num_in'] = data['x_dim']
        sys_data['num_out'] = data['y_dim']
        instance_data = ood.utils.InstanceData(
            ood.utils.load_model(sys_data['architecture'], sys_data['lr'], sys_data['device'], sys_data),
            data['dataloader'],
            num_endpoints,
            {"clip": 5, "kappa": 1}
        )
        start = 0
        stats = {}
    else:
        sys_data, instance_data, data, start, stats = load_checkpoint(checkpoint_fn)
        print(f"Loaded checkpoint, now starting training from round {start}")

    instance_data.other_params['histories'] = {
        i: torch.zeros(
            len(ood.flatten_params(instance_data.model.get_params(), instance_data.model.device)),
            device=instance_data.model.device
        ) for i in range(instance_data.total_nodes)
    }
    instance_data.other_params['rep'] = torch.zeros(instance_data.total_nodes, device=instance_data.model.device)
    instance_data.other_params['gmm'] = None
    instance_data.other_params['first'] = True

    # Run learning
    for r in range(start, sys_data["rounds"]):
        if r < sys_data['delay']:
            ood.fed_avg_srv(instance_data)
        else:
            fn(instance_data)
        bm = ood.utils.benchmark(instance_data.model, instance_data.dataset)
        for k, v in bm.items():
            if stats.get(k) is None:
                stats[k] = [v.item()]
            else:
                stats[k].append(v.item())
        print(f"\rRound {r + 1}/{sys_data['rounds']} loss: {bm['loss']}, acc: {bm['accuracy']}", end='')
        if sys_data['checkpoint'] > 0 and (r % sys_data['checkpoint']) == 0:
            print()
            filename = f"checkpoints/server/round_{r}"
            save_checkpoint(filename, sys_data, instance_data, data, num_endpoints, r, stats)
            print(f"Saved current checkpoint to {filename}")
    print()
    with open("results.pkl", "wb") as f:
        pickle.dump(stats, f)
    print("Saved results to results.pkl")


def init_endpoint(e_id, num_endpoints, fn, batch_size, checkpoint_fn=None, backend='gloo'):
    # Setup endpoint
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=e_id + 1, world_size=num_endpoints + 1)
    if checkpoint_fn is None:
        params = [None]
        dist.broadcast_object_list(params, 0)
        sys_data = params[0]
        data = ood.utils.load_data(sys_data['data'], batch_size, train=True, classes=[e_id])
        sys_data['num_in'] = data['x_dim']
        sys_data['num_out'] = data['y_dim']
        instance_data = ood.utils.InstanceData(
            ood.utils.load_model(sys_data['architecture'], sys_data['lr'], sys_data['device'], sys_data),
            data['dataloader'],
            num_endpoints,
            { "epochs": 1, "verbose": False}
        )
        start = 0
    else:
        sys_data, instance_data, data, start, _ = load_checkpoint(checkpoint_fn)

    # Run learning
    for r in range(start, sys_data["rounds"]):
        if r < sys_data['delay']:
            ood.bs_grads_end(instance_data)
        else:
            fn(instance_data)
        if sys_data['checkpoint'] > 0 and (r % sys_data['checkpoint']) == 0:
            filename = f"checkpoints/endpoint_{e_id}/round_{r}"
            save_checkpoint(filename, sys_data, instance_data, data, num_endpoints, r)


def init_adversary(e_id, num_endpoints, fn, batch_size, checkpoint_fn=None, backend='gloo'):
    # Setup adversary
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=e_id + 1, world_size=num_endpoints + 1)
    if checkpoint_fn is None:
        params = [None]
        dist.broadcast_object_list(params, 0)
        sys_data = params[0]
        data = ood.utils.load_data(sys_data['data'], batch_size, train=True, classes=[5])
        data['dataloader'].dataset.targets[:] = 2
        sys_data['num_in'] = data['x_dim']
        sys_data['num_out'] = data['y_dim']
        instance_data = ood.utils.InstanceData(
            ood.utils.load_model(sys_data['architecture'], sys_data['lr'], sys_data['device'], sys_data),
            data['dataloader'],
            num_endpoints,
            { "epochs": 1, "verbose": False}
        )
        start = 0
    else:
        sys_data, instance_data, data, start, _ = load_checkpoint(checkpoint_fn)

    # Run learning
    for r in range(start, sys_data["rounds"]):
        if r < sys_data['delay']:
            ood.bs_grads_end(instance_data)
        else:
            fn(instance_data)
        if sys_data['checkpoint'] > 0 and (r % sys_data['checkpoint']) == 0:
            filename = f"checkpoints/endpoint_{e_id}/round_{r}"
            save_checkpoint(filename, sys_data, instance_data, data, num_endpoints, r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform federated learning")
    parser.add_argument("--checkpoint", dest="checkpoint", metavar="C", type=int, default=0, help="Checkpoint at every C rounds (default: no checkpointing)")
    parser.add_argument("--endpoints", dest="endpoints", metavar="N", type=int, default=1, help="Number of endpoints/users (default: 1)")
    parser.add_argument("--load", dest="load", metavar="R", type=int, default=-1, help="Load a checkpoint from round R (default: do not load a checkpoint)")
    parser.add_argument("--adversaries", dest="adversaries", metavar="M", type=int, default=1, help="Number of adversaries (default: 0)")
    args = parser.parse_args()

    srv_alg, end_alg = ood.load_algorithm_pair("foolsgold")
    total_nodes = args.endpoints + args.adversaries
    if args.load < 0:
        processes = [mp.Process(target=init_server, args=(total_nodes, srv_alg, 128, args.checkpoint))]
    else:
        processes = [mp.Process(target=init_server, args=(total_nodes, srv_alg, 128, args.checkpoint, f"checkpoints/server/round_{args.load}"))]
    mp.set_start_method("spawn")
    batch_sizes = [128 for _ in range(total_nodes)]
    print(f"Starting federated learning training for {total_nodes} endpoints...")
    for e in range(args.endpoints):
        if args.load < 0:
            processes.append(mp.Process(target=init_endpoint, args=(e, total_nodes, end_alg, batch_sizes[e])))
        else:
            processes.append(mp.Process(target=init_endpoint, args=(e, total_nodes, end_alg, batch_sizes[e], f"checkpoints/endpoint_{e}/round_{args.load}")))
    for a in range(args.adversaries):
        e = args.endpoints + a
        if args.load < 0:
            processes.append(mp.Process(target=init_adversary, args=(e, total_nodes, end_alg, batch_sizes[e])))
        else:
            processes.append(mp.Process(target=init_adversary, args=(e, total_nodes, end_alg, batch_sizes[e], f"checkpoints/endpoint_{e}/round_{args.load}")))
    for p in processes:
        p.start()
    for p in processes:
        p.join()
