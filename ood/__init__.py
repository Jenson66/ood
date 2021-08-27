import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist
import torch.multiprocessing as mp

import numpy as np


def flatten_grads(grads, device):
    """Flatten gradients into vectors"""
    with torch.no_grad():
        flat_grads = None
        for i in range(len(grads[0])):
            t = torch.tensor([], device=device)
            for j in range(len(grads)):
                t = torch.cat((t, grads[j][i].flatten()))
            if flat_grads is None:
                flat_grads = t.unsqueeze(0)
            else:
                flat_grads = torch.cat((flat_grads, t.unsqueeze(0)), dim=0)
        return flat_grads


def gen_agg_data(alg_name, net, num_endpoints, device):
    """Generate the data for the aggregation algorithm"""
    agg_data = dict()
    if alg_name in ["foolsgold", "viceroy"]:
        with torch.no_grad():
            flat_params = torch.tensor([]).to(device)
            for p in net.module_.parameters():
                flat_params = torch.cat((flat_params, p.flatten()))
        agg_data["histories"] = {
            i: torch.zeros(
                len(flat_params),
                device="cpu"
            ) for i in range(num_endpoints)
        }
        agg_data["rep"] = torch.zeros(num_endpoints, device="cpu")
    if alg_name == "std-dagmm":
        agg_data["gmm"] = None
    return agg_data


def grads_end(net, data):
    """The endpoint/client end that shares gradients after training"""
    # receive a copy of the global model
    for param in net.module_.parameters():
        dist.broadcast(tensor=param, src=0)
    # do some local training
    net.optimizer_.zero_grad()
    X, y = next(iter(data))
    net.train_step_single(X, y)
    grads = [p.grad for p in net.module_.parameters()]
    net.optimizer_.step()
    # send the results
    for grad in grads:
        dist.gather(grad, dst=0)


def grads_srv(net, total_nodes, fn, **params):
    # send a copy of the global model
    for param in net.module_.parameters():
        dist.broadcast(tensor=param, src=0)
    # get the gradients
    grads = [[torch.zeros(p.shape, dtype=torch.float32) for _ in range(total_nodes + 1)] for p in net.module_.parameters()]
    for i, _ in enumerate(net.module_.parameters()):
        dist.gather(grads[i][0], gather_list=grads[i], dst=0)
        grads[i] = torch.tensor([g.tolist() for g in grads[i][1:]], dtype=torch.float32)
    # aggregate the gradients
    fn(net, grads, **params)


def bs_grads_end(net, data):
    """The endpoint/client end that shares gradients and batch sizes after training"""
    # receive a copy of the global model
    for param in net.module_.parameters():
        dist.broadcast(tensor=param, src=0)
    # do some local training
    net.optimizer_.zero_grad()
    X, y = next(iter(data))
    net.train_step_single(X, y)
    grads = [p.grad for p in net.module_.parameters()]
    net.optimizer_.step()
    # send the results
    dist.gather(torch.tensor([net.batch_size], dtype=float), dst=0)
    for grad in grads:
        dist.gather(grad, dst=0)


def bs_grads_srv(net, total_nodes, fn, **params):
    """The server end of federated averaging"""
    # send a copy of the global model
    for param in net.module_.parameters():
        dist.broadcast(tensor=param, src=0)
    # get the batch sizes
    batch_sizes = [torch.zeros(1, dtype=float) for _ in range(total_nodes + 1)]
    dist.gather(torch.zeros(1, dtype=float), gather_list=batch_sizes, dst=0)
    batch_sizes = torch.tensor([bs.item() for bs in batch_sizes[1:]]) # eliminate this batch size
    # get the gradients
    grads = [[torch.zeros(p.shape, dtype=torch.float32) for _ in range(total_nodes + 1)] for p in net.module_.parameters()]
    for i, _ in enumerate(net.module_.parameters()):
        dist.gather(grads[i][0], gather_list=grads[i], dst=0)
        grads[i] = torch.tensor([g.tolist() for g in grads[i][1:]], dtype=torch.float32)
    # perform federated aggregation
    fn(net, batch_sizes, grads, **params)


def fed_avg(net, batch_sizes, grads):
    total_dc = batch_sizes.sum()
    alpha = batch_sizes / total_dc
    net.optimizer_.zero_grad()
    for param, grad in zip(net.module_.parameters(), grads):
        for a, g in zip(alpha, grad):
            g *= a
        param.grad = grad.sum(dim=0)
    net.optimizer_.step()


def krum(net, grads, clip, device):
    with torch.no_grad():
        flat_grads = flatten_grads(grads, device)
        num_clients = len(flat_grads)
        scores = torch.zeros(num_clients)
        dists = torch.sum(flat_grads**2, axis=1)[:, None] + torch.sum(flat_grads**2, axis=1)[None] - 2 * torch.mm(flat_grads, flat_grads.T)
        for i in range(num_clients):
            scores[i] = torch.sum(
                torch.sort(dists[i]).values[1:num_clients - clip - 1]
            )
        idx = np.argpartition(scores, num_clients - clip)[:num_clients - clip]
        idx = idx.tolist()
    net.optimizer_.zero_grad()
    for k, p in enumerate(net.module_.parameters()):
        for i in idx:
            p.grad = (1/len(idx)) * grads[k][i] + (p.grad if p.grad is not None else 0)
    net.optimizer_.step()

        
def foolsgold(net, grads, kappa, histories, device):
    with torch.no_grad():
        flat_grads = flatten_grads(grads, device)
        idx = list(range(len(flat_grads)))
        num_clients = len(flat_grads)
        cs = torch.tensor(
            [[0 for _ in range(num_clients)] for _ in range(num_clients)],
            dtype=torch.float32
        )
        v = torch.tensor([0 for _ in range(num_clients)], dtype=torch.float32)
        alpha = torch.tensor([0 for _ in range(num_clients)], dtype=torch.float32)
        for i in idx:
            histories[i] += flat_grads[i]
        for i in idx:
            for j in {x for x in idx} - {i}:
                cs[i][j] = torch.cosine_similarity(
                    histories[i],
                    histories[j],
                    dim=0
                )
            v[i] = max(cs[i])
        for i in idx:
            for j in idx:
                if (v[j] > v[i]) and (v[j] != 0):
                    cs[i][j] *= v[i] / v[j]
            alpha[i] = 1 - max(cs[i])
        alpha = alpha / max(alpha)
        ids = alpha != 1
        alpha[ids] = kappa * (
            torch.log(alpha[ids] / (1 - alpha[ids])) + 0.5
        )
        alpha[alpha > 1] = 1
        alpha[alpha < 0] = 0
        alpha[alpha.isnan()] = 0
        alpha = alpha / alpha.sum()
    net.optimizer_.zero_grad()
    for k, p in enumerate(net.module_.parameters()):
        for i in idx:
            p.grad = alpha[i] * grads[k][i] + (p.grad if p.grad is not None else 0)
    net.optimizer_.step()


def viceroy(net, grads, histories, first, rep, kappa, device):
    with torch.no_grad():
        flat_grads = flatten_grads(grads, device)
        idx = list(range(len(flat_grads)))
        num_clients = len(flat_grads)
        alpha = torch.tensor([0 for _ in range(num_clients)], dtype=torch.float32, device=device)
        gamma = torch.tensor([0 for _ in range(num_clients)], dtype=torch.float32, device=device)
        b = torch.tensor([0 for _ in range(num_clients)], dtype=torch.float32, device=device)
        G = torch.zeros(histories[idx[0]].shape, device=device)
        for i in idx:
            if first:
                rep[i] = 1
            else:
                if rep[i] < 0:
                    rep[i] = 0.97**(abs(torch.cosine_similarity(histories[i], flat_grads[i], dim=0))) * rep[i]
                    histories[i] *= 0.9
                else:
                    rep[i] = torch.clip(0.9 * rep[i] + 0.5 *
                            torch.tanh(2 * abs(torch.cosine_similarity(histories[i],
                                flat_grads[i], dim=0)) - 1), 0, 1)
                    histories[i] = 0.9 * histories[i] + flat_grads[i]
                    if rep[i] < 0.1:
                        rep[i] = -1
            G += histories[i]
        G /= len(idx)
        S = kappa * G / 21.85
        B = torch.tensor(0.0, device=device)
        for i in idx:
            if first:
                gamma[i] = 1
            else:
                gamma[i] = 1 - abs(torch.cosine_similarity(histories[i], G, dim=0))
            for j in set(idx) - {i}:
                b[i] += torch.cdist(flat_grads[i].unsqueeze(0), flat_grads[j].unsqueeze(0)).item()
            b[i] /= len(idx) - 1
            B += torch.cdist(S.unsqueeze(0), flat_grads[i].unsqueeze(0)).item()
        B /= len(idx)
        alpha = torch.exp(-(2 / torch.norm(S)) * (b - B)**2)
        alpha = alpha * rep * gamma
        alpha[alpha > 1] = 1
        alpha[alpha < 0] = 0
        alpha[alpha.isnan()] = 0
        alpha = alpha / alphs if (alphs := alpha.sum()) > 0 else torch.zeros(alpha.shape)
    net.optimizer_.zero_grad()
    for k, p in enumerate(net.module_.parameters()):
        for i in idx:
            p.grad = alpha[i] * grads[k][i] + (p.grad if p.grad is not None else 0)
    net.optimizer_.step()
    return rep


# STD-DAGMM
class STD_DAGMM(nn.Module):
    """
    Based on https://github.com/danieltan07/dagmm
    and https://github.com/datamllab/pyodds
    """
    def __init__(self, in_len, device, n_gmm=2, latent_dim=4):
        super().__init__()
        # AC encode
        self.encoder = nn.ModuleList([
            nn.Linear(in_len, 60),
            nn.ReLU(),
            nn.Linear(60, 30),
            nn.ReLU(),
            nn.Linear(30, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        ]).eval()
        # AC decode
        self.decoder = nn.ModuleList([
            nn.Linear(1, 10),
            nn.Tanh(),
            nn.Linear(10, 30),
            nn.Tanh(),
            nn.Linear(30, 60),
            nn.Tanh(),
            nn.Linear(60, in_len)
        ]).eval()
        # GMM
        self.estimator = nn.ModuleList([
            nn.Linear(latent_dim, 10),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(10, n_gmm),
            nn.Softmax(dim=1)
        ]).eval()
        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm, latent_dim))
        self.register_buffer("cov", torch.zeros(n_gmm, latent_dim, latent_dim))
        # Other configuration
        self.device = device
        self.optimizer = optim.Adam(
            self.parameters(),
            weight_decay=0.0001
        )
        self.to(device)

    def to_var(self, x):
        return Variable(x).to(self.device)

    def relative_euclidean_distance(self, a, b, dim=1):
        return (a - b).norm(2, dim=dim) / torch.clamp(a.norm(2, dim=dim), min=1e-10)

    def encode(self, x):
        for f in self.encoder:
            x = f(x)
        return x

    def decode(self, x):
        for f in self.decoder:
            x = f(x)
        return x

    def estimate(self, x):
        for f in self.estimator:
            x = f(x)
        return x

    def forward(self, x):
        enc = self.encode(x)
        dec = self.decode(enc)
        rec_cosine = F.cosine_similarity(
            x.view(x.shape[0], -1), dec.view(dec.shape[0], -1), dim=1
        )
        rec_euclidean = self.relative_euclidean_distance(
            x.view(x.shape[0], -1), dec.view(dec.shape[0], -1), dim=1
        )
        rec_std = torch.std(
            x.view(x.shape[0], -1), dim=1
        )
        z = torch.cat(
            [
                enc,
                rec_euclidean.unsqueeze(-1),
                rec_cosine.unsqueeze(-1),
                rec_std.unsqueeze(-1)
            ],
            dim=1
        )
        gamma = self.estimate(z)
        return enc, dec, z, gamma

    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)
        sum_gamma = torch.sum(gamma, dim=0)
        phi = (sum_gamma / N)
        self.phi = phi.data
        mu = torch.sum(
            gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0
        ) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        cov = torch.sum(
            gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=0
        ) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data
        return phi, mu, cov

    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        if phi is None:
            phi = Variable(self.phi)
        if mu is None:
            mu = Variable(self.mu)
        if cov is None:
            cov = Variable(self.cov)
        k, d, _ = cov.size()
        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        for i in range(k):
            cov_k = cov[i] + self.to_var(torch.eye(d) * eps)
            pinv = np.linalg.pinv(cov_k.data.cpu().numpy())
            cov_inverse.append(Variable(torch.from_numpy(pinv)).unsqueeze(0))
            eigvals = np.linalg.eigvals(cov_k.data.cpu().numpy() * (2 * np.pi))
            determinant = np.prod(
                np.clip(eigvals, a_min=sys.float_info.epsilon, a_max=None)
            )
            det_cov.append(determinant)
            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())
        cov_inverse = torch.cat(cov_inverse, dim=0).to(self.device)
        det_cov = Variable(torch.from_numpy(np.float32(np.array(det_cov))))
        exp_term_tmp = -0.5 * torch.sum(
            torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0),
            dim=-2
        ) * z_mu, dim=-1)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]
        exp_term = torch.exp(exp_term_tmp - max_val)
        sample_energy = -max_val.squeeze() - torch.log(
                torch.sum(
                    self.to_var(phi.unsqueeze(0)) * exp_term /
                    (torch.sqrt(self.to_var(det_cov)) + eps).unsqueeze(0),
                    dim=1
                ) + eps
        )
        if size_average:
            sample_energy = torch.mean(sample_energy)
        return sample_energy, cov_diag

    def loss_function(self, x, x_hat, z, gamma, lambda_energy, lambda_cov_diag):
        recon_error = torch.mean((x.view(*x_hat.shape) - x_hat) ** 2)
        phi, mu, cov = self.compute_gmm_params(z, gamma)
        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)
        loss = recon_error + \
            lambda_energy * sample_energy + \
            lambda_cov_diag * cov_diag
        return loss, sample_energy, recon_error, cov_diag

    def predict(self, X):
        E = torch.tensor([], device=self.device)
        for x in X:
            _, _, z, _ = self(x.unsqueeze(0))
            e, _ = self.compute_energy(z, size_average=False)
            E = torch.cat((E, e))
        return E

    def fit(self, x, epochs=1):
        for i in range(epochs):
            self.train()
            enc, dec, z, gamma = self(x)
            loss, sample_energy, recon_error, cov_diag = self.loss_function(
                x, dec, z, gamma, 0.1, 0.005
            )
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
            self.optimizer.step()


def std_dagmm(net, batch_sizes, grads, gmm, device):
    flat_grads = flatten_grads(grads, device)
    num_clients = len(flat_grads)
    if gmm is None:
        gmm = STD_DAGMM(
            flat_grads.shape[1],
            device
        )
    gmm.fit(flat_grads)
    energies = gmm.predict(flat_grads)
    std = torch.std(energies).item()
    avg = torch.mean(energies).item()
    indices = (energies >= avg - std) * (energies <= avg + std)
    idx = torch.tensor(range(num_clients))[indices].tolist()
    with torch.no_grad():
        total_dc = sum([batch_sizes[i] for i in idx])
        alpha = torch.tensor([0.0 for _ in grads])
        net.optimizer_.zero_grad()
        for i in idx:
            alpha[i] = batch_sizes[i] / total_dc
            for k, p in enumerate(net.module_.parameters()):
                p.grad = alpha[i] * grads[k][i] + (p.grad if p.grad is not None else 0)
        net.optimizer_.step()
    return gmm


# Server functions defined for the sake of pickling

def fed_avg_srv(net, total_nodes):
    return bs_grads_srv(net, total_nodes, fed_avg)

def krum_srv(net, total_nodes, clip, device):
    return grads_srv(net, total_nodes, krum, clip=clip, device=device)

def foolsgold_srv(net, total_nodes, kappa, histories, device):
    return grads_srv(net, total_nodes, foolsgold, kappa=kappa, histories=histories, device=device)

def viceroy_srv(net, total_nodes, kappa, histories, rep, first, device):
    return grads_srv(net, total_nodes, viceroy, kappa=kappa, histories=histories, rep=rep, first=first, device=device)

def std_dagmm_srv(net, total_nodes, gmm, device):
    return bs_grads_srv(net, total_nodes, std_dagmm, gmm=gmm, device=device)


# Algorithm loader
def load_algorithm_pair(alg_name):
    """Load a federated learning aggregation algorithm pair as (server, endpoint)"""
    alg_pairs = {
        "federated averaging": (fed_avg_srv, bs_grads_end),
        "krum": (krum_srv, grads_end),
        "foolsgold": (foolsgold_srv, grads_end),
        "viceroy": (viceroy_srv, grads_end),
        "std-dagmm": (std_dagmm_srv, bs_grads_end)
    }
    if (alg_pair := alg_pairs.get(alg_name)) is None:
        raise Exception(
            f"Model '{alg_name}' does not exist, " +
            f"possible options: {set(alg_pairs.keys())}"
        )
    return alg_pair
