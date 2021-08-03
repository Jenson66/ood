import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist
import torch.multiprocessing as mp

import numpy as np

from ood.utils.errors import MisconfigurationError


def flatten_grads(grads, device):
    """Flatten gradients into vectors"""
    with torch.no_grad():
        flat_grads = None
        for i in range(len(grads[0])):
            t = torch.tensor([]).to(device)
            for j in range(len(grads)):
                t = torch.cat((t, grads[j][i].flatten()))
            if flat_grads is None:
                flat_grads = t.unsqueeze(0)
            else:
                flat_grads = torch.cat((flat_grads, t.unsqueeze(0)), dim=0)
        return flat_grads


def flatten_params(params, device):
    """Flatten params into a vector"""
    with torch.no_grad():
        flat_params = torch.tensor([]).to(device)
        for p in params:
            flat_params = torch.cat((flat_params, p.flatten()))
        return flat_params


def grads_end(instance_data):
    """The endpoint/client end that shares gradients after training"""
    # receive a copy of the global model
    for param in instance_data.model.get_params():
        dist.broadcast(tensor=param, src=0)
    # do some local training
    _, grads = instance_data.model.fit(instance_data.dataset, epochs=instance_data.other_params["epochs"], verbose=instance_data.other_params["verbose"])
    # send the results
    for grad in grads:
        dist.gather(grad, dst=0)


def bs_grads_end(instance_data):
    """The endpoint/client end that shares gradients and batch sizes after training"""
    # receive a copy of the global model
    for param in instance_data.model.get_params():
        dist.broadcast(tensor=param, src=0)
    # do some local training
    batch_size, grads = instance_data.model.fit(instance_data.dataset, epochs=instance_data.other_params["epochs"], verbose=instance_data.other_params["verbose"])
    # send the results
    dist.gather(torch.tensor([batch_size], dtype=float), dst=0)
    for grad in grads:
        dist.gather(grad, dst=0)

# Federated averaging

def fed_avg_srv(instance_data):
    """The server end of federated averaging"""
    with torch.no_grad():
        # send a copy of the global model
        for param in instance_data.model.get_params():
            dist.broadcast(tensor=param, src=0)
        # get the batch sizes
        batch_sizes = [torch.zeros(1, dtype=float) for _ in range(instance_data.total_nodes + 1)]
        dist.gather(torch.zeros(1, dtype=float), gather_list=batch_sizes, dst=0)
        batch_sizes = torch.tensor([bs.item() for bs in batch_sizes[1:]]) # eliminate this batch size
        batch_sizes /= batch_sizes.sum()
        # get the gradients
        grads = [[torch.zeros(p.shape, dtype=torch.float32) for _ in range(instance_data.total_nodes + 1)] for p in instance_data.model.get_params()]
        for i, _ in enumerate(instance_data.model.get_params()):
            dist.gather(grads[i][0], gather_list=grads[i], dst=0)
            grads[i] = torch.tensor([g.tolist() for g in grads[i][1:]], dtype=torch.float32)
        # perform federated averaging
        for param, grad in zip(instance_data.model.get_params(), grads):
            for b, g in zip(batch_sizes, grad):
                g *= b
            param.data.add_(grad.sum(dim=0))


# Krum

def krum_srv(instance_data):
    with torch.no_grad():
         # send a copy of the global model
        for param in instance_data.model.get_params():
            dist.broadcast(tensor=param, src=0)
        # get the gradients
        grads = [[torch.zeros(p.shape, dtype=torch.float32) for _ in range(instance_data.total_nodes + 1)] for p in instance_data.model.get_params()]
        for i, _ in enumerate(instance_data.model.get_params()):
            dist.gather(grads[i][0], gather_list=grads[i], dst=0)
            grads[i] = torch.tensor([g.tolist() for g in grads[i][1:]], dtype=torch.float32)
        # perform krum
        clip = instance_data.other_params["clip"]
        flat_grads = flatten_grads(grads, instance_data.model.device)
        num_clients = len(flat_grads)
        scores = torch.zeros(num_clients)
        dists = torch.sum(flat_grads**2, axis=1)[:, None] + torch.sum(flat_grads**2, axis=1)[None] - 2 * torch.mm(flat_grads, flat_grads.T)
        for i in range(num_clients):
            scores[i] = torch.sum(
                torch.sort(dists[i]).values[1:num_clients - clip - 1]
            )
        idx = np.argpartition(scores, num_clients - clip)[:num_clients - clip]
        idx = idx.tolist()
        for k, p in enumerate(instance_data.model.get_params()):
            for i in idx:
                p.data.add_((1/len(idx)) * grads[k][i])


# Foolsgold

def foolsgold_srv(instance_data):
    with torch.no_grad():
         # send a copy of the global model
        for param in instance_data.model.get_params():
            dist.broadcast(tensor=param, src=0)
        # get the gradients
        grads = [[torch.zeros(p.shape, dtype=torch.float32) for _ in range(instance_data.total_nodes + 1)] for p in instance_data.model.get_params()]
        for i, _ in enumerate(instance_data.model.get_params()):
            dist.gather(grads[i][0], gather_list=grads[i], dst=0)
            grads[i] = torch.tensor([g.tolist() for g in grads[i][1:]], dtype=torch.float32)
        # perform foolsgold
        flat_grads = flatten_grads(grads, instance_data.model.device)
        idx = list(range(len(flat_grads)))
        num_clients = len(flat_grads)
        cs = torch.tensor(
            [[0 for _ in range(num_clients)] for _ in range(num_clients)],
            dtype=torch.float32
        )
        v = torch.tensor([0 for _ in range(num_clients)], dtype=torch.float32)
        alpha = torch.tensor([0 for _ in range(num_clients)], dtype=torch.float32)
        for i in idx:
            instance_data.other_params['histories'][i] += flat_grads[i]
        for i in idx:
            for j in {x for x in idx} - {i}:
                cs[i][j] = torch.cosine_similarity(
                    instance_data.other_params['histories'][i],
                    instance_data.other_params['histories'][j],
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
        alpha[ids] = instance_data.other_params['kappa'] * (
            torch.log(alpha[ids] / (1 - alpha[ids])) + 0.5
        )
        alpha[alpha > 1] = 1
        alpha[alpha < 0] = 0
        alpha[alpha.isnan()] = 0
        alpha = alpha / alpha.sum()
        for k, p in enumerate(instance_data.model.get_params()):
            for i in idx:
                p.data.add_(alpha[i] * grads[k][i])



# Viceroy

def viceroy_srv(instance_data):
    with torch.no_grad():
         # send a copy of the global model
        for param in instance_data.model.get_params():
            dist.broadcast(tensor=param, src=0)
        # get the gradients
        grads = [[torch.zeros(p.shape, dtype=torch.float32) for _ in range(instance_data.total_nodes + 1)] for p in instance_data.model.get_params()]
        for i, _ in enumerate(instance_data.model.get_params()):
            dist.gather(grads[i][0], gather_list=grads[i], dst=0)
            grads[i] = torch.tensor([g.tolist() for g in grads[i][1:]], dtype=torch.float32)
        # perform viceroy
        flat_grads = flatten_grads(grads, instance_data.model.device)
        idx = list(range(len(flat_grads)))
        num_clients = len(flat_grads)
        alpha = torch.tensor([0 for _ in range(num_clients)], dtype=torch.float32, device=instance_data.model.device)
        gamma = torch.tensor([0 for _ in range(num_clients)], dtype=torch.float32, device=instance_data.model.device)
        b = torch.tensor([0 for _ in range(num_clients)], dtype=torch.float32, device=instance_data.model.device)
        G = torch.zeros(instance_data.other_params['histories'][idx[0]].shape, device=instance_data.model.device)
        for i in idx:
            if instance_data.other_params['first']:
                instance_data.other_params['rep'][i] = 1
            else:
                if instance_data.other_params['rep'][i] < 0:
                    instance_data.other_params['rep'][i] = 0.97**(abs(torch.cosine_similarity(instance_data.other_params['histories'][i], flat_grads[i], dim=0))) * instance_data.other_params['rep'][i]
                    instance_data.other_params['histories'][i] *= 0.9
                else:
                    instance_data.other_params['rep'][i] = torch.clip(0.9 * instance_data.other_params['rep'][i] + 0.5 *
                            torch.tanh(2 * abs(torch.cosine_similarity(instance_data.other_params['histories'][i],
                                flat_grads[i], dim=0)) - 1), 0, 1)
                    instance_data.other_params['histories'][i] = 0.9 * instance_data.other_params['histories'][i] + flat_grads[i]
                    if instance_data.other_params['rep'][i] < 0.1:
                        instance_data.other_params['rep'][i] = -1
            G += instance_data.other_params['histories'][i]
        G /= len(idx)
        S = instance_data.other_params['kappa'] * G / 21.85
        B = torch.tensor(0.0, device=instance_data.model.device)
        for i in idx:
            if instance_data.other_params['first']:
                gamma[i] = 1
            else:
                gamma[i] = 1 - abs(torch.cosine_similarity(instance_data.other_params['histories'][i], G, dim=0))
            for j in set(idx) - {i}:
                b[i] += torch.cdist(flat_grads[i].unsqueeze(0), flat_grads[j].unsqueeze(0)).item()
            b[i] /= len(idx) - 1
            B += torch.cdist(S.unsqueeze(0), flat_grads[i].unsqueeze(0)).item()
        B /= len(idx)
        alpha = torch.exp(-(2 / torch.norm(S)) * (b - B)**2)
        if instance_data.other_params['first']:
            instance_data.other_params['first'] = False
        alpha = alpha * instance_data.other_params['rep'] * gamma
        alpha[alpha > 1] = 1
        alpha[alpha < 0] = 0
        alpha[alpha.isnan()] = 0
        alpha = alpha / alphs if (alphs := alpha.sum()) > 0 else torch.zeros(alpha.shape)
        for k, p in enumerate(instance_data.model.get_params()):
            for i in idx:
                p.data.add_(alpha[i] * grads[k][i])

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


def std_dagmm_srv(instance_data):
    # send a copy of the global model
    for param in instance_data.model.get_params():
        dist.broadcast(tensor=param, src=0)
    # get the batch sizes
    batch_sizes = [torch.zeros(1, dtype=float) for _ in range(instance_data.total_nodes + 1)]
    dist.gather(torch.zeros(1, dtype=float), gather_list=batch_sizes, dst=0)
    print(f"batch sizes = {batch_sizes}")
    # get the gradients
    grads = [[torch.zeros(p.shape, dtype=torch.float32) for _ in range(instance_data.total_nodes + 1)] for p in instance_data.model.get_params()]
    for i, _ in enumerate(instance_data.model.get_params()):
        dist.gather(grads[i][0], gather_list=grads[i], dst=0)
        grads[i] = torch.tensor([g.tolist() for g in grads[i][1:]], dtype=torch.float32)
    print(f"grads = {grads}")
    # perform std-dagmm
    flat_grads = flatten_grads(grads, instance_data.model.device)
    num_clients = len(flat_grads)
    if instance_data.other_params['gmm'] is None:
        instance_data.other_params['gmm'] = STD_DAGMM(
            flat_grads.shape[1],
            instance_data.model.device
        )
    instance_data.other_params['gmm'].fit(flat_grads)
    energies = instance_data.other_params['gmm'].predict(flat_grads)
    std = torch.std(energies).item()
    avg = torch.mean(energies).item()
    indices = (energies >= avg - std) * (energies <= avg + std)
    idx = torch.tensor(range(num_clients))[indices].tolist()
    with torch.no_grad():
        total_dc = sum([batch_sizes[i] for i in idx])
        alpha = torch.tensor([0.0 for _ in grads.values()])
        for i in idx:
            alpha[i] = batch_sizes[i] / total_dc
            for k, p in enumerate(instance_data.model.get_params()):
                p.data.add_(alpha[i] * grads[k][i])


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
        raise MisconfigurationError(
            f"Model '{alg_name}' does not exist, " +
            f"possible options: {set(alg_pairs.keys())}"
        )
    return alg_pair
