"""
Models whose hidden layer is treated as a 'neural population'.

All models expose  forward(x) -> (output, hidden)  and a  loss(...)  that adds
biologically-motivated costs:
    act_l2  : mean squared firing rate   (metabolic cost of spikes)
    act_l1  : mean absolute rate         (sparsity)
    w_l2    : squared synaptic weights   (cost of synapses)
Nonnegativity of rates is a switch ('relu' / 'softplus' vs 'linear').
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _act(name):
    return {"relu": F.relu, "softplus": F.softplus, "linear": lambda z: z,
            "tanh": torch.tanh}[name]


class Autoencoder(nn.Module):
    """x -> z = act(enc(x)) -> x_hat = W_d z + b. enc is linear or an MLP."""

    def __init__(self, D, H, act="relu", enc_hidden=0, init_scale=1.0, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        if enc_hidden:
            self.enc = nn.Sequential(nn.Linear(D, enc_hidden), nn.ReLU(),
                                     nn.Linear(enc_hidden, H))
        else:
            self.enc = nn.Linear(D, H)
        self.dec = nn.Linear(H, D)
        self.act_name = act
        self.act = _act(act)
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.weight.mul_(init_scale)

    def forward(self, x):
        z = self.act(self.enc(x))
        return self.dec(z), z

    def weights(self):
        return [m.weight for m in self.modules() if isinstance(m, nn.Linear)]


class MultiTaskNet(nn.Module):
    """x -> h (population) -> many binary/linear read-outs."""

    def __init__(self, D, H, n_out, act="relu", init_scale=1.0, depth=1, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        layers, d = [], D
        for _ in range(depth):
            layers.append(nn.Linear(d, H)); d = H
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(H, n_out)
        self.act = _act(act)
        self.init_scale = init_scale
        with torch.no_grad():
            for l in self.layers:
                nn.init.normal_(l.weight, 0, init_scale / math.sqrt(l.in_features))
                l.bias.zero_()
            nn.init.normal_(self.out.weight, 0, init_scale / math.sqrt(H))

    def forward(self, x, return_all=False):
        hs = []
        h = x
        for l in self.layers:
            h = self.act(l(h)); hs.append(h)
        y = self.out(h)
        return (y, hs) if return_all else (y, h)

    def weights(self):
        return [l.weight for l in self.layers] + [self.out.weight]


def bio_costs(h, weights, act_l2=0.0, act_l1=0.0, w_l2=0.0):
    c = h.new_zeros(())
    if act_l2:
        c = c + act_l2 * h.pow(2).mean()
    if act_l1:
        c = c + act_l1 * h.abs().mean()
    if w_l2:
        c = c + w_l2 * sum(w.pow(2).sum() for w in weights)
    return c


def train(model, X, Y, task="recon", steps=3000, lr=3e-3, batch=256,
          act_l2=0.0, act_l1=0.0, w_l2=0.0, snapshots=None, seed=0,
          verbose=False, opt="adam", Xs_snap=None):
    """Generic trainer.
    task: 'recon' (MSE to X), 'mse' (MSE to Y), 'bce' (binary targets Y).
    snapshots: list of steps at which to record hidden activity on Xs_snap
    (or X) -> returned in history['snap'] as {step: H (numpy)}."""
    g = torch.Generator().manual_seed(seed)
    X = torch.as_tensor(X, dtype=torch.float32)
    Y = X if task == "recon" else torch.as_tensor(Y, dtype=torch.float32)
    Xs = X if Xs_snap is None else torch.as_tensor(Xs_snap, dtype=torch.float32)
    if opt == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optim = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {"step": [], "loss": [], "task_loss": [], "snap": {}}
    snapshots = set(snapshots or [])
    n = len(X)
    for it in range(steps + 1):
        if it in snapshots:
            with torch.no_grad():
                hist["snap"][it] = model(Xs)[1].numpy().copy()
        if it == steps:
            break
        idx = torch.randint(0, n, (min(batch, n),), generator=g)
        out, h = model(X[idx])
        if task == "bce":
            tl = F.binary_cross_entropy_with_logits(out, Y[idx])
        else:
            tl = F.mse_loss(out, Y[idx])
        loss = tl + bio_costs(h, model.weights(), act_l2, act_l1, w_l2)
        optim.zero_grad(); loss.backward(); optim.step()
        if it % 50 == 0:
            hist["step"].append(it); hist["loss"].append(loss.item())
            hist["task_loss"].append(tl.item())
            if verbose and it % (steps // 5 or 1) == 0:
                print(f"step {it}: task {tl.item():.4f} total {loss.item():.4f}")
    return hist


@torch.no_grad()
def hidden(model, X):
    return model(torch.as_tensor(X, dtype=torch.float32))[1].numpy()


@torch.no_grad()
def task_accuracy(model, X, Y):
    out = model(torch.as_tensor(X, dtype=torch.float32))[0].numpy()
    return float(((out > 0) == (Y > 0.5)).mean())
