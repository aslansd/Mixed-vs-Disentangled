"""
Neurons with dendritic subunits (two-layer neuron model; Poirazi, Brannon &
Mel 2003; Gidon et al. 2020 for non-monotonic dCaAP branches).

    branch_b(x) = g( w_b . (m_b * x) + beta_b )         (m_b: synaptic mask)
    soma(x)     = f( sum_b c_b * branch_b(x) + beta_0 )

g: 'relu' | 'sigmoid' | 'dcaap' (non-monotonic: peaks at threshold then decays,
   lets ONE branch compute XOR) | 'linear' (point neuron)
Masks allow 'factor-segregated' dendrites: every branch samples synapses from a
single input group (e.g. one stimulus factor), as for clustered inputs.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def dcaap(u):
    return torch.exp(-((u - 1.0) ** 2) / 0.5)


_G = {"relu": F.relu, "sigmoid": torch.sigmoid, "dcaap": dcaap,
      "linear": lambda u: u, "softplus": F.softplus}


class DendriticLayer(nn.Module):
    def __init__(self, D, N, n_branches=4, branch_act="relu", soma_act="relu",
                 groups=None, branch_groups=None, init_scale=1.0, seed=0):
        """groups: list of index arrays partitioning the D inputs.
        branch_groups: (n_branches,) which group each branch samples; if None
        every branch samples every input."""
        super().__init__()
        torch.manual_seed(seed)
        self.N, self.nb, self.D = N, n_branches, D
        self.W = nn.Parameter(torch.randn(N, n_branches, D) * init_scale / math.sqrt(D))
        self.bb = nn.Parameter(torch.zeros(N, n_branches))
        self.c = nn.Parameter(torch.ones(N, n_branches) / math.sqrt(n_branches)
                              + 0.1 * torch.randn(N, n_branches))
        self.b0 = nn.Parameter(torch.zeros(N))
        mask = torch.ones(N, n_branches, D)
        if groups is not None and branch_groups is not None:
            mask.zero_()
            for b, gi in enumerate(branch_groups):
                mask[:, b, torch.as_tensor(groups[gi])] = 1.0
        self.register_buffer("mask", mask)
        self.g = _G[branch_act]
        self.f = _G[soma_act]

    def forward(self, x, return_pre=False):
        u = torch.einsum("nbd,Bd->Bnb", self.W * self.mask, x) + self.bb
        br = self.g(u)
        s_in = (br * self.c).sum(-1) + self.b0
        out = self.f(s_in)
        if return_pre:
            return out, br, s_in
        return out, br


class SingleNeuron(nn.Module):
    """A single (dendritic or point) neuron used as a binary classifier; the
    somatic drive is the logit."""

    def __init__(self, D, n_branches=1, branch_act="linear", groups=None,
                 branch_groups=None, init_scale=0.1, seed=0):
        super().__init__()
        self.layer = DendriticLayer(D, 1, n_branches, branch_act, "linear",
                                    groups, branch_groups, init_scale, seed)

    def forward(self, x):
        _, br, s_in = self.layer(x, return_pre=True)
        return s_in[:, 0], br[:, 0]

    def weights(self):
        return [self.layer.W]


class DendriticNet(nn.Module):
    """x -> population of dendritic neurons -> linear read-out."""

    def __init__(self, D, N, n_out, n_branches=4, branch_act="relu",
                 soma_act="relu", groups=None, branch_groups=None, seed=0):
        super().__init__()
        self.hid = DendriticLayer(D, N, n_branches, branch_act, soma_act,
                                  groups, branch_groups, seed=seed)
        self.out = nn.Linear(N, n_out)

    def forward(self, x, return_branches=False):
        h, br = self.hid(x)
        y = self.out(h)
        return (y, h, br) if return_branches else (y, h)

    def weights(self):
        return [self.hid.W, self.out.weight]


def train_classifier(model, X, Y, steps=2000, lr=1e-2, w_l2=0.0, act_l2=0.0,
                     batch=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.as_tensor(X, dtype=torch.float32); Y = torch.as_tensor(Y, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for it in range(steps):
        if batch and batch < len(X):
            idx = torch.randint(0, len(X), (batch,), generator=g); xb, yb = X[idx], Y[idx]
        else:
            xb, yb = X, Y
        out, h = model(xb)[:2]
        loss = F.binary_cross_entropy_with_logits(out, yb)
        if w_l2:
            loss = loss + w_l2 * sum(w.pow(2).sum() for w in model.weights())
        if act_l2:
            loss = loss + act_l2 * h.pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 50 == 0:
            losses.append(loss.item())
    return losses


# ----------------------------------------------------------------------------
# Transitive inference (order relations)
# ----------------------------------------------------------------------------
def transitive_data(n_items=7, pairs="adjacent"):
    """Input: [one-hot(left), one-hot(right)]; target 1 if rank(left) > rank(right)
    (item 0 is the lowest). 'adjacent' -> training set of neighbouring pairs,
    'all' -> every ordered pair (i != j)."""
    X, Y, D = [], [], []
    for i in range(n_items):
        for j in range(n_items):
            if i == j:
                continue
            if pairs == "adjacent" and abs(i - j) != 1:
                continue
            x = np.zeros(2 * n_items, np.float32); x[i] = 1; x[n_items + j] = 1
            X.append(x); Y.append(float(i > j)); D.append(abs(i - j))
    return np.array(X), np.array(Y, np.float32), np.array(D)
