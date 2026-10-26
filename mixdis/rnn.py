"""
Multi-region rate RNN with two cell types per region.

Regions
    S1 : receives stimulus factor A ("sensory area 1", e.g. colour)
    S2 : receives stimulus factor B ("sensory area 2", e.g. motion)
    P  : receives context cue + S1, S2 ("association / PFC-like"); output read
         from P only.
Feedback P -> S1, S2 optional.

Cell types (per region): 'fast' (tau_fast) and 'slow' (tau_slow) units; with
dale=True each type is further split into excitatory and inhibitory units
(80/20) with sign-constrained outgoing weights.

Trial structure (dt = 10 ms): fixation | stimulus | delay | response.
Targets are delivered in the response epoch; the network must hold the
stimulus through the delay -> a role for slow cell types.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import discrete_conditions, random_dichotomy_tasks


class Trials:
    def __init__(self, levels=(3, 3, 2), n_nonlinear=4, include_linear=True,
                 include_context=True, D_stim=20, T_fix=3, T_stim=8, T_delay=8,
                 T_resp=4, noise=0.3, seed=0):
        """levels = (A levels, B levels, n contexts)."""
        self.levels = levels
        self.conds = discrete_conditions(levels)  # (C, 3)
        rng = np.random.default_rng(seed)
        nA, nB, nX = levels
        self.EA = rng.normal(0, 1, (nA, D_stim)).astype(np.float32)
        self.EB = rng.normal(0, 1, (nB, D_stim)).astype(np.float32)
        self.D_stim, self.nX = D_stim, nX
        self.T = (T_fix, T_stim, T_delay, T_resp)
        self.noise = noise
        tasks, names = [], []
        A, B, X = self.conds.T
        if include_linear:
            tasks += [(A >= nA / 2).astype(np.float32), (B >= nB / 2).astype(np.float32)]
            names += ["A-linear", "B-linear"]
        if include_context:
            # context gating (Mante et al. 2013): ctx 0 -> report A, ctx 1 -> report B
            tasks.append(np.where(X % 2 == 0, A >= nA / 2, B >= nB / 2).astype(np.float32))
            names.append("context-gated")
        if n_nonlinear:
            R = random_dichotomy_tasks(self.conds, n_nonlinear, seed=seed + 1,
                                       exclude_linear=True, levels=levels)
            tasks += list(R.T); names += [f"xor-{i}" for i in range(n_nonlinear)]
        self.Ycond = np.stack(tasks, 1).astype(np.float32)
        self.task_names = names
        self.n_out = self.Ycond.shape[1]

    @property
    def T_total(self):
        return sum(self.T)

    def sample(self, B, rng=None, cond_idx=None):
        rng = rng or np.random.default_rng()
        ci = rng.integers(0, len(self.conds), B) if cond_idx is None else cond_idx
        Tf, Ts, Td, Tr = self.T
        T = self.T_total
        c = self.conds[ci]
        uA = np.zeros((T, B, self.D_stim), np.float32)
        uB = np.zeros((T, B, self.D_stim), np.float32)
        uX = np.zeros((T, B, self.nX + 1), np.float32)  # context + go cue
        uA[Tf:Tf + Ts] = self.EA[c[:, 0]]
        uB[Tf:Tf + Ts] = self.EB[c[:, 1]]
        uX[:, np.arange(B), c[:, 2]] = 1.0
        uX[Tf + Ts + Td:, :, -1] = 1.0
        uA += self.noise * rng.normal(size=uA.shape)
        uB += self.noise * rng.normal(size=uB.shape)
        Y = self.Ycond[ci]
        mask = np.zeros(T, np.float32); mask[Tf + Ts + Td:] = 1.0
        return (torch.from_numpy(uA), torch.from_numpy(uB), torch.from_numpy(uX)), \
            torch.from_numpy(Y), torch.from_numpy(mask), ci


class MultiRegionRNN(nn.Module):
    def __init__(self, D_stim, n_ctx_in, n_out, n_per_region=64, frac_slow=0.5,
                 tau_fast=2.0, tau_slow=20.0, feedback=True, dale=False,
                 noise=0.05, g_rec=0.9, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.R = ["S1", "S2", "P"]
        n = n_per_region
        self.n = n
        N = 3 * n
        self.N = N
        # time constants (units of dt) per unit
        tau = torch.full((N,), tau_fast)
        slow = torch.zeros(N, dtype=torch.bool)
        for r in range(3):
            idx = torch.arange(r * n, r * n + int(n * frac_slow))
            slow[idx] = True
        tau[slow] = tau_slow
        self.register_buffer("alpha", 1.0 / tau)
        self.register_buffer("is_slow", slow)
        region = torch.arange(N) // n
        self.register_buffer("region", region)
        # connectivity mask: within-region dense; S1->P, S2->P; P->S1,S2 (fb)
        M = torch.zeros(N, N)  # M[post, pre]
        for r in range(3):
            M[r * n:(r + 1) * n, r * n:(r + 1) * n] = 1
        M[2 * n:, 0:n] = 1; M[2 * n:, n:2 * n] = 1
        if feedback:
            M[0:n, 2 * n:] = 1; M[n:2 * n, 2 * n:] = 1
        M.fill_diagonal_(0)
        self.register_buffer("M", M)
        self.dale = dale
        if dale:
            sign = torch.ones(N)
            for r in range(3):
                for typ in (True, False):
                    ids = torch.where((region == r) & (slow == typ))[0]
                    sign[ids[int(0.8 * len(ids)):]] = -1.0
            self.register_buffer("sign", sign)
            self.W_raw = nn.Parameter(torch.rand(N, N) * g_rec / math.sqrt(N) * 2)
        else:
            self.W_raw = nn.Parameter(torch.randn(N, N) * g_rec / math.sqrt(N / 2))
        self.WinA = nn.Parameter(torch.randn(n, D_stim) / math.sqrt(D_stim))
        self.WinB = nn.Parameter(torch.randn(n, D_stim) / math.sqrt(D_stim))
        self.WinX = nn.Parameter(torch.randn(n, n_ctx_in))
        self.b = nn.Parameter(torch.zeros(N))
        self.Wout = nn.Linear(n, n_out)
        self.noise = noise

    def W(self):
        if self.dale:
            return F.relu(self.W_raw) * self.M * self.sign[None, :]
        return self.W_raw * self.M

    def forward(self, inputs, return_rates=False):
        uA, uB, uX = inputs
        T, B, _ = uA.shape
        n = self.n
        x = torch.zeros(B, self.N)
        W = self.W()
        rates, outs = [], []
        for t in range(T):
            r = F.relu(x)
            inp = torch.cat([uA[t] @ self.WinA.T, uB[t] @ self.WinB.T,
                             uX[t] @ self.WinX.T], 1)
            drive = r @ W.T + inp + self.b
            if self.training and self.noise > 0:
                drive = drive + self.noise * torch.randn_like(drive) / self.alpha.sqrt()
            x = (1 - self.alpha) * x + self.alpha * drive
            r = F.relu(x)
            rates.append(r); outs.append(self.Wout(r[:, 2 * n:]))
        Y = torch.stack(outs)
        if return_rates:
            return Y, torch.stack(rates)
        return Y

    def weights(self):
        return [self.W_raw, self.WinA, self.WinB, self.WinX, self.Wout.weight]


def train_rnn(model, trials, steps=2000, batch=128, lr=2e-3, act_l2=1e-3,
              w_l2=1e-5, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for it in range(steps):
        u, Y, mask, _ = trials.sample(batch, rng)
        out, R = model(u, return_rates=True)
        m = mask[:, None, None]
        tl = (F.binary_cross_entropy_with_logits(out, Y[None].expand_as(out),
                                                 reduction="none") * m).sum() / (m.sum() * batch * out.shape[-1])
        loss = tl + act_l2 * R.pow(2).mean() + w_l2 * sum(w.pow(2).sum() for w in model.weights())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % 50 == 0:
            with torch.no_grad():
                acc = (((out[-1] > 0).float() == Y).float().mean()).item()
            hist.append((it, tl.item(), acc))
            if verbose and it % max(1, steps // 5) == 0:
                print(f"step {it}: task loss {tl.item():.3f}  acc {acc:.3f}")
    return hist


@torch.no_grad()
def collect(model, trials, n_per_cond=20, seed=123):
    """Run every condition n_per_cond times. Returns rates (T, trials, N),
    condition labels (trials, 3), per-task accuracy."""
    model.eval()
    rng = np.random.default_rng(seed)
    ci = np.repeat(np.arange(len(trials.conds)), n_per_cond)
    u, Y, mask, ci = trials.sample(len(ci), rng, cond_idx=ci)
    out, R = model(u, return_rates=True)
    acc = ((out[-1] > 0).float() == Y).float().mean(0).numpy()
    model.train()
    return R.numpy(), trials.conds[ci], acc
