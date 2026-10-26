"""
Latent-factor environments with controllable statistics.

Continuous factors (for autoencoder / efficient-coding experiments)
    sample_factors(n, K, corr, support, marginal)
      corr     : pairwise correlation via Gaussian copula (statistical dependence)
      support  : 'full' (rectangular) or 'corner' (a corner of factor space is
                 never visited -> non-rectangular support, cf. Whittington et
                 al. 2023 who show that disentangling needs rectangular support)
      marginal : 'uniform' | 'skewed' | 'cardinal' (for 1-D efficient coding)

Discrete factors (for mixed-selectivity / multitask experiments)
    discrete_conditions(levels) -> all combinations of factor levels
    random_dichotomy_tasks       -> Rigotti-style nonlinear (XOR-like) tasks
    linear_tasks                 -> tasks that are linear in single factors

Observation models
    RandomObservation: x = phi(A f) + noise, with A a random D x K matrix and
    phi = identity (linear mixing) or a fixed random 2-layer network
    (nonlinear mixing, i.e. a "sensory periphery").
"""
import itertools
import numpy as np
import torch
from scipy.stats import norm


def sample_factors(n, K=2, corr=0.0, support="full", marginal="uniform",
                   corner=0.5, band=0.4, rng=None):
    """Return (n, K) factors in [-1, 1].
    support: 'full'   rectangular (all combinations possible)
             'corner' the corner where all factors > 2*corner-1 is removed
             'band'   only |f1 - f2| < band is allowed (strongly dependent)
             'diamond' f = R45 (u, v) with u, v independent uniform: f1 and f2
                      are UNCORRELATED but dependent - the support is a
                      rectangle only in the rotated basis (f1+f2, f1-f2), so
                      the energy-optimal non-negative code is predicted to be
                      aligned with the rotated axes, i.e. mixed w.r.t. f
                      (range dependence, Whittington et al. 2023)."""
    rng = rng or np.random.default_rng()
    out = []
    need = n
    while need > 0:
        m = int(need * 1.6) + 16
        C = np.full((K, K), corr) + (1 - corr) * np.eye(K)
        z = rng.multivariate_normal(np.zeros(K), C, size=m)
        u = norm.cdf(z)  # uniform marginals, Gaussian copula dependence
        if marginal == "skewed":
            u = u ** 2.0
        f = 2 * u - 1
        if support == "corner":
            # remove the corner where every factor exceeds 2*corner-1
            # (corner=0.5 removes the positive orthant: 25% of space for K=2)
            keep = ~np.all(f > 2 * corner - 1, axis=1)
            f = f[keep]
        elif support == "band":
            f = f[np.abs(f[:, 0] - f[:, 1]) < band]
        elif support == "diamond":
            u = rng.uniform(-1, 1, (len(f), 2))
            f = f.copy()
            f[:, 0] = (u[:, 0] + u[:, 1]) / 2
            f[:, 1] = (u[:, 0] - u[:, 1]) / 2
        out.append(f); need -= len(f)
    return np.concatenate(out)[:n].astype(np.float32)


def sample_angles(n, prior="cardinal", kappa=2.0, rng=None):
    """1-D circular stimulus (orientation, in [0, pi)) with a non-uniform prior.
    'cardinal': p(theta) ~ exp(kappa*cos(4 theta)) -> peaks at 0 and 90 deg
    (natural-image orientation statistics; Girshick et al. 2011)."""
    rng = rng or np.random.default_rng()
    if prior == "uniform":
        return rng.uniform(0, np.pi, n).astype(np.float32)
    out = []
    while sum(len(o) for o in out) < n:
        th = rng.uniform(0, np.pi, 4 * n)
        p = np.exp(kappa * np.cos(4 * th)) / np.exp(kappa)
        out.append(th[rng.random(4 * n) < p])
    return np.concatenate(out)[:n].astype(np.float32)


def prior_density(theta_grid, prior="cardinal", kappa=2.0):
    if prior == "uniform":
        p = np.ones_like(theta_grid)
    else:
        p = np.exp(kappa * np.cos(4 * theta_grid))
    return p / np.trapezoid(p, theta_grid)


class RandomObservation:
    """Fixed random map from K latent factors to D-dimensional observations."""

    def __init__(self, K, D=50, kind="linear", hidden=100, noise=0.0, seed=0):
        g = np.random.default_rng(seed)
        self.kind, self.noise = kind, noise
        if kind == "linear":
            self.A = g.normal(0, 1 / np.sqrt(K), (K, D)).astype(np.float32)
        else:
            self.A1 = g.normal(0, 1.5 / np.sqrt(K), (K, hidden)).astype(np.float32)
            self.b1 = g.normal(0, 0.5, hidden).astype(np.float32)
            self.A2 = g.normal(0, 1 / np.sqrt(hidden), (hidden, D)).astype(np.float32)
        self.D = D

    def __call__(self, f, rng=None):
        if self.kind == "linear":
            x = f @ self.A
        else:
            x = np.tanh(f @ self.A1 + self.b1) @ self.A2
        if self.noise > 0:
            rng = rng or np.random.default_rng()
            x = x + self.noise * rng.normal(size=x.shape).astype(np.float32)
        return x.astype(np.float32)


def angle_observation(theta, n_channels=60, width=0.3, noise=0.0, rng=None):
    """Population of von-Mises 'input' channels tuned uniformly on [0, pi)."""
    pref = np.linspace(0, np.pi, n_channels, endpoint=False)
    x = np.exp((np.cos(2 * (theta[:, None] - pref[None])) - 1) / width)
    if noise > 0:
        rng = rng or np.random.default_rng()
        x = x + noise * rng.normal(size=x.shape)
    return x.astype(np.float32)


# ----------------------------------------------------------------------------
# Discrete factors and tasks
# ----------------------------------------------------------------------------
def discrete_conditions(levels):
    """levels: e.g. (2,2,2). Returns (C, K) integer array of all conditions."""
    return np.array(list(itertools.product(*[range(l) for l in levels])))


def condition_embedding(conds, levels, D=50, kind="linear", seed=0):
    """Map discrete conditions to input patterns: one-hot per factor then a
    fixed random projection (+ optional nonlinearity)."""
    oh = np.concatenate([np.eye(l)[conds[:, k]] for k, l in enumerate(levels)], 1)
    obs = RandomObservation(oh.shape[1], D, kind, seed=seed)
    return obs(oh.astype(np.float32)), oh


def linear_tasks(conds, levels):
    """One binary task per factor level split (e.g. 'is factor k >= mid')."""
    T = []
    for k, l in enumerate(levels):
        T.append((conds[:, k] >= l / 2).astype(np.float32))
    return np.stack(T, 1)


def random_dichotomy_tasks(conds, n_tasks, seed=0, exclude_linear=True,
                           levels=None):
    """Random balanced dichotomies of the C conditions (Rigotti et al. 2013).
    With exclude_linear, dichotomies linearly separable in the one-hot factor
    space are rejected, so every task demands a nonlinear (mixed) read-out."""
    from sklearn.svm import LinearSVC
    g = np.random.default_rng(seed)
    C = len(conds)
    oh = None
    if levels is not None:
        oh = np.concatenate([np.eye(l)[conds[:, k]] for k, l in enumerate(levels)], 1)
    tasks, tries = [], 0
    while len(tasks) < n_tasks and tries < 5000:
        tries += 1
        lab = np.zeros(C); lab[g.permutation(C)[:C // 2]] = 1
        if exclude_linear and oh is not None:
            clf = LinearSVC(C=100, max_iter=20000).fit(oh, lab)
            if (clf.predict(oh) == lab).all():
                continue
        if any((lab == t).all() or (lab == 1 - t).all() for t in tasks):
            continue
        tasks.append(lab)
    return np.stack(tasks, 1).astype(np.float32)


def trials_from_conditions(X_cond, Y_cond, n_per, noise=0.1, rng=None):
    rng = rng or np.random.default_rng()
    idx = np.repeat(np.arange(len(X_cond)), n_per)
    X = X_cond[idx] + noise * rng.normal(size=(len(idx), X_cond.shape[1]))
    return X.astype(np.float32), Y_cond[idx], idx


def to_t(*a):
    return [torch.as_tensor(x, dtype=torch.float32) for x in a]
