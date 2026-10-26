"""
Metrics that place a population code on the pure <-> mixed <-> disentangled map.

Single-neuron level
    selectivity_continuous / selectivity_discrete
        -> class of each neuron: 'unselective', 'pure', 'linear_mixed',
           'nonlinear_mixed' (Rigotti et al. 2013 taxonomy) and a continuous
           nonlinear-mixing index  NMS = (R2_full - R2_additive) / R2_full
    mi_matrix  -> normalised mutual information between each neuron and factor

Population level
    modularity  (Ridgeway & Mozer 2018; Whittington et al. 2023): does each
                neuron carry information about only one factor?
    mig         mutual-information gap (Chen et al. 2018)
    participation_ratio  linear dimensionality
    shattering_dim       fraction of random dichotomies linearly decodable
                         (Rigotti 2013; Bernardi et al. 2020)
    ccgp                 cross-condition generalisation (abstraction;
                         Bernardi et al. 2020)
    energy               mean rate, mean squared rate, fraction active
"""
import itertools
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC


def active_mask(Z, tol=1e-3):
    return Z.std(0) > tol


# ----------------------------------------------------------------------------
# Mutual information
# ----------------------------------------------------------------------------
def _discretize(v, n_bins):
    if np.unique(v).size <= n_bins:
        _, d = np.unique(v, return_inverse=True)
        return d
    edges = np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1])
    edges = np.unique(edges)
    return np.digitize(v, edges)


def _mi(a, b):
    ca, cb = np.unique(a, return_inverse=True)[1], np.unique(b, return_inverse=True)[1]
    joint = np.zeros((ca.max() + 1, cb.max() + 1))
    np.add.at(joint, (ca, cb), 1)
    p = joint / joint.sum()
    pa, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (pa @ pb)[nz])).sum())


def _H(a):
    _, c = np.unique(a, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def mi_matrix(Z, F, n_bins=12):
    """(N, K) mutual information normalised by factor entropy."""
    N, K = Z.shape[1], F.shape[1]
    Fd = [_discretize(F[:, k], n_bins) for k in range(K)]
    M = np.zeros((N, K))
    for i in range(N):
        zi = _discretize(Z[:, i], n_bins)
        for k in range(K):
            M[i, k] = _mi(zi, Fd[k]) / max(_H(Fd[k]), 1e-12)
    return M


def modularity(Z, F, n_bins=12, M=None, min_info=0.02):
    """Mean modularity of informative neurons (1 = each neuron codes 1 factor)."""
    M = mi_matrix(Z, F, n_bins) if M is None else M
    K = M.shape[1]
    keep = M.max(1) > min_info
    if keep.sum() == 0 or K < 2:
        return np.nan
    m = M[keep]
    th = m.max(1)
    tmpl = np.zeros_like(m); tmpl[np.arange(len(m)), m.argmax(1)] = th
    dev = ((m - tmpl) ** 2).sum(1) / (th ** 2 * (K - 1))
    return float(np.mean(1 - dev))


def factor_contributions(Z, F, deg=3):
    """Variance decomposition of each neuron into per-factor additive
    components: z ~ sum_k g_k(f_k) with g_k a degree-`deg` polynomial.
    Returns C (N, K) with rows summing to 1 (share of each factor) and the
    additive-fit R2 per neuron. Robust alternative to MI-based modularity."""
    K = F.shape[1]
    Fs = (F - F.mean(0)) / (F.std(0) + 1e-9)
    blocks = [PolynomialFeatures(deg, include_bias=False).fit_transform(Fs[:, [k]]) for k in range(K)]
    X = np.concatenate(blocks, 1)
    sizes = [b.shape[1] for b in blocks]
    C = np.zeros((Z.shape[1], K)); R2 = np.zeros(Z.shape[1])
    for i in range(Z.shape[1]):
        z = Z[:, i]
        if z.std() < 1e-6:
            C[i] = np.nan; continue
        lr = LinearRegression().fit(X, z)
        R2[i] = lr.score(X, z)
        o = 0
        for k, s in enumerate(sizes):
            C[i, k] = np.var(X[:, o:o + s] @ lr.coef_[o:o + s]); o += s
        C[i] /= max(C[i].sum(), 1e-12)
    return C, R2


def disentanglement(Z, F, deg=3, min_r2=0.1):
    """Variance-based modularity in [0,1]: 1 = every informative neuron
    depends on exactly one factor, 0 = every neuron depends equally on all.
    Weighted by each neuron's variance. Also returns per-neuron purity."""
    C, R2 = factor_contributions(Z, F, deg)
    K = F.shape[1]
    keep = (R2 > min_r2) & np.isfinite(C).all(1)
    if keep.sum() == 0:
        return np.nan, np.full(Z.shape[1], np.nan)
    pur = np.full(Z.shape[1], np.nan)
    pur[keep] = (K * C[keep].max(1) - 1) / (K - 1)
    w = Z.var(0)
    return float((pur[keep] * w[keep]).sum() / w[keep].sum()), pur


def mig(Z, F, n_bins=12, M=None):
    M = mi_matrix(Z, F, n_bins) if M is None else M
    s = np.sort(M, 0)[::-1]
    return float(np.mean(s[0] - s[1])) if M.shape[0] > 1 else np.nan


# ----------------------------------------------------------------------------
# Single-neuron selectivity classes
# ----------------------------------------------------------------------------
def _r2(X, y):
    return LinearRegression().fit(X, y).score(X, y)


def _classify(r2_single, r2_add, r2_full, min_r2=0.1, pure_frac=0.9, nms_thr=0.1):
    if r2_full < min_r2:
        return "unselective", np.nan
    nms = max(0.0, (r2_full - r2_add) / r2_full)
    if max(r2_single) >= pure_frac * r2_full:
        return "pure", nms
    return ("nonlinear_mixed" if nms > nms_thr else "linear_mixed"), nms


def selectivity_continuous(Z, F, deg=3, **kw):
    """Polynomial regression on each factor alone / additive / full."""
    K = F.shape[1]
    singles = [PolynomialFeatures(deg, include_bias=False).fit_transform(F[:, [k]])
               for k in range(K)]
    Xadd = np.concatenate(singles, 1)
    Xfull = PolynomialFeatures(deg, include_bias=False).fit_transform(F)
    classes, nms, r2f = [], [], []
    for i in range(Z.shape[1]):
        z = Z[:, i]
        if z.std() < 1e-6:
            classes.append("unselective"); nms.append(np.nan); r2f.append(0); continue
        rs = [_r2(s, z) for s in singles]
        ra, rf = _r2(Xadd, z), _r2(Xfull, z)
        c, n = _classify(rs, ra, rf, **kw)
        classes.append(c); nms.append(n); r2f.append(rf)
    return np.array(classes), np.array(nms), np.array(r2f)


def selectivity_discrete(Z, C, **kw):
    """Z: (trials, N); C: (trials, K) integer factor levels per trial.
    Uses one-hot ANOVA-style regressions: single factor / additive /
    full conjunction (all K-way interactions)."""
    K = C.shape[1]

    def oh(v):
        u = np.unique(v); return (v[:, None] == u[None]).astype(float)
    singles = [oh(C[:, k]) for k in range(K)]
    Xadd = np.concatenate([singles[0]] + [s[:, 1:] for s in singles[1:]], 1)
    key = np.zeros(len(C), dtype=np.int64)
    for k in range(K):
        key = key * 1000 + C[:, k]
    Xfull = oh(key)
    classes, nms, r2f = [], [], []
    for i in range(Z.shape[1]):
        z = Z[:, i]
        if z.std() < 1e-6:
            classes.append("unselective"); nms.append(np.nan); r2f.append(0); continue
        rs = [_r2(s, z) for s in singles]
        ra, rf = _r2(Xadd, z), _r2(Xfull, z)
        c, n = _classify(rs, ra, rf, **kw)
        classes.append(c); nms.append(n); r2f.append(rf)
    return np.array(classes), np.array(nms), np.array(r2f)


CLASSES = ["unselective", "pure", "linear_mixed", "nonlinear_mixed"]


def class_fractions(classes):
    return {c: float(np.mean(classes == c)) for c in CLASSES}


# ----------------------------------------------------------------------------
# Population geometry
# ----------------------------------------------------------------------------
def participation_ratio(Z):
    Zc = Z - Z.mean(0)
    ev = np.linalg.eigvalsh(np.cov(Zc.T) + 1e-12 * np.eye(Z.shape[1]))
    ev = np.clip(ev, 0, None)
    return float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-12))


def _split(n, rng, frac=0.5):
    p = rng.permutation(n); k = int(n * frac)
    return p[:k], p[k:]


def shattering_dim(Z, cond_idx, n_dich=60, seed=0):
    """Mean held-out accuracy over random balanced dichotomies of conditions."""
    rng = np.random.default_rng(seed)
    conds = np.unique(cond_idx); C = len(conds)
    accs = []
    tr, te = _split(len(Z), rng)
    for _ in range(n_dich):
        pos = rng.permutation(conds)[:C // 2]
        y = np.isin(cond_idx, pos).astype(int)
        clf = LinearSVC(C=1.0, max_iter=5000).fit(Z[tr], y[tr])
        accs.append(clf.score(Z[te], y[te]))
    return float(np.mean(accs))


def ccgp(Z, C, k, max_splits=40, seed=0):
    """Cross-condition generalisation for binary factor k (C: trials x K)."""
    rng = np.random.default_rng(seed)
    others = [j for j in range(C.shape[1]) if j != k]
    combos = np.unique(C[:, others], axis=0) if others else np.zeros((1, 0), int)
    if len(combos) < 2:
        return np.nan
    accs = []
    pairs = list(itertools.combinations(range(len(combos)), max(1, len(combos) // 2)))
    rng.shuffle(pairs)
    for held in pairs[:max_splits]:
        test_c = combos[list(held)]
        is_test = np.zeros(len(C), bool)
        for tc in test_c:
            is_test |= np.all(C[:, others] == tc, axis=1)
        y = C[:, k]
        if len(np.unique(y[~is_test])) < 2 or len(np.unique(y[is_test])) < 2:
            continue
        clf = LinearSVC(C=1.0, max_iter=5000).fit(Z[~is_test], y[~is_test])
        accs.append(clf.score(Z[is_test], y[is_test]))
    return float(np.mean(accs)) if accs else np.nan


def energy(Z):
    return dict(mean_rate=float(np.abs(Z).mean()), mean_sq=float((Z ** 2).mean()),
                frac_active=float(active_mask(Z).mean()))


def summary_continuous(Z, F, n_bins=12):
    M = mi_matrix(Z, F, n_bins)
    cls, nms, _ = selectivity_continuous(Z, F)
    out = dict(disent=disentanglement(Z, F)[0],
               modularity=modularity(Z, F, M=M), mig=mig(Z, F, M=M),
               pr=participation_ratio(Z), nms_mean=float(np.nanmean(nms)) if np.isfinite(nms).any() else np.nan)
    out.update(class_fractions(cls)); out.update(energy(Z))
    return out
