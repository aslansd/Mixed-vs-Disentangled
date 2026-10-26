# Project 2 — When does the brain mix and when does it disentangle?
### Energy, stimulus statistics, gradient descent, task demands and single-neuron computation

Answers **Questions 3, 4 and 5**:

3. Why mixed selectivity sometimes and disentanglement other times (mechanistically: energy, statistics)? Which regions?
   Which cell types are involved when selectivity is mixed?
4. How do single-neuron abilities (dendritic XOR, learning order relations) relate to mixed vs disentangled codes?
5. How do mixing and disentangling relate to stimulus statistics and gradient descent, as for efficient coding?

**Central claim (supported by the final results below).** Mixing and disentangling are outcomes of one optimisation.
Non-negative rates plus energy costs make the cheapest code *factorised along the axes in which the joint support of the
latent variables is rectangular* — correlation alone does not matter. Nonlinear downstream readouts, sparse energy-limited
classification and feedback push towards mixing; frequency-weighted gradient descent allocates sensitivity within factors
(efficient coding), transiently. Inside single neurons, dendritic nonlinearities trade abstraction for conjunction capacity
unless they saturate.

## Notebooks (run in order)

| # | notebook | question | main result it produces |
|---|---|---|---|
| 00 | `00_metrics_pure_mixed_disentangled` | all | one validated yardstick: neuron classes (pure / linear / nonlinear mixed), NMS index, disentanglement, PR, shattering dim, CCGP |
| 01 | `01_energy_nonnegativity_and_disentangling` | Q3 | analytic energy landscape of non-negative codes (support decides; a zero-correlation "diamond" support makes the optimal code mixed) + trained autoencoders across constraints |
| 02 | `02_statistics_gradient_descent_efficient_coding` | Q5 | correlation vs support; learning trajectories (transient mixing; rich vs lazy); efficient coding of orientation emerging through SGD; Fisher vs energy allocation |
| 03 | `03_task_demands_energy_and_learning_regime` | Q3 | NMS, SD, CCGP, disentanglement vs # nonlinear tasks × energy budget × init scale |
| 04 | `04_regions_and_cell_types_multiregion_rnn` | Q3 | sensory S1/S2 → association P RNN with fast/slow cell types (± Dale): where and in which type mixing emerges; delay memory by type |
| 05 | `05_single_neuron_capacity_vs_abstraction` | Q4 | Boolean capacity of point vs dendritic neurons, transitive inference from adjacent pairs, capacity–abstraction plane, branch vs soma selectivity |
| 06 | `06_synthesis_and_predictions` | — | six-panel paper figure + table of testable predictions |

Package `mixdis/`: `data.py` (factor environments: correlation, corner/band/diamond supports, skewed marginals,
orientation with cardinal prior, discrete conditions, Rigotti-style nonlinear dichotomies) · `models.py` (autoencoder,
multitask net, biological costs, trainer with snapshots) · `metrics.py` · `dendrite.py` (dendritic neurons incl. dCaAP
branches, factor-segregated synapses, transitive-inference data) · `rnn.py` (multi-region rate RNN with cell types) · `utils.py`.

## Setup (macOS)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd notebooks && jupyter lab
```

`FAST=1` (env var, or set `FAST = True` in the first cell) runs a smoke test in < 1 min per notebook. The default full
mode takes roughly 5–20 min per notebook on a laptop CPU (≈ 1 h in total; nb 01 and nb 04 are the longest).
Results (JSON) and figures (PNG + PDF) go to `results/` and `results/figures/`.

## Final results (full runs, Oct 2026)

All notebooks were executed in full mode (`FAST = False`). Each notebook now contains a **Result** cell after every
analysis and a summary at the end; `results/figures/06_paper_figure.png` is the draft main figure and notebook 06 holds the
full table of mechanisms and predictions.

### Question 3 — why mixing vs disentangling, where, and in which cell types

| finding | key numbers | notebook |
|---|---|---|
| Energy-optimal non-negative codes align with the axes in which the joint support is rectangular | analytic optimum: rectangular, correlated (ρ = 0.8) and corner-missing supports → factor axes (disent 0.99–1.00); **uncorrelated diamond support → 45°/315°** (0.01 w.r.t. factors, 1.00 w.r.t. f1 ± f2) | 01 |
| Trained ReLU autoencoders follow the prediction; linear ones have no preferred basis | diamond: 0.28–0.41 w.r.t. factors vs 0.88–0.97 w.r.t. f1 ± f2 at every energy level | 01 |
| Energy makes factorised codes reliable for rectangular supports | 0.62 ± 0.16 (λ = 0) → 0.80 ± 0.03 (λ = 0.1) | 01 |
| Correlation alone does not decide | no trend from ρ = 0 to 0.95 (4 seeds) | 01, 02 |
| Nonlinear readout demands buy mixing, at an energy cost, without losing abstraction | 0 → 16 nonlinear tasks: nonlinear-mixed 0.54 → 1.00, disentanglement 0.62 → 0.31, dimensionality ×3.2, energy ×3, CCGP ≈ 0.98 | 03 |
| In classification networks an activity cost creates sparse conjunctive units | linear tasks only: nonlinear-mixed 0.01 → 0.78 as λ goes 0 → 0.3 | 03 |
| Sensory → association gradient of mixing emerges from wiring | no feedback: sensory NMS 0.00 (disent 1.00), association 0.97–1.00 | 04 |
| Feedback makes sensory regions mixed late in the trial | sensory NMS 0.25–0.47 during the stimulus vs 0.65–1.00 in the delay | 04 |
| Cell types divide the work | sensory slow units: factorised delay memory of their own factor (decoding 0.98–1.00; NMS 0.33–0.80); fast units mixed (0.97–1.00) | 04 |

### Question 5 — statistics and gradient descent

| finding | key numbers | notebook |
|---|---|---|
| Correlated factors pass through a transient mixed phase | disentanglement 0.73 → 0.44 (step ≈ 126) → 0.85 | 02 |
| A rotated support drives learning towards mixing | 0.63 → 0.17 | 02 |
| Efficient coding of orientation emerges transiently via SGD | FI cardinal/oblique 1.36 and corr(√FI, prior) 0.78 at step ≈ 180–330; fades by step 6000 | 02 |
| With an activity cost, frequent values are coded by silence | corr(√sensitivity, density) = −0.83 / −0.86 | 02 |

### Question 4 — single-neuron computation

| finding | key numbers | notebook |
|---|---|---|
| Dendritic branches raise one neuron's conjunction capacity | Boolean functions of 3 inputs: point 40 % (= the 102 linearly separable ones), dCaAP 57 %, 2 ReLU 93 %, 4 sigmoid 98 %, 4 ReLU 100 % | 05 |
| Point neurons learn order relations as a linear value axis | transitive inference 1.00 at every distance; value-vs-rank R² 1.00 | 05 |
| ReLU and dCaAP branches trade abstraction for capacity; saturating branches do not | accuracy at symbolic distance 3: 0.62–0.69 (ReLU), 0.34 (dCaAP), 1.00 (sigmoid) | 05 |
| Clustered synapses give pure branches and a mixed soma | branches 83–85 % pure; soma 25–44 % nonlinear-mixed (≥ 4 neurons) | 05 |

### Not supported / open
* Rich vs lazy learning: no clear dissociation with Adam (only a weak lazy signature with linear tasks). Repeat with plain SGD and NTK scaling.
* Corner-missing support: trained 6-unit codes align with the diagonal (0.84–0.96 in the rotated basis), unlike the 2-neuron optimum.
* Dale's-law analyses pool excitatory and inhibitory units; split them.
* Several analyses use 2 seeds (trajectories, energy × task) or 1 seed per size (dendritic populations): use ≥ 5 for the paper.

## How the reading list maps onto the code

* *Disentanglement with biological constraints: a theory of functional cell types* (Whittington et al.) → nb 01–02 (non-negativity, energy, range dependence).
* *Efficient neural codes naturally emerge through gradient descent learning* (Benjamin, Zhang & Kording) → nb 02 sections 3–4.
* *Mixed selectivity: cellular computations for complexity* → nb 05 (dendritic sources of nonlinear mixing).
* *The inherent capacity of neurons to learn order relations and support abstract reasoning* → nb 05 transitive inference.
* *The neuron as a direct data-driven controller* → nb 05 discussion: an abstract linear value axis is what a controller neuron needs; natural extension with closed-loop objectives.
* *How heterogeneity shapes dynamics and computation* / *Adapting to time* → fast vs slow cell types in nb 04 (link to Project 1).
* *Neural spiking for causal inference and learning* → extension: replace backprop with a spike-discontinuity estimator and check whether the same mixing regimes appear.
