# Revising the paper without new experiments

Everything below is derived from the code and frozen artifacts in
`yield-curve-geometric-sde/` (config `paper_best.yaml` = `reports/best_model/config.yaml`,
scorecard `reports/best_model/scorecard.{csv,json}`). No model was retrained and no
new numbers were produced.

Read Part 0 first. Two things I found in the code change what the paper can claim,
independently of the reviewer's list.

---

## Part 0 — Two code-level findings that outrank the TODO list

### 0.1 The penalty is **not** evaluated on the Jacobian-linearized forecast

The paper's central premise (§2.5, §4.3, abstract, conclusion) is that the constraint is
evaluated on

$$\tilde y = D(z_t) + J(z_t)(\hat z - z_t).$$

That expression is `training/manifold_ops.py::linearized_curve_forecast`. In
`train_stage_b.py::compute_stage_b_loss` it is reached only through the `else` branch:

```python
use_persistence_residual = bool(training_cfg.get("use_persistence_residual", True))
...
if use_persistence_residual:
    y_constraint = persistence_residual_curve_forecast(decode_fn, z_hist, y_hist, z_t)
else:
    y_constraint = linearized_curve_forecast(decode_fn, z_start, z_t)
```

The locked config sets `use_persistence_residual: true`. A repo-wide grep confirms
`linearized_curve_forecast` is called nowhere else except `tests/test_manifold_ops.py`.

**So the run that produced every table penalizes**

$$\hat y = y_t + D(\hat z) - D(z_t), \qquad \text{projected against the base point } D(z_t),$$

i.e. the deviation being projected is

$$\underbrace{\big(y_t - D(z_t)\big)}_{r_t,\ \text{Stage-A residual, } \perp\text{-large, } \theta\text{-independent}} \;+\; \underbrace{\big(D(\hat z) - D(z_t)\big)}_{m(\theta),\ \text{the decoded move}}.$$

Consequences:

- The $(w\varepsilon/4)\lVert\delta\rVert^2$ bound, exactly as stated, describes a
  configuration that was **not run**. It remains true as a corollary about the
  linearized variant.
- The claim still survives in a corrected and arguably stronger form (Part 2, TODO 8).
  You are not losing the result; you are restating it about the code you ran.
- The penalty's *value* is dominated by $\lVert(I-P_\varepsilon)r_t\rVert^2$, which does
  not depend on the dynamics at all. With Stage-A RMSE $0.659$ and a 5-of-11-dimensional
  tangent space, that constant is roughly an order of magnitude larger than the fit loss
  — the penalty dominates the loss numerically while carrying almost no gradient.
  State this as an implication of the algebra, not as a logged measurement, unless you
  still have the Colab stdout (`constraint` column of the epoch printout) to quote.

This is a correctness fix, not a wording fix. Do not submit the current §4.3.

### 0.2 "Bond prices" are computed from robust-scaled yields

`constraints/bond_math.py::yield_to_discount` computes $P=\exp(-y\tau)$, and the $y$ it
receives is `decode_fn(z)` — the decoder output, which lives in **robust-scaled units**
($\tilde x = (x-\mathrm{median})/\mathrm{IQR}$), not decimals and not even percent.
`short_rate_from_curve` likewise returns a scaled value as $r$.

So $P$ is $\exp(-\text{scaled yield}\times\tau)$, which exceeds 1 whenever the scaled
yield is negative (i.e. below the training median — roughly half the sample). These are
not discount factors, and the residual $-\partial_\tau P + \nabla_z P^\top\mu_\mathbb{Q} - rP$
is not a no-arbitrage condition in any unit system.

Separately, `compute_dP_dtau` returns $-yP$, which is $\partial_\tau P$ only if $y$ is
constant in $\tau$; the true derivative is $-(y+\tau\,\partial_\tau y)P$. The slope term
is dropped, on top of the omitted Hessian.

This settles the reviewer's point 3 in the strong direction: **you cannot call it
no-arbitrage.** The zero-fix, no-rerun move is to rename it throughout — it is a
smoothness/monotonicity regularizer on a monotone transform of scaled yields — and to
say so plainly. Everything you currently report about it (that it destabilizes
multi-step rollouts) stays valid under the new name.

### 0.3 Three smaller code facts you will need

| Fact | Source | Consequence |
|---|---|---|
| `align_and_impute` drops incomplete rows, **then** reindexes to `freq="B"` over `[min, max]`, **then** `interpolate(method="time", limit_direction="both")` | `data/preprocess_curves.py` | The 30Y hole (2002-02-19 → 2006-02-08, **1,037 business days**) is dropped and then re-created and linearly interpolated. That is ~16% of the panel. See TODO 3 for the honest disclosure — the affected window falls entirely inside the train split. |
| `rolling_forecast_pca_var` and `rolling_forecast_nss` do `history = np.vstack([history, window_scores])` with `window = test[start:start+lookback]`, `lookback=21` | `baselines/pca_var.py:70`, `baselines/nelson_siegel.py:186` | Each step appends 21 rows of which 20 duplicate the previous step. Within ~24 steps the 500-row VAR estimation window is entirely overlapping duplicates. Both baselines are handicapped by construction. |
| `PCA(n_components=3)`, VAR lag 1 | `baselines/pca_var.py` | The baseline is rank-3, not rank-5. It is not rank-matched to the $d{=}5$ VAE anywhere in the paper. |

---

## Part 1 — The framing that survives with zero new runs

Four moves. Each one trades a claim you cannot support for one you can.

**1. Demote the theorem's scope, keep its force.**
From "decoder-Jacobian projection cannot train" to: *a tangent-space penalty evaluated at
the linearization point supplies no first-order off-tangent signal; what survives is
bounded increment shrinkage plus a decoder-curvature term weighted by a fixed vector
unrelated to the forecast.* This is provable from the code you ran, needs no seeds, and
is the paper's actual contribution.

**2. Demote every empirical comparison to a point estimate.**
You have one seed (42), no determinism flags, one GPU run. Say "point estimates from a
single training run; we did not conduct a multi-seed study and therefore do not claim
statistical indistinguishability." Delete "within run-to-run variation" everywhere — it
is currently doing inferential work that no measurement backs. A $5\times10^{-5}$ gap on
$0.122$ is $0.04\%$; you can simply show it and let the reader judge, which is honest and
costs you nothing, because your thesis *predicts* a null.

**3. Drop `sde_both` entirely.**
It is bit-identical to `sde_pde` across all eight metrics at all four horizons. From the
code, the Jacobian penalty has a nonzero gradient from epoch 1 (warmup scale $1/40$, not
0), so identical weights should be impossible. The distinguishing evidence — per-variant
best-checkpoint epoch and the logged penalty — lives in
`reports/checkpoints/stage_b/stage_b_*_history.json`, which the repo does not ship
("Checkpoints are not in git (Colab-only)"). You cannot adjudicate it, so report three
variants and one sentence: *a fourth configuration combining both penalties returned
metrics numerically identical to the PDE-only configuration; we could not determine from
the retained artifacts whether this reflects checkpoint selection during the Jacobian
warmup or a configuration fault, and we therefore exclude it.* Section 4.4 disappears.
This costs you nothing — §4.4 already concedes the equality is "corroborating evidence at
best."

**4. Retitle the residual, retitle the comparison.**
Not "no-arbitrage PDE residual" (§0.2) and not "the comparison [cited] pose." Their
benchmark uses the decoder Jacobian as a volatility basis with nonlinear re-encoding
$D(E(\cdot))$; yours is a ridge projection of a decoded move against a linearized base
point. Call it what it is: *a linearized tangent-penalty surrogate inspired by decoder
geometry.* Then your negative result is about **your** construction, which you can fully
defend, instead of about theirs, which you did not implement.

Net effect on the paper: the title becomes something like *"A tangent-space penalty that
cannot see off-manifold error: a constraint ablation on latent yield-curve dynamics"*;
the abstract's four claims become one theorem plus three descriptive observations; and
nothing you assert requires a number you do not have.

---

## Part 2 — TODO-by-TODO resolution

Four have real values. Seven cannot be filled without runs and are resolved by deletion
or restatement — which is the point: a paper with no TODOs and narrower claims is
submittable; a paper with 11 TODOs is not.

### TODO 1 — §2.3, PCA-5 reconstruction RMSE
**Cannot fill.** Requires the processed panel (`data/processed/` is gitignored).
Also note the implemented PCA baseline is rank-3, so even a rerun would not be
rank-matched without changing `n_components`.
**Action:** delete the sentence. Add to Limitations: *"We do not report a rank-matched
linear reconstruction baseline, so we cannot separate the share of the Stage-A
reconstruction error attributable to the rank constraint from that attributable to the
decoder."*

### TODO 2 and TODO 5 — the ridge constant $\varepsilon$
**FILLED: $\varepsilon = 10^{-5}$.**
Reproducibility trap worth a footnote: `constraints.projection_eps: 1e-5` in the config is
**not** threaded into training. `total_constraint_loss` calls `manifold_projection_loss`
without an `eps` argument, so training uses the Python default `eps=1e-5` in
`project_curve_to_manifold`. The two happen to coincide, so the number is right, but
changing the config alone would not change training.

### TODO 3 — §2.6 seeds, determinism, checkpoint epochs
**Partly filled.**
- Seed: **42**, single seed (`project.seed`, applied by `set_seed`).
- Determinism: **not enabled.** `set_seed` calls only `torch.manual_seed` and
  `np.random.seed`; there is no `torch.use_deterministic_algorithms`, no
  `cudnn.deterministic`, no dataloader worker seeding. Training ran on a Colab GPU, so
  runs are not bit-reproducible.
- Best-checkpoint epochs: **not recoverable.** Written to
  `reports/checkpoints/stage_b/stage_b_{ablation}_history.json`, not in the repo.

Paste-ready:

```latex
\paragraph{Reproducibility.} All reported runs use a single seed ($42$), set via
\texttt{torch.manual\_seed} and \texttt{numpy.random.seed}. Deterministic kernels were
not enabled and training ran on a cloud GPU, so runs are not bit-reproducible. Per-variant
best-checkpoint epochs are written to the Stage~B history files, which are not retained in
the released artifacts; we therefore cannot report them, and we draw no conclusion that
depends on them.
```

### TODO 4 — §2.6 multi-seed standard deviation
**Cannot fill.** One seed exists.
**Action:** delete the sentence and remove every downstream appeal to it — abstract
("scale of run-to-run noise"), §3.1 table caption, §3.2, §4.2, §4.5, §4.7 bullet 2.
Replacement sentence: *"All comparisons below are point estimates from a single training
run per configuration. We did not vary the seed, and we therefore make no claim about
whether the sub-$10^{-4}$ differences between configurations are reproducible."*

If — and only if — you actually observed rerun variation during development (the comment
in `paper_best.yaml` says "run-to-run RMSE may vary ~4th decimal"), you may write:
*"informal reruns during development varied in the fourth decimal; this was not a
controlled study and we do not treat it as a variance estimate."* Only keep this if it is
a true recollection.

### TODO 6 — the numeric bound
**FILLED.** $w\varepsilon/4 = 0.3\times10^{-5}/4 = 7.5\times10^{-7}$.
So the linearized-variant penalty obeys
$w\lVert B_\varepsilon\delta\rVert^2 \le 7.5\times10^{-7}\,\lVert\delta\rVert^2$.

### TODO 7 — typical $\mathcal{L}_{\mathrm{fit}}$ magnitude
**Fill by derivation, labelled as such.** The paper misstates the objective: the code is

$$\mathcal{L}_{\mathrm{fit}} = 0.05\,\lVert\hat z_{t+1}-z_{t+1}\rVert^2 + 10\,\lVert\hat y_{t+h}-y_{t+h}\rVert^2$$

(`latent_fit_weight: 0.05`, `curve_loss_weight: 10.0`) — fix §2.4 to show the $0.05$.
At the reported $h{=}1$ curve RMSE of $0.029$, the curve term is
$10\times(0.029)^2\approx 8.5\times10^{-3}$. Write "of order $10^{-2}$, implied by the
reported RMSE" — not "logged."

### TODO 8 — the $J^\top J$ spectrum
**Cannot fill** (no Stage-A checkpoint in the repo). This TODO also encodes a claim that
is **backwards**: $\lambda\varepsilon^2/(\lambda+\varepsilon)^2$ peaks at
$\lambda=\varepsilon$ and vanishes as $\lambda\to0$ *and* as $\lambda\to\infty$. Collapsed
latent directions ($\lambda\to0$) are where the surviving force is *weakest*, not
strongest. Delete the posterior-collapse sentence; it argues against you and is wrong.

Paste-ready replacement for the whole of §4.3, corrected to match the code
(§0.1) and stated so that no unmeasured quantity appears:

```latex
\subsection{Why the tangent penalty carries no off-tangent signal}
\label{sec:analysis-inert}

Write $r_t = y_t - D_\theta(z_t)$ for the Stage~A re-encoding residual at the forecast
origin and $m(\theta) = D_\theta(\hat z) - D_\theta(z_t)$ for the decoded move. The
penalty is evaluated on the persistence-residual forecast
$\hat y = y_t + m(\theta)$ against the base point $D_\theta(z_t)$, so with
$P_\varepsilon = J(J^\top J + \varepsilon I)^{-1}J^\top$ it equals
\begin{equation}
  w\big\lVert (I - P_\varepsilon)\,\big(r_t + m(\theta)\big)\big\rVert^2 .
  \label{eq:penalty-actual}
\end{equation}
Only $m$ depends on the dynamics parameters. Three consequences follow.

\emph{(i) The value is dominated by a term the dynamics cannot affect.} $r_t$ is fixed by
the frozen Stage~A model, and its off-tangent component enters
Eq.~\eqref{eq:penalty-actual} as an additive constant in $\theta$.

\emph{(ii) The gradient carries no first-order off-tangent information.} Differentiating,
$\nabla_\theta = 2w\,\big[(I-P_\varepsilon)(r_t+m)\big]^\top (I-P_\varepsilon)\,
\partial_\theta m$, with $\partial_\theta m = J(\hat z)\,\partial_\theta \hat z$. Writing
$A = J^\top J$ and using $I - (A+\varepsilon I)^{-1}A = \varepsilon(A+\varepsilon I)^{-1}$,
\begin{equation}
  (I - P_\varepsilon)\,J(z_t) \;=\; \varepsilon\,J(A+\varepsilon I)^{-1} ,
  \label{eq:offtangent}
\end{equation}
so the off-tangent sensitivity of the decoded move is $O(\varepsilon)$ plus a decoder
curvature term $O(\lVert\delta\rVert)$ arising from $J(\hat z) - J(z_t)$. The force the
drift receives is therefore second order, and its direction is set by the fixed vector
$(I-P_\varepsilon)r_t$ --- a property of Stage~A reconstruction, not of whether the
forecast left the manifold.

\emph{(iii) In the exactly linearized variant the signal vanishes identically.} If the
move is replaced by its first-order form $J\delta$, Eq.~\eqref{eq:offtangent} gives
residual $\varepsilon J(A+\varepsilon I)^{-1}\delta \equiv B_\varepsilon\delta$, and
diagonalizing $A$ with eigenvalues $\lambda_i$,
\begin{equation}
  \lVert B_\varepsilon\delta\rVert^2
  = \sum_{i=1}^{d} \frac{\lambda_i\varepsilon^2}{(\lambda_i+\varepsilon)^2}\,\delta_i^2
  \;\leq\; \frac{\varepsilon}{4}\,\lVert\delta\rVert^2 ,
\end{equation}
uniformly in $J$, since the scalar coefficient is maximized at $\lambda_i = \varepsilon$
and vanishes as $\lambda_i \to 0$ and as $\lambda_i \to \infty$. With $w = 0.3$ and
$\varepsilon = 10^{-5}$ the weighted penalty is bounded by
$7.5\times10^{-7}\lVert\delta\rVert^2$, against a fit loss of order $10^{-2}$; in the
limit $\varepsilon \to 0$ it is identically zero for every $\delta$ and every parameter
setting. What remains is an anisotropic shrinkage of the latent increment --- output
regularization, not a geometric constraint.

Two modifications would restore a nondegenerate signal: evaluating the projection away
from the linearization point, or replacing the tangent map with the nonlinear re-encoding
$\Pi_{\mathcal{M}}(y) = D_\theta(E_\phi(y))$, which is the form used as a benchmark by
\citet{NoArbVAEYieldCurves}. Note that with piecewise-linear decoder activations even the
nonlinear move stays in $\operatorname{span}(J)$ whenever $\hat z$ and $z_t$ fall in the
same linear region.
```

### TODO 9 — §3.2 placeholder figure (penalty / gradient-norm trace)
**Cannot produce** (needs `stage_b_*_history.json`; and the history dict logs only the
combined `constraint` scalar, never a gradient norm — so even with the JSONs the figure as
captioned is not reconstructible).
**Action:** delete the figure and its caption. You already have real figures in
`reports/best_model/figures/`. If you want a second figure, use
`stage_b_sde_pde_rmse_vs_horizon.png` beside `stage_b_sde_only_rmse_vs_horizon.png` — the
divergence contrast is your sharpest empirical result and those PNGs exist. The inertness
claim now rests on Eq.~\eqref{eq:penalty-actual}, which needs no figure.

### TODO 10 — §4.4 `sde_both` = `sde_pde`
**Cannot resolve.** See Part 1, move 3: drop the variant, drop §4.4, add the one-sentence
exclusion note. Remove `sde_both` from Table 1, Tables 2–4, §3.3 item 3, and the
abstract's "four soft-training configurations" (now three).

### TODO 11 — Diebold–Mariano tests
**Cannot run:** requires per-date forecast errors; `reports/forecasts/` is not in the repo
and the scorecard stores only aggregates.
**Action:** delete the TODO, keep the limitation, and make it do real work:

```latex
\item \textbf{No formal significance test.} We report point estimates from a single run
per configuration and perform no paired inference. Differences of order $10^{-5}$ between
the constrained and unconstrained variants, and the $1.3\%$ margin over persistence at
$h{=}21$, are therefore descriptive. A Diebold--Mariano test with HAC standard errors is
the appropriate instrument, and the overlapping forecast windows at $h>1$ induce serial
correlation that such a test would need to accommodate.
```

---

## Part 3 — Other passages that must change (not TODOs, but wrong as written)

| Location | Current | Correct |
|---|---|---|
| §2.5 opening | "Constraints are evaluated on a Jacobian-linearized forecast $\tilde y = D(z_t)+J(z_t)(\hat z - z_t)$" | Evaluated on the persistence-residual forecast $y_t + D(\hat z) - D(z_t)$, projected against base point $D(z_t)$ (§0.1). |
| §2.4 $\mathcal{L}_\mathrm{fit}$ | latent term unweighted | $0.05\cdot$latent $+\ 10\cdot$curve |
| §2.5, §3, §4.6, abstract | "no-arbitrage residual / PDE penalty" | Rename. Prices are $\exp(-\tilde y\tau)$ on robust-scaled yields, $r$ is a scaled yield, and $\partial_\tau P$ drops the $\tau\partial_\tau y$ term. Describe as a heuristic smoothness/monotonicity regularizer (§0.2). |
| §2.2 | "the sample beginning 2001-07-01, the earliest date at which all tenors are jointly available" | DGS1MO starts 2001-07-31. Also disclose the 30Y gap: 1,037 business days (2002-02-19 → 2006-02-08) are dropped by the completeness filter and then reinstated by the business-day reindex and linearly interpolated — about 16% of the panel. **Mitigation you can state truthfully:** at the 70/15/15 chronological split that window lies entirely inside the training split, so it contaminates the learned representation and the robust scaler but not the held-out evaluation. State the retrieval date, $T$, and the three split date ranges. |
| §2.2 | "imputation by time interpolation followed by forward/back-filling" | `limit_direction="both"` plus `bfill` is applied before splitting, so interpolated values near split boundaries can use future observations. Disclose. |
| §2.2 | "reindex to a regular business-day calendar" | `freq="B"` also creates rows for US market holidays, which are then interpolated into synthetic observations. Disclose. |
| §2.2 LevelScript | "level/shape decomposition after robust scaling" | Accurate as to the code, but each tenor has its own IQR, so subtracting the scaled 1Y is not a parallel level shift, and it differs from the cited implementation. Say so; do not claim it separates level from shape economically. |
| §2.7 baselines | "rolling, expanding-window" | Contradictory *and* both baselines re-append the full 21-day window each step, duplicating ~20 of 21 rows into a 500-row VAR window (§0.3). Either disclose this as a defect that inflates baseline error, or drop the claim that parametric baselines are uncompetitive. The safest sentence: *"our NSS and PCA--VAR implementations refit on an overlapping-window history that duplicates observations; their errors should be read as upper bounds, and we do not claim these model classes are uncompetitive in general."* |
| §2.7 / §4.6 | "PCA + VAR" | rank-3, VAR(1); not rank-matched to $d{=}5$. |
| Throughout | "off-manifold distance", "manifold floor" | "re-encoding error" — $D(E(y))$ is not the nearest point on the decoder manifold. |
| §4.3 | "weight decay on the drift in the costume of a geometric constraint" | "output-space regularization of the latent increment." (Good line; wrong term.) |
| Abstract, Intro, Conclusion | "we carry out that comparison", "answering the comparison \citet{} pose" | You implemented a different construction. Reframe as a linearized tangent-penalty surrogate (Part 1, move 4). |
| §3.2, §4.5 | Jacobian "attains nominally the lowest tangent residual" | Also report `manifold_correction_gain`, which is in your own scorecard and goes the other way at $h{=}21$ ($-0.0065$ jacobian vs $-0.0074$ only; more negative is better by the repo's own definition). Reporting only the metric that favors the Jacobian variant is selection you don't need — your thesis predicts a null, so a mixed result is evidence *for* you. |

---

## What the paper claims after all this

1. **Theorem (provable, no runs needed).** A tangent-space penalty of this form supplies
   no first-order off-tangent training signal; the surviving gradient is bounded increment
   shrinkage plus a curvature term weighted by the fixed Stage-A reconstruction residual.
   In the exactly linearized limit it is identically zero.
2. **Observation (single run, point estimates).** Adding the penalty moves test RMSE by
   $\le 7\times10^{-5}$ at all trained horizons and reverses sign at $h{=}63$, which is
   what the theorem predicts.
3. **Observation (single run).** The $h{=}1$-gated regularizer bundle degrades multi-step
   rollouts by two orders of magnitude by $h{=}63$; the design does not identify which
   term is responsible, and the residual is misspecified in units.
4. **Scope.** One market, one seed, soft penalties only, a re-encoding error of $0.659$
   that bounds what any genuine manifold projection could achieve, and a data pipeline
   whose interpolation defect is confined to the training split.

That is a coherent short paper. It is smaller than the current draft and every sentence
in it is one you can defend.
