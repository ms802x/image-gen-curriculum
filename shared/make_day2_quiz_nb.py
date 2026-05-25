"""Generate day2/day2_ddpm_quiz.ipynb -- self-test notebook to check DDPM understanding.

Format per quiz:
- Markdown cell: task description
- Markdown cell: hint inside a <details> collapsible block
- Code cell: stub with `YOUR CODE HERE` for the user to fill in
- Markdown cell: solution inside a <details> collapsible block

Progressive difficulty from per-step forward all the way to full Algorithm 1 + 2
implementation, plus conceptual questions and a final full-pipeline challenge."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "day2" / "day2_ddpm_quiz.ipynb"
OUT.parent.mkdir(parents=True, exist_ok=True)


def md(text):  return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}
def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


cells = []

# ============================================================================
# Title + how to use
# ============================================================================
cells.append(md(r"""\
# DDPM — self-check quiz

A test of your DDPM understanding through code. Each quiz follows the same format:

1. **Task** — what you need to write.
2. **Hint** — collapsible. Click to reveal if stuck. (Try without first.)
3. **Code cell** — replace `# YOUR CODE HERE` with your implementation. Run to test.
4. **Solution** — collapsible reference implementation. **Try yours first, then compare.**

**You will not learn by clicking solutions immediately.** Click them after you've genuinely tried and either succeeded or got stuck. Comparing your working code to a reference teaches more than reading the reference cold.

Companion material:
- [`ddpm_paper_tutorial.md`](./ddpm_paper_tutorial.md) — the prose walkthrough of the paper
- [`day2_paper_math_to_code.ipynb`](./day2_paper_math_to_code.ipynb) — math + visualizations
- [`day2_diffusion.ipynb`](./day2_diffusion.ipynb) — full training on anime
"""))

# ============================================================================
# Setup
# ============================================================================
cells.append(md("## Setup"))
cells.append(code("""\
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0); np.random.seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

# Spiral data (same as the math notebook) -- 2D toy you'll train on at the end
def make_spiral(n=5000, noise=0.03):
    theta = torch.linspace(0, 4 * math.pi, n)
    r = theta / (4 * math.pi)
    x = r * torch.cos(theta) + noise * torch.randn(n)
    y = r * torch.sin(theta) + noise * torch.randn(n)
    return torch.stack([x, y], dim=1) * 2.0

x0_data = make_spiral()
print("x0_data shape:", x0_data.shape)
"""))

# ============================================================================
# Quiz 1 — Per-step forward
# ============================================================================
cells.append(md(r"""\
## Quiz 1 — Per-step forward $q(x_t \mid x_{t-1})$

**Task.** Write a function `per_step_forward(x_prev, beta)` that takes the previous state `x_prev` (a tensor) and a scalar `beta`, and returns a sample from the per-step forward distribution.

<details>
<summary>💡 <b>Hint</b> (click to reveal)</summary>

Paper equation 2:
$$x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)$$

In code:
- Use `torch.randn_like(x_prev)` for $\varepsilon$.
- Use `torch.sqrt(...)` on the scalars.

</details>
"""))
cells.append(code("""\
def per_step_forward(x_prev, beta):
    # YOUR CODE HERE
    pass

# --- Test: variance preservation ---
torch.manual_seed(0)
x_prev = torch.randn(10000)                    # 10k samples, unit variance
for b in [0.01, 0.1, 0.5]:
    x_t = per_step_forward(x_prev, torch.tensor(b))
    print(f"beta={b}:  var(x_prev)={x_prev.var():.3f}  var(x_t)={x_t.var():.3f}  (theory: 1.0)")
"""))
cells.append(md(r"""\
<details>
<summary>✅ <b>Solution</b> — try yours first, then compare</summary>

```python
def per_step_forward(x_prev, beta):
    eps = torch.randn_like(x_prev)
    return torch.sqrt(1 - beta) * x_prev + torch.sqrt(beta) * eps
```

Three line. The two scalars together preserve unit variance:
$\text{Var}(x_t) = (1-\beta) \cdot 1 + \beta \cdot 1 = 1$.

</details>
"""))

# ============================================================================
# Quiz 2 — Schedule
# ============================================================================
cells.append(md(r"""\
## Quiz 2 — Compute the schedule

**Task.** Given `betas` (a tensor of length $T$), compute `alphas` and `alpha_bars`.

<details>
<summary>💡 <b>Hint</b></summary>

Paper definitions:
$$\alpha_t := 1 - \beta_t, \qquad \bar\alpha_t := \prod_{s=1}^{t} \alpha_s$$

The product over indices is what `torch.cumprod(..., dim=0)` returns as a vector.

</details>
"""))
cells.append(code("""\
T = 200
betas = torch.linspace(1e-4, 0.05, T)

def compute_schedule(betas):
    # YOUR CODE HERE
    alphas = None
    alpha_bars = None
    return alphas, alpha_bars

alphas, alpha_bars = compute_schedule(betas)
if alphas is not None and alpha_bars is not None:
    print(f"alphas[0]      = {alphas[0]:.6f}    (expected ~0.9999)")
    print(f"alpha_bars[0]  = {alpha_bars[0]:.6f}    (expected ~0.9999, same as alphas[0])")
    print(f"alpha_bars[T-1]= {alpha_bars[-1]:.6f}    (expected ~0.006 -- the chain has consumed nearly all signal)")
"""))
cells.append(md(r"""\
<details>
<summary>✅ <b>Solution</b></summary>

```python
def compute_schedule(betas):
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return alphas, alpha_bars
```

`alpha_bars` is the **cumulative** product so we can index `alpha_bars[t]` directly at any timestep instead of recomputing the running product.

</details>
"""))

# ============================================================================
# Quiz 3 — Closed-form forward
# ============================================================================
cells.append(md(r"""\
## Quiz 3 — Closed-form forward $q(x_t \mid x_0)$

**Task.** Write `q_sample(x_0, t, alpha_bars)` that samples $x_t$ in one shot, without iterating through the chain.

<details>
<summary>💡 <b>Hint</b></summary>

Paper equation 4 in sample form:
$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$$

- `t` may be a scalar or a 1-D tensor of timestep indices (so the function should index `alpha_bars[t]`).
- Return both $x_t$ and the noise $\varepsilon$ you sampled (you'll need both for training).

</details>
"""))
cells.append(code("""\
def q_sample(x_0, t, alpha_bars):
    # YOUR CODE HERE
    # Return (x_t, eps)
    pass

# --- Test: match mean/variance against theory at several t values ---
torch.manual_seed(0)
x_0_test = torch.full((50_000,), 0.5)
for t_check in [10, 100, 199]:
    x_t, eps = q_sample(x_0_test, torch.tensor([t_check]).expand(50_000), alpha_bars)
    ab_t = alpha_bars[t_check]
    theory_mean = ab_t.sqrt().item() * 0.5
    theory_std = (1 - ab_t).sqrt().item()
    print(f"t={t_check:3d}   yours: mean={x_t.mean():+.4f} std={x_t.std():.4f}    theory: mean={theory_mean:+.4f} std={theory_std:.4f}")
"""))
cells.append(md(r"""\
<details>
<summary>✅ <b>Solution</b></summary>

```python
def q_sample(x_0, t, alpha_bars):
    ab = alpha_bars[t]
    while ab.dim() < x_0.dim():
        ab = ab.unsqueeze(-1)
    eps = torch.randn_like(x_0)
    x_t = ab.sqrt() * x_0 + (1 - ab).sqrt() * eps
    return x_t, eps
```

The `while ab.dim() < x_0.dim()` handles broadcasting for higher-dimensional data (e.g., images). For 1-D `x_0` like the test, the loop runs zero times.

</details>
"""))

# ============================================================================
# Quiz 4 — Predict x_0 from x_t and noise
# ============================================================================
cells.append(md(r"""\
## Quiz 4 — Algebraic recovery: predict $x_0$ from $x_t$ and $\varepsilon$

**Task.** Write `predict_x0_from_eps(x_t, eps, t, alpha_bars)` that recovers $x_0$ exactly given the noisy $x_t$, the noise $\varepsilon$ that was used to construct it, and the timestep $t$.

This is the same formula used in the DDIM sampler and as part of the clip-x0 trick at sample time.

<details>
<summary>💡 <b>Hint</b></summary>

Invert the equation from Quiz 3. Starting from $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$, solve for $x_0$:
$$x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\varepsilon}{\sqrt{\bar\alpha_t}}$$

</details>
"""))
cells.append(code("""\
def predict_x0_from_eps(x_t, eps, t, alpha_bars):
    # YOUR CODE HERE
    pass

# --- Test: round-trip should give back x_0 to machine precision ---
torch.manual_seed(0)
x_0 = torch.randn(4, 3, 8, 8)
t = 100
eps = torch.randn_like(x_0)
ab = alpha_bars[t]
x_t = ab.sqrt() * x_0 + (1 - ab).sqrt() * eps
x_0_recovered = predict_x0_from_eps(x_t, eps, torch.tensor(t), alpha_bars)
if x_0_recovered is not None:
    print(f"max |x_0_recovered - x_0_true| = {(x_0_recovered - x_0).abs().max():.2e}    (should be ~1e-6 or smaller)")
"""))
cells.append(md(r"""\
<details>
<summary>✅ <b>Solution</b></summary>

```python
def predict_x0_from_eps(x_t, eps, t, alpha_bars):
    ab = alpha_bars[t]
    while ab.dim() < x_t.dim():
        ab = ab.unsqueeze(-1)
    return (x_t - (1 - ab).sqrt() * eps) / ab.sqrt()
```

</details>
"""))

# ============================================================================
# Quiz 5 — Reverse step mean (epsilon-parameterization)
# ============================================================================
cells.append(md(r"""\
## Quiz 5 — Reverse step mean (paper equation 11)

**Task.** Implement `reverse_mean(x_t, eps_pred, t, alphas, alpha_bars, betas)` returning the mean of $p_\theta(x_{t-1} \mid x_t)$ using $\varepsilon$-parameterization.

This is what gets called inside the sampling loop.

<details>
<summary>💡 <b>Hint</b></summary>

Paper equation 11:
$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta(x_t, t)\right)$$

Three coefficients: $\frac{1}{\sqrt{\alpha_t}}$ outside, $\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}$ on the noise term.

</details>
"""))
cells.append(code("""\
def reverse_mean(x_t, eps_pred, t, alphas, alpha_bars, betas):
    # YOUR CODE HERE
    pass

# --- Test: if eps_pred = eps_true (perfect model), the mean should equal the true x_{t-1} on average ---
# This is hard to test directly without a trained model, but we can at least check shapes & no NaN.
torch.manual_seed(0)
x_t_test = torch.randn(8, 3, 8, 8)
eps_test = torch.randn_like(x_t_test)
mean = reverse_mean(x_t_test, eps_test, torch.tensor(50), alphas, alpha_bars, betas)
if mean is not None:
    print(f"output shape: {mean.shape}   range: [{mean.min():.3f}, {mean.max():.3f}]   NaN? {torch.isnan(mean).any().item()}")
"""))
cells.append(md(r"""\
<details>
<summary>✅ <b>Solution</b></summary>

```python
def reverse_mean(x_t, eps_pred, t, alphas, alpha_bars, betas):
    a_t = alphas[t]
    ab_t = alpha_bars[t]
    b_t = betas[t]
    return (x_t - (b_t / (1 - ab_t).sqrt()) * eps_pred) / a_t.sqrt()
```

Or equivalently, write it via `predict_x0_from_eps` and the posterior mean formula (paper eq 7). Both give the same answer.

</details>
"""))

# ============================================================================
# Quiz 6 — Algorithm 1 (one training step)
# ============================================================================
cells.append(md(r"""\
## Quiz 6 — Algorithm 1: one training step

**Task.** Implement `train_step(model, x_0_batch, T, alpha_bars, optim)` that does one step of DDPM training. Returns the loss (as a float).

<details>
<summary>💡 <b>Hint</b></summary>

Paper Algorithm 1:
1. Sample timesteps `t ~ Uniform({0, ..., T-1})` for each sample in the batch.
2. Sample noise `eps ~ N(0, I)` per sample.
3. Build `x_t = sqrt(ab_t) * x_0 + sqrt(1-ab_t) * eps` using `q_sample` (re-use!).
4. Run `model(x_t, t)` to get `eps_pred`.
5. Compute MSE between `eps_pred` and the *true* `eps`.
6. Standard PyTorch backward: zero grads, backward, step.
7. Return loss.

You'll need `q_sample` from Quiz 3.

</details>
"""))
cells.append(code("""\
def train_step(model, x_0_batch, T, alpha_bars, optim):
    # YOUR CODE HERE -- 5 lines of computation + 3 lines of optimizer
    pass

print("Quiz 6 function defined. It will be exercised by the final challenge at the end.")
"""))
cells.append(md(r"""\
<details>
<summary>✅ <b>Solution</b></summary>

```python
def train_step(model, x_0_batch, T, alpha_bars, optim):
    B = x_0_batch.size(0)
    t = torch.randint(0, T, (B,), device=x_0_batch.device)
    x_t, eps = q_sample(x_0_batch, t, alpha_bars)
    eps_pred = model(x_t, t)
    loss = F.mse_loss(eps_pred, eps)
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.item()
```

</details>
"""))

# ============================================================================
# Quiz 7 — Algorithm 2 (sampling loop)
# ============================================================================
cells.append(md(r"""\
## Quiz 7 — Algorithm 2: full sampling loop

**Task.** Implement `sample(model, shape, T, alphas, alpha_bars, betas, device)` that generates samples by running the reverse process from $x_T \sim \mathcal{N}(0, I)$ down to $x_0$.

<details>
<summary>💡 <b>Hint</b></summary>

Paper Algorithm 2:
1. `x = torch.randn(shape, device=device)` (this is $x_T$).
2. For `t` going from $T-1$ down to $0$:
    - Run model to predict `eps_pred`.
    - Compute mean via `reverse_mean` from Quiz 5.
    - If $t > 0$: add stochastic noise `sigma_t * z` where $\sigma_t = \sqrt{\beta_t}$.
    - If $t = 0$: just use the mean (no noise).
3. Return the final `x`.

Wrap the whole function in `@torch.no_grad()` so gradients aren't tracked.

</details>
"""))
cells.append(code("""\
@torch.no_grad()
def sample(model, shape, T, alphas, alpha_bars, betas, device="cuda"):
    # YOUR CODE HERE
    pass

print("Quiz 7 function defined. Will be exercised by the final challenge.")
"""))
cells.append(md(r"""\
<details>
<summary>✅ <b>Solution</b></summary>

```python
@torch.no_grad()
def sample(model, shape, T, alphas, alpha_bars, betas, device="cuda"):
    model.eval()
    x = torch.randn(shape, device=device)
    for t in reversed(range(T)):
        t_b = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps_pred = model(x, t_b)
        mean = reverse_mean(x, eps_pred, t, alphas, alpha_bars, betas)
        if t > 0:
            x = mean + betas[t].sqrt() * torch.randn_like(x)
        else:
            x = mean
    return x
```

</details>
"""))

# ============================================================================
# Quiz 8 — Conceptual
# ============================================================================
cells.append(md(r"""\
## Quiz 8 — Conceptual questions

For each, think your answer first, then click for the canonical one. (No code cell here — it's a mental check.)

### 8.1 — Why predict $\varepsilon$ instead of $x_0$ or $\mu$ directly?

<details>
<summary>✅ <b>Answer</b></summary>

Two reasons:

1. **Consistent target scale across all $t$.** $\varepsilon \sim \mathcal{N}(0, I)$ has the same magnitude at every timestep. Predicting $x_0$ would mean target magnitudes that vary by orders of magnitude depending on noise level — harder to fit with one network.

2. **Direct connection to score matching.** $\varepsilon_\theta(x_t, t)$ is proportional to $-\nabla_{x_t} \log q(x_t \mid x_0)$ — the score of the noised distribution. This unifies DDPM with score-based generative models (Yang Song 2019).

Empirically, the original paper found ε-prediction with $L_\text{simple}$ gave the best FID — see paper Table 2.

</details>

### 8.2 — Why is the forward noise covariance $\beta_t I$ (identity matrix)?

<details>
<summary>✅ <b>Answer</b></summary>

It's a deliberate **isotropy** choice: independent Gaussian noise on every dimension, with the same variance per dimension, and zero correlation between dimensions.

Concretely: the $(i, j)$ entry of the covariance is the covariance of noise added to coordinate $i$ and coordinate $j$. The identity matrix means "$\beta_t$ on the diagonal, zero off-diagonal" — independent noise per pixel, no structured (e.g., spatial) correlations.

If we used a non-identity covariance — say a Gaussian blur kernel — the forward process would be a *smoothing* instead of a noise-addition, and the reverse process would need to learn to "unblur" rather than "denoise." The identity is the simplest and most generic choice; alternative covariances exist (cold diffusion, blur diffusion) but require a different reverse process.

</details>

### 8.3 — What is the "score function" in the context of DDPM, and how does it relate to $\varepsilon_\theta$?

<details>
<summary>✅ <b>Answer</b></summary>

The **score** is the gradient of the log-density:
$$\nabla_x \log p(x)$$
It's a vector field pointing toward higher density.

For Gaussian-noised data $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$:
$$\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{\varepsilon}{\sqrt{1-\bar\alpha_t}}$$

So **the score is the noise, up to a known scalar**. Training the model to predict $\varepsilon$ is equivalent to training it to predict the score field (one is a scaled version of the other). This is the bridge between DDPM and score-based generative modeling.

</details>

### 8.4 — What would happen if you removed the $\sigma_t\,z$ term from the reverse step?

<details>
<summary>✅ <b>Answer</b></summary>

You'd be running the reverse process deterministically — that's **DDIM with $\eta = 0$** (Song et al. 2020).

Pros of removing it:
- Reproducible: same starting noise → same output.
- Allows step-skipping (DDIM lets you use 20–50 steps instead of 1000 at similar quality).
- Most modern fast samplers (DPM-Solver, EDM) are deterministic.

Cons:
- Reduced sample diversity. Multiple starting noises that are close can map to the same output point — you lose coverage of the data distribution.
- Sometimes slightly lower fine-detail quality at full step count (compounding model errors aren't averaged out by the noise injection).

</details>

### 8.5 — In paper equation 14, $L_\text{simple} = \mathbb{E}[\|\varepsilon - \varepsilon_\theta(x_t, t)\|^2]$, the *weighting* from equation 12 is dropped. Why does this empirically give *better* sample quality?

<details>
<summary>✅ <b>Answer</b></summary>

The full equation 12 has a weight $\frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar\alpha_t)}$ in front of the MSE. For the paper's linear $\beta$ schedule, this weight is **largest at small $t$** (near-clean data) and small at large $t$ (near-pure noise).

Predicting noise at small $t$ is easy — the input is close to clean and the noise is small. Spending most of the training capacity on easy timesteps wastes it. Dropping the weight effectively *down-weights* small-$t$ terms and lets the network focus on harder mid-to-large-$t$ denoising tasks.

Empirically (paper §3.4): this rebalancing improves visual sample quality even though the simplified objective is no longer a proper variational bound on log-likelihood.

</details>
"""))

# ============================================================================
# Quiz 9 — Final challenge: train your own DDPM
# ============================================================================
cells.append(md(r"""\
## Quiz 9 — Final challenge: train a working DDPM end-to-end

Using your functions from Quizzes 1–7, train a tiny MLP on the spiral and verify the samples actually match the data distribution.

We provide:
- The spiral data (already loaded as `x0_data`)
- The schedule (already computed: `betas`, `alphas`, `alpha_bars`)
- A `ResMLP` architecture stub for the noise predictor $\varepsilon_\theta$

You write:
- The training loop (call `train_step` repeatedly)
- The sampling call (call `sample` once after training)
- The verification: compute the median NN distance from samples to the spiral

**Pass criterion: ratio (samples to spiral / real-to-real) < 2.0×.** Excellent < 1.2×.

<details>
<summary>💡 <b>Hint — training recipe that works</b></summary>

- Use `AdamW(lr=2e-3, weight_decay=1e-4)` with `CosineAnnealingLR(T_max=TRAIN_STEPS)`.
- Batch size 512.
- 30k training steps.
- Move data and schedule to DEVICE.

Wait time: ~70 seconds on GPU, ~10 minutes on CPU.

</details>
"""))

cells.append(code("""\
# Provided: noise-predictor architecture
class TimeEmb(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class ResMLP(nn.Module):
    def __init__(self, hidden=256, t_dim=128, n_blocks=4):
        super().__init__()
        self.t_emb = TimeEmb(t_dim)
        self.in_proj = nn.Linear(2 + t_dim, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                          nn.Linear(hidden, hidden), nn.SiLU())
            for _ in range(n_blocks)
        ])
        self.out = nn.Linear(hidden, 2)
    def forward(self, x, t):
        h = self.in_proj(torch.cat([x, self.t_emb(t)], dim=-1))
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)

torch.manual_seed(0)
model = ResMLP().to(DEVICE)
print(f"model params: {sum(p.numel() for p in model.parameters()):,}")
"""))

cells.append(code("""\
# Move data + schedule to DEVICE for training
x0_data_dev = x0_data.to(DEVICE)
alpha_bars_dev = alpha_bars.to(DEVICE)
alphas_dev = alphas.to(DEVICE)
betas_dev = betas.to(DEVICE)

# YOUR CODE HERE
#  1. Build optimizer + LR scheduler.
#  2. Loop TRAIN_STEPS times, calling train_step(model, batch, T, alpha_bars_dev, optim) each iteration.
#  3. Print loss every few thousand steps.

TRAIN_STEPS = 30000
# ...
"""))

cells.append(md(r"""\
<details>
<summary>✅ <b>Training-loop solution</b></summary>

```python
TRAIN_STEPS = 30000
optim = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=TRAIN_STEPS)

losses = []
t0 = time.time()
for step in range(TRAIN_STEPS):
    idx = torch.randint(0, len(x0_data_dev), (512,), device=DEVICE)
    batch = x0_data_dev[idx]
    loss = train_step(model, batch, T, alpha_bars_dev, optim)
    sched.step()
    losses.append(loss)
    if step % 5000 == 0:
        avg = sum(losses[-200:]) / max(1, len(losses[-200:]))
        print(f"step {step}  recent avg loss={avg:.4f}  time={time.time()-t0:.1f}s")
```

</details>
"""))

cells.append(code("""\
# YOUR CODE HERE
#  1. Call sample() to get 2000 samples.
#  2. Move samples to CPU.
#  3. Compute median NN distance from samples to x0_data.
#  4. Compute baseline: median NN distance real-to-real (exclude self).
#  5. Print ratio. Pass: < 2.0.

# ...
"""))

cells.append(md(r"""\
<details>
<summary>✅ <b>Verification solution</b></summary>

```python
samples = sample(model, (2000, 2), T, alphas_dev, alpha_bars_dev, betas_dev, device=DEVICE).cpu()

nn_samp = torch.cdist(samples, x0_data).min(dim=1).values.median().item()
nn_real_mat = torch.cdist(x0_data, x0_data); nn_real_mat.fill_diagonal_(float('inf'))
nn_real = nn_real_mat.min(dim=1).values.median().item()
ratio = nn_samp / nn_real
print(f"NN dist (samples -> spiral): {nn_samp:.4f}")
print(f"NN dist (real -> real):      {nn_real:.4f}")
print(f"Ratio: {ratio:.2f}x   --> {'PASS (excellent)' if ratio < 1.2 else 'PASS' if ratio < 2.0 else 'FAIL'}")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].scatter(x0_data[:, 0], x0_data[:, 1], s=3, alpha=0.5, c="tab:blue"); axes[0].set_title("real"); axes[0].set_aspect("equal")
axes[1].scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.5, c="tab:green"); axes[1].set_title(f"yours (ratio {ratio:.2f}x)"); axes[1].set_aspect("equal")
plt.tight_layout(); plt.show()
```

</details>
"""))

# ============================================================================
# Wrap-up
# ============================================================================
cells.append(md(r"""\
## Wrap-up

If your final ratio is < 2.0× and the scatter plot shows a clear spiral, you've successfully implemented DDPM end-to-end from the math:

- Forward process (per-step and closed-form)
- Schedule computation
- Algebraic recovery of $x_0$ from $\varepsilon$ prediction
- Reverse step mean formula
- Algorithm 1 (training loop)
- Algorithm 2 (sampling loop)
- Conceptual understanding of ε-parameterization, isotropic noise, score-matching connection, deterministic vs stochastic sampling, and simplified loss

The pieces here are the same ones inside `day2_diffusion.ipynb` for the anime images, scaled up to a UNet on 64×64 RGB. The math is identical; only the dimensions change.

### Where to go next

- `day2_diffusion.ipynb` — full DDPM training on anime, using the YHL04/ddpm reference implementation
- `day2_paper_math_to_code.ipynb` — visualizations of the same math (forward dissolution, score field, conditional posterior)
- `ddpm_paper_tutorial.md` — Q&A on subtler conceptual points

After Day 2 in the curriculum:
- Day 3 — classifier-free guidance, faster samplers (DDIM, DPM-Solver)
- Day 4 — Stable Diffusion, latent diffusion, LoRA fine-tuning
"""))


# ============================================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT}  ({len(cells)} cells)")
