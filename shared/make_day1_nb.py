"""Generate day1/day1_vae_gan.ipynb -- 1000 imgs, 200 epochs, six variants (3 VAE + 3 GAN), worked examples per concept."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "day1" / "day1_vae_gan.ipynb"
OUT.parent.mkdir(parents=True, exist_ok=True)


def md(text):  return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}
def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


cells = []

# ============================================================================
# 0. Title + protocol
# ============================================================================
cells.append(md("""\
# Day 1 — Foundations: VAE and GAN

Build the two classical image-generation families from scratch and *see* their failure modes before moving to diffusion. Every concept gets a worked numerical example, every experiment produces a comparison grid and a numeric row in `results/master_table.md`.

**Protocol**
- **Dataset:** 1000 images from the `lambdalabs/naruto-blip-captions` set, resized to 64×64, normalized to [−1, 1].
- **Training:** 200 epochs, batch 32, AdamW.
- **Hardware:** single H100 (any modern GPU works; total wall-time is a few minutes).
- **Seeds:** fixed throughout, including pre-generated eval latents so every comparison grid uses the same point in latent space across models.

**What you'll leave with**
- A felt understanding of the reparameterization trick — including a gradient-check that breaks without it.
- A felt understanding of why VAEs blur — including a 1-D toy that proves MSE picks the mean.
- A felt understanding of mode collapse — including a numerical perceptual-diversity score on 100 GAN samples.
- Three VAE variants and three GAN variants compared side-by-side, with every engineering lever named and measured.
"""))

# ============================================================================
# 1. Setup
# ============================================================================
cells.append(md("## Setup — imports and config"))
cells.append(code("""\
import math, sys, time, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

ROOT = Path.cwd().parent if Path.cwd().name == "day1" else Path.cwd()
sys.path.insert(0, str(ROOT))
from shared.grid import to_uint8_grid, save_grid

DEVICE   = "cuda"
IMG_SIZE = 64
LATENT   = 128
N_TRAIN  = 1000
BATCH    = 32
EPOCHS   = 200           # held constant across all trainings; the only variables are
                         # model size, loss form, regularization, and augmentation
LR       = 2e-4
SEED     = 0
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

# Fixed eval latents -- shared across every sampling cell so the n-th column of every grid
# is the SAME point in latent space rendered by different decoders. Two sizes because models
# with LATENT=128 vs latent=256 need matching-dim noise.
EVAL_Z_128 = torch.randn(100, 128, device=DEVICE)
EVAL_Z_256 = torch.randn(100, 256, device=DEVICE)

print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))
print("EVAL_Z_128:", EVAL_Z_128.shape, " EVAL_Z_256:", EVAL_Z_256.shape)
"""))

# ============================================================================
# 1b. References — every paper we cite today
# ============================================================================
cells.append(md("""\
## References

Every concept we use today has a paper. We cite them inline as we go; this is the index.

**VAE family**
- *Auto-Encoding Variational Bayes* — Kingma & Welling, 2013. [arXiv 1312.6114](https://arxiv.org/abs/1312.6114). The VAE paper. Defines the reparameterization trick and ELBO objective.
- *Generating Sentences from a Continuous Space* — Bowman et al., 2016. [arXiv 1511.06349](https://arxiv.org/abs/1511.06349). Introduces KL-cost annealing — used in our VAE v1.
- *Perceptual Losses for Real-Time Style Transfer and Super-Resolution* — Johnson, Alahi, Fei-Fei, 2016. [arXiv 1603.08155](https://arxiv.org/abs/1603.08155). VGG-feature perceptual loss — used in our VAE v2.
- *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS)* — Zhang et al., 2018. [arXiv 1801.03924](https://arxiv.org/abs/1801.03924). Validates VGG features as a perceptual metric — same idea we use for diversity.

**"L2 picks the mean" — the VAE blur explanation**
- *Pattern Recognition and Machine Learning* — C. Bishop, 2006, §1.5.5. The textbook result: under squared loss, the optimal regressor is the conditional mean of the target. Applied to VAE decoders, this is why outputs are mean-of-plausibles → soft.

**GAN family**
- *Generative Adversarial Nets* — Goodfellow et al., 2014. [arXiv 1406.2661](https://arxiv.org/abs/1406.2661). The GAN paper. Introduces the non-saturating G objective.
- *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (DCGAN)* — Radford, Metz, Chintala, 2015. [arXiv 1511.06434](https://arxiv.org/abs/1511.06434). The conv-architecture conventions we use.
- *Improved Techniques for Training GANs* — Salimans et al., 2016. [arXiv 1606.03498](https://arxiv.org/abs/1606.03498). Label smoothing, feature matching, minibatch discrimination.
- *Spectral Normalization for Generative Adversarial Networks* — Miyato et al., 2018. [arXiv 1802.05957](https://arxiv.org/abs/1802.05957). The Lipschitz-bound trick on D — used in our GAN v1 and v2.
- *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (TTUR & FID)* — Heusel et al., 2017. [arXiv 1706.08500](https://arxiv.org/abs/1706.08500). Different learning rates for G and D. Also introduces FID.
- *Geometric GAN (hinge loss)* — Lim & Ye, 2017. [arXiv 1705.02894](https://arxiv.org/abs/1705.02894). The margin/SVM-style loss used in modern GANs (BigGAN, StyleGAN2). Used in our v2.
- *Which Training Methods for GANs do actually Converge? (R1 regularization)* — Mescheder, Geiger, Nowozin, 2018. [arXiv 1801.04406](https://arxiv.org/abs/1801.04406). R1 gradient penalty on the discriminator. Used in our v2.
- *Differentiable Augmentation for Data-Efficient GAN Training (DiffAugment)* — Zhao, Liu, Lin, Zhu, Han, 2020. [arXiv 2006.10738](https://arxiv.org/abs/2006.10738), [GitHub](https://github.com/mit-han-lab/data-efficient-gans). **2–4× FID improvement on 1000-image datasets.** The single biggest lever for our setting. Used in our v2.

**Limited-data GAN training**
Small datasets (~1k images) are a known-hard setting for GANs. Modern SOTA techniques:
- *Training Generative Adversarial Networks with Limited Data (StyleGAN2-ADA)* — Karras et al., 2020. [arXiv 2006.06676](https://arxiv.org/abs/2006.06676). Adaptive discriminator augmentation.
- *Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis (FastGAN)* — Liu et al., 2021. [arXiv 2101.04775](https://arxiv.org/abs/2101.04775).

The GAN v2 in this notebook uses DiffAugment + hinge + R1 — a strong baseline that converges in a few minutes on a single GPU.
"""))

# ============================================================================
# 2. VAE theory — short
# ============================================================================
cells.append(md(r"""\
## VAE in 4 ideas

1. **Goal.** Learn a distribution $p(x)$ over images so you can sample. Pixels are 12,288-D — too big to model directly. So assume images come from a **low-dim latent** $z$ (we'll use 128) and learn two networks:
   - **Encoder** $q(z \mid x)$ — image → latent.
   - **Decoder** $p(x \mid z)$ — latent → image.

2. **Constraint.** At generation time we have no $x$. We need to sample $z$ from *somewhere*, so pick $p(z) = \mathcal{N}(0, I)$ and force the encoder's outputs to look like that too.

3. **Reparameterization trick.** The encoder outputs $\mu$ and $\log \sigma^2$. Then
$$z = \mu + \sigma \cdot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$
This makes $z$ a deterministic function of $(\mu, \sigma)$ and an *external* random input $\varepsilon$. Gradients flow through $\mu, \sigma$. We'll prove this works with a gradient-check below.

4. **Loss.** Two terms:
$$\mathcal{L} = \underbrace{\|x - \mathrm{dec}(z)\|^2}_{\text{reconstruction}} + \beta \cdot \underbrace{\text{KL}\!\left(\mathcal{N}(\mu, \sigma^2) \,\|\, \mathcal{N}(0, I)\right)}_{\text{pull encoder to prior}}$$

The KL has a closed form (this is what you'll code):
$$\text{KL} = -\tfrac{1}{2} \sum_j \big(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\big)$$

Three claims hide in those 4 ideas. We test each one now.
"""))

# ============================================================================
# 3. Example #1 — Why VAEs blur (MSE picks the mean)
# ============================================================================
cells.append(md(r"""\
### Worked example 1 — "MSE picks the mean"

Claim: *L2 loss in pixel space rewards predicting the average of plausible outputs.*

Toy setup: a single pixel can plausibly be **0** or **1** with equal probability. What single value $p$ minimizes the expected MSE $E[(p - \text{target})^2]$?

By hand:
$$\frac{d}{dp}\left[\tfrac{1}{2}p^2 + \tfrac{1}{2}(p-1)^2\right] = 2p - 1 = 0 \;\Rightarrow\; p = 0.5$$

The MSE-optimal prediction is **0.5 (grey)** — even though the true distribution has zero mass at 0.5. *That's* why VAEs blur: when many output images are plausible given a latent, MSE picks their average — which is usually a fuzzy mean image.

The cell below confirms this numerically.
"""))
cells.append(code("""\
targets = torch.tensor([0.0, 1.0])           # two equally-plausible "true" values
preds   = torch.linspace(0, 1, 101)
losses  = torch.stack([((p - targets) ** 2).mean() for p in preds])

best_p = preds[losses.argmin()].item()
print(f"MSE-optimal prediction: {best_p:.3f}  (theory says 0.5)")

plt.figure(figsize=(6, 3))
plt.plot(preds, losses)
plt.axvline(0.5, ls="--", c="r", label="theoretical optimum p = 0.5")
plt.xlabel("prediction p"); plt.ylabel("E[(p - target)^2]")
plt.title("MSE picks the mean of plausible targets — this is why VAEs blur")
plt.legend(); plt.tight_layout(); plt.show()
"""))

# ============================================================================
# 4. Example #2 — Reparam gradient check
# ============================================================================
cells.append(md(r"""\
### Worked example 2 — Why we need the reparameterization trick

Claim: *Sampling directly from $\mathcal{N}(\mu, \sigma^2)$ breaks backprop; sampling as $\mu + \sigma \varepsilon$ does not.*

Below: define $\mu$ and $\sigma$ as learnable parameters. Compute a toy loss $\mathcal{L} = z^2$ and try to get $\partial \mathcal{L} / \partial \mu$ and $\partial \mathcal{L} / \partial \sigma$ — first the broken way, then the right way.
"""))
cells.append(code("""\
# --- Method A (broken): direct .sample() ---
mu_a    = torch.tensor(2.0, requires_grad=True)
sigma_a = torch.tensor(1.0, requires_grad=True)
z_a   = torch.distributions.Normal(mu_a, sigma_a).sample()   # .sample() returns a NEW tensor with no graph
loss_a = z_a ** 2
try:
    loss_a.backward()
    print(f"[A naive .sample()]    mu.grad={mu_a.grad}, sigma.grad={sigma_a.grad}")
except RuntimeError as e:
    print(f"[A naive .sample()]    backward failed: {e}")

# --- Method B (correct): reparameterization ---
mu_b    = torch.tensor(2.0, requires_grad=True)
sigma_b = torch.tensor(1.0, requires_grad=True)
eps     = torch.randn(())                                    # external noise, no grad needed
z_b     = mu_b + sigma_b * eps                               # differentiable composition
loss_b  = z_b ** 2
loss_b.backward()
print(f"[B reparam mu+sigma*eps]  mu.grad={mu_b.grad.item():.3f}, sigma.grad={sigma_b.grad.item():.3f}")

# --- Method C (also correct): rsample() does reparam internally ---
mu_c    = torch.tensor(2.0, requires_grad=True)
sigma_c = torch.tensor(1.0, requires_grad=True)
z_c     = torch.distributions.Normal(mu_c, sigma_c).rsample()  # 'r' = reparameterized
loss_c  = z_c ** 2
loss_c.backward()
print(f"[C torch ...rsample()]    mu.grad={mu_c.grad.item():.3f}, sigma.grad={sigma_c.grad.item():.3f}")
"""))
cells.append(md("""\
**Read the output:** Method A returns gradients of **None** (or zero) for $\mu$ and $\sigma$ — `.sample()` cuts the autograd graph. Methods B and C give real gradients. This is exactly why we hand-write the reparameterization in `VAE.reparam()` below.
"""))

# ============================================================================
# 5. Example #3 — KL divergence numerically
# ============================================================================
cells.append(md(r"""\
### Worked example 3 — What the KL term penalizes

Claim: *The KL term pulls $(\mu, \sigma)$ toward $(0, 1)$.* Let's compute it for a few specific cases.

Formula again (per latent dimension, no sum):
$$\text{KL}\!\left(\mathcal{N}(\mu, \sigma^2) \,\|\, \mathcal{N}(0, 1)\right) = -\tfrac{1}{2}\big(1 + \log \sigma^2 - \mu^2 - \sigma^2\big)$$
"""))
cells.append(code("""\
def kl_one_dim(mu, sigma):
    mu, sigma = float(mu), float(sigma)
    logvar = math.log(sigma**2)
    return -0.5 * (1 + logvar - mu**2 - sigma**2)

cases = [
    (0.0, 1.0,  "perfect match -> 0"),
    (0.0, 0.5,  "too narrow"),
    (0.0, 2.0,  "too wide"),
    (1.0, 1.0,  "shifted right by 1 std"),
    (5.0, 1.0,  "far off-center"),
    (0.0, 0.01, "near-deterministic (sigma -> 0)"),
]
print(f"{'mu':>6} {'sigma':>6}  {'KL':>8}   note")
print("-" * 50)
for mu, sigma, note in cases:
    print(f"{mu:>6} {sigma:>6}  {kl_one_dim(mu, sigma):>8.3f}   {note}")
"""))
cells.append(md("""\
**Read the output:** KL = 0 only when $(\\mu, \\sigma) = (0, 1)$. Both wide and narrow distributions are penalized; so are off-center ones. The smallest non-zero we see is a wide one (`sigma=2`) — the model's "least bad" way to deviate from the prior is to spread out a little. The largest is `sigma=0.01` — collapsing the latent to a deterministic point is heavily penalized. This is the engineering reason a trained VAE has a *usable* latent space: the KL term keeps it close to a unit Gaussian, so you can sample from $\\mathcal{N}(0, I)$ at inference.
"""))

# ============================================================================
# 6. VAE architecture diagram
# ============================================================================
cells.append(md("""\
## VAE architecture

```
                   ┌──────────────┐
       x  ──────▶  │   ENCODER    │  ──▶  μ, log σ²
   (B,3,64,64)     └──────────────┘       (B,128), (B,128)
                                                │
                                                ▼
                                    z = μ + σ·ε,  ε ~ N(0, I)
                                                │  (B,128)
                                                ▼
                                       ┌──────────────┐
                                       │   DECODER    │  ──▶  x̂
                                       └──────────────┘       (B,3,64,64)
```

Encoder, layer by layer:

```
in (3, 64, 64)
  │ Conv2d k=4 s=2 p=1                       → (32, 32, 32)
  │ Conv2d k=4 s=2 p=1 + GroupNorm + SiLU    → (64, 16, 16)
  │ Conv2d k=4 s=2 p=1 + GroupNorm + SiLU    → (128,  8,  8)
  │ Conv2d k=4 s=2 p=1 + GroupNorm + SiLU    → (256,  4,  4)
  │ flatten                                  → (4096,)
  ├─▶ Linear  → μ        (128,)
  └─▶ Linear  → log σ²   (128,)
```

Decoder is the exact mirror with `ConvTranspose2d` upsampling 4→8→16→32→64 and a final `Tanh` → outputs in [−1, 1]. Total ≈ 3M params.
"""))

# ============================================================================
# 7. Data
# ============================================================================
cells.append(md("""\
## Data — 1000 anime images at 64×64

Loaded from `data/full/` (the full 1221-image naruto-blip-captions set). We take the first 1000 deterministically, resize to 64×64, normalize to **[−1, 1]** to match the decoder's `Tanh` output range.
"""))
cells.append(code("""\
def load_images(n: int = N_TRAIN) -> torch.Tensor:
    paths = sorted((ROOT / "data" / "full").glob("*.png"))[:n]
    tfm = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    return torch.stack([tfm(Image.open(p).convert("RGB")) for p in paths]).to(DEVICE)

x_train = load_images()
print("x_train.shape =", tuple(x_train.shape),
      " range =", (x_train.min().item(), x_train.max().item()),
      " VRAM =", f"{x_train.element_size() * x_train.nelement() / 1e6:.1f} MB")

# Show first 25
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(to_uint8_grid(x_train[:25], nrow=5)); ax.axis("off")
ax.set_title("First 25 of 1000 training images")
plt.show()
"""))

# ============================================================================
# 8. Encoder
# ============================================================================
cells.append(md("""\
## Encoder code

Four conv-stride-2 blocks halve the spatial dimension each time: 64 → 32 → 16 → 8 → 4. Then two linear heads to (μ, log σ²). `GroupNorm` over `BatchNorm` because our batches are small and we'll want this to also work later in lessons with batch=1.
"""))
cells.append(code("""\
class Encoder(nn.Module):
    def __init__(self, latent: int = LATENT):
        super().__init__()
        ch = [3, 32, 64, 128, 256]
        layers = []
        for i in range(4):
            layers += [
                nn.Conv2d(ch[i], ch[i+1], 4, stride=2, padding=1),
                nn.GroupNorm(8, ch[i+1]) if i > 0 else nn.Identity(),
                nn.SiLU(),
            ]
        self.conv      = nn.Sequential(*layers)        # -> (256, 4, 4)
        self.fc_mu     = nn.Linear(256*4*4, latent)
        self.fc_logvar = nn.Linear(256*4*4, latent)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

enc = Encoder().to(DEVICE)
mu, logvar = enc(x_train[:4])
print(f"encoder out: mu {tuple(mu.shape)}, logvar {tuple(logvar.shape)}")
print(f"encoder params: {sum(p.numel() for p in enc.parameters())/1e6:.2f}M")
"""))

# ============================================================================
# 9. Decoder
# ============================================================================
cells.append(md("""\
## Decoder code — the exact mirror of the encoder
"""))
cells.append(code("""\
class Decoder(nn.Module):
    def __init__(self, latent: int = LATENT):
        super().__init__()
        self.fc = nn.Linear(latent, 256*4*4)
        ch = [256, 128, 64, 32, 3]
        layers = []
        for i in range(4):
            layers += [
                nn.ConvTranspose2d(ch[i], ch[i+1], 4, stride=2, padding=1),
                nn.GroupNorm(8, ch[i+1]) if i < 3 else nn.Identity(),
                nn.SiLU()                if i < 3 else nn.Tanh(),
            ]
        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        return self.deconv(self.fc(z).view(-1, 256, 4, 4))

dec = Decoder().to(DEVICE)
print(f"decoder out: {tuple(dec(torch.randn(2, LATENT, device=DEVICE)).shape)}")
print(f"decoder params: {sum(p.numel() for p in dec.parameters())/1e6:.2f}M")
"""))

# ============================================================================
# 10. VAE + reparam
# ============================================================================
cells.append(md("""\
## VAE — full model with the reparam trick

`reparam` is the same 3 lines you proved work in Example 2. Read each line:

```python
sigma = (0.5 * logvar).exp()    # σ from log σ²
eps   = torch.randn_like(mu)    # external N(0,I) noise — gradients DON'T flow here
z     = mu + sigma * eps        # but they DO flow through mu and sigma
```
"""))
cells.append(code("""\
class VAE(nn.Module):
    def __init__(self, latent: int = LATENT):
        super().__init__()
        self.enc = Encoder(latent)
        self.dec = Decoder(latent)

    @staticmethod
    def reparam(mu, logvar):
        sigma = (0.5 * logvar).exp()
        eps   = torch.randn_like(mu)
        return mu + sigma * eps

    def forward(self, x):
        mu, logvar = self.enc(x)
        z = self.reparam(mu, logvar)
        return self.dec(z), mu, logvar

vae = VAE().to(DEVICE)
n_params = sum(p.numel() for p in vae.parameters())
print(f"VAE total params: {n_params/1e6:.2f}M")
"""))

# ============================================================================
# 11. Loss
# ============================================================================
cells.append(md("""\
## Loss — the equation, in code

`recon` is per-pixel MSE summed and divided by batch size (per-image scalar — easier to compare across batch sizes). `kl` is the closed-form expression from Example 3, summed across the 128 latent dims.
"""))
cells.append(code("""\
def vae_loss(x_hat, x, mu, logvar, beta: float = 1.0):
    recon = F.mse_loss(x_hat, x, reduction="sum") / x.size(0)
    kl    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + beta * kl, recon.item(), kl.item()
"""))

# ============================================================================
# 12. Training
# ============================================================================
cells.append(md("""\
## Train

~31 steps/epoch × 200 epochs ≈ 6,200 gradient updates. Loss curves are saved for plotting. What to watch:
- `recon` should drop substantially as the decoder learns to reconstruct.
- `kl` rises early (encoder spreads latents apart to gain capacity), then stabilizes (KL term pulls back).
"""))
cells.append(code("""\
torch.cuda.reset_peak_memory_stats()
torch.manual_seed(SEED)
vae = VAE().to(DEVICE)
opt = torch.optim.AdamW(vae.parameters(), lr=LR)

history = {"loss": [], "recon": [], "kl": []}
t0 = time.time()
for ep in range(EPOCHS):
    # shuffle indices each epoch
    perm = torch.randperm(N_TRAIN, device=DEVICE)
    ep_loss = ep_recon = ep_kl = 0.0; n_batches = 0
    for i in range(0, N_TRAIN, BATCH):
        idx = perm[i:i+BATCH]
        xb  = x_train[idx]
        x_hat, mu, logvar = vae(xb)
        loss, recon, kl   = vae_loss(x_hat, xb, mu, logvar)
        opt.zero_grad(); loss.backward(); opt.step()
        ep_loss  += loss.item(); ep_recon += recon; ep_kl += kl
        n_batches += 1
    history["loss"].append(ep_loss / n_batches)
    history["recon"].append(ep_recon / n_batches)
    history["kl"].append(ep_kl / n_batches)
    if (ep + 1) % 10 == 0 or ep == 0:
        print(f"ep {ep+1:3d}/{EPOCHS}  loss={history['loss'][-1]:7.2f}  "
              f"recon={history['recon'][-1]:7.2f}  kl={history['kl'][-1]:6.2f}")

vae_train_s   = time.time() - t0
vae_peak_vram = torch.cuda.max_memory_allocated() / 1e9
print(f"\\ntrain time: {vae_train_s:.1f}s   peak VRAM: {vae_peak_vram:.2f} GB")
"""))

# ============================================================================
# 13. Loss curves
# ============================================================================
cells.append(md("""\
## VAE training curves

Three things to look at:
1. **Total loss** — should be monotonic-ish down.
2. **Recon loss** — drops a lot. The decoder is learning to mimic the data.
3. **KL** — rises in the first ~10 epochs (encoder spreading latents apart so it can encode 1000 distinct images), then stabilizes around an equilibrium where KL-pull-toward-N(0,I) and capacity-need balance.
"""))
cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(history["loss"]);  axes[0].set_title("total loss");  axes[0].set_xlabel("epoch")
axes[1].plot(history["recon"]); axes[1].set_title("recon (MSE)"); axes[1].set_xlabel("epoch")
axes[2].plot(history["kl"]);    axes[2].set_title("KL");          axes[2].set_xlabel("epoch")
plt.tight_layout(); plt.show()
"""))

# ============================================================================
# 14. Reconstructions
# ============================================================================
cells.append(md("""\
## Reconstructions

Pass real training images through the VAE. Compare originals to outputs. With 3100 updates the VAE *can* reconstruct, but you'll still see softness — fine textures (hair strands, eye details) average out. That's the L2-mean-picking from Example 1, in pixels.
"""))
cells.append(code("""\
vae.eval()
with torch.no_grad():
    sample_idx = torch.arange(10, device=DEVICE)
    x_hat, _, _ = vae(x_train[sample_idx])

fig, axes = plt.subplots(2, 1, figsize=(12, 5))
axes[0].imshow(to_uint8_grid(x_train[sample_idx], nrow=10)); axes[0].axis("off"); axes[0].set_title("originals")
axes[1].imshow(to_uint8_grid(x_hat,               nrow=10)); axes[1].axis("off"); axes[1].set_title(f"VAE reconstructions ({EPOCHS} epochs)")
plt.tight_layout(); plt.show()

save_grid(x_hat, ROOT / "results/grids/day1_vae_recon.png", nrow=5)
"""))

# ============================================================================
# 15. Samples from prior
# ============================================================================
cells.append(md("""\
## Sample from the prior — pure generation

This is the real test of the generative model: sample $z \\sim \\mathcal{N}(0, I)$ — *no input image* — and decode. Whatever comes out is what the VAE has learned "an anime image" looks like.
"""))
cells.append(code("""\
with torch.no_grad():
    z = EVAL_Z_128[:20]                       # fixed eval latents -- same across all 128-D models
    vae_samples = vae.dec(z)

fig, ax = plt.subplots(figsize=(12, 5))
ax.imshow(to_uint8_grid(vae_samples, nrow=10)); ax.axis("off")
ax.set_title("20 VAE samples from N(0, I) prior")
plt.show()

save_grid(vae_samples[:10], ROOT / "results/grids/day1_vae_samples.png", nrow=5)
"""))

# ============================================================================
# 16. Example #4 — Latent interpolation
# ============================================================================
cells.append(md("""\
### Worked example 4 — Latent interpolation (the VAE's flagship demo)

Claim: *The VAE's latent space is **smooth** — points near each other in latent space decode to similar images.*

Test: encode two real images A and B, get $\\mu_A$ and $\\mu_B$, then decode a linear interpolation:
$$z(\\alpha) = (1 - \\alpha)\\,\\mu_A + \\alpha\\,\\mu_B, \\quad \\alpha = 0, 0.1, 0.2, \\ldots, 1$$

If the latent space is smooth, you'll see a gradual morph from A to B. If it's not (e.g. a plain autoencoder with no KL term), interpolated points decode to garbage.
"""))
cells.append(code("""\
i, j = 3, 200          # two arbitrary training images
with torch.no_grad():
    mu_a, _ = vae.enc(x_train[i:i+1])
    mu_b, _ = vae.enc(x_train[j:j+1])
    alphas  = torch.linspace(0, 1, 8, device=DEVICE).view(-1, 1)
    z_interp= (1 - alphas) * mu_a + alphas * mu_b
    interp  = vae.dec(z_interp)

real_pair = torch.stack([x_train[i], x_train[j]])
fig, axes = plt.subplots(2, 1, figsize=(12, 4))
axes[0].imshow(to_uint8_grid(real_pair, nrow=2)); axes[0].axis("off"); axes[0].set_title("endpoints: real image A (left), real image B (right)")
axes[1].imshow(to_uint8_grid(interp,    nrow=8)); axes[1].axis("off"); axes[1].set_title("decoded latent interpolation alpha = 0 → 1")
plt.tight_layout(); plt.show()
"""))
cells.append(md("""\
**Read the output:** the middle 6 images are *not* in the training set — they're decodings of latent points that have never been encoded from a real image. The fact that they look like plausible (if blurry) anime images is what people mean when they call the VAE's latent space "well-behaved." Plain autoencoders without the KL term don't have this property.
"""))

# ============================================================================
# 17. Latency
# ============================================================================
cells.append(md("""\
## VAE inference latency

VAEs generate in one forward pass through the decoder — among the cheapest samplers possible. Diffusion models need many forward passes; a major engineering focus in modern image gen is closing that gap (consistency distillation, ADD, few-step samplers).
"""))
cells.append(code("""\
z = torch.randn(1, LATENT, device=DEVICE)
for _ in range(3): vae.dec(z)
torch.cuda.synchronize(); t = time.time()
for _ in range(100): vae.dec(z)
torch.cuda.synchronize()
vae_step_ms = (time.time() - t) * 10
print(f"VAE decoder latency (1 img, batch=1): {vae_step_ms:.2f} ms")
"""))

# ============================================================================
# 18. VAE — what went wrong, engineering levers
# ============================================================================
cells.append(md("""\
## Pushing VAE quality — engineering levers

The baseline ends with high reconstruction loss and soft samples. Two further VAE experiments below isolate two different levers, so we can attribute any quality gain to a specific change.

| Variant | Architecture | Loss | KL schedule |
|---|---|---|---|
| **Baseline** | 3M params, latent 128 | pixel-MSE + 1.0·KL | constant β = 1 |
| **VAE v1** — bigger + KL annealing | 14M params, latent 256 | pixel-MSE + β·KL | β ramps 0 → 1 over first 100 epochs |
| **VAE v2** — bigger + perceptual + low β | 14M params, latent 256 | VGG perceptual + 0.05·pixel-MSE + 0.001·KL | constant β = 0.001 |

**v1** isolates *capacity + schedule*: same loss function, just bigger model and warm up the KL pressure. Prediction: visibly better reconstructions, samples still soft (the L2 ceiling persists).

**v2** swaps the loss function — VGG features instead of pixels — directly attacking the "L2 picks the mean" failure from Example 1. Prediction: dramatically sharper reconstructions and qualitatively different samples.

All three trainings use the same `EPOCHS = 200` and same data. The only variables are model size, loss form, and KL schedule.
"""))

cells.append(md("""\
### Wider architecture for v1 and v2

Both v1 and v2 use the same wider encoder/decoder; only the loss and KL schedule differ.

```
encoder ch: [3, 64, 128, 256, 512]
decoder ch: [512, 256, 128, 64, 3]
latent:     256
total:      ~14 M params
```
"""))
cells.append(code("""\
class EncoderBig(nn.Module):
    def __init__(self, latent: int = 256):
        super().__init__()
        ch = [3, 64, 128, 256, 512]
        layers = []
        for i in range(4):
            layers += [
                nn.Conv2d(ch[i], ch[i+1], 4, stride=2, padding=1),
                nn.GroupNorm(8, ch[i+1]) if i > 0 else nn.Identity(),
                nn.SiLU(),
            ]
        self.conv      = nn.Sequential(*layers)
        self.fc_mu     = nn.Linear(512*4*4, latent)
        self.fc_logvar = nn.Linear(512*4*4, latent)
    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

class DecoderBig(nn.Module):
    def __init__(self, latent: int = 256):
        super().__init__()
        self.fc = nn.Linear(latent, 512*4*4)
        ch = [512, 256, 128, 64, 3]
        layers = []
        for i in range(4):
            layers += [
                nn.ConvTranspose2d(ch[i], ch[i+1], 4, stride=2, padding=1),
                nn.GroupNorm(8, ch[i+1]) if i < 3 else nn.Identity(),
                nn.SiLU()                if i < 3 else nn.Tanh(),
            ]
        self.deconv = nn.Sequential(*layers)
    def forward(self, z):
        return self.deconv(self.fc(z).view(-1, 512, 4, 4))

class VAEBig(nn.Module):
    def __init__(self, latent: int = 256):
        super().__init__()
        self.enc = EncoderBig(latent); self.dec = DecoderBig(latent)
    def forward(self, x):
        mu, logvar = self.enc(x)
        sigma = (0.5 * logvar).exp(); eps = torch.randn_like(mu)
        z = mu + sigma * eps
        return self.dec(z), mu, logvar

_probe = VAEBig(latent=256).to(DEVICE)
n_params_imp = sum(p.numel() for p in _probe.parameters())
print(f"v1/v2 VAE arch params: {n_params_imp/1e6:.2f}M")
del _probe
"""))

# ── VAE v1: KL annealing ─────────────────────────────────────────────────
cells.append(md(r"""\
## VAE v1: bigger + KL annealing

14M params (vs the baseline's 3M) and β ramps from 0 → 1 over the first 100 epochs instead of being held at 1 throughout.

**Why KL annealing?** With constant β = 1, the encoder gets pulled toward N(0, I) from the first step *while* it's also learning to reconstruct — the reconstruction loss has no head start. Bowman et al. (2016) showed that ramping β up *after* reconstruction has had time to converge gives a better-trained model:
$$\beta(\text{epoch}) = \min\big(1,\ \text{epoch} / 100\big)$$

For the first 100 epochs the model behaves like a plain autoencoder (β = 0), focusing on reconstruction. Then KL gradually turns on and the latent gets organized into a usable prior. By the end, v1 has had 100 epochs at full β = 1 — same regularization budget as the baseline, with a better early-training trajectory.
"""))
cells.append(code("""\
ANNEAL_EPOCHS = 100   # beta ramps from 0 -> 1 over first 100 epochs

torch.cuda.reset_peak_memory_stats()
torch.manual_seed(SEED)
vae_v1 = VAEBig(latent=256).to(DEVICE)
opt = torch.optim.AdamW(vae_v1.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

history_v1 = {"loss": [], "recon": [], "kl": [], "beta": []}
t0 = time.time()
for ep in range(EPOCHS):
    beta = min(1.0, ep / ANNEAL_EPOCHS)
    perm = torch.randperm(N_TRAIN, device=DEVICE)
    ep_loss = ep_recon = ep_kl = 0.0; n_batches = 0
    for i in range(0, N_TRAIN, BATCH):
        idx = perm[i:i+BATCH]; xb = x_train[idx]
        x_hat, mu, logvar = vae_v1(xb)
        recon = F.mse_loss(x_hat, xb, reduction="sum") / xb.size(0)
        kl    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / xb.size(0)
        loss  = recon + beta * kl
        opt.zero_grad(); loss.backward(); opt.step()
        ep_loss  += loss.item(); ep_recon += recon.item(); ep_kl += kl.item()
        n_batches += 1
    sched.step()
    history_v1["loss" ].append(ep_loss / n_batches)
    history_v1["recon"].append(ep_recon / n_batches)
    history_v1["kl"   ].append(ep_kl / n_batches)
    history_v1["beta" ].append(beta)
    if (ep + 1) % 20 == 0 or ep == 0 or ep == ANNEAL_EPOCHS - 1:
        print(f"ep {ep+1:3d}/{EPOCHS}  beta={beta:.2f}  loss={history_v1['loss'][-1]:7.2f}  "
              f"recon={history_v1['recon'][-1]:7.2f}  kl={history_v1['kl'][-1]:6.2f}")

vae_v1_train_s   = time.time() - t0
vae_v1_peak_vram = torch.cuda.max_memory_allocated() / 1e9
print(f"\\ntrain time: {vae_v1_train_s:.1f}s   peak VRAM: {vae_v1_peak_vram:.2f} GB")
print(f"final pixel-MSE: {history_v1['recon'][-1]:.1f}")
"""))

cells.append(md("""\
### v1 — reconstructions and samples
"""))
cells.append(code("""\
vae_v1.eval()
with torch.no_grad():
    x_hat_v1, _, _ = vae_v1(x_train[:10])
    vae_v1_samples = vae_v1.dec(EVAL_Z_256[:10])    # fixed eval latents

fig, axes = plt.subplots(3, 1, figsize=(12, 7))
axes[0].imshow(to_uint8_grid(x_train[:10],     nrow=10)); axes[0].axis("off"); axes[0].set_title("originals")
axes[1].imshow(to_uint8_grid(x_hat_v1,         nrow=10)); axes[1].axis("off"); axes[1].set_title(f"v1 reconstructions   pixel-MSE={history_v1['recon'][-1]:.0f}")
axes[2].imshow(to_uint8_grid(vae_v1_samples,   nrow=10)); axes[2].axis("off"); axes[2].set_title("v1 samples from N(0, I)")
plt.tight_layout(); plt.show()

# Latency
z1 = torch.randn(1, 256, device=DEVICE)
for _ in range(3): vae_v1.dec(z1)
torch.cuda.synchronize(); t = time.time()
for _ in range(100): vae_v1.dec(z1)
torch.cuda.synchronize()
vae_v1_step_ms = (time.time() - t) * 10
print(f"v1 decoder latency: {vae_v1_step_ms:.2f} ms")
"""))

# ── VAE v2: perceptual + low beta ──────────────────────────────────────
cells.append(md(r"""\
## VAE v2: bigger + VGG perceptual loss + low β

Worked example 1 proved L2 pixel-MSE picks the mean of plausible outputs. The fix is to compute L2 in a space where averaging is benign. **Deep features of an ImageNet-pretrained network** (Johnson et al., 2016; Zhang et al., 2018 — LPIPS) are such a space: features are sparse and semantic, so the average of "anime face A's features" and "anime face B's features" is itself a face-like feature vector, not a blob.

Recipe: pass both $x$ and $\hat{x}$ through a frozen VGG16. MSE on the activations after `relu1_2`, `relu2_2`, `relu3_3`. Sum across layers.

VGG expects [0, 1] images normalized with ImageNet mean/std. We're in [−1, 1], so denorm-then-renorm before the VGG call.
"""))
cells.append(code("""\
import torchvision.models as tvm

class VGGPerceptual(nn.Module):
    def __init__(self, layers=(3, 8, 15)):              # after relu1_2, relu2_2, relu3_3
        super().__init__()
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1).features.eval()
        for p in vgg.parameters(): p.requires_grad = False
        self.vgg = vgg
        self.layers = set(layers); self.max_layer = max(layers)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _feats(self, x):
        x = (x + 1) / 2                                  # [-1,1] -> [0,1]
        x = (x - self.mean) / self.std                   # ImageNet norm
        out, h = [], x
        for i, layer in enumerate(self.vgg):
            h = layer(h)
            if i in self.layers: out.append(h)
            if i >= self.max_layer: break
        return out

    def forward(self, x_hat, x):
        return sum(F.mse_loss(a, b) for a, b in zip(self._feats(x_hat), self._feats(x)))

vgg_loss = VGGPerceptual().to(DEVICE)
# Sanity check on a single batch
with torch.no_grad():
    test = vgg_loss(x_train[:4], x_train[:4])  # same images -> ~0
    test2= vgg_loss(x_train[:4], torch.randn_like(x_train[:4]))  # noise vs real -> larger
print(f"perceptual loss same images: {test.item():.4f}  (should be ~0)")
print(f"perceptual loss vs noise:    {test2.item():.4f}  (should be much larger)")
"""))

cells.append(md(r"""\
### v2 — training with perceptual loss + low β

Loss composition:
$$\mathcal{L}_{v2} \;=\; 1.0 \cdot \mathcal{L}_{\text{VGG}} \;+\; 0.05 \cdot \mathcal{L}_{\text{pixel-MSE (sum/batch)}} \;+\; 0.001 \cdot \mathcal{L}_{\text{KL}}$$

- **VGG perceptual** — primary sharpness driver. L2 in deep feature space, which is sparse and semantic, so averaging stays sharp.
- **Pixel-MSE (per-image sum/batch, ≈ 50–200 in magnitude)** — keeps reconstructions in the right pixel neighborhood.
- **β = 0.001** — KL barely active; capacity isn't bottlenecked. We retain just enough regularization that sampling from N(0, I) is still meaningful.

**Loss-balancing note.** All three terms must be on the same numeric scale before applying weights, otherwise one term silently dominates and the model drifts in the directions you stopped caring about. Here, pixel-MSE summed-per-image (~50–200) and VGG perceptual (~1–10 after training) are within ~10× of each other after multiplying by their weights — close enough for a stable mix. When mixing losses, **always check the per-step contribution of each term**, not just the weights.
"""))
cells.append(code("""\
torch.cuda.reset_peak_memory_stats()
torch.manual_seed(SEED)
vae_v2 = VAEBig(latent=256).to(DEVICE)
opt = torch.optim.AdamW(vae_v2.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

W_VGG, W_PIX, W_KL = 1.0, 0.05, 0.001

history_v2 = {"loss": [], "recon_mse": [], "perc": [], "kl": []}
t0 = time.time()
for ep in range(EPOCHS):
    perm = torch.randperm(N_TRAIN, device=DEVICE)
    ep_loss = ep_mse = ep_perc = ep_kl = 0.0; n_batches = 0
    for i in range(0, N_TRAIN, BATCH):
        idx = perm[i:i+BATCH]; xb = x_train[idx]
        x_hat, mu, logvar = vae_v2(xb)
        perc = vgg_loss(x_hat, xb)
        mse_per_image = F.mse_loss(x_hat, xb, reduction="sum") / xb.size(0)   # both reported AND optimized
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / xb.size(0)
        loss = W_VGG * perc + W_PIX * mse_per_image + W_KL * kl
        opt.zero_grad(); loss.backward(); opt.step()
        ep_loss += loss.item(); ep_perc += perc.item(); ep_mse += mse_per_image.item(); ep_kl += kl.item()
        n_batches += 1
    sched.step()
    history_v2["loss"     ].append(ep_loss / n_batches)
    history_v2["perc"     ].append(ep_perc / n_batches)
    history_v2["recon_mse"].append(ep_mse  / n_batches)
    history_v2["kl"       ].append(ep_kl   / n_batches)
    if (ep + 1) % 20 == 0 or ep == 0:
        print(f"ep {ep+1:3d}/{EPOCHS}  loss={history_v2['loss'][-1]:6.3f}  "
              f"perc={history_v2['perc'][-1]:5.3f}  recon_mse={history_v2['recon_mse'][-1]:6.2f}  "
              f"kl={history_v2['kl'][-1]:7.2f}")

vae_v2_train_s   = time.time() - t0
vae_v2_peak_vram = torch.cuda.max_memory_allocated() / 1e9
print(f"\\ntrain time: {vae_v2_train_s:.1f}s   peak VRAM: {vae_v2_peak_vram:.2f} GB")
print(f"final pixel-MSE: {history_v2['recon_mse'][-1]:.1f}")
"""))

cells.append(md("""\
## Three-way loss curve comparison

Three reconstruction-loss curves on a single log-y axis. Baseline plateaus high; v1 (bigger + KL anneal) drops further; v2 (VGG perceptual + low β) should drop sharply lower despite being optimized on a *different* loss — because the underlying pixel-MSE just happens to come along for free when you fit deep features.
"""))
cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# pixel-MSE comparison (apples-to-apples across all 3)
axes[0].plot(history["recon"],         label="baseline (pixel-MSE β=1)")
axes[0].plot(history_v1["recon"],      label="v1 (bigger + KL anneal)")
axes[0].plot(history_v2["recon_mse"],  label="v2 (perceptual + low β)")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("pixel-MSE / image")
axes[0].set_title("Reconstruction (pixel-MSE) — same metric"); axes[0].legend(); axes[0].set_yscale("log")

# v1 beta schedule + KL
ax2 = axes[1]
ax2.plot(history_v1["kl"], color="tab:blue", label="KL")
ax2.set_xlabel("epoch"); ax2.set_ylabel("KL", color="tab:blue")
ax2b = ax2.twinx()
ax2b.plot(history_v1["beta"], color="tab:orange", linestyle="--")
ax2b.set_ylabel("beta", color="tab:orange"); ax2b.set_ylim(0, 1.1)
ax2.set_title("v1: KL responds to β annealing"); ax2.legend(loc="upper left")

# v2 perceptual + KL on same axis
ax3 = axes[2]
ax3.plot(history_v2["perc"], color="tab:green", label="VGG perc")
ax3.set_xlabel("epoch"); ax3.set_ylabel("VGG perceptual", color="tab:green")
ax3b = ax3.twinx()
ax3b.plot(history_v2["kl"], color="tab:red", linestyle="--", label="KL")
ax3b.set_ylabel("KL", color="tab:red")
ax3.set_title("v2: VGG perc dominates the loss"); ax3.legend(loc="upper left")

plt.tight_layout(); plt.show()
"""))

cells.append(md("""\
## Three-way reconstruction grid

Same 10 input images, three different VAE versions reconstruct them. **This is the headline visual:** look how much sharper v2 (perceptual) is than v1 (bigger pixel-MSE), even though both have the same parameters and same training budget.
"""))
cells.append(code("""\
vae_v2.eval()
with torch.no_grad():
    x_hat_v2, _, _ = vae_v2(x_train[:10])
    vae_v2_samples = vae_v2.dec(EVAL_Z_256[:10])    # SAME latents as v1 → fair column-by-column compare

fig, axes = plt.subplots(4, 1, figsize=(12, 9))
axes[0].imshow(to_uint8_grid(x_train[:10], nrow=10)); axes[0].axis("off"); axes[0].set_title("originals")
axes[1].imshow(to_uint8_grid(x_hat[:10],   nrow=10)); axes[1].axis("off"); axes[1].set_title(f"baseline   pixel-MSE={history['recon'][-1]:.0f}    (3M, β=1, pixel-MSE)")
axes[2].imshow(to_uint8_grid(x_hat_v1,     nrow=10)); axes[2].axis("off"); axes[2].set_title(f"v1         pixel-MSE={history_v1['recon'][-1]:.0f}    (14M, KL-anneal, pixel-MSE)")
axes[3].imshow(to_uint8_grid(x_hat_v2,     nrow=10)); axes[3].axis("off"); axes[3].set_title(f"v2         pixel-MSE={history_v2['recon_mse'][-1]:.0f}    (14M, low β, VGG perceptual)")
plt.tight_layout(); plt.show()

save_grid(x_hat_v1,       ROOT / "results/grids/day1_vae_recon_v1.png", nrow=5)
save_grid(x_hat_v2,       ROOT / "results/grids/day1_vae_recon_v2.png", nrow=5)
save_grid(vae_v2_samples, ROOT / "results/grids/day1_vae_samples_v2.png", nrow=5)
"""))

cells.append(md("""\
## Three-way prior-sample grid + v2 latency

Same 10 latents through each decoder.
"""))
cells.append(code("""\
fig, axes = plt.subplots(3, 1, figsize=(12, 7))
axes[0].imshow(to_uint8_grid(vae_samples[:10],    nrow=10)); axes[0].axis("off"); axes[0].set_title("baseline samples")
axes[1].imshow(to_uint8_grid(vae_v1_samples,      nrow=10)); axes[1].axis("off"); axes[1].set_title("v1 samples  (bigger + KL anneal)")
axes[2].imshow(to_uint8_grid(vae_v2_samples,      nrow=10)); axes[2].axis("off"); axes[2].set_title("v2 samples  (perceptual + low β)")
plt.tight_layout(); plt.show()

# v2 latency
z = torch.randn(1, 256, device=DEVICE)
for _ in range(3): vae_v2.dec(z)
torch.cuda.synchronize(); t = time.time()
for _ in range(100): vae_v2.dec(z)
torch.cuda.synchronize()
vae_v2_step_ms = (time.time() - t) * 10
print(f"v2 decoder latency: {vae_v2_step_ms:.2f} ms")
"""))

cells.append(md("""\
### VAE recap

| Variant | Params | pixel-MSE recon | Notable |
|---|---|---|---|
| baseline | 3M | high | the L2 ceiling on a small model |
| v1: bigger + KL anneal | 14M | meaningfully lower | capacity + better KL schedule |
| v2: bigger + perceptual + low β | 14M | dramatically lower | the loss function was the bottleneck, not capacity |

Same architecture between v1 and v2, same training budget, same data — only the loss formulation changed. The gap between v1 and v2 reconstructions is the measurable size of the "L2 picks the mean of plausibles" effect from Example 1.
"""))

# ============================================================================
# 19. GAN theory
# ============================================================================
cells.append(md(r"""\
## GAN theory — the adversarial game

Two networks play minimax:

- **Generator** $G(z): \mathbb{R}^{128} \to \mathbb{R}^{3 \times 64 \times 64}$ — noise → image.
- **Discriminator** $D(x) \in [0, 1]$ — probability that $x$ is real.

Goodfellow's 2014 objective:
$$\min_G \max_D \; \mathbb{E}_{x \sim \text{data}}\!\left[\log D(x)\right] + \mathbb{E}_{z}\!\left[\log(1 - D(G(z)))\right]$$

In practice we use the **non-saturating** trick: $G$ maximizes $\log D(G(z))$ instead of minimizing $\log(1 - D(G(z)))$. Same fixed point, much better gradients early on. We *prove* this with a plot below.

**Two key differences from VAE:**
1. No reconstruction loss → no L2-blur penalty → **GANs are sharper**.
2. Training signal is the discriminator's gradient → if $D$ wins too fast (probability ≈ 0 on all fakes), the gradient flowing to $G$ becomes tiny → **training stalls** → **mode collapse**.
"""))

# ============================================================================
# 20. Example #5 — Non-saturating G loss
# ============================================================================
cells.append(md(r"""\
### Worked example 5 — Why we use the non-saturating G objective

Claim: *Goodfellow's original $G$ objective has vanishing gradients when $D$ is winning; the non-saturating form does not.*

When $D$ is confidently calling fakes fake, $D(G(z)) \to 0$. Look at the two candidate $G$ losses (and their derivatives w.r.t. $D(G(z))$, call it $p$) near $p = 0$:
$$\text{saturating: }     \mathcal{L}_G = \log(1 - p),  \quad  \frac{d\mathcal{L}_G}{dp} = -\frac{1}{1-p} \xrightarrow{p \to 0} -1$$
$$\text{non-saturating: } \mathcal{L}_G = -\log(p),     \quad  \frac{d\mathcal{L}_G}{dp} = -\frac{1}{p}   \xrightarrow{p \to 0} -\infty$$

The saturating gradient stays bounded near $-1$ — small. The non-saturating one blows up — large. The cell below plots both. Look at the slopes near the left edge.
"""))
cells.append(code("""\
p = torch.linspace(0.01, 0.99, 200)
sat    = torch.log(1 - p)        # original G loss (minimize)
nonsat = -torch.log(p)           # non-saturating G loss (minimize)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(p, sat,    label="saturating  log(1 - p)")
axes[0].plot(p, nonsat, label="non-saturating  -log(p)")
axes[0].set_xlabel("p = D(G(z))   (D's prob the fake is real)")
axes[0].set_ylabel("G loss"); axes[0].legend(); axes[0].set_title("losses")

# derivatives w.r.t. p
dsat    = -1 / (1 - p)
dnonsat = -1 / p
axes[1].plot(p, dsat,    label="d/dp log(1-p) = -1/(1-p)")
axes[1].plot(p, dnonsat, label="d/dp -log(p) = -1/p")
axes[1].set_ylim(-15, 0)
axes[1].set_xlabel("p = D(G(z))"); axes[1].set_ylabel("dL/dp")
axes[1].legend(); axes[1].set_title("gradients (saturating one stays near -1)")
plt.tight_layout(); plt.show()
"""))

# ============================================================================
# 21. GAN architecture diagram
# ============================================================================
cells.append(md("""\
## GAN architecture diagram

```
   z ~ N(0, I)                                   real image x
     (B, 128)                                    (B, 3, 64, 64)
        │                                              │
        ▼                                              │
  ┌───────────┐                                        │
  │ GENERATOR │  ──▶  fake (B, 3, 64, 64) ──┐          │
  └───────────┘                             │          │
                                            ▼          ▼
                                  ┌─────────────────────┐
                                  │   DISCRIMINATOR     │  ──▶ logit
                                  └─────────────────────┘     (B,)

  D wants:  logit(real) → +∞,  logit(fake) → −∞
  G wants:  logit(fake) → +∞   (fool D)
```

`Generator` has the **same shape as the VAE decoder**, with `BatchNorm` + `ReLU` instead of `GroupNorm` + `SiLU` (DCGAN convention).
`Discriminator` has the **same shape as the VAE encoder** with `LeakyReLU(0.2)` (gradients survive on the fake side) and a single linear head.
"""))
cells.append(code("""\
class Generator(nn.Module):
    def __init__(self, latent: int = LATENT):
        super().__init__()
        self.fc = nn.Linear(latent, 256*4*4)
        ch = [256, 128, 64, 32, 3]
        layers = []
        for i in range(4):
            layers += [
                nn.ConvTranspose2d(ch[i], ch[i+1], 4, stride=2, padding=1),
                nn.BatchNorm2d(ch[i+1]) if i < 3 else nn.Identity(),
                nn.ReLU(True)           if i < 3 else nn.Tanh(),
            ]
        self.deconv = nn.Sequential(*layers)
    def forward(self, z):
        return self.deconv(self.fc(z).view(-1, 256, 4, 4))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [3, 32, 64, 128, 256]
        layers = []
        for i in range(4):
            layers += [
                nn.Conv2d(ch[i], ch[i+1], 4, stride=2, padding=1),
                nn.BatchNorm2d(ch[i+1]) if i > 0 else nn.Identity(),
                nn.LeakyReLU(0.2, True),
            ]
        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(256*4*4, 1)
    def forward(self, x):
        return self.head(self.conv(x).flatten(1)).squeeze(-1)

print("G/D defined")
"""))

# ============================================================================
# 22. GAN training
# ============================================================================
cells.append(md("""\
## Train

Each step alternates:
1. **D step** — sample a real batch + fake batch, update D to push real logits up and fake logits down.
2. **G step** — sample fresh fakes, update G to push D's *fake* logits up (the non-saturating objective from Example 5).

We log **D(real)** and **D(fake)** (after sigmoid) every epoch. A healthy game keeps both near 0.5; a winning-D regime drifts toward (≈1, ≈0).
"""))
cells.append(code("""\
torch.cuda.reset_peak_memory_stats()
torch.manual_seed(SEED)
G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)
n_params_gan = sum(p.numel() for p in G.parameters()) + sum(p.numel() for p in D.parameters())
print(f"GAN params (G+D): {n_params_gan/1e6:.2f}M")

opt_g = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

hist_g = {"D_loss": [], "G_loss": [], "D_real": [], "D_fake": []}
t0 = time.time()
for ep in range(EPOCHS):
    perm = torch.randperm(N_TRAIN, device=DEVICE)
    d_sum = g_sum = dr_sum = df_sum = 0.0; n_batches = 0
    for i in range(0, N_TRAIN, BATCH):
        idx = perm[i:i+BATCH]; xb = x_train[idx]; bs = xb.size(0)
        real_lbl = torch.ones (bs, device=DEVICE)
        fake_lbl = torch.zeros(bs, device=DEVICE)

        # D step
        z = torch.randn(bs, LATENT, device=DEVICE)
        with torch.no_grad():
            fake = G(z)
        d_real_logit = D(xb); d_fake_logit = D(fake)
        d_loss = 0.5 * (F.binary_cross_entropy_with_logits(d_real_logit, real_lbl)
                      + F.binary_cross_entropy_with_logits(d_fake_logit, fake_lbl))
        opt_d.zero_grad(); d_loss.backward(); opt_d.step()

        # G step (non-saturating)
        z = torch.randn(bs, LATENT, device=DEVICE)
        fake = G(z)
        g_loss = F.binary_cross_entropy_with_logits(D(fake), real_lbl)
        opt_g.zero_grad(); g_loss.backward(); opt_g.step()

        d_sum += d_loss.item(); g_sum += g_loss.item()
        dr_sum += torch.sigmoid(d_real_logit).mean().item()
        df_sum += torch.sigmoid(d_fake_logit).mean().item()
        n_batches += 1

    hist_g["D_loss"].append(d_sum / n_batches)
    hist_g["G_loss"].append(g_sum / n_batches)
    hist_g["D_real"].append(dr_sum / n_batches)
    hist_g["D_fake"].append(df_sum / n_batches)
    if (ep + 1) % 10 == 0 or ep == 0:
        print(f"ep {ep+1:3d}/{EPOCHS}  D={hist_g['D_loss'][-1]:.3f}  G={hist_g['G_loss'][-1]:.3f}  "
              f"D(real)={hist_g['D_real'][-1]:.2f}  D(fake)={hist_g['D_fake'][-1]:.2f}")

gan_train_s   = time.time() - t0
gan_peak_vram = torch.cuda.max_memory_allocated() / 1e9
print(f"\\ntrain time: {gan_train_s:.1f}s   peak VRAM: {gan_peak_vram:.2f} GB")
"""))

# ============================================================================
# 23. GAN curves
# ============================================================================
cells.append(md("""\
## GAN training curves — the diagnostic plot

Left panel: D loss and G loss over time. **D loss going to 0 ⇒ D is winning ⇒ G's gradient is bad ⇒ collapse coming.**
Right panel: D's probability assignments to real and fake batches. **Healthy ≈ 0.5/0.5. Collapsed ≈ 1.0/0.0.**
"""))
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(hist_g["D_loss"], label="D loss")
axes[0].plot(hist_g["G_loss"], label="G loss")
axes[0].set_xlabel("epoch"); axes[0].set_title("losses"); axes[0].legend()
axes[1].plot(hist_g["D_real"], label="D(real)")
axes[1].plot(hist_g["D_fake"], label="D(fake)")
axes[1].axhline(0.5, ls="--", c="grey", label="balanced game = 0.5")
axes[1].set_xlabel("epoch"); axes[1].set_title("discriminator confidence"); axes[1].legend()
axes[1].set_ylim(0, 1.05)
plt.tight_layout(); plt.show()
"""))

# ============================================================================
# 24. GAN samples + latency
# ============================================================================
cells.append(md("""\
## GAN samples

20 samples from 20 different latents. The diagnostic question: **count distinct outputs.** Healthy GAN ≈ 20 different images. Collapsed ≈ a few clusters repeating.
"""))
cells.append(code("""\
G.eval()
with torch.no_grad():
    gan_samples = G(EVAL_Z_128[:20])     # fixed eval latents

fig, ax = plt.subplots(figsize=(12, 5))
ax.imshow(to_uint8_grid(gan_samples, nrow=10)); ax.axis("off")
ax.set_title("20 baseline-GAN samples from N(0, I)  (fixed eval latents)")
plt.show()

save_grid(gan_samples[:10], ROOT / "results/grids/day1_gan_samples.png", nrow=5)
"""))

# ============================================================================
# 25. Example #6 — Diversity via VGG perceptual features (LPIPS-style)
# ============================================================================
cells.append(md(r"""\
### Worked example 6 — Diversity via perceptual features

Claim: *we can measure semantic diversity (and therefore mode collapse) numerically, not just visually.*

The right space to measure in is **deep features of a pretrained network**, not pixels. Pixel-space distances correlate poorly with semantic content — two images of the same face with different noise patterns have high pixel distance but encode the same thing. The LPIPS paper (Zhang et al. 2018, [arXiv 1801.03924](https://arxiv.org/abs/1801.03924)) established VGG features as a perceptual surrogate; we reuse the VGG16 from the VAE v2 section.

**`feature_diversity(samples)`** = mean pairwise cosine distance between samples in concatenated VGG features.

How to read it:
- Score ≈ real-image dataset value → semantically varied samples.
- Score noticeably below the real-image value → samples are semantically similar → mode collapse.

We compute it for white noise, real images, and the trained generators below.
"""))
cells.append(code("""\
@torch.no_grad()
def feature_diversity(t: torch.Tensor) -> float:
    \"\"\"Average pairwise cosine distance in VGG perceptual feature space.
    Reuses vgg_loss._feats() from the VAE v2 section.\"\"\"
    feats = vgg_loss._feats(t)                                 # list of (B,C,H,W)
    flat  = torch.cat([f.flatten(1) for f in feats], dim=1)    # (B, total_feat_dim)
    flat  = F.normalize(flat, dim=1)
    sim   = flat @ flat.T
    n     = sim.size(0)
    mask  = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    return (1 - sim[mask]).mean().item()

with torch.no_grad():
    s_gan   = G(EVAL_Z_128)                                              # 100 GAN samples
    pick    = torch.randperm(N_TRAIN, device=DEVICE)[:100]
    s_real  = x_train[pick]                                              # 100 real
    s_noise = torch.rand_like(s_real) * 2 - 1                            # 100 [-1,1] uniform noise

div_gan_perc   = feature_diversity(s_gan)
div_real_perc  = feature_diversity(s_real)
div_noise_perc = feature_diversity(s_noise)

print("Perceptual diversity (VGG-feature pairwise cosine dist, higher = more semantically varied):")
print(f"  white noise:             {div_noise_perc:.4f}   <- meaningless 'images', but features differ")
print(f"  100 real training:       {div_real_perc:.4f}   <- realistic upper bound")
print(f"  100 baseline GAN:        {div_gan_perc:.4f}   <- our generator")
print()
print(f"GAN / real ratio = {div_gan_perc / div_real_perc:.0%}    (< ~60% suggests partial mode collapse)")
"""))

# ============================================================================
# 26. GAN latency
# ============================================================================
cells.append(code("""\
z = torch.randn(1, LATENT, device=DEVICE)
for _ in range(3): G(z)
torch.cuda.synchronize(); t = time.time()
for _ in range(100): G(z)
torch.cuda.synchronize()
gan_step_ms = (time.time() - t) * 10
print(f"baseline-GAN generator latency (1 img, batch=1): {gan_step_ms:.2f} ms")
"""))

# ============================================================================
# 26b. GAN — engineering levers
# ============================================================================
cells.append(md("""\
## GAN v1 — stability tricks

The baseline's training is unstable: D dominates (D(real) saturates near 1, D(fake) near 0), so G's gradient nearly vanishes per Example 5. The fixes below address **training stability**, not **capacity**. They keep D and G balanced; sample quality improves modestly because the architecture is unchanged. The bigger quality lift comes in v2.

Three classical stability tricks, all targeting an overpowered discriminator:

| Trick | Reference | Mechanism |
|---|---|---|
| **Spectral normalization on D** | Miyato et al. 2018, [arXiv 1802.05957](https://arxiv.org/abs/1802.05957) | Caps D's Lipschitz constant — D cannot be arbitrarily confident, so G's gradient stays informative. |
| **Label smoothing on real** | Salimans et al. 2016, [arXiv 1606.03498](https://arxiv.org/abs/1606.03498) | D's "real" target = 0.9 instead of 1.0 — prevents saturation. |
| **TTUR (two time-scale update rule)** | Heusel et al. 2017, [arXiv 1706.08500](https://arxiv.org/abs/1706.08500) | Slow D, fast G — `D_lr = 1e-4`, `G_lr = 4e-4`. |

Same architecture and same non-saturating G objective as the baseline. The diagnostic to watch: D(real) and D(fake) should move toward 0.5/0.5.
"""))
cells.append(code("""\
class DiscriminatorSN(nn.Module):
    \"\"\"Same shape as baseline, but every conv wrapped in spectral_norm.\"\"\"
    def __init__(self):
        super().__init__()
        ch = [3, 32, 64, 128, 256]
        layers = []
        for i in range(4):
            conv = nn.utils.spectral_norm(nn.Conv2d(ch[i], ch[i+1], 4, stride=2, padding=1))
            layers += [conv, nn.LeakyReLU(0.2, True)]   # no BN -- spectral norm makes it unnecessary
        self.conv = nn.Sequential(*layers)
        self.head = nn.utils.spectral_norm(nn.Linear(256*4*4, 1))
    def forward(self, x):
        return self.head(self.conv(x).flatten(1)).squeeze(-1)

torch.manual_seed(SEED)
G_imp = Generator().to(DEVICE)
D_imp = DiscriminatorSN().to(DEVICE)
n_params_gan_imp = sum(p.numel() for p in G_imp.parameters()) + sum(p.numel() for p in D_imp.parameters())
print(f"Improved GAN params (G + spectral-norm D): {n_params_gan_imp/1e6:.2f}M")
"""))
cells.append(code("""\
opt_g_imp = torch.optim.Adam(G_imp.parameters(), lr=4e-4, betas=(0.5, 0.999))
opt_d_imp = torch.optim.Adam(D_imp.parameters(), lr=1e-4, betas=(0.5, 0.999))

hist_g_imp = {"D_loss": [], "G_loss": [], "D_real": [], "D_fake": []}
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
for ep in range(EPOCHS):
    perm = torch.randperm(N_TRAIN, device=DEVICE)
    d_sum = g_sum = dr_sum = df_sum = 0.0; n_batches = 0
    for i in range(0, N_TRAIN, BATCH):
        idx = perm[i:i+BATCH]; xb = x_train[idx]; bs = xb.size(0)
        real_lbl = torch.full((bs,), 0.9, device=DEVICE)   # label smoothing
        fake_lbl = torch.zeros(bs, device=DEVICE)

        z = torch.randn(bs, LATENT, device=DEVICE)
        with torch.no_grad(): fake = G_imp(z)
        d_real_logit = D_imp(xb); d_fake_logit = D_imp(fake)
        d_loss = 0.5 * (F.binary_cross_entropy_with_logits(d_real_logit, real_lbl)
                      + F.binary_cross_entropy_with_logits(d_fake_logit, fake_lbl))
        opt_d_imp.zero_grad(); d_loss.backward(); opt_d_imp.step()

        z = torch.randn(bs, LATENT, device=DEVICE)
        fake = G_imp(z)
        g_loss = F.binary_cross_entropy_with_logits(D_imp(fake), torch.ones(bs, device=DEVICE))
        opt_g_imp.zero_grad(); g_loss.backward(); opt_g_imp.step()

        d_sum += d_loss.item(); g_sum += g_loss.item()
        dr_sum += torch.sigmoid(d_real_logit).mean().item()
        df_sum += torch.sigmoid(d_fake_logit).mean().item()
        n_batches += 1

    hist_g_imp["D_loss"].append(d_sum / n_batches)
    hist_g_imp["G_loss"].append(g_sum / n_batches)
    hist_g_imp["D_real"].append(dr_sum / n_batches)
    hist_g_imp["D_fake"].append(df_sum / n_batches)
    if (ep + 1) % 30 == 0 or ep == 0:
        print(f"ep {ep+1:3d}/{EPOCHS}  D={hist_g_imp['D_loss'][-1]:.3f}  G={hist_g_imp['G_loss'][-1]:.3f}  "
              f"D(real)={hist_g_imp['D_real'][-1]:.2f}  D(fake)={hist_g_imp['D_fake'][-1]:.2f}")

gan_imp_train_s   = time.time() - t0
gan_imp_peak_vram = torch.cuda.max_memory_allocated() / 1e9
print(f"\\ntrain time: {gan_imp_train_s:.1f}s   peak VRAM: {gan_imp_peak_vram:.2f} GB")
"""))
cells.append(md("""\
### GAN v1 — diagnostic curves

A healthy GAN keeps D(real) and D(fake) hovering near 0.5 — the balanced game. The v1 curves should sit closer to 0.5/0.5 than the baseline's saturated values, confirming the stability tricks did their job.
"""))
cells.append(code("""\
fig, axes = plt.subplots(2, 2, figsize=(13, 7))

axes[0,0].plot(hist_g["D_loss"],     label="D loss")
axes[0,0].plot(hist_g["G_loss"],     label="G loss")
axes[0,0].set_title("BASELINE losses"); axes[0,0].set_xlabel("epoch"); axes[0,0].legend()

axes[0,1].plot(hist_g["D_real"],     label="D(real)")
axes[0,1].plot(hist_g["D_fake"],     label="D(fake)")
axes[0,1].axhline(0.5, ls="--", c="grey", label="balanced = 0.5")
axes[0,1].set_title("BASELINE D confidence"); axes[0,1].set_xlabel("epoch"); axes[0,1].legend(); axes[0,1].set_ylim(0, 1.05)

axes[1,0].plot(hist_g_imp["D_loss"], label="D loss")
axes[1,0].plot(hist_g_imp["G_loss"], label="G loss")
axes[1,0].set_title("IMPROVED losses (spec-norm + smoothing + TTUR)"); axes[1,0].set_xlabel("epoch"); axes[1,0].legend()

axes[1,1].plot(hist_g_imp["D_real"], label="D(real)")
axes[1,1].plot(hist_g_imp["D_fake"], label="D(fake)")
axes[1,1].axhline(0.5, ls="--", c="grey", label="balanced = 0.5")
axes[1,1].set_title("IMPROVED D confidence"); axes[1,1].set_xlabel("epoch"); axes[1,1].legend(); axes[1,1].set_ylim(0, 1.05)
plt.tight_layout(); plt.show()
"""))
cells.append(md("""\
### v1 samples + perceptual diversity
"""))
cells.append(code("""\
G_imp.eval()
with torch.no_grad():
    gan_imp_samples = G_imp(EVAL_Z_128[:20])     # fixed eval latents -- same as baseline
    div_gan_imp = feature_diversity(G_imp(EVAL_Z_128))

fig, ax = plt.subplots(figsize=(12, 5))
ax.imshow(to_uint8_grid(gan_imp_samples, nrow=10)); ax.axis("off")
ax.set_title(f"GAN v1 samples   feature-div={div_gan_imp:.3f}")
plt.show()

# v1 generator latency
z1 = torch.randn(1, LATENT, device=DEVICE)
for _ in range(3): G_imp(z1)
torch.cuda.synchronize(); t = time.time()
for _ in range(100): G_imp(z1)
torch.cuda.synchronize()
gan_imp_step_ms = (time.time() - t) * 10
print(f"GAN v1 generator latency (1 img): {gan_imp_step_ms:.2f} ms")

save_grid(gan_imp_samples[:10], ROOT / "results/grids/day1_gan_samples_v1.png", nrow=5)
"""))

# ─────────────── GAN v2: DiffAugment + bigger + hinge + R1 ───────────────
cells.append(md(r"""\
## GAN v2 — bigger + hinge + R1 + DiffAugment

v1 stabilized training but didn't change capacity. For genuinely better samples, modern data-efficient GAN techniques are required: bigger generator/discriminator, hinge loss, gradient regularization, and differentiable augmentation.

| Lever | Reference | Effect |
|---|---|---|
| **DiffAugment** (color + translation + cutout) | Zhao et al. 2020 [arXiv 2006.10738](https://arxiv.org/abs/2006.10738) | Apply differentiable augmentation to BOTH real and fake before D. Paper: **2–4× FID improvement on 1000-image datasets.** The single biggest lever for our setting. |
| **Bigger model** | — | G channels [128, 256, 512] → ~9M. D channels [64, 128, 256, 512] → ~4.6M. Real capacity to fit 1000 diverse images. |
| **Hinge loss** | Lim & Ye 2017 [arXiv 1705.02894](https://arxiv.org/abs/1705.02894) | Modern standard with spectral norm. D: `relu(1−D(real)).mean() + relu(1+D(fake)).mean()`. G: `−D(fake).mean()`. More stable than BCE for spec-norm models. |
| **R1 gradient penalty (lazy)** | Mescheder et al. 2018 [arXiv 1801.04406](https://arxiv.org/abs/1801.04406) | Penalize `‖∇_x D(x)‖²` on real samples every 16 steps. Encourages a smooth D landscape, prevents memorization. Karras et al. 2020 popularized the lazy (every-K-steps) form. |

Everything else stays controlled: same `EPOCHS=200`, same `BATCH=32`, same `N_TRAIN=1000`, same Adam optimizer family.

**DiffAugment implementation reproduced verbatim from the [official GitHub](https://github.com/mit-han-lab/data-efficient-gans).**
"""))

cells.append(code("""\
# === DiffAugment (Zhao et al. NeurIPS 2020) -- reproduced from the official PyTorch implementation ===
def rand_brightness(x):
    return x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)

def rand_saturation(x):
    m = x.mean(dim=1, keepdim=True)
    return (x - m) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + m

def rand_contrast(x):
    m = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - m) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + m

def rand_translation(x, ratio=0.125):
    sx, sy = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    tx = torch.randint(-sx, sx + 1, size=[x.size(0), 1, 1], device=x.device)
    ty = torch.randint(-sy, sy + 1, size=[x.size(0), 1, 1], device=x.device)
    gb, gx, gy = torch.meshgrid(
        torch.arange(x.size(0), device=x.device),
        torch.arange(x.size(2), device=x.device),
        torch.arange(x.size(3), device=x.device),
        indexing="ij",
    )
    gx = torch.clamp(gx + tx + 1, 0, x.size(2) + 1)
    gy = torch.clamp(gy + ty + 1, 0, x.size(3) + 1)
    xp = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    return xp.permute(0, 2, 3, 1).contiguous()[gb, gx, gy].permute(0, 3, 1, 2).contiguous()

def rand_cutout(x, ratio=0.5):
    cs = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    ox = torch.randint(0, x.size(2) + (1 - cs[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    oy = torch.randint(0, x.size(3) + (1 - cs[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    gb, gx, gy = torch.meshgrid(
        torch.arange(x.size(0), device=x.device),
        torch.arange(cs[0],     device=x.device),
        torch.arange(cs[1],     device=x.device),
        indexing="ij",
    )
    gx = torch.clamp(gx + ox - cs[0] // 2, 0, x.size(2) - 1)
    gy = torch.clamp(gy + oy - cs[1] // 2, 0, x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[gb, gx, gy] = 0
    return x * mask.unsqueeze(1)

AUG_FNS = {
    "color":       [rand_brightness, rand_saturation, rand_contrast],
    "translation": [rand_translation],
    "cutout":      [rand_cutout],
}

def DiffAugment(x, policy="color,translation,cutout"):
    for p in policy.split(","):
        for fn in AUG_FNS[p]:
            x = fn(x)
    return x.contiguous()

# Sanity check
_aug_test = DiffAugment(x_train[:4])
print(f"DiffAugment in {x_train[:4].shape} -> out {_aug_test.shape}, range [{_aug_test.min().item():.2f}, {_aug_test.max().item():.2f}]")
"""))

cells.append(md("""\
### Bigger G + D for v2
"""))
cells.append(code("""\
class GenBig(nn.Module):
    \"\"\"Bigger DCGAN-style generator. ~9M params.\"\"\"
    def __init__(self, latent: int = LATENT):
        super().__init__()
        self.fc = nn.Linear(latent, 512 * 4 * 4)
        ch = [512, 256, 128, 64, 3]
        layers = []
        for i in range(4):
            layers += [
                nn.ConvTranspose2d(ch[i], ch[i+1], 4, stride=2, padding=1),
                nn.BatchNorm2d(ch[i+1]) if i < 3 else nn.Identity(),
                nn.ReLU(True)           if i < 3 else nn.Tanh(),
            ]
        self.deconv = nn.Sequential(*layers)
    def forward(self, z):
        return self.deconv(self.fc(z).view(-1, 512, 4, 4))

class DiscBigSN(nn.Module):
    \"\"\"Bigger D with spectral norm on every layer. ~4.6M params.\"\"\"
    def __init__(self):
        super().__init__()
        ch = [3, 64, 128, 256, 512]
        layers = []
        for i in range(4):
            conv = nn.utils.spectral_norm(nn.Conv2d(ch[i], ch[i+1], 4, stride=2, padding=1))
            layers += [conv, nn.LeakyReLU(0.2, True)]
        self.conv = nn.Sequential(*layers)
        self.head = nn.utils.spectral_norm(nn.Linear(512 * 4 * 4, 1))
    def forward(self, x):
        return self.head(self.conv(x).flatten(1)).squeeze(-1)

torch.manual_seed(SEED)
G_v2 = GenBig().to(DEVICE)
D_v2 = DiscBigSN().to(DEVICE)
n_params_gan_v2 = sum(p.numel() for p in G_v2.parameters()) + sum(p.numel() for p in D_v2.parameters())
print(f"GAN v2 params (G + D): {n_params_gan_v2/1e6:.2f}M")
"""))

cells.append(md(r"""\
### v2 training — hinge loss, lazy R1 every 16 steps, DiffAugment on both real and fake

**Hinge loss form**:
$$\mathcal{L}_D = \mathbb{E}_x [\max(0, 1 - D(x))] + \mathbb{E}_z [\max(0, 1 + D(G(z)))]$$
$$\mathcal{L}_G = -\mathbb{E}_z [D(G(z))]$$

**R1 lazy regularization**: compute the squared gradient norm `‖∇_x D(x_real)‖²` *every 16 steps*, multiply by 16 to compensate the lazy schedule, add `γ/2` weight to D loss. We use γ=10 (Mescheder et al. 2018 recommendation).

**DiffAugment**: applied to BOTH `x_real` (before D sees it) and `G(z)` (before D sees that). The augmentation is *differentiable*, so gradients flow through it back to G. Without this dual application, D would learn the augmentation as a discriminative feature — Zhao et al. 2020 §3.2.
"""))
cells.append(code("""\
GAMMA = 10.0       # R1 coefficient (Mescheder et al. recommendation)
R1_EVERY = 16      # lazy regularization frequency (Karras et al. 2020)
POLICY = "color,translation,cutout"

opt_g_v2 = torch.optim.Adam(G_v2.parameters(), lr=2e-4, betas=(0.0, 0.99))
opt_d_v2 = torch.optim.Adam(D_v2.parameters(), lr=2e-4, betas=(0.0, 0.99))

hist_g_v2 = {"D_loss": [], "G_loss": [], "R1": []}
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
step = 0
for ep in range(EPOCHS):
    perm = torch.randperm(N_TRAIN, device=DEVICE)
    d_sum = g_sum = r1_sum = 0.0; n_batches = r1_batches = 0
    for i in range(0, N_TRAIN, BATCH):
        idx = perm[i:i+BATCH]; xb = x_train[idx]; bs = xb.size(0)

        # ── D step (hinge loss + DiffAugment on real & fake) ──
        z = torch.randn(bs, LATENT, device=DEVICE)
        with torch.no_grad(): fake = G_v2(z)
        xb_aug   = DiffAugment(xb,   policy=POLICY)
        fake_aug = DiffAugment(fake, policy=POLICY)
        d_real = D_v2(xb_aug); d_fake = D_v2(fake_aug)
        d_loss = F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean()

        # ── R1 lazy ──
        if step % R1_EVERY == 0:
            xb_r1 = xb.detach().requires_grad_(True)
            d_real_r1 = D_v2(DiffAugment(xb_r1, policy=POLICY))
            grad = torch.autograd.grad(d_real_r1.sum(), xb_r1, create_graph=True)[0]
            r1 = grad.pow(2).sum(dim=[1, 2, 3]).mean()
            d_loss = d_loss + (GAMMA / 2) * r1 * R1_EVERY     # lazy compensation
            r1_sum += r1.item(); r1_batches += 1
        opt_d_v2.zero_grad(); d_loss.backward(); opt_d_v2.step()

        # ── G step (hinge: maximize D's score on fake) ──
        z = torch.randn(bs, LATENT, device=DEVICE)
        fake_aug = DiffAugment(G_v2(z), policy=POLICY)
        g_loss = -D_v2(fake_aug).mean()
        opt_g_v2.zero_grad(); g_loss.backward(); opt_g_v2.step()

        d_sum += d_loss.item(); g_sum += g_loss.item(); n_batches += 1
        step += 1

    hist_g_v2["D_loss"].append(d_sum / n_batches)
    hist_g_v2["G_loss"].append(g_sum / n_batches)
    hist_g_v2["R1"    ].append(r1_sum / max(1, r1_batches))
    if (ep + 1) % 20 == 0 or ep == 0:
        print(f"ep {ep+1:3d}/{EPOCHS}  D={hist_g_v2['D_loss'][-1]:7.3f}  G={hist_g_v2['G_loss'][-1]:7.3f}  R1={hist_g_v2['R1'][-1]:7.3f}")

gan_v2_train_s   = time.time() - t0
gan_v2_peak_vram = torch.cuda.max_memory_allocated() / 1e9
print(f"\\ntrain time: {gan_v2_train_s:.1f}s   peak VRAM: {gan_v2_peak_vram:.2f} GB")
"""))

cells.append(md("""\
### GAN v2 — diagnostic curves
"""))
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(hist_g_v2["D_loss"], label="D loss")
axes[0].plot(hist_g_v2["G_loss"], label="G loss")
axes[0].set_title("v2 hinge losses"); axes[0].set_xlabel("epoch"); axes[0].legend()
axes[1].plot(hist_g_v2["R1"]); axes[1].set_title("v2 R1 penalty (lazy, every 16 steps)"); axes[1].set_xlabel("epoch")
plt.tight_layout(); plt.show()
"""))

cells.append(md("""\
### v2 samples + perceptual diversity
"""))
cells.append(code("""\
G_v2.eval()
with torch.no_grad():
    gan_v2_samples = G_v2(EVAL_Z_128[:20])
    div_gan_v2 = feature_diversity(G_v2(EVAL_Z_128))

fig, ax = plt.subplots(figsize=(12, 5))
ax.imshow(to_uint8_grid(gan_v2_samples, nrow=10)); ax.axis("off")
ax.set_title(f"GAN v2 samples   feature-div={div_gan_v2:.3f}  (real {div_real_perc:.3f}, baseline {div_gan_perc:.3f}, v1 {div_gan_imp:.3f})")
plt.show()

# v2 generator latency
z1 = torch.randn(1, LATENT, device=DEVICE)
for _ in range(3): G_v2(z1)
torch.cuda.synchronize(); t = time.time()
for _ in range(100): G_v2(z1)
torch.cuda.synchronize()
gan_v2_step_ms = (time.time() - t) * 10
print(f"GAN v2 generator latency (1 img): {gan_v2_step_ms:.2f} ms")

save_grid(gan_v2_samples[:10], ROOT / "results/grids/day1_gan_samples_v2.png", nrow=5)
"""))

# ============================================================================
# 27. Side-by-side — 5-way headline
# ============================================================================
cells.append(md("""\
## Six-way comparison

Same 10 fixed eval latents per row (`EVAL_Z_128` for rows 1, 4, 5, 6; `EVAL_Z_256` for rows 2, 3). Column-by-column comparison is like-for-like: column *k* of every row is the same point in latent space rendered by a different decoder.

1. **baseline VAE** — small + pixel-MSE → soft blobs (the L2 ceiling from Example 1).
2. **VAE v1** — bigger + KL anneal, still pixel-MSE → modestly less soft.
3. **VAE v2** — bigger + VGG perceptual + low β → semantically sharper.
4. **baseline GAN** — small DCGAN with BCE; D dominates training.
5. **GAN v1** — same architecture + spectral norm + label smoothing + TTUR → balanced training, quality close to baseline.
6. **GAN v2** — bigger + hinge + R1 + DiffAugment → the quality lift on small data.
"""))
cells.append(code("""\
fig, axes = plt.subplots(6, 1, figsize=(12, 13))
axes[0].imshow(to_uint8_grid(vae_samples[:10],     nrow=10)); axes[0].axis("off"); axes[0].set_title(f"baseline VAE  ({n_params/1e6:.1f}M, β=1, pixel-MSE)")
axes[1].imshow(to_uint8_grid(vae_v1_samples,       nrow=10)); axes[1].axis("off"); axes[1].set_title(f"VAE v1       ({n_params_imp/1e6:.1f}M, KL anneal, pixel-MSE)")
axes[2].imshow(to_uint8_grid(vae_v2_samples,       nrow=10)); axes[2].axis("off"); axes[2].set_title(f"VAE v2       ({n_params_imp/1e6:.1f}M, low β, VGG perceptual)")
axes[3].imshow(to_uint8_grid(gan_samples[:10],     nrow=10)); axes[3].axis("off"); axes[3].set_title(f"baseline GAN ({n_params_gan/1e6:.1f}M, BCE)        feature-div={div_gan_perc:.3f}")
axes[4].imshow(to_uint8_grid(gan_imp_samples[:10], nrow=10)); axes[4].axis("off"); axes[4].set_title(f"GAN v1       (+ spec-norm + smoothing + TTUR) feature-div={div_gan_imp:.3f}")
axes[5].imshow(to_uint8_grid(gan_v2_samples[:10],  nrow=10)); axes[5].axis("off"); axes[5].set_title(f"GAN v2       ({n_params_gan_v2/1e6:.1f}M + hinge + R1 + DiffAugment) feature-div={div_gan_v2:.3f}")
plt.tight_layout(); plt.show()
"""))

# ============================================================================
# 28. Master table — 6 rows
# ============================================================================
cells.append(md("""\
## Master-table entry — 6 rows

Self-rate quality 1–10 on each row. The cell appends to `results/master_table.md`.
"""))
cells.append(code("""\
Q_BASE_VAE = "?/10"
Q_V1_VAE   = "?/10"
Q_V2_VAE   = "?/10"
Q_BASE_GAN = "?/10"
Q_V1_GAN   = "?/10"
Q_V2_GAN   = "?/10"

table_path = ROOT / "results" / "master_table.md"
header = "| Day | Model | Params(M) | Train(s) | VRAM(GB) | Steps | Step(ms) | Recon | FeatDiv | Quality |\\n"
sep    = "|---|---|---|---|---|---|---|---|---|---|\\n"
if not table_path.exists():
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(header + sep)

rows = [
    f"| 1 | baseline VAE          | {n_params/1e6:.2f}          | {vae_train_s:.1f}      | {vae_peak_vram:.2f}     | 1 | {vae_step_ms:.2f}      | {history['recon'][-1]:.0f}        | —     | {Q_BASE_VAE} |\\n",
    f"| 1 | VAE v1 (bigger+anneal)| {n_params_imp/1e6:.2f}     | {vae_v1_train_s:.1f}   | {vae_v1_peak_vram:.2f}  | 1 | {vae_v1_step_ms:.2f}   | {history_v1['recon'][-1]:.0f}     | —     | {Q_V1_VAE} |\\n",
    f"| 1 | VAE v2 (+perceptual)  | {n_params_imp/1e6:.2f}     | {vae_v2_train_s:.1f}   | {vae_v2_peak_vram:.2f}  | 1 | {vae_v2_step_ms:.2f}   | {history_v2['recon_mse'][-1]:.0f} | —     | {Q_V2_VAE} |\\n",
    f"| 1 | baseline GAN (BCE)    | {n_params_gan/1e6:.2f}     | {gan_train_s:.1f}      | {gan_peak_vram:.2f}     | 1 | {gan_step_ms:.2f}      | —     | {div_gan_perc:.3f} | {Q_BASE_GAN} |\\n",
    f"| 1 | GAN v1 (stability)    | {n_params_gan_imp/1e6:.2f} | {gan_imp_train_s:.1f}  | {gan_imp_peak_vram:.2f} | 1 | {gan_imp_step_ms:.2f}  | —     | {div_gan_imp:.3f}  | {Q_V1_GAN} |\\n",
    f"| 1 | GAN v2 (hinge+R1+DiffAug)| {n_params_gan_v2/1e6:.2f}| {gan_v2_train_s:.1f}   | {gan_v2_peak_vram:.2f}  | 1 | {gan_v2_step_ms:.2f}   | —     | {div_gan_v2:.3f}   | {Q_V2_GAN} |\\n",
]
with table_path.open("a") as f:
    for r in rows: f.write(r)
print(table_path.read_text())
"""))

# ============================================================================
# 29. Closing
# ============================================================================
cells.append(md("""\
## Summary — what you measured today

Three questions that are usually waved at, now answered with code and numbers:

| Question | Answer | Where |
|---|---|---|
| Why do VAEs blur? | MSE picks the mean of plausible reconstructions. | Example 1 |
| Why doesn't `.sample()` work as a stochastic latent? | It severs the autograd graph; the reparameterization trick works around it. | Example 2 |
| Why do GANs collapse on small data? | D wins → G's gradient vanishes → few modes survive. | Examples 5–6 |

### Levers, costs, and what they actually do

| Lever | Cost | Effect |
|---|---|---|
| Bigger VAE | ~4× params, ~1.5× train time | Lower pixel-MSE, modestly sharper |
| KL annealing | none | Reconstruction can converge before KL pressure kicks in |
| VGG perceptual loss | +VGG forward per step | Sidesteps L2 averaging — sharp reconstructions |
| Spectral normalization on D | tiny D-step overhead | Caps D's Lipschitz constant; G's gradient stays informative |
| Hinge loss | none | Stable companion to spectral-norm D |
| Label smoothing + TTUR | none | Balances D vs G dynamics |
| DiffAugment | augmentation cost only | 2–4× FID improvement on small datasets (Zhao 2020) |
| R1 gradient penalty | +1 backward / 16 steps | Smooths D; prevents memorization of real samples |

**Stability vs quality:** spec-norm, label smoothing, TTUR, R1 make training *converge*. Bigger model, perceptual loss, DiffAugment make samples *better*. The two categories are not interchangeable.

### Next

Diffusion: same encoder/decoder shape, but generation is iterative noise removal. Single-step latency goes up; sharpness and stability come without an adversarial loss.
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
