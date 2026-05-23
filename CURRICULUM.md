# Image Generation — 7-Day Engineering Curriculum

**Hardware:** H100 80GB for training; deployment target is **low-resource compute** generally (small GPUs like L4 / T4 / A10, consumer cards, or on-device — the curriculum doesn't pin to a single chip).
**Budget:** 6 hours/day × 7 days = 42 hours.
**Data:** `lambdalabs/naruto-blip-captions` — 1,221 native-512² anime images with BLIP captions. First 1,000 used for from-scratch training (Days 1–3, 5); full set in `data/full/` for the Day 4 style-LoRA fine-tune. This is the canonical SD fine-tune demo dataset, so your numbers will be directly comparable to public benchmarks.
**Protocol every lesson (revised after Day 1):** train ~1,000 images for enough epochs to *visibly converge* (typically 50–200, batch 32), sample the same fixed prompts/seeds, save a comparison grid, log params / peak-VRAM / step-latency / total-train-time / training-loss-curves to `results/master_table.md`. You rate quality; the table holds the numbers. **Every concept gets a worked numerical example** — toy demos, gradient checks, parameter-sweep tables. No vague prose.
**Math depth:** intuition + the equations that appear in code. No SDE/Fokker-Planck.
**Code style:** from-scratch PyTorch days 1–3 and small models on days 5–7; HuggingFace `diffusers` + `peft` + `optimum` for SDXL/Flux on days 4–7.

---

## The arc (why this order)

The field's history is also the right pedagogical order:

1. **VAE/GAN (2014–2019)** — set up the two classical families and their failure modes. You see why we needed something else.
2. **DDPM (2020)** — diffusion arrives. You implement it tiny and *see* it work.
3. **Score-based + samplers + CFG (2020–2022)** — the unification that made diffusion controllable and 50× faster at inference.
4. **Latent diffusion / SDXL (2022–2023)** — the engineering insight (compress, then diffuse) that made it deployable. LoRA fine-tuning.
5. **DiT + Flow Matching / Flux + SD3 (2023–2024)** — UNet→Transformer transition; rectified flow as the new training objective that enables fewer sampling steps natively.
6. **Distillation / few-step (2023–2025)** — Turbo, Lightning, LCM, DMD. The "make it fast" lineage that brings diffusion latency to small-GPU-friendly numbers.
7. **Autoregressive resurgence + deployment (2024–2026)** — VAR/Infinity-class AR models reclaim parts of SOTA; quantization & TensorRT to ship on low-resource compute.

By Day 7 you have a master table with every approach side-by-side. That's the artifact.

---

## Day 1 (6h) — Foundations: VAE & GAN

**Theory (1.5h)** — Generative modeling landscape: likelihood-based (VAE, AR, diffusion-as-ELBO), implicit (GAN), score-based. The three things every generative model must do: parametrize a distribution, define a tractable loss, define a sampler. **VAE:** encoder→latent→decoder, reparametrization trick `z = μ + σ·ε`, ELBO = reconstruction + β·KL. Why VAEs blur (Gaussian likelihood + mean of plausible reconstructions). **GAN:** generator vs discriminator, BCE/non-saturating loss, mode collapse explained as "discriminator gradient only points to *a* mode."

**Build (4h)** —
- `day1/day1_vae_gan.ipynb` (notebook): VAE encoder/decoder ~3M params, DCGAN ~2M params, both trained 100 epochs on 1000 images at 64×64. Includes worked examples (MSE→mean, reparam gradient-check, KL numerical table, latent interpolation, non-saturating G derivation, mode-collapse diversity metric).
- `day1/gan.py`: DCGAN-style generator/discriminator, same images same epochs.

**Compare (0.5h)** — Sample grid: 16 VAE samples + 16 GAN samples + originals. Fill row 1 of master table. Notice VAE blur vs GAN sharpness-but-collapse.

---

## Day 2 (6h) — Diffusion from scratch: DDPM

**Theory (1.5h)** — Forward process `q(xₜ|x₀) = N(√ᾱₜ x₀, (1-ᾱₜ)I)` — closed form, you can sample any `t` directly. Reverse process learned. The trick: **predict the noise ε instead of x₀**, loss is MSE — derive in 5 lines, see why it's elegant. β-schedule (linear vs cosine) — engineering choice that matters. DDIM: deterministic sampler, far fewer steps.

**Build (3.5h)** —
- `day2/unet.py`: small UNet ~5M params, sinusoidal timestep embedding, GroupNorm, self-attn at low resolution.
- `day2/ddpm.py`: training loop (sample t, noise, predict, MSE), DDPM 1000-step sampler, DDIM 50/20/5-step samplers.

**Compare (1h)** — Same 4 fixed seeds, sample at 1000/50/20/5 steps. Build the **step-vs-latency-vs-quality** curve you'll keep extending all week. Add to master table.

---

## Day 3 (6h) — Score view, samplers, conditioning, CFG

**Theory (1.5h)** — Score = ∇log p(x); ε-prediction ⇔ score (same network, different parametrization). Karras (EDM) unified view: σ schedule, preconditioning — why you'll see σ instead of t in modern code. **Samplers:** Euler (first-order, cheap), Heun (second-order, ~2× cost, much better), DPM-Solver++ (state of the art for ≤20 steps). **CFG:** train with random label dropout (~10%), sample `ε = ε_uncond + s·(ε_cond − ε_uncond)`, scale `s` trades diversity↔fidelity.

**Build (3h)** —
- `day3/ddpm_cond.py`: add CLIP text encoder (frozen), cross-attn or AdaLN conditioning, 10% dropout. Captions are already in `data/raw/captions.json` (BLIP-generated by the dataset authors). Train.
- `day3/samplers.py`: implement Euler, Heun, DPM-Solver++ on top of your UNet.

**Compare (1.5h)** — Same prompts, sweep sampler ∈ {DDIM, Euler, Heun, DPM++} × steps ∈ {5, 10, 20, 50} × CFG ∈ {1, 3, 7, 12}. Big grid. The CFG sweep is the most important plot you'll make this week — saturation point is real.

---

## Day 4 (6h) — Latent diffusion, SDXL, LoRA

**Theory (1h)** — Why latent diffusion: image space is 512×512×3 ≈ 800k dims, latent space ~64×64×4 = 16k dims, 50× cheaper per step *and* the VAE already removed imperceptible detail. SDXL architecture: two text encoders (CLIP-L + OpenCLIP-G), conditioning on size/crop/aspect, base + refiner. ControlNet & IP-Adapter at concept level.

**Build (4h)** —
- `day4/sdxl_baseline.py`: pretrained SDXL inference grid on fixed prompts (some "in naruto style" and some neutral). Note VRAM, latency per image.
- `day4/lora_finetune_small.ipynb`: LoRA fine-tune SDXL UNet (rank 16, ~10M trainable params) on a **small 50-image** subset. This will overfit lightly — that's expected. You'll see "naruto-flavored" outputs on neutral prompts after training. Compare grid pre- vs post-LoRA.
- `day4/lora_finetune_full.py`: same recipe on the **full 1,221 images** in `data/full/` (~1h on H100). This is the *real* style-LoRA experience — generate any prompt in coherent naruto style. Compare against the 10-image version side-by-side to see the data-scale effect.
- Use `diffusers` + `peft`.

**Compare (1h)** — SD1.5 vs SDXL vs SDXL+LoRA. Params, VRAM, latency, 1024² quality. Master table grows.

---

## Day 5 (6h) — DiT, Flow Matching, Flux, SD3

**Theory (2h)** — **DiT:** patchify image latent (like ViT), AdaLN-zero conditioning, pure transformer. Why it scales better than UNet (no inductive bias bottleneck, attention reuse). **MM-DiT (SD3):** text and image tokens in shared sequence with separate weights. **Flow matching / rectified flow:** instead of denoising, learn the velocity field `v_t = x_1 − x_0` along a straight interpolation `x_t = (1−t)x_0 + t·x_1`. Straighter trajectories ⇒ ODE solver needs fewer steps natively. This is why Flux samples well in 4 steps without distillation.

**Build (2.5h)** —
- `day5/dit_tiny.ipynb`: ~10M-param DiT trained on the 1000-image set at 64×64, ~100 epochs. Compare against Day 2 UNet-DDPM head-to-head (same params, same FLOPs budget).
- `day5/flux_sd3_baseline.py`: run pretrained `black-forest-labs/FLUX.1-schnell` (4-step) and `stabilityai/stable-diffusion-3-medium` on the Day 4 prompts.

**Compare (1.5h)** — Master table: SDXL vs SD3 vs Flux-schnell — params, FLOPs/image, H100 latency, your quality rating. This is the moment "modern" lands.

---

## Day 6 (6h) — Acceleration: distillation & few-step

**Theory (1.5h)** — Progressive distillation (Salimans & Ho): student predicts what teacher does in 2 steps, halve iteratively, 1024→512→…→4 steps. **Consistency models (CM):** train so any point on the ODE trajectory maps to the same endpoint — single-step sampling works. **LCM:** consistency in latent space, applied to SD. **ADD / SDXL Turbo:** adversarial distillation, score-distillation + GAN loss. **DMD:** distribution matching distillation, currently SOTA for 1-step. Cost: diversity collapse, mode dropping, fine-detail loss — the engineering trade-off.

**Build (3h)** —
- `day6/cm_distill.ipynb`: distill your Day 2 DDPM into a consistency model on the 1000-image set. Sample at 1/2/4/8 steps.
- `day6/turbo_lightning_lcm.py`: run SDXL-Turbo, SDXL-Lightning (2/4/8-step variants), LCM-LoRA on SDXL, Flux-schnell. All on the same prompts.

**Compare (1.5h)** — The **money plot of the week:** x-axis = sampling steps (1 to 50, log scale), y-axis = latency *and* your quality rating, one line per model. This plot tells you which model to pick under your resource constraints.

---

## Day 7 (6h) — Autoregressive + low-resource deployment

**Theory (1.5h)** — **VQ-VAE:** discrete tokens via vector quantization, image becomes a sequence of codes. **MaskGIT/Muse:** parallel masked decoding, ~10× faster than raster AR. **VAR (next-scale prediction, NeurIPS '24 best paper):** AR over scales not pixels — currently beats diffusion on some ImageNet metrics with lower latency. **Infinity (2024):** bitwise tokenization, scales VAR. Why AR is interesting again: KV-cache + speculative decoding tricks from LLMs transfer directly.

**Build (1.5h)** —
- `day7/vqvae_maskgit.ipynb`: tiny VQ-VAE + MaskGIT-style parallel decoder on the 1000-image set. Didactic — won't beat your diffusion models, but you'll feel the token-grid mental model.

**Deploy (2.5h)** —
- `day7/quantize_low_resource.py`: take your best fast model from Day 6 (likely SDXL-Lightning 4-step or Flux-schnell). Quantize to INT8 weights via `bitsandbytes` and FP8 via `torchao` where applicable. Optionally export ONNX and benchmark with TensorRT.
- Project latency on a low-resource target. Reference numbers (compute vs H100's ~990 TFLOPS bf16): L4 ~30 TFLOPS FP16 (~33× slower); T4 ~8 TFLOPS FP16; A10 ~125 TFLOPS FP16. **Rule of thumb to verify empirically:** small-batch SDXL on a small data-center GPU is typically closer to ~10× slower than H100, not 33×, because memory bandwidth dominates inference, not raw FLOPs.
- Benchmark on H100, multiply by your measured slowdown ratio against the target GPU, and project. For ground truth, rent the target on `vast.ai` for an hour (≈ $0.40 for L4, less for T4).

**Wrap (0.5h)** — Fill the master table's final rows. Pick the model you'd ship under your resource constraints and write 3 sentences on why.

---

## What `master_table.md` looks like

| Day | Model | Params | Train time (H100) | Peak VRAM | Sample steps | Step latency | Total/img | Your quality |
|-----|-------|--------|-------------------|-----------|--------------|--------------|-----------|--------------|
| 1 | tiny VAE | … | … | … | 1 | … | … | …/10 |
| 1 | tiny GAN | … | … | … | 1 | … | … | …/10 |
| 2 | DDPM (UNet) | … | … | … | 1000 / 50 / 20 / 5 | … | … | …/10 |
| 3 | DDPM+CFG | … | … | … | … | … | … | …/10 |
| 4 | SDXL base | 2.6B | — | … | 30 | … | … | …/10 |
| 4 | SDXL+LoRA (50 imgs) | 2.6B + 10M | … | … | 30 | … | … | …/10 |
| 4 | SDXL+LoRA (1.2k imgs) | 2.6B + 10M | … | … | 30 | … | … | …/10 |
| 5 | tiny DiT | 10M | … | … | … | … | … | …/10 |
| 5 | SD3 medium | 2B | — | … | 28 | … | … | …/10 |
| 5 | Flux-schnell | 12B | — | … | 4 | … | … | …/10 |
| 6 | SDXL-Turbo | 2.6B | — | … | 1 | … | … | …/10 |
| 6 | SDXL-Lightning | 2.6B | — | … | 4 | … | … | …/10 |
| 6 | LCM-SDXL | 2.6B | — | … | 4 | … | … | …/10 |
| 6 | CM (yours) | 5M | … | … | 1 / 4 | … | … | …/10 |
| 7 | VQ-VAE + MaskGIT | … | … | … | 8 | … | … | …/10 |
| 7 | SDXL-Lightning INT8 | 2.6B | — | … | 4 | … | … | …/10 |
| 7 | Flux-schnell (low-resource projected) | 12B | — | … | 4 | … | … | …/10 |

---

## Repo layout

```
/mnt/local-fast/aalhejab/
├── CURRICULUM.md            (this file)
├── data/
│   ├── raw/                 (10 sampled naruto-blip images, 512x512, + captions.json)
│   ├── full/                (full 1,221 naruto-blip images, for Day 4 full LoRA)
│   └── eval/                (fixed prompts + seeds, shared across lessons)
├── shared/
│   ├── eval.py              (grid sampler + latency timer, used every day)
│   ├── captions.py          (auto-caption with BLIP-2)
│   └── master_table.py      (appender)
├── day1_foundations/
├── day2_ddpm/
├── day3_score_cfg/
├── day4_sdxl_lora/
├── day5_dit_flux/
├── day6_distill/
├── day7_ar_deploy/
└── results/
    ├── master_table.md
    └── grids/               (PNG per lesson)
```

---

## What I'll do at the start of each day

1. ~15 min theory recap (text, key equations only).
2. Walk through the code I'll write *before* writing it — you'll know what each block does.
3. Write the code, run it on the H100, you watch latency/VRAM live.
4. Generate the grid, you rate, we fill the table.
5. ~15 min closing: what surprised you, what to watch for tomorrow.

---

## Before Day 1

Data is loaded — 10 anime images in `data/raw/` (with captions), 1,221 in `data/full/`. Just say "start Day 1" and I'll begin with the VAE + GAN theory and code.

## Stretch / cut points

If we have spare time: ControlNet hands-on, IP-Adapter, Textual Inversion, FID/CLIP-Score automation, training a tiny rectified-flow model from scratch, attention-map visualization.

If we're behind: cut the Day 7 VQ-VAE/MaskGIT *build* (keep theory only), cut the second SDXL-Lightning quantization path, skip the tiny-DiT build (keep theory + comparison to Day 2).
