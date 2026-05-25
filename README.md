# Image Generation — Engineering Curriculum

> ⚠️ **Status: under active development.** This curriculum is incomplete. Day 1 and Day 2 are published; later days are planned but not written. Expect breaking changes.

A from-scratch, code-first journey through modern image generation. The goal is **engineering competence for low-resource deployment** — understanding latency, memory, parameter counts, sampling speed, training compute, and the engineering levers that move quality on small GPUs and edge-class compute. Theory is included only where it changes an implementation choice.

## Curriculum

| Day | Topic | Artifacts |
|---|---|---|
| 1 | VAE & GAN foundations (3 VAE variants, 3 GAN variants, worked examples) | [`day1/day1_vae_gan.ipynb`](./day1/day1_vae_gan.ipynb) |
| 2 | DDPM from scratch + paper deep-dive | [`day2/day2_diffusion.ipynb`](./day2/day2_diffusion.ipynb) (trained model on anime data), [`day2/day2_paper_math_to_code.ipynb`](./day2/day2_paper_math_to_code.ipynb) (math + visualizations on 2D toy data), [`day2/ddpm_paper_tutorial.md`](./day2/ddpm_paper_tutorial.md) (paper walkthrough + Q&A) |
| 3 | Score view, samplers, classifier-free guidance | planned |
| 4 | Latent diffusion, SDXL, LoRA fine-tune | planned |
| 5 | DiT, flow matching, FLUX, SD3 | planned |
| 6 | Distillation — Turbo, Lightning, LCM, consistency models | planned |
| 7 | Autoregressive (VAR/MaskGIT) + low-resource deployment | planned |

See [`CURRICULUM.md`](./CURRICULUM.md) for the full week plan.

## What's here

Every model family is built up from a small, runnable PyTorch implementation, then compared side-by-side with fixed eval latents. Each lesson appends a row to [`results/master_table.md`](./results/master_table.md) so the trade-offs (params, train time, peak VRAM, sampler latency, quality) are directly comparable across approaches.

Day 2 additionally includes a full reading of the original DDPM paper (Ho, Jain, Abbeel 2020) with:
- 7 prose steps explaining the math of §2–§4 from first principles, suitable for following along with paper and pen
- A companion notebook running every equation on 2D spiral data with numerical verification
- Detailed Q&A sections on common questions: what `q(x_0)` means, what the identity matrix is doing as a covariance parameter, what `torch.cumprod` is computing, how the noise term appears in the sample form (reparameterization trick), and what would break if we removed the isotropic noise from the forward process

## Dataset

[`lambdalabs/naruto-blip-captions`](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions) — 1,221 anime images at 512×512 with BLIP captions. Resized to 64×64 for from-scratch experiments; native resolution for SDXL/Flux fine-tunes later in the curriculum.

Data is not committed (it's too large for git). Regenerate locally with:

```bash
python shared/fetch_dataset.py
```

This downloads the full set into `data/full/` and saves a 10-image sample into `data/raw/`. Notebooks expect these paths.

## Hardware notes

Notebooks target a single H100 (80GB) but are deliberately small enough to run on any modern GPU. Per-cell wall-times are reported in each notebook's training output. **End deployment target: low-resource compute** — small data-center GPUs (L4, T4, A10), consumer cards, or on-device inference — reached via quantization, step-distilled diffusion, and architecture choices (covered Day 7).

## Layout

```
.
├── CURRICULUM.md                          # full week plan
├── README.md
├── data/                                  # not committed; created by shared/fetch_dataset.py
├── day1/
│   └── day1_vae_gan.ipynb                 # 3 VAE variants + 3 GAN variants
├── day2/
│   ├── day2_diffusion.ipynb               # DDPM trained on anime (YHL04/ddpm reference)
│   ├── day2_paper_math_to_code.ipynb      # paper math visualized on 2D toy
│   └── ddpm_paper_tutorial.md             # paper walkthrough + Q&A
├── results/
│   ├── master_table.md                    # numeric comparison across lessons
│   └── grids/                             # sample grids per lesson
└── shared/
    ├── grid.py                            # image-grid helper
    ├── fetch_dataset.py                   # reproducible dataset download
    ├── make_day1_nb.py                    # Day 1 notebook generator
    ├── make_day2_nb.py                    # Day 2 main notebook generator
    └── make_day2_math_nb.py               # Day 2 math notebook generator
```

## References

Every notebook cites the papers it depends on. Day 2 references include Ho, Jain, Abbeel (DDPM), Nichol & Dhariwal (Improved DDPM), Peebles & Xie (DiT), Hang et al. (Min-SNR), Song et al. (DDIM, score-based SDE), and Karras et al. (EDM). Day 1 references include Kingma & Welling (VAE), Goodfellow (GAN), Radford (DCGAN), Bowman (KL annealing), Johnson (perceptual loss), Zhang (LPIPS), Miyato (spectral norm), Heusel (TTUR), Lim & Ye (hinge loss), Mescheder (R1), Zhao (DiffAugment), Karras (StyleGAN-ADA), and Liu (FastGAN).

## License

MIT.
