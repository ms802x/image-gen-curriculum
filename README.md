# Image Generation — Engineering Curriculum

A from-scratch, code-first journey through modern image generation. The goal is **engineering competence** — understanding latency, memory, parameter counts, sampling speed, training compute, and the engineering levers that move quality on real hardware. Theory is included only where it changes an implementation choice.

## What's here

Every model family is built up from a small, runnable PyTorch implementation, then compared side-by-side on the same 1000-image dataset with fixed eval latents. Each lesson appends a row to `results/master_table.md` so the trade-offs (params, train time, peak VRAM, sampler latency, quality) are directly comparable across approaches.

## Curriculum (week-long sprint)

See [`CURRICULUM.md`](./CURRICULUM.md) for the full plan. Summary:

| Day | Topic | Status |
|---|---|---|
| 1 | VAE & GAN foundations (3 VAE variants, 3 GAN variants, worked examples) | [notebook](./day1/day1_vae_gan.ipynb) |
| 2 | DDPM from scratch | planned |
| 3 | Score view, samplers, classifier-free guidance | planned |
| 4 | Latent diffusion, SDXL, LoRA fine-tune | planned |
| 5 | DiT, flow matching, FLUX, SD3 | planned |
| 6 | Distillation — Turbo, Lightning, LCM, consistency models | planned |
| 7 | Autoregressive (VAR/MaskGIT) + L4 deployment | planned |

## Dataset

[`lambdalabs/naruto-blip-captions`](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions) — 1,221 anime images at 512×512 with BLIP captions. Resized to 64×64 for from-scratch experiments; native resolution for SDXL/Flux fine-tunes later in the curriculum.

Data is not committed (it's too large for git). Regenerate locally with one command:

```bash
python shared/fetch_dataset.py
```

This downloads the full set into `data/full/` and saves a 10-image sample into `data/raw/`. Notebooks expect these paths.

## Hardware notes

The notebooks are written for a single H100 (80GB) but are deliberately small enough to run on any modern GPU. Per-cell wall-times are reported in each notebook's training output. End deployment target: L4 (24GB) via quantization and step-distilled diffusion (covered Day 7).

## Layout

```
.
├── CURRICULUM.md           # full week plan with theory + engineering goals
├── README.md
├── data/                   # not committed; created by `python shared/fetch_dataset.py`
│   ├── raw/                # 10-image sample
│   └── full/               # full 1,221-image set
├── day1/
│   └── day1_vae_gan.ipynb  # VAE + GAN foundations, 3 variants each
├── results/
│   ├── master_table.md     # numeric comparison across all lessons
│   └── grids/              # sample grids per lesson
└── shared/
    ├── grid.py             # image-grid helper (used every day)
    ├── fetch_dataset.py    # reproducible dataset download
    └── make_day1_nb.py     # notebook generator for Day 1
```

## References

Each notebook cites the papers it depends on. The Day 1 references cell lists the full bibliography for the VAE / GAN material (Kingma & Welling, Goodfellow, Radford, Bowman, Johnson, Zhang (LPIPS), Miyato, Heusel, Lim & Ye, Mescheder, Zhao (DiffAugment), Karras, Liu). Future days will follow the same pattern.

## License

MIT.
