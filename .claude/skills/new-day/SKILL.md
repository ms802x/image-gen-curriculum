---
name: new-day
description: Create a new day's notebook for the image-gen curriculum following the project's established conventions (fair-comparison experiments, worked numerical examples, references, master table). Auto-executes the notebook before considering the task done.
---

# new-day — notebook creation for the image-gen curriculum

When this skill is loaded, the user wants a new lesson notebook for the [image-gen-curriculum](https://github.com/ms802x/image-gen-curriculum) project. Follow the conventions below by default — only deviate if the user explicitly asks for it.

## Audience and tone

- The reader is an engineer who values **engineering competence for low-resource deployment** (small GPUs, edge devices) over theoretical novelty.
- Theory is included **only where it changes an implementation choice**.
- The notebook must be **self-contained and shareable**. No autobiographical "dev-log" voice — no phrases like `(was 10)`, `Honest correction:`, `the first version of this notebook`, `tomorrow we'll`, `Revised after feedback`. A colleague reading it cold should understand it without context from any chat.

## File layout

```
day{N}/
└── day{N}_{topic}.ipynb           # the lesson notebook (executed, with outputs)
shared/
├── grid.py                        # already exists — image grid helper
├── fetch_dataset.py               # already exists — pulls naruto-blip
└── make_day{N}_nb.py              # new: generator script that builds the .ipynb
results/
├── grids/day{N}_*.png             # saved sample grids
└── master_table.md                # appended to with new rows
```

**Workflow:** write a Python generator script (`shared/make_day{N}_nb.py`) that emits the notebook JSON, then `jupyter nbconvert --execute --inplace` it. This is reproducible and easy to iterate on. Do not hand-write `.ipynb` JSON.

## Notebook contents (in order)

1. **Title cell** — `# Day {N} — {Topic}`, one-paragraph goal, **Protocol** block (dataset, epochs, batch, hardware), **What you'll leave with** bullets. No autobiographical language.

2. **Setup cell** — imports + config:
    ```python
    DEVICE = "cuda"; IMG_SIZE = 64; BATCH = 32; EPOCHS = 200; LR = 2e-4; SEED = 0
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    EVAL_Z_128 = torch.randn(100, 128, device=DEVICE)   # fixed eval latents
    EVAL_Z_256 = torch.randn(100, 256, device=DEVICE)   # for models with latent=256
    ```
    Hold `EPOCHS` and `BATCH` constant across every variant in the notebook so comparisons are fair.

3. **References cell** — arXiv-linked bibliography for every paper the notebook cites. Add new references as you go; do not invent citations.

4. **Theory cell(s)** — short. The 4-idea / N-idea framing works well: lay out the core claims with the equations that appear in code. Skip full derivations unless the user explicitly asks.

5. **Worked example cell(s) — one per non-obvious concept.** This is non-negotiable. Every concept introduced (a loss formula, a gradient property, a metric, an architecture choice) gets a numerical demo cell that *proves* it with code, not just describes it. Examples that worked in Day 1:
    - "MSE picks the mean" → 1-D toy with target 0 OR 1, plot the loss curve, optimal predictor = 0.5
    - Reparameterization trick → 3-way gradient check (`.sample()` breaks, `mu + sigma*eps` works, `.rsample()` works)
    - KL divergence intuition → numerical table for `(mu, sigma)` ∈ {(0,1), (0,0.5), (0,2), (1,1), (5,1), (0,0.01)}
    - Non-saturating G loss → plot saturating vs non-saturating loss and their derivatives near `p=0`
    Verbal descriptions alone are flagged "vague."

6. **Architecture diagram (ASCII) in a markdown cell.** Tensor shapes layer-by-layer. Easier to read than a code listing of the model.

7. **Data loading** — use `data/full/` (the 1000-image set fetched via `shared/fetch_dataset.py`). Resize to 64×64, normalize to `[-1, 1]`. Show first 25 inline so the reader sees what they're training on.

8. **Model + loss code cells** — small, readable, one class per cell.

9. **Training cell** — track loss history in a `dict` for later plotting. Print epoch summaries every 20 epochs. After training, print `train time`, `peak VRAM`, `final {metric}`. Use `torch.cuda.reset_peak_memory_stats()` before the loop.

10. **Loss-curve plot cell** — required. Even the simplest notebook benefits from seeing convergence.

11. **Reconstructions / samples** — display inline with `matplotlib`. Save the grid to `results/grids/day{N}_*.png` via `shared.grid.save_grid`. Use `EVAL_Z_*[:N]` for samples so columns are comparable across variants.

12. **Latency benchmark cell** — `torch.cuda.synchronize`-bracketed timing loop, 100 forward passes, report ms/image.

13. **Variants (if applicable)** — when comparing alternatives (e.g. baseline vs improved), keep `EPOCHS`, `BATCH`, `data`, and `SEED` identical. The only variables across variants are model size, loss form, regularization, and augmentation. State this explicitly in a comparison table at the top of the variants section.

14. **Side-by-side comparison panel** — final figure. Same `EVAL_Z_*` latents for every row so column *k* is the same point in latent space rendered by different decoders.

15. **Master-table cell** — append rows to `results/master_table.md`. Schema: `| Day | Model | Params(M) | Train(s) | VRAM(GB) | Steps | Step(ms) | {Recon/FeatDiv/...} | Quality |`. Leave `Quality` as `?/10` for the user to fill in.

16. **Summary cell** — a "what you measured" table mapping each question/lever to the cell that demonstrated it. No "tomorrow we'll" / "next week"-style multi-day-course phrasing.

## Metrics

- **Reconstruction quality** → pixel-MSE summed over pixels then divided by batch size (so the magnitude is per-image, easier to reason about).
- **Sample diversity / mode collapse** → use VGG-feature pairwise cosine distance (LPIPS-style — Zhang et al. 2018). **Never use pixel-space cosine distance.** Reference implementation: reuse `vgg_loss._feats()` from Day 1's VAE v2 section, average pairwise `1 - cos(sim)` over feature vectors. Compare against real-image dataset value.
- **Latency** → measure with `torch.cuda.synchronize()` brackets, batch=1, 100 iterations, report mean ms.

## References discipline

Cite the paper for any technique imported from the literature. Use arXiv links. Examples (already in Day 1's references — extend as needed):
- VAE — Kingma & Welling 2013 ([arXiv 1312.6114](https://arxiv.org/abs/1312.6114))
- DDPM — Ho et al. 2020 ([arXiv 2006.11239](https://arxiv.org/abs/2006.11239))
- DDIM — Song et al. 2020 ([arXiv 2010.02502](https://arxiv.org/abs/2010.02502))
- DiT — Peebles & Xie 2022 ([arXiv 2212.09748](https://arxiv.org/abs/2212.09748))
- Flow Matching — Lipman et al. 2022 ([arXiv 2210.02747](https://arxiv.org/abs/2210.02747))
- Classifier-Free Guidance — Ho & Salimans 2022 ([arXiv 2207.12598](https://arxiv.org/abs/2207.12598))
- Consistency Models — Song et al. 2023 ([arXiv 2303.01469](https://arxiv.org/abs/2303.01469))
- LCM — Luo et al. 2023 ([arXiv 2310.04378](https://arxiv.org/abs/2310.04378))
- ADD (SDXL Turbo) — Sauer et al. 2023 ([arXiv 2311.17042](https://arxiv.org/abs/2311.17042))
- LoRA — Hu et al. 2021 ([arXiv 2106.09685](https://arxiv.org/abs/2106.09685))

If unsure of a paper or citation, do a web search to verify. **Never invent citations.**

## Execute before declaring done

After writing the generator and emitting the notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace day{N}/day{N}_*.ipynb \
    --ExecutePreprocessor.timeout=2400
```

If any cell fails, fix the underlying cause and re-execute. Do not declare the notebook ready until it runs clean end-to-end with outputs embedded.

## Forbidden patterns

These produce a worse reader experience. Do not include them:

- Phrases comparing to "prior versions of this notebook" or "the first time I built this"
- `(was X)` / `(baseline X)` annotations inline in titles or prints — let the comparison plots and master table speak
- `Honest correction:`, `Honest framing first:` — just write the correct thing
- `Tomorrow we'll`, `next week`, `the floor for the week` — the notebook should stand alone
- `the money plot of the week` and similar hype
- Pixel-space cosine distance for diversity
- Hand-written `.ipynb` JSON (always use a generator script)
- Inventing paper citations

## After the notebook is done

Hand off to the user. If they want to push, they invoke `/push-curriculum` separately — this skill does **not** auto-push.
