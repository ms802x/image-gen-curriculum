# DDPM Paper Tutorial — Read-Along Q&A

A guided walk through **Denoising Diffusion Probabilistic Models** (Ho, Jain, Abbeel, NeurIPS 2020). The format is Q&A: each chunk introduces one idea or one equation from the paper, then captures the questions that came up while reading it. Everything below is grounded in the paper text — no invented details.

Paper: [arXiv 2006.11239](https://arxiv.org/abs/2006.11239)

---

## Step 1 — What problem is this paper solving?

**Setup.** The title is *Denoising Diffusion Probabilistic Models*. Before touching any equation, ground the abstract's vocabulary.

> *"We present high quality image synthesis results using diffusion probabilistic models, a class of latent variable models inspired by considerations from nonequilibrium thermodynamics."*

Two terms to anchor on:

### "Image synthesis"

Generate new images that look like they came from a real dataset. Concretely: we have data $x^{(1)}, x^{(2)}, \ldots$ assumed to be samples from an unknown distribution $q(x_0)$. We want to build a *model* $p_\theta(x_0)$ that closely matches $q(x_0)$. Once we have that, sampling from $p_\theta$ yields new images.

> **Generate an image = draw a sample from a probability distribution over pixel arrays.**

### "Latent variable model"

Modeling $p_\theta(x_0)$ for high-dimensional images directly is intractable. The latent-variable trick: introduce hidden variables $z$ that we never observe, define the joint $p_\theta(x_0, z)$, and recover the marginal we care about by integrating:

$$p_\theta(x_0) \;=\; \int p_\theta(x_0, z)\, dz$$

This is the same machinery as a VAE — encoder $q(z|x)$, decoder $p_\theta(x|z)$.

### What this paper proposes (one-sentence preview)

Instead of a single latent vector $z$ like a VAE, use a *sequence* of latents $x_1, x_2, \ldots, x_T$ — a "diffusion chain" — that progressively destroys an image into pure noise. Then learn to *reverse* that destruction.

---

### Q1.1 — What does $q(x_0)$ actually mean? Is it one image, a set of images, or something else?

**Not** one image. **Not** a set either. It's a **function** that assigns a probability density to *every possible image* of a given size.

**Concrete picture.** The space of all possible 64×64 RGB images contains $256^{12{,}288}$ candidates (astronomical). $q(x_0)$ takes any one of them as input and outputs a number:
- $q(\text{real cat photo}) \to$ high
- $q(\text{real face photo}) \to$ high
- $q(\text{pure random noise}) \to$ essentially zero
- $q(\text{cat photo with green sky}) \to$ low but nonzero
- $q(\text{unicorn riding tricycle}) \to$ near zero

So $q(x_0)$ is **"how natural-looking is this image, viewed as a number?"** It encodes the entire concept of "the distribution our data comes from." We never have $q$ as an explicit function — we only have *samples* from it (our training images).

**Notation $x_0 \sim q(x_0)$** means "$x_0$ is drawn from $q$." Each draw gives a different image; high-probability regions get sampled more often.

**Why the subscript 0?** Because images are about to become the *first* element of a chain $x_0, x_1, x_2, \ldots, x_T$. The clean image is $x_0$; the noisier latents are $x_1, x_2, \ldots$ in order of increasing noise. The subscript 0 just means "step 0 — the original, no-noise state." This sets up Step 2 cleanly.

**$q$ vs $p_\theta$ convention.**
- $q$ = the true distribution (data, forward process). Unknown function, only samples available.
- $p_\theta$ = our **model**, parameterized by neural-network weights $\theta$. We can evaluate and sample from it.

Training's goal: make $p_\theta(x_0)$ close to $q(x_0)$.

### Q1.2 — What does the integral $\int p_\theta(x_0, z)\, dz$ mean? Why is there an integral in a definition of probability?

It's **marginalization** — the standard machinery for going from a joint distribution over two variables to a distribution over just one of them.

**Discrete toy example.** Two random variables: $x \in \{\text{rain}, \text{no rain}\}$, $z \in \{\text{spring}, \text{summer}, \text{fall}, \text{winter}\}$. The joint $p(x, z)$ is a 2×4 table of probabilities that sum to 1. To get *just* $p(x = \text{rain})$ — ignoring season — you sum over all seasons:

$$p(x = \text{rain}) = \sum_z p(x = \text{rain}, z)$$

For *continuous* variables, summing becomes integrating:

$$p(x) = \int p(x, z)\, dz$$

The integral is the continuous version of "sum over all possible values of $z$, weighted by their joint probability with $x$."

**Why introduce $z$ at all?** Because the joint $p_\theta(x_0, z)$ is *easier to define* than $p_\theta(x_0)$ directly. We pick the prior $p_\theta(z)$ to be a simple Gaussian and the conditional $p_\theta(x_0 | z)$ to be a neural network. Together they give a usable joint. The integral is the cost of admission.

**In DDPM specifically**, the latent isn't a single $z$ but the whole chain $z = (x_1, \ldots, x_T)$ — $T$ latent images. So:

$$p_\theta(x_0) = \int p_\theta(x_0, x_1, \ldots, x_T)\, dx_1 \cdots dx_T$$

Looks scary; the paper structures the joint so this integral is tractable. We'll see how next.

---

## Step 2 — The diffusion chain (the forward process)

### The picture before the math

The paper's Figure 2 shows the whole game on one line:

```
  x_T ──── ... ────► x_t ────► x_{t-1} ────► ... ────► x_0
  noise                                                clean image
       ◄────── reverse process (learned, p_θ) ──────
       ──────► forward process (fixed, q) ──────────►
```

Two processes operate on the same chain of $T+1$ variables ($x_0$ through $x_T$):
- **Forward $q$**: starts from a clean image $x_0$, gradually adds noise step by step, ends at $x_T$ which is approximately pure Gaussian noise. **Fixed** — no learnable parameters. We *define* it.
- **Reverse $p_\theta$**: starts from pure noise $x_T$, gradually denoises step by step, ends at a clean image $x_0$. **Learned** — this is what the neural network does.

For Step 2 we focus only on the forward process $q$.

### The Markov chain notation

The paper writes (equation 2):

$$q(x_{1:T} \mid x_0) := \prod_{t=1}^{T} q(x_t \mid x_{t-1})$$

This is the **Markov property**: the joint distribution of all latents given $x_0$ factors as a product of per-step transitions, and each per-step transition $q(x_t \mid x_{t-1})$ depends *only on the previous step*, not on the history before it.

Notation decoded:
- $x_{1:T}$ is shorthand for the tuple $(x_1, x_2, \ldots, x_T)$.
- $q(x_{1:T} \mid x_0)$ is a **conditional distribution**: given the original image $x_0$, what's the joint probability of the noisy latent sequence?
- $q(x_t \mid x_{t-1})$ is the **per-step transition**: given $x_{t-1}$, what's the distribution of $x_t$?
- The product $\prod_{t=1}^{T}$ multiplies $T$ such transitions together.

The Markov assumption matters because it makes the chain *factorize* into independent per-step pieces. Without it, $q(x_t \mid x_{t-1}, x_{t-2}, \ldots)$ would have to remember the whole history — intractable.

### The actual per-step transition

This is equation 2 in the paper:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\ \sqrt{1 - \beta_t}\,x_{t-1},\ \beta_t I\right)$$

Decoding $\mathcal{N}(\cdot;\ \mu,\ \Sigma)$ notation:
- $\mathcal{N}$ is the Gaussian (normal) distribution.
- First arg is the **variable** being assigned a distribution (here, $x_t$).
- Second arg is the **mean** $\mu$.
- Third arg is the **covariance matrix** $\Sigma$.

For multivariate $x_t$ (an image with many pixels):
- Mean $\mu = \sqrt{1 - \beta_t}\,x_{t-1}$ — a vector (same shape as $x_{t-1}$): every pixel gets shrunk by the scalar $\sqrt{1 - \beta_t}$.
- Covariance $\Sigma = \beta_t I$ — $I$ is the identity matrix. This means *every pixel gets independent Gaussian noise with variance $\beta_t$*, no correlation between pixels.

To **sample** $x_t$ from this distribution (in code):

$$x_t = \sqrt{1 - \beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

That's a one-liner. Shrink $x_{t-1}$ by $\sqrt{1 - \beta_t}$, then add independent Gaussian noise scaled by $\sqrt{\beta_t}$.

### Why this *specific* form? (Variance-preserving Markov chain)

This isn't arbitrary. The coefficients $\sqrt{1-\beta_t}$ on the mean and $\beta_t$ on the variance are chosen so that **if $x_{t-1}$ has variance 1, then $x_t$ also has variance 1**.

Proof in one line, assuming $\text{Var}(x_{t-1}) = 1$ and noise is independent of $x_{t-1}$:
$$\text{Var}(x_t) = (\sqrt{1 - \beta_t})^2 \cdot \text{Var}(x_{t-1}) + (\sqrt{\beta_t})^2 \cdot 1 = (1 - \beta_t) + \beta_t = 1$$

So the chain "preserves variance" — the magnitude of the latent stays roughly constant from step to step. Without this, repeated noise additions would blow up.

### The noise schedule $\beta_t$

The numbers $\beta_1, \beta_2, \ldots, \beta_T$ are called the **variance schedule**. The original DDPM paper chooses them to grow linearly from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$ (page 5, Experiments section). Properties:
- Each $\beta_t$ is small ($\ll 1$). So each step adds only a *small* amount of noise.
- $\beta_t$ grows with $t$. Later steps add more noise per step than early steps.
- Schedule total length $T = 1000$ in the paper's experiments.

After $T = 1000$ such tiny noise additions, $x_T$ is approximately $\mathcal{N}(0, I)$ — indistinguishable from pure standard Gaussian noise.

### What we've built so far

- $q(x_t \mid x_{t-1})$ — a known Gaussian transition that adds a small amount of noise.
- $q(x_{1:T} \mid x_0)$ — the full forward chain, factorized as $T$ transitions.
- Nothing here is learned. The forward process is a *fixed mathematical construction* the paper designs.

Next step: a **trick** that lets us sample $x_t$ from $x_0$ in *one shot* (no Markov chain needed), in closed form. This is what makes training feasible.

---

## Step 3 — The closed-form forward (jump from $x_0$ to any $x_t$ in one shot)

### The trick in one sentence

Even though the forward chain is defined as $T$ small Gaussian steps, the *cumulative* result $x_t$ given $x_0$ turns out to **also be Gaussian**, with a known mean and variance. So we don't have to simulate $t$ steps — we can sample $x_t$ in one line of code.

### New notation: $\alpha_t$ and $\bar\alpha_t$

The paper introduces two abbreviations (just above equation 4):

$$\alpha_t := 1 - \beta_t \qquad \text{and} \qquad \bar\alpha_t := \prod_{s=1}^{t} \alpha_s$$

- $\alpha_t = 1 - \beta_t$ is "the fraction of signal that survives one step." If $\beta_t$ is small, $\alpha_t$ is close to 1.
- $\bar\alpha_t$ (read: "alpha-bar at $t$") is the **product** of all $\alpha_s$ from $s = 1$ up to $t$. It's the cumulative signal that survives after $t$ steps.

Concretely:
- $\bar\alpha_1 = \alpha_1$
- $\bar\alpha_2 = \alpha_1 \alpha_2$
- $\bar\alpha_3 = \alpha_1 \alpha_2 \alpha_3$
- ...
- $\bar\alpha_T = \alpha_1 \alpha_2 \cdots \alpha_T$

Since each $\alpha_s$ is close to 1 (a little less, because $\beta_s$ is small positive), $\bar\alpha_t$ starts near 1 at $t=1$ and shrinks toward 0 by $t = T$. **$\bar\alpha_t$ is the "signal scale" at step $t$.**

### The closed-form equation (paper equation 4)

$$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\ \sqrt{\bar\alpha_t}\,x_0,\ (1 - \bar\alpha_t)\,I\right)$$

To **sample** in one line:

$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)$$

That's it. Pick any $t \in \{1, \ldots, T\}$, draw a single Gaussian noise vector $\varepsilon$, combine. No loop.

### Where does this formula come from? (Light derivation)

We have the per-step rule (Step 2):

$$x_t = \sqrt{1 - \beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\varepsilon_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{1 - \alpha_t}\,\varepsilon_t$$

where each $\varepsilon_t$ is a fresh independent $\mathcal{N}(0, I)$.

**One substitution.** Replace $x_{t-1}$ using the same rule one step earlier:

$$x_t = \sqrt{\alpha_t}\left(\sqrt{\alpha_{t-1}}\,x_{t-2} + \sqrt{1 - \alpha_{t-1}}\,\varepsilon_{t-1}\right) + \sqrt{1 - \alpha_t}\,\varepsilon_t$$

$$= \sqrt{\alpha_t \alpha_{t-1}}\,x_{t-2} + \sqrt{\alpha_t (1 - \alpha_{t-1})}\,\varepsilon_{t-1} + \sqrt{1 - \alpha_t}\,\varepsilon_t$$

Now use a **standard fact**: a sum of independent Gaussians is Gaussian. Specifically, $a\,\varepsilon_a + b\,\varepsilon_b$ (with $\varepsilon_a, \varepsilon_b$ independent unit Gaussians) is itself Gaussian with variance $a^2 + b^2$. So we can collapse the two noise terms into one:

$$x_t = \sqrt{\alpha_t \alpha_{t-1}}\,x_{t-2} + \sqrt{\alpha_t(1 - \alpha_{t-1}) + (1 - \alpha_t)}\,\tilde\varepsilon$$

The inside of the noise sqrt simplifies:
$$\alpha_t(1 - \alpha_{t-1}) + (1 - \alpha_t) = \alpha_t - \alpha_t \alpha_{t-1} + 1 - \alpha_t = 1 - \alpha_t \alpha_{t-1}$$

So:
$$x_t = \sqrt{\alpha_t \alpha_{t-1}}\,x_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}}\,\tilde\varepsilon$$

**Notice the pattern.** That's *exactly the same shape* as the one-step rule, but with the product $\alpha_t \alpha_{t-1}$ in place of a single $\alpha$.

**Continue substituting** all the way back to $x_0$. The signal coefficient becomes the full product $\sqrt{\alpha_t \alpha_{t-1} \cdots \alpha_1} = \sqrt{\bar\alpha_t}$, and the combined noise variance is $1 - \bar\alpha_t$:

$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\varepsilon$$

That's the closed form. The whole chain of $t$ Gaussian transitions collapses into a single Gaussian transition.

### Sanity check at the extremes

- **At $t = 1$**: $\bar\alpha_1 = \alpha_1 = 1 - \beta_1$. With $\beta_1 = 10^{-4}$, $\bar\alpha_1 \approx 0.9999$. So $x_1 \approx x_0$ (almost identical to clean image), with a tiny bit of noise added.
- **At $t = T$**: $\bar\alpha_T = \prod_{s=1}^{T}(1 - \beta_s) \approx 0$ for the paper's linear schedule with $T=1000$. So $x_T \approx 0 \cdot x_0 + 1 \cdot \varepsilon = \varepsilon$, pure noise — no information about $x_0$ left.

These match the qualitative picture: the chain starts at the clean image and ends in pure noise.

### Why this is *the* enabling trick for training

Without the closed form, training a denoiser would require simulating the full forward chain for every gradient step:
1. Sample $x_0$ from data.
2. Pick a random $t$.
3. Loop $t$ times: $x_1 \leftarrow x_0$, then $x_2 \leftarrow x_1$, then ..., then $x_t \leftarrow x_{t-1}$.
4. Compute model output and loss.

With the closed form, step 3 becomes a one-line tensor op:
```python
x_t = sqrt(alpha_bar[t]) * x_0 + sqrt(1 - alpha_bar[t]) * eps
```

For $T = 1000$, that's a **1000× speedup** per training step. This is what makes DDPM training feasible at scale.

### What we have so far (recap)

- **Forward process** $q$ is fully defined by the variance schedule $\beta_1, \ldots, \beta_T$.
- The per-step transition $q(x_t \mid x_{t-1})$ is a Gaussian (Step 2).
- The closed-form $q(x_t \mid x_0)$ is also a Gaussian, with mean $\sqrt{\bar\alpha_t}\,x_0$ and variance $(1 - \bar\alpha_t)\,I$ (Step 3).
- Both processes are *fixed* — no learnable parameters.

The forward process is now fully specified. **Nothing about $q$ is learned.** Everything learnable lives in the *reverse* process $p_\theta$, which is what we tackle in Step 4.

---

## Step 4 — The reverse process (what we actually learn)

### What the reverse process *is*

The forward process $q$ takes a clean image and destroys it into noise. The reverse process $p_\theta$ should do the opposite: take pure noise and reconstruct a plausible clean image, one denoising step at a time.

The paper writes the reverse process (equation 1) as a Markov chain too, but going the other direction:

$$p_\theta(x_{0:T}) := p(x_T)\,\prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t)$$

- $x_{0:T}$ is shorthand for the whole tuple $(x_0, x_1, \ldots, x_T)$.
- $p(x_T) = \mathcal{N}(x_T;\ 0, I)$ is the **starting distribution** — just standard Gaussian noise. No parameters, no learning here. We picked this because, as Step 3 showed, $q(x_T \mid x_0) \approx \mathcal{N}(0, I)$ for any $x_0$.
- $p_\theta(x_{t-1} \mid x_t)$ is the **learned per-step reverse transition**: given the noisier latent $x_t$ at step $t$, produce a (slightly less noisy) $x_{t-1}$.

So the joint $p_\theta(x_{0:T})$ is: start from random noise, then take $T$ learned denoising steps. Sampling from this joint gives you a sequence $(x_0, x_1, \ldots, x_T)$, but the only one we care about is the final $x_0$ — the generated image.

### What does $p_\theta(x_{t-1} \mid x_t)$ look like?

The paper writes (equation 1):

$$p_\theta(x_{t-1} \mid x_t) := \mathcal{N}\!\left(x_{t-1};\ \mu_\theta(x_t, t),\ \Sigma_\theta(x_t, t)\right)$$

Same Gaussian notation as Step 2, but with two key differences:
1. **The mean and covariance are functions output by a neural network** $\theta$. That is, given the current noisy image $x_t$ and the current timestep $t$, the network produces a mean vector $\mu_\theta(x_t, t)$ and a covariance matrix $\Sigma_\theta(x_t, t)$ that describe where $x_{t-1}$ likely is.
2. Both $\mu$ and $\Sigma$ depend on $t$. The same network handles all timesteps; we just tell it which timestep we're at.

In code, you'd feed the network an image $x_t$ and an integer $t$, and it outputs the parameters of a Gaussian distribution. To sample $x_{t-1}$, you draw from that Gaussian.

### Why Gaussian?

This is a subtle but important point the paper makes in its **§2 Background** discussion (citing Sohl-Dickstein et al. 2015, reference [53]):

> "When the diffusion consists of small amounts of Gaussian noise, it is sufficient to set the sampling chain transitions to conditional Gaussians too, allowing for a particularly simple neural network parameterization."

In other words: **because each forward step adds only a *small* amount of noise (small $\beta_t$), the true reverse posterior $q(x_{t-1} \mid x_t)$ is approximately Gaussian**. Therefore modeling the reverse transition $p_\theta(x_{t-1} \mid x_t)$ as Gaussian is a *good enough* approximation. This is the engineering rationale for the Gaussian parameterization of the reverse process.

(If $\beta_t$ were large — say each step destroyed half the signal — the true reverse posterior would *not* be Gaussian, and this whole approach would break. The "many small steps" structure of diffusion is what makes Gaussian reverse transitions work.)

### Sampling — what an image generation actually does

Given a trained model with parameters $\theta$, here is how you generate one new image (paper Algorithm 2):

```
1. x_T ~ N(0, I)           # start from pure noise
2. for t = T, T-1, ..., 1:
3.     Compute mu, sigma from the network: mu, sigma = model(x_t, t)
4.     z ~ N(0, I) if t > 1, else z = 0
5.     x_{t-1} = mu + sigma * z      # sample from p_theta(x_{t-1} | x_t)
6. return x_0
```

That's the entire inference pipeline. Each iteration is one neural network forward pass. With $T = 1000$, sampling one image takes 1000 network evaluations.

### What's the loss?

So far we've defined the *architecture* of the reverse process — Gaussian transitions parameterized by a neural network — but we haven't said *how to train* it. Training means picking $\theta$ such that the model's joint $p_\theta(x_{0:T})$ explains the data well: the marginal $p_\theta(x_0)$ should be close to $q(x_0)$.

The standard tool for this is the **variational lower bound (ELBO)** on the negative log-likelihood. That's what Step 5 covers.

### What we have so far (recap)

| Object | Definition | Learned? |
|---|---|---|
| Forward $q(x_t \mid x_{t-1})$ | Gaussian with mean $\sqrt{1-\beta_t}\,x_{t-1}$ and variance $\beta_t I$ | No (fixed) |
| Closed-form forward $q(x_t \mid x_0)$ | Gaussian with mean $\sqrt{\bar\alpha_t}\,x_0$ and variance $(1-\bar\alpha_t) I$ | No (fixed) |
| Prior $p(x_T)$ | $\mathcal{N}(0, I)$ | No (fixed) |
| Reverse $p_\theta(x_{t-1} \mid x_t)$ | Gaussian with mean $\mu_\theta(x_t, t)$ and covariance $\Sigma_\theta(x_t, t)$ | **Yes — neural network** |

Step 5 builds the training loss that lets us *learn* $\mu_\theta$ and $\Sigma_\theta$.

---

## Step 5 — The training objective (variational lower bound)

### The problem: $\log p_\theta(x_0)$ is intractable

We want to *maximize* $\log p_\theta(x_0)$ (equivalently, minimize the negative log-likelihood $-\log p_\theta(x_0)$) so the model assigns high probability to real images. But from Step 1 we know:

$$p_\theta(x_0) = \int p_\theta(x_0, x_1, \ldots, x_T)\, dx_1\,dx_2\,\cdots\,dx_T$$

That's a $T$-fold integral over **all possible noisy latent sequences** — totally intractable. We cannot evaluate it, so we cannot directly optimize $\log p_\theta(x_0)$.

### The fix: a *bound* we can compute

We bound the negative log-likelihood from above by a quantity we *can* compute, and minimize that bound. This is the **variational lower bound (VLB)** trick, also called the **ELBO** (evidence lower bound). The paper's equation 3 states it:

$$\mathbb{E}\bigl[-\log p_\theta(x_0)\bigr] \;\leq\; \mathbb{E}_q\!\left[-\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\right] \;=:\; L$$

The left side is what we *want* to minimize (the true negative log-likelihood). The right side $L$ is the variational bound — an upper bound on the left, and tractable. **Minimizing $L$ pushes the true negative log-likelihood down too.**

### Where does the bound come from? (Jensen's inequality in one move)

$$-\log p_\theta(x_0) = -\log \int p_\theta(x_{0:T})\, dx_{1:T}$$

Multiply and divide by $q(x_{1:T} \mid x_0)$ inside the integral (multiplying by 1, since $\int q\,dx = 1$):

$$= -\log \int q(x_{1:T} \mid x_0)\, \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\, dx_{1:T} \;=\; -\log \mathbb{E}_q\!\left[\frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\right]$$

Now apply **Jensen's inequality**: $-\log \mathbb{E}[X] \leq \mathbb{E}[-\log X]$ because $-\log$ is convex. So:

$$-\log p_\theta(x_0) \;\leq\; \mathbb{E}_q\!\left[-\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\right] \;=\; L$$

That's the bound. (Don't worry if Jensen's inequality is new — the result is what matters: we have a tractable upper bound on the thing we wanted to minimize.)

### Why is $L$ tractable when $\log p_\theta(x_0)$ wasn't?

Because both numerator and denominator inside the log **factorize** over the chain:

$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t) \qquad \text{(Step 4)}$$
$$q(x_{1:T} \mid x_0) = \prod_{t=1}^{T} q(x_t \mid x_{t-1}) \qquad \text{(Step 2)}$$

So the ratio's log is just a *sum* of per-step log terms — no integral over all $T$ latents. This is what the paper's equation 3 rewrites as:

$$L = \mathbb{E}_q\!\left[-\log p(x_T) - \sum_{t \geq 1} \log \frac{p_\theta(x_{t-1} \mid x_t)}{q(x_t \mid x_{t-1})}\right]$$

A computable expectation over per-timestep terms.

### Further simplification: condition the posterior on $x_0$ (paper equation 5)

The form above works but has high variance during stochastic optimization. The paper applies one more rewrite, summarized in equation 5:

$$L = \mathbb{E}_q\!\Bigl[\underbrace{D_{KL}\!\bigl(q(x_T \mid x_0) \,\|\, p(x_T)\bigr)}_{L_T} \;+\; \sum_{t > 1} \underbrace{D_{KL}\!\bigl(q(x_{t-1} \mid x_t, x_0) \,\|\, p_\theta(x_{t-1} \mid x_t)\bigr)}_{L_{t-1}} \;-\; \underbrace{\log p_\theta(x_0 \mid x_1)}_{L_0}\Bigr]$$

This decomposes the bound into **$T + 1$ named pieces**:

| Term | Name | What it measures |
|---|---|---|
| $L_T = D_{KL}\bigl(q(x_T \mid x_0) \,\|\, p(x_T)\bigr)$ | Final-state mismatch | How close is the noised endpoint to the assumed pure-noise prior $\mathcal{N}(0,I)$? |
| $L_{t-1} = D_{KL}\bigl(q(x_{t-1} \mid x_t, x_0) \,\|\, p_\theta(x_{t-1} \mid x_t)\bigr)$, for $t > 1$ | Per-step reverse mismatch | How close is each learned reverse step to the true reverse posterior (given $x_0$)? |
| $L_0 = -\log p_\theta(x_0 \mid x_1)$ | Decoder term | Log-likelihood of the original image given the first latent. |

**Why is this form better?**
1. $L_T$ depends only on the noise schedule, not on $\theta$. With a long enough chain and small enough $\beta$, $q(x_T \mid x_0) \approx \mathcal{N}(0, I) = p(x_T)$, so $L_T \approx 0$. **It's effectively a constant — ignored during training.**
2. Each $L_{t-1}$ is a **KL between two Gaussians** (because both arguments are Gaussian, as shown in §3.2 of the paper). KL between Gaussians has a **closed-form** expression — no Monte Carlo needed within the KL.
3. The decomposition gives one well-behaved loss per timestep, allowing stochastic optimization: pick a random $t$, compute *one* term, take a gradient step. (Lots more efficient than summing all $T$ terms each step.)

### The conditional posterior $q(x_{t-1} \mid x_t, x_0)$ — the magic ingredient

The key object inside each $L_{t-1}$ is $q(x_{t-1} \mid x_t, x_0)$ — the **true reverse posterior**, *conditional* on knowing the clean image $x_0$.

You might ask: aren't we trying to *recover* $x_0$? Why condition on it?

Answer: during *training* we have $x_0$ (it's in our dataset). We use the *known* clean image to construct an exact target for the reverse transition. The neural network then learns to produce the same transition *without* needing $x_0$.

The paper derives (equation 6, 7):
$$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\!\left(x_{t-1};\ \tilde\mu_t(x_t, x_0),\ \tilde\beta_t I\right)$$
where
$$\tilde\mu_t(x_t, x_0) := \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1 - \bar\alpha_t}\,x_0 + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t}\,x_t$$
$$\tilde\beta_t := \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\,\beta_t$$

This is a Gaussian with closed-form mean and variance in terms of $x_0$, $x_t$, and the schedule. The whole point of the next-step parameterization (covered in Step 6) is to *match* this distribution efficiently.

### Putting it together — what training actually does

Training a DDPM amounts to:
1. Sample $x_0$ from the training data.
2. Pick a random timestep $t \in \{1, \ldots, T\}$.
3. Sample $x_t$ in one shot using the closed form $q(x_t \mid x_0)$ (Step 3).
4. Compute the loss term $L_{t-1}$ — a KL between the **true reverse posterior** $q(x_{t-1} \mid x_t, x_0)$ (known closed-form Gaussian) and the **learned reverse transition** $p_\theta(x_{t-1} \mid x_t)$ (the network's Gaussian).
5. Backprop through $\theta$.

The network learns, timestep by timestep, to predict the correct reverse Gaussian *without* seeing $x_0$ directly.

The next refinement — Step 6 — shows that this KL between Gaussians can be rewritten as a **simple MSE** on a transformed quantity, leading to the famously elegant DDPM loss.

---

## Step 6 — The ε-parameterization and the famous "simple" loss

Step 5 left us with: train the network so $p_\theta(x_{t-1} \mid x_t)$ matches the true reverse posterior $q(x_{t-1} \mid x_t, x_0)$, measured by KL between two Gaussians. The paper now makes a sequence of parameterization choices that collapse this KL into a clean **MSE on noise**. This is what made DDPM practical.

### Choice 1 — Fix the variance, don't learn it

The paper's §3.2 chooses (for this paper's experiments):

$$\Sigma_\theta(x_t, t) = \sigma_t^2\, I$$

where $\sigma_t^2$ is a **fixed scalar per timestep**, not a function of $x_t$ and not learned. They tried two choices: $\sigma_t^2 = \beta_t$ and $\sigma_t^2 = \tilde\beta_t$ (the closed-form posterior variance from Step 5). Both gave similar sample quality. So the only thing the network needs to predict is the **mean** $\mu_\theta(x_t, t)$.

(Aside: a later paper — Nichol & Dhariwal 2021, the "Improved DDPM" we used in the Day 2 notebook — *did* make the variance learnable. The reference repo's $v$-head is that. But the original paper here fixes it.)

### Why this collapses KL into MSE

A standard fact: the KL between two Gaussians with the *same* variance $\sigma^2 I$ is just MSE on their means:

$$D_{KL}\bigl(\mathcal{N}(\mu_1, \sigma^2 I) \,\|\, \mathcal{N}(\mu_2, \sigma^2 I)\bigr) = \frac{1}{2\sigma^2}\|\mu_1 - \mu_2\|^2$$

(The general Gaussian-KL formula has variance-difference terms; they vanish when the variances are equal.)

Apply this to $L_{t-1}$ from Step 5. Both Gaussians have variance $\sigma_t^2 I$ (we just fixed the model's variance to match), so equation 8 of the paper becomes:

$$L_{t-1} = \mathbb{E}_q\!\left[\frac{1}{2\sigma_t^2}\,\|\tilde\mu_t(x_t, x_0) - \mu_\theta(x_t, t)\|^2\right] + C$$

where $C$ is a constant. **Goal: predict $\tilde\mu_t$ given $x_t$ and $t$ alone (the network doesn't see $x_0$).**

### Choice 2 — Reparameterize $\tilde\mu_t$ in terms of $\varepsilon$ instead of $x_0$

Here's the elegant trick. From Step 3 we know:

$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)$$

Solve for $x_0$:

$$x_0 = \frac{1}{\sqrt{\bar\alpha_t}}\!\left(x_t - \sqrt{1 - \bar\alpha_t}\,\varepsilon\right)$$

Now substitute this expression for $x_0$ into the formula for $\tilde\mu_t$ (Step 5):

$$\tilde\mu_t(x_t, x_0) = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1 - \bar\alpha_t}\,x_0 + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t}\,x_t$$

After the substitution and algebraic simplification (paper's equation 10), this collapses to:

$$\tilde\mu_t(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\,\varepsilon\right)$$

**Crucial observation.** The right side is a function of $x_t$ and $\varepsilon$ — *not* $x_0$. Since the network sees $x_t$, if it can predict $\varepsilon$, it can compute this exact mean. So the paper parameterizes (equation 11):

$$\mu_\theta(x_t, t) := \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\,\varepsilon_\theta(x_t, t)\right)$$

Now the network's job is to **output a noise prediction $\varepsilon_\theta(x_t, t)$**, and we mechanically derive the mean from it.

### The per-step loss in terms of $\varepsilon$

Substitute the new $\mu_\theta$ and the new form of $\tilde\mu_t$ into the MSE-on-means above. The $x_t$ terms cancel, the $\frac{1}{\sqrt{\alpha_t}}$ and $\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}$ factors come out, and we get the paper's equation 12:

$$L_{t-1} - C = \mathbb{E}_{x_0,\,\varepsilon}\!\left[\frac{\beta_t^2}{2\sigma_t^2\,\alpha_t\,(1 - \bar\alpha_t)}\,\|\varepsilon - \varepsilon_\theta(x_t, t)\|^2\right]$$

(with $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$.)

**Read this carefully.** The loss is just **MSE between the true noise $\varepsilon$ and the model's predicted noise $\varepsilon_\theta(x_t, t)$**, scaled by a timestep-dependent weight $\frac{\beta_t^2}{2\sigma_t^2\,\alpha_t\,(1-\bar\alpha_t)}$.

### Choice 3 — Drop the weight (equation 14, the "simple" loss)

The paper observes (§3.4) that **dropping the weighting factor empirically gives better sample quality**:

$$L_\text{simple}(\theta) := \mathbb{E}_{t,\,x_0,\,\varepsilon}\!\left[\,\|\varepsilon - \varepsilon_\theta(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon,\ t)\|^2\,\right]$$

This is the famous DDPM training loss. It's a *re-weighted* variational bound (technically not the same as the original VLB), but it gives sharper samples in practice.

The paper's justification (§3.4):

> "The simplified objective discards the weighting in Eq. (12) ... [Our] diffusion process setup ... causes the simplified objective to down-weight loss terms corresponding to small $t$. These terms train the network to denoise data with very small amounts of noise, so it is beneficial to down-weight them so that the network can focus on more difficult denoising tasks at larger $t$ terms."

In other words: under the linear $\beta$ schedule, the original weighting overemphasizes easy (small-$t$) denoising. Dropping the weight is effectively a rebalancing that improves visual quality.

### The training algorithm (paper Algorithm 1)

The complete training procedure (one gradient step) becomes:

```
1. x_0 ~ q(x_0)              # sample a real image from the data
2. t ~ Uniform({1, ..., T})  # pick a random timestep
3. eps ~ N(0, I)             # sample fresh standard Gaussian noise
4. x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1 - alpha_bar_t)*eps   # forward in one shot
5. Take gradient step on ||eps - eps_theta(x_t, t)||^2
6. Repeat until converged
```

No sampling loop during training. One forward pass, one backward pass. **This is what made DDPM cheap enough to actually scale.**

### The sampling algorithm (paper Algorithm 2)

Sampling uses the learned $\varepsilon_\theta$ and the mean formula from Choice 2:

```
1. x_T ~ N(0, I)
2. for t = T, T-1, ..., 1:
3.     z ~ N(0, I) if t > 1, else z = 0
4.     x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps_theta(x_t, t)) + sigma_t * z
5. return x_0
```

Each iteration: one network forward pass for $\varepsilon_\theta$, plug into the closed-form mean formula, add Gaussian noise with the fixed $\sigma_t$, advance.

### Why predict $\varepsilon$ instead of $x_0$ or $\tilde\mu$ directly?

This is a parameterization choice with three options:
1. Predict $\tilde\mu_t$ directly.
2. Predict $x_0$ directly.
3. Predict $\varepsilon$ (the paper's choice).

The paper investigates this in §4 Experiments (Table 2) and finds **$\varepsilon$-prediction with $L_\text{simple}$ gives the best FID/IS**. Two reasons usually cited:
- $\varepsilon$ has the same scale ($\mathcal{N}(0, I)$) at *every* timestep, so the network's output target is consistently scaled. Predicting $x_0$ or $\mu$ would require the network to output very different magnitudes at different $t$.
- $\varepsilon$-prediction has a natural connection to **denoising score matching** (§3.2 of the paper), which is the score-based generative modeling family (Song & Ermon 2019). This connection — score matching ↔ noise prediction — is the unification that motivates ε-parameterization.

### The whole DDPM story so far

| What | Where |
|---|---|
| Forward process: Gaussian noise added in $T$ steps | Step 2 |
| Closed-form $q(x_t \mid x_0)$ | Step 3 |
| Reverse process: Gaussian transitions, neural network parameterization | Step 4 |
| Variational bound $L = L_T + \sum L_{t-1} + L_0$ | Step 5 |
| KL → MSE → MSE on $\varepsilon$ → $L_\text{simple}$ | Step 6 |
| Training (Algorithm 1) and sampling (Algorithm 2) | Step 6 |

We've covered everything in §2 (Background) and §3 (Diffusion models and denoising autoencoders) of the paper. What's left in the paper:
- §3.3 — the decoder term $L_0$ (the discretized Gaussian likelihood for pixel values). Small detail; we can cover it briefly in Step 7 if you want.
- §4 — Experiments and results.

---

## Step 7 — Experiments and the rest of the paper

This step covers what's left in the paper: the small technical detail in §3.3 about the pixel decoder, the experimental setup in §4, and the headline results (§4.1–§4.4).

### §3.3 — The pixel decoder ($L_0$, the tiny remaining detail)

Recall from Step 5 the loss decomposition had three pieces: $L_T$, $L_{t-1}$ for $t > 1$, and $L_0 = -\log p_\theta(x_0 \mid x_1)$. Steps 5–6 covered $L_T$ (effectively constant) and $L_{t-1}$ (the ε-MSE term). What about $L_0$?

**The issue.** Pixel intensities are discrete integers in $\{0, 1, \ldots, 255\}$, scaled to $[-1, 1]$ before training. But our model $p_\theta(x_0 \mid x_1)$ is a *Gaussian* — a continuous distribution. The variational bound term $-\log p_\theta(x_0 \mid x_1)$ would mix continuous and discrete log-likelihoods, which is theoretically awkward (and the bound wouldn't be a proper bits/dim measurement).

**The fix (paper equation 13).** Replace the Gaussian density at $x_0$ with the *integral* of the Gaussian density over the small bin around $x_0$ that the discrete pixel value represents:

$$p_\theta(x_0 \mid x_1) = \prod_{i=1}^{D} \int_{\delta_-(x_0^i)}^{\delta_+(x_0^i)} \mathcal{N}\!\bigl(x;\ \mu_\theta^i(x_1, 1),\ \sigma_1^2\bigr)\, dx$$

with $\delta_+(x) = x + 1/255$ for $x < 1$, $\delta_-(x) = x - 1/255$ for $x > -1$, and ±∞ at the edges.

The $i$ index is per-coordinate (per pixel-channel); $D$ is the total dimensionality. The integral is the *probability mass* the Gaussian assigns to the bin of width $1/255$ centered on $x_0^i$ — a discrete probability, not a density.

**Why bother?** This makes the variational bound a proper *lossless codelength* on the discrete data (measured in bits/dim), so it's directly comparable to other density-based models like PixelCNN, NICE, RealNVP, etc.

**Practical relevance.** The simplified loss $L_\text{simple}$ used for actual training **ignores this detail** — at $t = 1$, $L_\text{simple}$ just uses MSE on $\varepsilon$, treating $x_0$ as continuous. The discretized decoder is only used when evaluating bits/dim for benchmarking. (Paper §3.4: "the $t = 1$ case corresponds to $L_0$ with the integral in the discrete decoder definition (13) approximated by the Gaussian probability density function times the bin width, ignoring $\sigma_1^2$ and edge effects.")

So the L_0 decoder is a technical detail for proper likelihood evaluation; for training and sampling you can mostly ignore it.

### §4 — Experimental setup

The paper's fixed choices for all CIFAR10 experiments:

| Parameter | Value | Notes |
|---|---|---|
| $T$ | 1000 | Number of diffusion steps |
| Schedule | linear $\beta_t$ | From $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$ |
| $L_T$ | $\approx 10^{-5}$ bits/dim | Forward end-state is very close to $\mathcal{N}(0,I)$ |
| Backbone | U-Net (PixelCNN++-style) | Group normalization throughout |
| Time conditioning | sinusoidal position embedding | Same idea as Transformers |
| Attention | self-attention at 16×16 feature map | Captures long-range dependencies |
| Variance | fixed isotropic $\sigma_t^2 I$ | Either $\sigma_t^2 = \beta_t$ or $\sigma_t^2 = \tilde\beta_t$ |

Schedule comment: $\beta_t$ is *small* relative to data scaled to $[-1, 1]$. This keeps the per-step signal-to-noise ratio close to 1 (variance preservation, Step 2) while still ending at near-pure-noise after 1000 steps.

### §4.1 — Sample quality (Table 1)

The headline result: on **unconditional CIFAR10**, their model achieves:

- **Inception Score (IS): 9.46** — higher is better; measures sample quality and diversity via a pretrained Inception classifier.
- **Fréchet Inception Distance (FID): 3.17** — lower is better; compares the distribution of generated features to real features.
- **NLL bits/dim: ≤ 3.75 (3.72 test/train)** — lower is better; the variational bound on negative log-likelihood, in bits per dimension.

**What this beats.** Most other models in the table:
- *Unconditional* models: NCSN (25.32), NCSNv2 (31.75), SNGAN (21.7), SNGAN-DDLS (15.42), StyleGAN2+ADA (3.26 — slightly better FID, but a GAN).
- *Class-conditional* models: BigGAN (14.73), StyleGAN2+ADA (2.67 — best on table, but uses class labels).

So DDPM is competitive even with class-conditional GANs while being unconditional and being a likelihood-based model.

**The qualitative caveat (§4.1).** Despite competitive FID, the NLL bits/dim isn't best — other likelihood models like PixelCNN++ get lower bits/dim. The paper interprets this in §4.3: "the majority of our models' lossless codelengths are consumed to describe imperceptible image details" — most of the bits-per-dim budget goes to noise/textures the human eye doesn't care about, so the model uses likelihood capacity differently from PixelCNN.

### §4.2 — The parameterization ablation (Table 2)

This is the experimental support for the design choices in Step 6. Different ways to parameterize the reverse process:

| Objective | Trained on | IS | FID |
|---|---|---|---|
| $\tilde\mu$ prediction (baseline) | $L$ (variational bound), learned diagonal $\Sigma$ | 7.28 ± 0.10 | 23.69 |
| $\tilde\mu$ prediction | $L$, fixed isotropic $\Sigma$ | 8.06 ± 0.09 | 13.22 |
| $\tilde\mu$ prediction | $\|\tilde\mu - \tilde\mu_\theta\|^2$ (MSE on means) | unstable | unstable |
| **$\varepsilon$ prediction (ours)** | $L$, fixed isotropic $\Sigma$ | 7.67 ± 0.13 | 13.51 |
| **$\varepsilon$ prediction (ours)** | $\|\tilde\varepsilon - \varepsilon_\theta\|^2$ ($L_\text{simple}$) | **9.46 ± 0.11** | **3.17** |

**What this ablation says (reading down the rows):**

1. **Predicting $\tilde\mu$ directly works only with the variational bound, not with raw MSE on means.** The blank "unstable" rows mean training diverged or produced out-of-range scores.
2. **Learning the variance (top row) actually hurt.** It led to unstable training and worse FID than fixed variance. (This is what motivated later work like Nichol & Dhariwal 2021 to revisit and fix the learned-variance setup.)
3. **$\varepsilon$ prediction with the full VLB performs comparably to $\tilde\mu$ prediction with the full VLB.** Both around FID 13.
4. **$\varepsilon$ prediction with $L_\text{simple}$ blows everything else away.** FID drops from 13 → 3.17, IS jumps from ~7.7 → 9.46. **This is the combination that defines DDPM.**

The takeaway: predict $\varepsilon$, train with the simple unweighted MSE, fix the variance. Three choices, dramatic empirical win.

### §4.3 — Progressive coding (the rate-distortion view)

This is the paper's most conceptually interesting (and most easily skipped) section. They reinterpret DDPM as a **lossy compression scheme**.

**The idea.** Treat the diffusion chain as a sequence of progressively-decoded reconstructions. At any timestep $t$, the receiver has $x_t$ and the bits transmitted so far; they can estimate $\hat x_0$ via:

$$\hat x_0 = \bigl(x_t - \sqrt{1 - \bar\alpha_t}\,\varepsilon_\theta(x_t)\bigr) / \sqrt{\bar\alpha_t}$$

(That's just inverting the closed-form forward equation using the model's ε prediction.)

**Result.** Plotting distortion (RMSE on $\hat x_0$ vs true $x_0$) against cumulative bits transmitted gives a steep rate-distortion curve: most of the bits go to imperceptible high-frequency details late in the reverse process; the early reverse steps establish large-scale structure with very few bits.

**Quote from the paper:** "Our CIFAR10 model with the highest quality samples has a rate of 1.78 bits/dim and a distortion of 1.97 bits/dim, which amounts to a root mean squared error of 0.95 on a scale from 0 to 255. More than half of the lossless codelength describes imperceptible distortions."

This is also what motivates their connection (eq 16) between DDPM and **autoregressive coding** — in a degenerate limit, a diffusion chain with $T$ equal to the data dimensionality and masking-style noise reduces to autoregressive image generation. So DDPM is a *generalization* of autoregressive decoding with a bit-ordering induced by the noise schedule rather than raster scan.

### §4.4 — Interpolation

A neat application: smoothly interpolate between two real images by:

1. Encode both into noisy latents: $x_t, x'_t \sim q(x_t \mid x_0), q(x_t \mid x'_0)$ (using the closed-form forward).
2. Linearly interpolate in latent space: $\bar x_t = (1-\lambda)\,x_t + \lambda\,x'_t$.
3. Decode by running the reverse process from $\bar x_t$ to $\hat x_0$.

Result (Figure 8 in the paper): smooth morphs of CelebA-HQ faces — pose, skin tone, hair, expression all vary continuously. Larger $t$ gives coarser interpolations (more semantic blending); smaller $t$ stays closer to the original images.

This is a side effect of the smooth, hierarchical structure of the diffusion latent space — different timesteps encode different scales of features.

### §5 — Related work (briefly)

The key positioning point from §5 and the paper as a whole:

> *"Our ε-prediction reverse process parameterization establishes a connection between diffusion models and denoising score matching over multiple noise levels with annealed Langevin dynamics for sampling. Diffusion models, however, admit straightforward log likelihood evaluation, and the training procedure explicitly trains the Langevin dynamics sampler using variational inference."*

In plain English: by choosing ε-prediction, the paper *unifies* two previously-separate threads — likelihood-based diffusion modeling (Sohl-Dickstein 2015) and score-based generative modeling with Langevin sampling (Song & Ermon 2019). The model trained with $L_\text{simple}$ is *simultaneously* a likelihood model (because it minimizes a weighted VLB) and a score-matching model (because predicting ε is equivalent to predicting the data score up to a constant).

This is the main *theoretical* contribution of the paper, alongside the empirical SOTA results.

### §6 — Conclusion

Three sentences from the conclusion that summarize the contribution:

> *"We have presented high quality image samples using diffusion models, and we have found connections among diffusion models and variational inference for training Markov chains, denoising score matching and annealed Langevin dynamics (and energy-based models by extension), autoregressive models, and progressive lossy compression."*

So the paper does three things:
1. **Empirical:** Shows diffusion models can produce SOTA-quality image samples (FID 3.17 on CIFAR10 unconditional, comparable to ProgressiveGAN on CelebA-HQ 256×256).
2. **Conceptual unification:** Shows that the ε-prediction parameterization links diffusion, score matching, Langevin dynamics, energy-based models, autoregressive coding, and progressive compression.
3. **Practical recipe:** The simple training algorithm (Algorithm 1) and sampling algorithm (Algorithm 2) make diffusion usable at scale.

### What the paper does *not* address (subsequent work fills these in)

For context, here's what's missing from this 2020 paper that later papers added:

- **Faster sampling.** DDPM needs 1000 NN evaluations to sample one image. Later work (DDIM, DPM-Solver, Consistency Models, Distillation, SDXL Turbo) brings this down to 1–50 evaluations.
- **Learned variance.** Fixed $\sigma_t^2$ leaves likelihood quality on the table. Nichol & Dhariwal 2021 (Improved DDPM) addresses this with the $v$-head we used in the Day 2 notebook.
- **Better noise schedules.** Linear $\beta$ is suboptimal at high resolutions. Cosine schedule (Nichol & Dhariwal 2021) and Karras EDM schedule (2022) improve sample quality.
- **Class conditioning / text conditioning.** DDPM is unconditional. Classifier guidance (Dhariwal & Nichol 2021), classifier-free guidance (Ho & Salimans 2022), and cross-attention text conditioning (Rombach et al. 2022 — Stable Diffusion) extend this.
- **Latent diffusion.** Operating in pixel space at $256\times256$ is expensive. Stable Diffusion runs diffusion in a learned VAE latent space, $64\times64\times4$, dramatically cheaper.

That's the full lineage you've been seeing in the curriculum. DDPM is the foundation; everything else is one of these improvements stacked on top.

---

See the companion notebook **`day2/day2_paper_math_to_code.ipynb`** for code-level translations of every equation in this tutorial: numerical verifications of the closed-form forward process, KL divergence between Gaussians, the $\tilde\mu_t$ posterior, ε-parameterization equivalence, and the training/sampling algorithms.
