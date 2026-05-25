# ComfyUI-AnimaDynamicCFG

Custom ComfyUI nodes for dynamic CFG control and realism enhancement. Originally designed for [Anima](https://huggingface.co/circlestone-labs/Anima) realism finetunes, but works with any diffusion model.

## Problem

Static CFG forces a compromise: low values produce realistic textures but structural artifacts, while high values improve coherence but amplify the base model's stylistic bias (e.g. anime look). These nodes let you have both.

## Nodes

### CFG Control (`Anima/sampling`)

**Anima CFG Schedule** — Interpolates CFG from `cfg_start` to `cfg_end` across denoising steps using a chosen curve (linear, cosine, sine, ease_in, ease_out, ease_in_out). Use higher CFG at the start for structure, lower at the end for realism.

**Anima Adaptive CFG** — Monitors the norm of `cond - uncond` divergence at each step. When the model pushes too hard in one direction, CFG is automatically reduced. When divergence is low, CFG increases to maintain structure.

**Anima CFG Rescale** — Post-CFG std normalization. Rescales the CFG output so its standard deviation matches the conditional prediction's std, preventing over-saturation at higher CFG values. Based on [arXiv:2305.08891](https://arxiv.org/abs/2305.08891).

**Anima CFG Schedule (Advanced)** — All-in-one node combining scheduled CFG curve + adaptive divergence damping + post-CFG rescaling.

### Realism Enhancement (`Anima/realism`)

**Anima Frequency Boost** — Post-CFG FFT filter that selectively boosts high-frequency components (skin textures, hair detail, fabric weave) while preserving low-frequency structure. Strength ramps up in later steps where fine details form.

**Anima Latent Mean Correction** — Subtracts per-channel spatial mean from the denoised prediction. CFG causes latent mean drift away from zero, leading to over-saturated and unnatural colors. This correction restores natural color balance.

**Anima Noise Sculpt** — Injects controlled micro-noise during late denoising steps with a sinusoidal fade envelope. Creates the natural surface grain (skin pores, fabric texture) that models with anime priors tend to over-smooth.

## Installation

Clone into your ComfyUI `custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/f-tuners/ComfyUI-AnimaDynamicCFG.git
```

Restart ComfyUI. Nodes appear under `Anima/sampling` and `Anima/realism`.

No additional dependencies required — uses only `torch` and `math`.

## Usage

All nodes are model patches. Connect them between your model loader and sampler:

```
Model Loader → Anima CFG Schedule (Advanced) → Anima Frequency Boost → KSampler
```

Multiple nodes can be chained. They use different hook points and don't conflict:
- CFG Schedule/Adaptive use `sampler_cfg_function` (replaces CFG formula)
- FrequencyBoost, MeanCorrection, NoiseSculpt, Rescale use `post_cfg_function` (post-processing chain)

Compatible with KSampler, SamplerCustomAdvanced, ClownsharkKSampler, SharkSampler, and other custom samplers that go through ComfyUI's standard sampling pipeline.

## Recommended Settings

For anime-to-realism finetunes:

| Parameter | Value |
|-----------|-------|
| cfg_start | 5.5–7.0 |
| cfg_end | 3.0–4.0 |
| schedule | cosine |
| rescale | 0.5–0.7 |
| high_freq_scale | 1.2–1.4 |
| mean correction strength | 0.3–0.5 |
| noise sculpt strength | 0.01–0.03 |

## License

MIT
