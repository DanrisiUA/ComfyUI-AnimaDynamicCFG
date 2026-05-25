import torch
import torch.fft as fft
import math


def _get_denoising_progress(sigma, model_options):
    sample_sigmas = model_options.get("transformer_options", {}).get("sample_sigmas", None)
    if sample_sigmas is not None and len(sample_sigmas) > 1:
        sigma_max = float(sample_sigmas[0])
        sigma_min = float(sample_sigmas[-1])
    else:
        sigma_max = 1.0
        sigma_min = 0.0

    s = float(sigma.reshape(-1)[0])
    if sigma_max == sigma_min:
        return 1.0
    progress = 1.0 - (s - sigma_min) / (sigma_max - sigma_min)
    return max(0.0, min(1.0, progress))


def _interpolate_cfg(progress, cfg_start, cfg_end, schedule, transition_start, transition_end):
    if progress <= transition_start:
        return cfg_start
    if progress >= transition_end:
        return cfg_end

    span = transition_end - transition_start
    if span <= 0:
        return cfg_end
    t = (progress - transition_start) / span

    if schedule == "linear":
        pass
    elif schedule == "cosine":
        t = (1.0 - math.cos(t * math.pi)) / 2.0
    elif schedule == "sine":
        t = math.sin(t * math.pi / 2.0)
    elif schedule == "ease_in":
        t = t * t
    elif schedule == "ease_out":
        t = 1.0 - (1.0 - t) ** 2
    elif schedule == "ease_in_out":
        t = 3 * t * t - 2 * t * t * t

    return cfg_start + (cfg_end - cfg_start) * t


class AnimaCFGSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "cfg_start": ("FLOAT", {"default": 5.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "cfg_end": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "schedule": (["linear", "cosine", "sine", "ease_in", "ease_out", "ease_in_out"],),
                "transition_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "transition_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Anima/sampling"
    DESCRIPTION = (
        "Schedules CFG dynamically across denoising steps. "
        "Use higher cfg_start for structure, lower cfg_end for realism. "
        "Recommended for Anima realism finetunes: cfg_start=5.5, cfg_end=3.5, cosine."
    )

    def patch(self, model, cfg_start, cfg_end, schedule, transition_start, transition_end):
        m = model.clone()

        def scheduled_cfg_function(args):
            cond = args["cond"]
            uncond = args["uncond"]
            sigma = args["sigma"]
            model_opts = args["model_options"]

            progress = _get_denoising_progress(sigma, model_opts)
            cfg = _interpolate_cfg(progress, cfg_start, cfg_end, schedule,
                                   transition_start, transition_end)

            return uncond + cfg * (cond - uncond)

        m.set_model_sampler_cfg_function(scheduled_cfg_function)
        return (m,)


class AnimaAdaptiveCFG:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "cfg_base": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "min_cfg": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "max_cfg": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "adaptation_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "target_divergence": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01, "round": 0.001}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Anima/sampling"
    DESCRIPTION = (
        "Adaptive CFG that monitors the divergence between conditional and unconditional "
        "predictions. When divergence is high (model pushes hard toward anime style), "
        "CFG is reduced to let realism through. When divergence is low, CFG is increased "
        "to maintain structure."
    )

    def patch(self, model, cfg_base, min_cfg, max_cfg, adaptation_strength, target_divergence):
        m = model.clone()

        def adaptive_cfg_function(args):
            cond = args["cond"]
            uncond = args["uncond"]

            diff = cond - uncond
            actual_div = diff.flatten(1).norm(dim=1).mean().item()

            if actual_div < 1e-8:
                effective_cfg = cfg_base
            else:
                ratio = target_divergence / actual_div
                effective_cfg = cfg_base * (ratio ** adaptation_strength)

            effective_cfg = max(min_cfg, min(max_cfg, effective_cfg))
            return uncond + effective_cfg * (cond - uncond)

        m.set_model_sampler_cfg_function(adaptive_cfg_function)
        return (m,)


class AnimaCFGRescale:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "rescale": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Anima/sampling"
    DESCRIPTION = (
        "Post-CFG rescaling that normalizes the CFG output amplitude to match the "
        "conditional prediction's std. Prevents over-saturation at higher CFG values. "
        "Based on 'Common Diffusion Noise Schedules' (arXiv:2305.08891)."
    )

    def patch(self, model, rescale):
        m = model.clone()

        def rescale_post_cfg(args):
            denoised = args["denoised"]
            cond_denoised = args["cond_denoised"]

            std_cond = torch.std(cond_denoised, dim=tuple(range(1, cond_denoised.ndim)), keepdim=True)
            std_cfg = torch.std(denoised, dim=tuple(range(1, denoised.ndim)), keepdim=True)

            std_cfg = torch.clamp(std_cfg, min=1e-8)

            rescaled = denoised * (std_cond / std_cfg)
            result = rescale * rescaled + (1.0 - rescale) * denoised
            return result

        m.set_model_sampler_post_cfg_function(rescale_post_cfg, disable_cfg1_optimization=True)
        return (m,)


class AnimaCFGScheduleAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "cfg_start": ("FLOAT", {"default": 5.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "cfg_end": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "schedule": (["linear", "cosine", "sine", "ease_in", "ease_out", "ease_in_out"],),
                "transition_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "transition_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "adaptive_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01,
                                                 "tooltip": "0 = pure schedule, 1 = heavy adaptive damping on top"}),
                "target_divergence": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01, "round": 0.001}),
                "rescale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "Post-CFG std rescaling (0 = off, 0.7 = recommended)"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Anima/sampling"
    DESCRIPTION = (
        "All-in-one node: scheduled CFG curve + optional adaptive divergence damping + "
        "optional post-CFG rescaling. For Anima realism finetunes, try: "
        "cfg_start=5.5, cfg_end=3.5, cosine, adaptive_strength=0.3, rescale=0.7."
    )

    def patch(self, model, cfg_start, cfg_end, schedule, transition_start, transition_end,
              adaptive_strength, target_divergence, rescale):
        m = model.clone()

        def advanced_cfg_function(args):
            cond = args["cond"]
            uncond = args["uncond"]
            sigma = args["sigma"]
            model_opts = args["model_options"]

            progress = _get_denoising_progress(sigma, model_opts)
            scheduled_cfg = _interpolate_cfg(progress, cfg_start, cfg_end, schedule,
                                             transition_start, transition_end)

            if adaptive_strength > 0.0:
                diff = cond - uncond
                actual_div = diff.flatten(1).norm(dim=1).mean().item()
                if actual_div > 1e-8:
                    ratio = target_divergence / actual_div
                    scheduled_cfg = scheduled_cfg * (ratio ** adaptive_strength)
                scheduled_cfg = max(1.0, scheduled_cfg)

            return uncond + scheduled_cfg * (cond - uncond)

        m.set_model_sampler_cfg_function(advanced_cfg_function)

        if rescale > 0.0:
            def rescale_post_cfg(args):
                denoised = args["denoised"]
                cond_denoised = args["cond_denoised"]
                std_cond = torch.std(cond_denoised, dim=tuple(range(1, cond_denoised.ndim)), keepdim=True)
                std_cfg = torch.std(denoised, dim=tuple(range(1, denoised.ndim)), keepdim=True)
                std_cfg = torch.clamp(std_cfg, min=1e-8)
                rescaled = denoised * (std_cond / std_cfg)
                return rescale * rescaled + (1.0 - rescale) * denoised

            m.set_model_sampler_post_cfg_function(rescale_post_cfg, disable_cfg1_optimization=True)

        return (m,)


def _fourier_filter(x, scale_low, scale_high, freq_cutoff):
    dtype = x.dtype
    x = x.to(torch.float32)

    spatial_dims = tuple(range(2, x.ndim))
    x_freq = fft.fftn(x, dim=spatial_dims)
    x_freq = fft.fftshift(x_freq, dim=spatial_dims)

    mask = torch.ones(x_freq.shape, device=x.device) * scale_high
    m = mask
    for d in spatial_dims:
        cc = x_freq.shape[d] // 2
        f_c = min(freq_cutoff, cc)
        m = m.narrow(d, cc - f_c, f_c * 2)
    m[:] = scale_low

    x_freq = x_freq * mask
    x_freq = fft.ifftshift(x_freq, dim=spatial_dims)
    x_filtered = fft.ifftn(x_freq, dim=spatial_dims).real

    return x_filtered.to(dtype)


class AnimaFrequencyBoost:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "high_freq_scale": ("FLOAT", {"default": 1.3, "min": 0.5, "max": 3.0, "step": 0.05}),
                "low_freq_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.05}),
                "freq_cutoff": ("INT", {"default": 20, "min": 1, "max": 200, "step": 1}),
                "start_percent": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Anima/realism"
    DESCRIPTION = (
        "Post-CFG frequency filter that boosts high-frequency detail (skin textures, "
        "hair, fabric) in denoised output. Stronger in later steps where details form. "
        "Addresses the flat shading typical of anime models."
    )

    def patch(self, model, high_freq_scale, low_freq_scale, freq_cutoff, start_percent, end_percent):
        m = model.clone()

        def freq_boost_post_cfg(args):
            denoised = args["denoised"]
            sigma = args["sigma"]
            model_opts = args["model_options"]

            progress = _get_denoising_progress(sigma, model_opts)
            if progress < start_percent or progress > end_percent:
                return denoised

            blend = (progress - start_percent) / max(end_percent - start_percent, 1e-8)
            blend = min(1.0, blend)

            current_high = 1.0 + (high_freq_scale - 1.0) * blend
            current_low = 1.0 + (low_freq_scale - 1.0) * blend

            return _fourier_filter(denoised, current_low, current_high, freq_cutoff)

        m.set_model_sampler_post_cfg_function(freq_boost_post_cfg, disable_cfg1_optimization=True)
        return (m,)


class AnimaLatentMeanCorrection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Anima/realism"
    DESCRIPTION = (
        "Corrects per-channel mean drift caused by CFG. High CFG shifts the latent "
        "mean away from zero, causing over-saturation and unnatural colors. "
        "This subtracts the excess mean to restore natural color balance."
    )

    def patch(self, model, strength, start_percent, end_percent):
        m = model.clone()

        def mean_correction_post_cfg(args):
            denoised = args["denoised"]
            sigma = args["sigma"]
            model_opts = args["model_options"]

            progress = _get_denoising_progress(sigma, model_opts)
            if progress < start_percent or progress > end_percent:
                return denoised

            spatial_dims = tuple(range(2, denoised.ndim))
            channel_mean = denoised.mean(dim=spatial_dims, keepdim=True)
            return denoised - channel_mean * strength

        m.set_model_sampler_post_cfg_function(mean_correction_post_cfg, disable_cfg1_optimization=True)
        return (m,)


class AnimaNoiseSculpt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "strength": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.2, "step": 0.005}),
                "start_percent": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Anima/realism"
    DESCRIPTION = (
        "Injects controlled micro-noise into late denoising steps to create natural "
        "skin texture, fabric weave, and surface grain that anime models tend to "
        "over-smooth. Use very low strength (0.01-0.03)."
    )

    def patch(self, model, strength, start_percent, end_percent):
        m = model.clone()

        def noise_sculpt_post_cfg(args):
            denoised = args["denoised"]
            sigma = args["sigma"]
            model_opts = args["model_options"]

            progress = _get_denoising_progress(sigma, model_opts)
            if progress < start_percent or progress > end_percent:
                return denoised

            span = end_percent - start_percent
            if span <= 0:
                t = 1.0
            else:
                t = (progress - start_percent) / span

            fade = math.sin(t * math.pi)
            noise = torch.randn_like(denoised) * strength * fade
            return denoised + noise

        m.set_model_sampler_post_cfg_function(noise_sculpt_post_cfg, disable_cfg1_optimization=True)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "AnimaCFGSchedule": AnimaCFGSchedule,
    "AnimaAdaptiveCFG": AnimaAdaptiveCFG,
    "AnimaCFGRescale": AnimaCFGRescale,
    "AnimaCFGScheduleAdvanced": AnimaCFGScheduleAdvanced,
    "AnimaFrequencyBoost": AnimaFrequencyBoost,
    "AnimaLatentMeanCorrection": AnimaLatentMeanCorrection,
    "AnimaNoiseSculpt": AnimaNoiseSculpt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaCFGSchedule": "Anima CFG Schedule",
    "AnimaAdaptiveCFG": "Anima Adaptive CFG",
    "AnimaCFGRescale": "Anima CFG Rescale",
    "AnimaCFGScheduleAdvanced": "Anima CFG Schedule (Advanced)",
    "AnimaFrequencyBoost": "Anima Frequency Boost",
    "AnimaLatentMeanCorrection": "Anima Latent Mean Correction",
    "AnimaNoiseSculpt": "Anima Noise Sculpt",
}
