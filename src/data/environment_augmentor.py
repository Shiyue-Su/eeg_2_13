import math
from typing import List

import torch


class EnvironmentAugmentor:
    """Deterministic environment perturbations for ICER pretraining."""

    def __init__(self, env_cfg, fs: int, channels: List[str]):
        self.enable = bool(getattr(env_cfg, "enable", False))
        self.seed = int(getattr(env_cfg, "seed", 0))
        self.fs = int(fs)
        self.channels = list(channels)

        self.band_cfg = getattr(env_cfg, "band", None)
        self.channel_cfg = getattr(env_cfg, "channel", None)
        self.noise_cfg = getattr(env_cfg, "noise", None)

        env_types = list(getattr(env_cfg, "types", ["band", "channel", "noise"]))
        requested_num_envs = max(1, int(getattr(env_cfg, "num_envs", 1)))
        self.env_ops = self._build_env_ops(env_types, requested_num_envs)
        self.num_envs = len(self.env_ops)

        self.dropout_ratio = float(getattr(self.channel_cfg, "dropout_ratio", 0.2))
        self.noise_small = float(getattr(self.noise_cfg, "sigma_small", 0.01))
        self.noise_large = float(getattr(self.noise_cfg, "sigma_large", 0.05))
        self.high_cut_hz = float(getattr(self.band_cfg, "high_cut_hz", 30.0))
        self.alpha_band = tuple(getattr(self.band_cfg, "alpha_band", [8.0, 13.0]))
        self.beta_band = tuple(getattr(self.band_cfg, "beta_band", [13.0, 30.0]))

    def _build_env_ops(self, env_types, requested_num_envs):
        ops = ["identity"]
        if "band" in env_types:
            ops.extend(["band_suppress_high", "band_suppress_alpha_or_beta"])
        if "channel" in env_types:
            ops.extend(["channel_dropout", "channel_region_mask"])
        if "noise" in env_types:
            ops.extend(["noise_small", "noise_large"])

        if len(ops) >= requested_num_envs:
            return ops[:requested_num_envs]

        base_ops = list(ops)
        while len(ops) < requested_num_envs:
            ops.append(base_ops[len(ops) % len(base_ops)])
        return ops

    def _make_generator(self, sample_id: int, env_id: int, view_id: int):
        mix = (
            (self.seed + 1) * 1000003
            + (sample_id + 17) * 9176
            + (env_id + 31) * 3571
            + (view_id + 13) * 1871
        )
        seed = int(mix % (2**31 - 1))
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        return g

    def apply_batch(self, x, env_ids, sample_ids=None, view_id=0):
        if not self.enable or self.num_envs <= 1:
            return x

        out = x.clone()
        b = out.shape[0]
        if sample_ids is None:
            sample_ids = torch.arange(b, dtype=torch.long)

        for i in range(b):
            env_id = int(env_ids[i].item()) % self.num_envs
            sample_id = int(sample_ids[i].item())
            out[i] = self.apply_single(out[i], env_id, sample_id, view_id)
        return out

    def apply_single(self, x, env_id: int, sample_id: int, view_id: int = 0):
        if not self.enable:
            return x

        op = self.env_ops[env_id % self.num_envs]
        g = self._make_generator(sample_id, env_id, view_id)

        if op == "identity":
            return x
        if op == "band_suppress_high":
            return self._suppress_high_freq(x)
        if op == "band_suppress_alpha_or_beta":
            # Deterministic branch selection.
            suppress_alpha = ((sample_id + view_id) % 2 == 0)
            band = self.alpha_band if suppress_alpha else self.beta_band
            return self._suppress_band(x, band[0], band[1])
        if op == "channel_dropout":
            return self._channel_dropout(x, g)
        if op == "channel_region_mask":
            return self._channel_region_mask(x, sample_id, view_id)
        if op == "noise_small":
            return self._add_noise(x, self.noise_small, g)
        if op == "noise_large":
            return self._add_noise(x, self.noise_large, g)
        return x

    def _suppress_high_freq(self, x):
        return self._apply_fft_mask(x, keep_low_hz=0.0, keep_high_hz=self.high_cut_hz)

    def _suppress_band(self, x, band_low_hz: float, band_high_hz: float):
        b, c, t = x.shape
        spectrum = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(t, d=1.0 / float(self.fs)).to(x.device)
        stop_mask = (freqs >= band_low_hz) & (freqs <= band_high_hz)
        spectrum[..., stop_mask] = 0
        return torch.fft.irfft(spectrum, n=t, dim=-1).to(x.dtype)

    def _apply_fft_mask(self, x, keep_low_hz: float, keep_high_hz: float):
        b, c, t = x.shape
        spectrum = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(t, d=1.0 / float(self.fs)).to(x.device)
        keep_mask = (freqs >= keep_low_hz) & (freqs <= keep_high_hz)
        spectrum[..., ~keep_mask] = 0
        return torch.fft.irfft(spectrum, n=t, dim=-1).to(x.dtype)

    def _channel_dropout(self, x, generator):
        _, c, _ = x.shape
        n_drop = max(1, int(math.floor(c * self.dropout_ratio)))
        perm = torch.randperm(c, generator=generator)
        drop_idx = perm[:n_drop]
        out = x.clone()
        out[:, drop_idx, :] = 0
        return out

    def _channel_region_mask(self, x, sample_id: int, view_id: int):
        channel_names = [str(ch).upper() for ch in self.channels]
        frontal_idx = [
            i for i, name in enumerate(channel_names)
            if name.startswith(("FP", "AF", "F", "FC"))
        ]
        temporal_idx = [
            i for i, name in enumerate(channel_names)
            if name.startswith(("FT", "T", "TP"))
        ]

        use_frontal = ((sample_id + view_id) % 2 == 0)
        keep_idx = frontal_idx if use_frontal else temporal_idx
        if len(keep_idx) == 0:
            return x

        out = torch.zeros_like(x)
        out[:, keep_idx, :] = x[:, keep_idx, :]
        return out

    def _add_noise(self, x, sigma: float, generator):
        noise = torch.randn(x.shape, generator=generator, dtype=x.dtype)
        return x + sigma * noise
