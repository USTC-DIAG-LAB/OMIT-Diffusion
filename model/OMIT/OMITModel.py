import math
from numbers import Integral, Real

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

from model.OMIT.OMITCore import OMITCore
from model.utils import default, extract


class OMITModel(OMITCore):
    """OMIT-Diffusion with transition-interval conditioning."""

    def __init__(self, model_config):
        # Reject unsupported modes before allocating the denoiser.
        shortcut_cfg = model_config.OMIT.params.shortcut
        if shortcut_cfg.enabled is not True:
            raise NotImplementedError(
                "OMITModel supports shortcut.enabled=True only; "
                "a base-only model needs a separate training and sampling contract"
            )
        self.shortcut_enabled = shortcut_cfg.enabled
        self.shortcut_denoise_timesteps = shortcut_cfg.denoise_timesteps
        if (
            isinstance(self.shortcut_denoise_timesteps, bool)
            or not isinstance(self.shortcut_denoise_timesteps, Integral)
        ):
            raise TypeError("shortcut.denoise_timesteps must be an integer")
        if self.shortcut_denoise_timesteps < 2 or (
            self.shortcut_denoise_timesteps & (self.shortcut_denoise_timesteps - 1)
        ):
            raise ValueError("shortcut.denoise_timesteps must be a power of two and at least 2")
        fraction = shortcut_cfg.shortcut_fraction
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, Real)
            or not math.isfinite(fraction)
            or not 0.0 < fraction < 1.0
        ):
            raise ValueError("shortcut.shortcut_fraction must be finite and in (0, 1)")
        self.shortcut_fraction = float(fraction)
        self.shortcut_lambda_base = shortcut_cfg.lambda_base
        self.shortcut_lambda = shortcut_cfg.lambda_shortcut

        num_timesteps = model_config.OMIT.params.num_timesteps
        if (
            isinstance(num_timesteps, bool)
            or not isinstance(num_timesteps, Integral)
            or num_timesteps < 2
        ):
            raise ValueError("num_timesteps must be an integer of at least 2")
        if self.shortcut_denoise_timesteps > num_timesteps:
            raise ValueError("shortcut.denoise_timesteps cannot exceed num_timesteps")
        self.shortcut_max_dt_base = int(math.log2(self.shortcut_denoise_timesteps))

        tau = self._build_reverse_tau(num_timesteps, self.shortcut_denoise_timesteps)
        if any(start <= end for start, end in zip(tau, tau[1:])):
            raise ValueError("shortcut time grid must be strictly decreasing")
        super().__init__(model_config)
        self.register_buffer("shortcut_tau", torch.tensor(tau, dtype=torch.long), persistent=False)

    def apply(self, weight_init):
        self.denoise_fn.apply(weight_init)
        self._zero_dt_embed_if_needed()
        return self

    def _zero_dt_embed_if_needed(self):
        if not getattr(self.denoise_fn, "use_dt_embed", False):
            return
        if not getattr(self.denoise_fn, "dt_embed_zero_init", True):
            return
        if not hasattr(self.denoise_fn, "dt_embed"):
            return
        last = self.denoise_fn.dt_embed[-1]
        if hasattr(last, "weight"):
            torch.nn.init.zeros_(last.weight)
        if hasattr(last, "bias") and last.bias is not None:
            torch.nn.init.zeros_(last.bias)

    @staticmethod
    def _build_reverse_tau(num_timesteps, denoise_timesteps):
        tau = np.rint(np.linspace(num_timesteps, 0, denoise_timesteps + 1)).astype(np.int64)
        tau[0] = num_timesteps
        tau[-1] = 0
        for i in range(1, len(tau) - 1):
            tau[i] = min(num_timesteps - 1, max(1, tau[i]))
        return tau.tolist()

    def _model_t(self, t):
        return t.clamp(0, self.num_timesteps - 1).long()

    def _schedule_values(self, t, x_shape):
        t_index = self._model_t(t)
        m = extract(self.m_t, t_index, x_shape)
        var = extract(self.variance_t, t_index, x_shape)

        endpoint_t = (t == self.num_timesteps).reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))
        endpoint_0 = (t == 0).reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))
        m = torch.where(endpoint_t, torch.ones_like(m), m)
        var = torch.where(endpoint_t, torch.zeros_like(var), var)
        m = torch.where(endpoint_0, torch.zeros_like(m), m)
        var = torch.where(endpoint_0, torch.zeros_like(var), var)
        return m, var

    def _skip_update_coefficients(self, x_shape, t, n, eta=0.0):
        """Coefficients for the residual-preserving transition in Eq. (14).

        The t=1 boundary is handled explicitly in ``omit_skip_update`` using
        Eq. (28), so these coefficients are only consumed for finite-variance
        interior states.
        """
        m_t, var_t = self._schedule_values(t, x_shape)
        m_n, var_n = self._schedule_values(n, x_shape)
        var_t = var_t.clamp_min(1e-12)

        sigma2_full = (
            var_t - var_n * (1.0 - m_t).pow(2) / (1.0 - m_n).clamp_min(1e-6).pow(2)
        ) * var_n / var_t
        sigma2_eta = (float(eta) ** 2) * sigma2_full.clamp_min(0.0)
        residual_scale = torch.sqrt(((var_n - sigma2_eta).clamp_min(0.0)) / var_t)
        return m_t, var_t, m_n, var_n, sigma2_eta, residual_scale

    def _loss(self, pred, target):
        if self.loss_type == "l1":
            return (pred - target).abs().mean()
        if self.loss_type == "l2":
            return F.mse_loss(pred, target)
        raise NotImplementedError()

    def q_sample_shortcut(self, x0, y, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x0))
        m_t, var_t = self._schedule_values(t, x0.shape)
        sigma_t = torch.sqrt(var_t.clamp_min(0.0))
        x_t = (1.0 - m_t) * x0 + m_t * y + sigma_t * noise

        if self.objective == "grad":
            objective = x_t - x0
        elif self.objective == "ysubx":
            objective = y - x0
        else:
            raise NotImplementedError()
        return x_t, objective

    def objective_from_x0(self, x_t, y, t, x0_recon):
        if self.objective == "grad":
            return x_t - x0_recon
        if self.objective == "ysubx":
            return y - x0_recon
        raise NotImplementedError()

    def predict_x0_from_objective_shortcut(self, x_t, y, t, objective_recon):
        if self.objective == "grad":
            return x_t - objective_recon
        if self.objective == "ysubx":
            return y - objective_recon
        raise NotImplementedError()

    def denoise_objective(self, x_t, y, t, context=None, dt_base=None):
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context
        if dt_base is None:
            dt_base = torch.full(
                (x_t.shape[0],),
                self.shortcut_max_dt_base,
                device=x_t.device,
                dtype=torch.long,
            )
        return self.denoise_fn(
            x_t,
            timesteps=self._model_t(t),
            context=context,
            dt_base=dt_base,
        )

    def omit_skip_update(self, x_t, y, t, n, objective, eta=0.0, clip_x0=False):
        x0_recon = self.predict_x0_from_objective_shortcut(x_t, y, t, objective)
        if clip_x0:
            x0_recon = x0_recon.clamp(-1.0, 1.0)
        endpoint_t = (t == self.num_timesteps).reshape(t.shape[0], *((1,) * (len(x_t.shape) - 1)))
        endpoint_0 = (n == 0).reshape(n.shape[0], *((1,) * (len(x_t.shape) - 1)))

        m_t, _, m_n, _, sigma2_eta, a = self._skip_update_coefficients(x_t.shape, t, n, eta=eta)
        x_n = (1.0 - m_n) * x0_recon + m_n * y
        interior_x_n = x_n + a * (x_t - (1.0 - m_t) * x0_recon - m_t * y)
        if eta != 0.0:
            interior_x_n = interior_x_n + torch.sqrt(sigma2_eta.clamp_min(0.0)) * torch.randn_like(x_t)
        # Eq. (28): at t=1 the residual variance is zero, so use the bridge
        # mean rather than substituting coefficients from a neighboring time.
        x_n = torch.where(endpoint_t, x_n, interior_x_n)
        return torch.where(endpoint_0, x0_recon, x_n)

    @torch.no_grad()
    def invert_skip_objective(self, x_t, y, t, n, x_n):
        endpoint_t = (t == self.num_timesteps).reshape(t.shape[0], *((1,) * (len(x_t.shape) - 1)))
        endpoint_0 = (n == 0).reshape(n.shape[0], *((1,) * (len(x_t.shape) - 1)))
        m_t, _, m_n, _, _, beta = self._skip_update_coefficients(x_t.shape, t, n, eta=0.0)
        alpha = (1.0 - m_n) - beta * (1.0 - m_t)
        gamma = m_n - beta * m_t
        alpha_floor = torch.where(alpha >= 0.0, torch.full_like(alpha, 1e-6), torch.full_like(alpha, -1e-6))
        alpha_safe = torch.where(alpha.abs() < 1e-6, alpha_floor, alpha)
        x0_from_inverse = (x_n - beta * x_t - gamma * y) / alpha_safe
        # Eq. (29): invert the deterministic boundary mean at t=1.
        x0_from_boundary = (x_n - m_n * y) / (1.0 - m_n).clamp_min(1e-6)
        x0_from_inverse = torch.where(endpoint_t, x0_from_boundary, x0_from_inverse)
        x0_recon = torch.where(endpoint_0, x_n, x0_from_inverse)
        return self.objective_from_x0(x_t, y, t, x0_recon)

    def make_anchor_targets(self, x0, y):
        b = x0.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=x0.device).long()
        x_t, objective = self.q_sample_shortcut(x0, y, t)
        dt_base = torch.full(
            (b,),
            self.shortcut_max_dt_base,
            device=x0.device,
            dtype=torch.long,
        )
        return {
            "x_t": x_t,
            "t": t,
            "dt_base": dt_base,
            "objective": objective,
        }

    @torch.no_grad()
    def make_shortcut_targets(self, x0, y, context=None):
        """Construct Eq. (22) from two length-d bridge transitions.

        A large interval 2d is sampled from {2/N, 4/N, ..., 1}.  Its
        supervision is obtained only through the two smaller transitions and
        endpoint inversion; paired x0 is used solely to sample the bridge
        state x_t, as specified by Algorithm 1.
        """
        b = x0.shape[0]
        if b == 0:
            return None

        device = x0.device
        # large_dt_code=0 is a full one-step transition; the largest code here
        # is N/64 steps for N=128, whose two half transitions are length 1/N.
        large_dt_code = torch.randint(
            0,
            self.shortcut_max_dt_base,
            (b,),
            device=device,
        ).long()
        starts, mids, ends = [], [], []
        for code in large_dt_code.tolist():
            large_grid_steps = self.shortcut_denoise_timesteps // (2 ** int(code))
            half_grid_steps = large_grid_steps // 2
            max_start = self.shortcut_denoise_timesteps - large_grid_steps
            start_index = int(torch.randint(0, max_start + 1, (1,), device=device).item())
            starts.append(int(self.shortcut_tau[start_index].item()))
            mids.append(int(self.shortcut_tau[start_index + half_grid_steps].item()))
            ends.append(int(self.shortcut_tau[start_index + large_grid_steps].item()))

        t_start = torch.tensor(starts, device=device, dtype=torch.long)
        t_mid = torch.tensor(mids, device=device, dtype=torch.long)
        t_end = torch.tensor(ends, device=device, dtype=torch.long)
        x_start, _ = self.q_sample_shortcut(x0, y, t_start)

        small_dt_code = large_dt_code + 1
        objective_1 = self.denoise_objective(
            x_start, y, t_start, context, small_dt_code
        )
        x_mid = self.omit_skip_update(
            x_start, y, t_start, t_mid, objective_1, eta=0.0
        )
        objective_2 = self.denoise_objective(
            x_mid, y, t_mid, context, small_dt_code
        )
        x_two = self.omit_skip_update(
            x_mid, y, t_mid, t_end, objective_2, eta=0.0
        )
        target_objective = self.invert_skip_objective(
            x_start, y, t_start, t_end, x_two
        )

        return {
            "x_t": x_start,
            "t": t_start,
            "dt_base": large_dt_code,
            "objective": target_objective.detach(),
        }

    def _sample_shortcut_loop(
        self,
        y,
        context=None,
        num_steps=128,
        clip_denoised=True,
        eta=0.0,
        progress=True,
    ):
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context

        if isinstance(num_steps, bool) or not isinstance(num_steps, Integral):
            raise TypeError("num_steps must be an integer")
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        if self.shortcut_denoise_timesteps % num_steps != 0:
            raise ValueError("num_steps must divide shortcut.denoise_timesteps")
        dt_base = int(math.log2(num_steps))
        if 2 ** dt_base != num_steps:
            raise ValueError("num_steps must be a power of two")

        h = self.shortcut_denoise_timesteps // num_steps
        img = y
        j = 0
        iterator = range(num_steps)
        if progress:
            iterator = tqdm(iterator, total=num_steps, desc=f"shortcut sampling {num_steps} steps")
        for _ in iterator:
            t = torch.full((y.shape[0],), int(self.shortcut_tau[j].item()), device=y.device, dtype=torch.long)
            n = torch.full((y.shape[0],), int(self.shortcut_tau[j + h].item()), device=y.device, dtype=torch.long)
            dt = torch.full((y.shape[0],), dt_base, device=y.device, dtype=torch.long)
            objective = self.denoise_objective(img, y, t, context, dt)
            img = self.omit_skip_update(img, y, t, n, objective, eta=eta, clip_x0=clip_denoised)
            j += h
        return img

    def training_losses(
        self,
        x0,
        y,
        context=None,
        shortcut_targets=None,
        shortcut_y=None,
        shortcut_context=None,
    ):
        """Eq. (26) on disjoint base and shortcut batch fractions."""
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context

        anchor = self.make_anchor_targets(x0, y)
        pred_anchor = self.denoise_objective(
            anchor["x_t"],
            y,
            anchor["t"],
            context,
            anchor["dt_base"],
        )
        loss_base = self._loss(pred_anchor, anchor["objective"])

        loss_shortcut = pred_anchor.new_tensor(0.0)
        if self.shortcut_enabled and shortcut_targets is not None:
            if shortcut_y is None:
                raise ValueError("shortcut_y is required when shortcut_targets are provided")
            if self.condition_key == "nocond":
                shortcut_context = None
            elif shortcut_context is None:
                shortcut_context = shortcut_y
            pred_shortcut = self.denoise_objective(
                shortcut_targets["x_t"],
                shortcut_y,
                shortcut_targets["t"],
                shortcut_context,
                shortcut_targets["dt_base"],
            )
            loss_shortcut = self._loss(pred_shortcut, shortcut_targets["objective"])

        total = (
            self.shortcut_lambda_base * loss_base
            + self.shortcut_lambda * loss_shortcut
        )
        x0_recon = self.predict_x0_from_objective_shortcut(
            anchor["x_t"],
            y,
            anchor["t"],
            pred_anchor,
        )
        return total, {
            "loss": total,
            "loss_base": loss_base.detach(),
            "loss_shortcut": loss_shortcut.detach(),
            "x0_recon": x0_recon.detach(),
        }

    def forward(self, x, y, context=None):
        requested_shortcut_count = x.shape[0] * self.shortcut_fraction
        shortcut_count = int(round(requested_shortcut_count))
        if (
            not math.isclose(requested_shortcut_count, shortcut_count, abs_tol=1e-8)
            or shortcut_count <= 0
            or shortcut_count >= x.shape[0]
        ):
            raise ValueError(
                f"Batch size {x.shape[0]} cannot realize shortcut_fraction="
                f"{self.shortcut_fraction} exactly; use a batch size divisible "
                "by 4 for the paper's 3:1 base/shortcut split"
            )
        base_count = x.shape[0] - shortcut_count
        context = y if context is None else context
        shortcut_targets = self.make_shortcut_targets(
            x[base_count:], y[base_count:], context=context[base_count:]
        )
        return self.training_losses(
            x[:base_count],
            y[:base_count],
            context=context[:base_count],
            shortcut_targets=shortcut_targets,
            shortcut_y=y[base_count:],
            shortcut_context=context[base_count:],
        )

    @torch.no_grad()
    def sample_shortcut(
        self,
        y,
        context=None,
        num_steps=1,
        clip_denoised=True,
        eta=0.0,
        progress=True,
    ):
        return self._sample_shortcut_loop(
            y,
            context=context,
            num_steps=num_steps,
            clip_denoised=clip_denoised,
            eta=eta,
            progress=progress,
        )

    @torch.no_grad()
    def sample(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        if sample_mid_step:
            raise NotImplementedError("sample_mid_step is not implemented for OMIT-Diffusion.")
        return self.sample_shortcut(
            y,
            context=context,
            num_steps=1,
            clip_denoised=clip_denoised,
            eta=0.0,
            progress=True,
        )
