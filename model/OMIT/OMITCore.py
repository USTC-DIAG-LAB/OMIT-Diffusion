"""Shared schedule and denoiser construction for OMIT-Diffusion.

The six schedule buffers retain their historical names for checkpoint
compatibility. The legacy BBDM loss and sampling implementations are removed;
OMITModel owns all supported training and sampling paths.
"""

import math
from functools import partial
from numbers import Integral, Real

import numpy as np
import torch
import torch.nn as nn

from model.OMIT.base.modules.diffusionmodules.OMITUNet import UNetModel


GENERATED_SCHEDULE_BUFFERS = frozenset({
    "m_t", "m_tminus", "variance_t", "variance_tminus",
    "variance_t_tminus", "posterior_variance_t",
})


class OMITCore(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        model_params = model_config.OMIT.params
        self.num_timesteps = model_params.num_timesteps
        if isinstance(self.num_timesteps, bool) or not isinstance(self.num_timesteps, Integral):
            raise TypeError("num_timesteps must be an integer")
        if self.num_timesteps < 2:
            raise ValueError("num_timesteps must be at least 2")

        self.mt_type = model_params.mt_type
        if self.mt_type not in {"linear", "sin"}:
            raise ValueError("mt_type must be 'linear' or 'sin'")
        self.max_var = getattr(model_params, "max_var", 1.0)
        if (
            isinstance(self.max_var, bool)
            or not isinstance(self.max_var, Real)
            or not math.isfinite(self.max_var)
            or self.max_var <= 0
        ):
            raise ValueError("max_var must be a finite positive number")

        self.loss_type = model_params.loss_type
        if self.loss_type not in {"l1", "l2"}:
            raise ValueError("loss_type must be 'l1' or 'l2'")
        self.objective = model_params.objective
        if self.objective not in {"grad", "ysubx"}:
            raise NotImplementedError(
                "OMIT supports objective='grad' or 'ysubx'; noise prediction "
                "is not invertible at the exact source endpoint"
            )

        unet_params = model_params.UNetParams
        if getattr(unet_params, "use_spatial_transformer", False):
            raise NotImplementedError(
                "SpatialTransformer is not a supported OMIT configuration; "
                "use the validated pixel-conditioned UNet"
            )
        if getattr(unet_params, "num_classes", None) is not None:
            raise NotImplementedError("OMIT does not accept class labels; num_classes must be None")
        if getattr(unet_params, "n_embed", None) is not None:
            raise NotImplementedError("OMIT predicts images, not codebook IDs; n_embed must be None")
        self.image_size = unet_params.image_size
        self.channels = unet_params.in_channels
        self.condition_key = unet_params.condition_key
        if self.condition_key not in {"concat", "SpatialRescaler", "nocond"}:
            raise ValueError("condition_key must be 'concat' or 'nocond' (legacy 'SpatialRescaler' is accepted)")

        self.register_schedule()
        self.denoise_fn = UNetModel(**vars(unet_params))

    def register_schedule(self):
        # Preserve the original schedules and persistent buffer names exactly.
        T = self.num_timesteps
        if self.mt_type == "linear":
            m_t = np.linspace(0.001, 0.999, T)
        else:
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        m_tminus = np.append(0, m_t[:-1])
        variance_t = 2. * (m_t - m_t ** 2) * self.max_var
        variance_tminus = np.append(0., variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer("m_t", to_torch(m_t))
        self.register_buffer("m_tminus", to_torch(m_tminus))
        self.register_buffer("variance_t", to_torch(variance_t))
        self.register_buffer("variance_tminus", to_torch(variance_tminus))
        self.register_buffer("variance_t_tminus", to_torch(variance_t_tminus))
        self.register_buffer("posterior_variance_t", to_torch(posterior_variance_t))

    def get_parameters(self):
        return self.denoise_fn.parameters()
