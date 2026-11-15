#!/usr/bin/env python3
"""Evaluate a paired OMIT checkpoint on the independent test split."""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm.autonotebook import tqdm

from datasets import custom  # noqa: F401,E402 - registers paired datasets
from Register import Registers  # noqa: E402
from model.OMIT.OMITModel import (  # noqa: E402
    OMITModel,
)
from model.OMIT.OMITCore import GENERATED_SCHEDULE_BUFFERS  # noqa: E402
from utils import dict2namespace  # noqa: E402


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError('must be a nonnegative integer')
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults to checkpoints/<region>/<direction>/inference_model.pth.",
    )
    parser.add_argument("--weights", choices=("auto", "model", "ema"), default="auto")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=positive_int, default=None)
    parser.add_argument("--num_workers", type=nonnegative_int, default=4)
    parser.add_argument("--max_samples", type=positive_int, default=None)
    parser.add_argument("--num_steps", type=positive_int, default=1, help="Sampling steps; defaults to one-step evaluation")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--save_predictions", type=Path, default=None)
    return parser.parse_args(argv)


def resolve_project_path(path):
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def describe_path(path):
    """Keep paths inside the repository concise, and accept external files."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def infer_checkpoint(config_path):
    task = config_path.stem
    try:
        region, direction = task.split("_", 1)
    except ValueError as exc:
        raise ValueError(f"Cannot infer checkpoint directory from config name {task!r}") from exc
    checkpoint_path = ROOT / "checkpoints" / region / direction / "inference_model.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Default checkpoint not found: {checkpoint_path}. "
            "Place the trained weight there or pass --checkpoint explicitly."
        )
    return checkpoint_path


def select_state_dict(checkpoint, requested):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state dictionary")
    if requested == "ema":
        if "ema" not in checkpoint:
            raise KeyError("--weights ema was requested, but the checkpoint has no 'ema' entry")
        return checkpoint["ema"], "ema"
    if requested == "model":
        if "model" in checkpoint:
            return checkpoint["model"], "model"
        if all(isinstance(value, torch.Tensor) for value in checkpoint.values()) and checkpoint:
            return checkpoint, "state_dict"
        raise KeyError("--weights model was requested, but the checkpoint has no 'model' entry or raw state dictionary")
    if "ema" in checkpoint:
        return checkpoint["ema"], "ema"
    if "model" in checkpoint:
        return checkpoint["model"], "model"
    return checkpoint, "state_dict"


def to_unit_interval(image, normalized):
    return image.mul(0.5).add(0.5).clamp(0.0, 1.0) if normalized else image.clamp(0.0, 1.0)


def per_image_ssim(prediction, target, window_size=11, sigma=1.5):
    """Per-slice SSIM using an 11x11 Gaussian window and valid convolution.

    Inputs must be in [0, 1].  Constants use K1=0.01, K2=0.03 and data_range=1.
    SSIM is averaged over valid spatial positions and image channels.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if prediction.shape[-2] < window_size or prediction.shape[-1] < window_size:
        raise ValueError(f"SSIM inputs must be at least {window_size}x{window_size}")
    channels = prediction.shape[1]
    coordinates = torch.arange(window_size, device=prediction.device, dtype=prediction.dtype)
    coordinates = coordinates - (window_size - 1) / 2
    kernel_1d = torch.exp(-(coordinates.square()) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    window = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    mu_x = F.conv2d(prediction, window, padding=0, groups=channels)
    mu_y = F.conv2d(target, window, padding=0, groups=channels)
    mu_x2, mu_y2, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    sigma_x2 = F.conv2d(prediction.square(), window, padding=0, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(target.square(), window, padding=0, groups=channels) - mu_y2
    sigma_xy = F.conv2d(prediction * target, window, padding=0, groups=channels) - mu_xy
    numerator = (2.0 * mu_xy + 0.01 ** 2) * (2.0 * sigma_xy + 0.03 ** 2)
    denominator = (mu_x2 + mu_y2 + 0.01 ** 2) * (sigma_x2 + sigma_y2 + 0.03 ** 2)
    return (numerator / denominator.clamp_min(1e-12)).flatten(1).mean(1)


def save_image(image, path):
    array = image.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    Image.fromarray(array).save(path)


def main(argv=None):
    args = parse_args(argv)
    # CLI paths in the supplied configs have always been relative to the repo.
    # Do not change another application's cwd merely by importing this module.
    os.chdir(ROOT)
    config_path = resolve_project_path(args.config)
    checkpoint_path = resolve_project_path(args.checkpoint) if args.checkpoint else infer_checkpoint(config_path)
    with config_path.open() as handle:
        config = dict2namespace(yaml.load(handle, Loader=yaml.FullLoader))

    device = torch.device(args.device)
    model = OMITModel(config.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict, selected_weights = select_state_dict(checkpoint, args.weights)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Selected checkpoint entry {selected_weights!r} is not a state dictionary")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    material_missing = [name for name in missing if name not in GENERATED_SCHEDULE_BUFFERS]
    if material_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={material_missing}, unexpected={unexpected}"
        )
    model.to(device).eval()
    if args.num_steps > model.shortcut_denoise_timesteps or model.shortcut_denoise_timesteps % args.num_steps:
        raise ValueError("--num_steps must divide shortcut.denoise_timesteps")

    dataset = Registers.datasets[config.data.dataset_type](config.data.dataset_config, stage="test")
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    if len(dataset) == 0:
        raise ValueError("Evaluation dataset is empty")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or config.data.test.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    if args.save_predictions:
        args.save_predictions = resolve_project_path(args.save_predictions)
        args.save_predictions.mkdir(parents=True, exist_ok=True)

    sums = torch.zeros(5, dtype=torch.float64, device=device)
    normalized = bool(config.data.dataset_config.to_normal)
    clip_denoised = bool(getattr(getattr(config, "testing", None), "clip_denoised", True))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        for (target, names), (source, _) in tqdm(loader, desc=f"{args.num_steps}-step evaluation"):
            source = source.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            prediction = model.sample_shortcut(
                source,
                num_steps=args.num_steps,
                clip_denoised=clip_denoised,
                eta=0.0,
                progress=False,
            )
            prediction_01 = to_unit_interval(prediction, normalized)
            target_01 = to_unit_interval(target, normalized)
            error = prediction_01 - target_01
            mse = error.square().flatten(1).mean(1)
            mae = error.abs().flatten(1).mean(1)
            psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
            ssim = per_image_ssim(prediction_01, target_01)
            batch_sums = torch.stack((mae.sum(), mse.sum(), psnr.sum(), ssim.sum()))
            sums[:4] += batch_sums.double()
            sums[4] += len(names)
            if args.save_predictions:
                for index, name in enumerate(names):
                    save_image(prediction_01[index], args.save_predictions / f"{name}.png")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    count = int(sums[4].item())
    result = {
        "config": describe_path(config_path),
        "checkpoint": describe_path(checkpoint_path),
        "checkpoint_weights": selected_weights,
        "steps": args.num_steps,
        "data_split": "test",
        "dataset_stage": "test",
        "count": count,
        "mae": float((sums[0] / count).item()),
        "mse": float((sums[1] / count).item()),
        "psnr": float((sums[2] / count).item()),
        "ssim": float((sums[3] / count).item()),
        "elapsed_seconds": elapsed,
        "seconds_per_slice": elapsed / count,
        "nfe_per_slice": args.num_steps,
        "timing_scope": "End-to-end loop: data loading, transfer, sampling, metrics, and optional image saving; no warm-up excluded.",
        "metric_protocol": {
            "aggregation": "arithmetic mean of per-slice metrics",
            "clip_denoised": clip_denoised,
            "psnr": "-10*log10(per-slice MSE), data_range=1",
            "ssim": "11x11 Gaussian window, sigma=1.5, valid convolution, K1=0.01, K2=0.03, data_range=1",
        },
    }
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    output = resolve_project_path(args.output) if args.output else ROOT / "artifacts" / f"evaluation_{config_path.stem}_{run_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved evaluation: {output}")
    return result


if __name__ == "__main__":
    main()
