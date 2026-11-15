import os
import copy
import warnings
from pathlib import Path

import torch
from tqdm.autonotebook import tqdm

from Register import Registers
from model.OMIT.OMITCore import GENERATED_SCHEDULE_BUFFERS
from model.OMIT.OMITModel import OMITModel
from runners.BaseRunner import BaseRunner
from runners.utils import get_image_grid, get_optimizer, make_dir, save_single_image, weights_init


@Registers.runners.register_with_name("OMITRunner")
class OMITRunner(BaseRunner):
    def print_model_summary(self, net):
        total = sum(parameter.numel() for parameter in net.parameters())
        trainable = sum(
            parameter.numel() for parameter in net.parameters() if parameter.requires_grad
        )
        self.logger(f"Total parameters: {total / 1e6:.2f}M")
        self.logger(f"Trainable parameters: {trainable / 1e6:.2f}M")

    def initialize_model(self, config):
        if config.model.model_type != "OMIT":
            raise NotImplementedError("OMITRunner requires model.model_type='OMIT'")
        if not config.args.train and not getattr(config.model, "model_load_path", None):
            raise ValueError("Inference requires a trained checkpoint: set --resume_model "
                             "or model.model_load_path")
        net = OMITModel(config.model).to(config.training.device[0])
        net.apply(weights_init)
        return net

    def initialize_optimizer_scheduler(self, net, config):
        optimizer = get_optimizer(config.model.OMIT.optimizer, net.get_parameters())
        # The paper uses a fixed learning rate of 2e-6 for OMIT-Diffusion.
        return [optimizer], None

    @staticmethod
    def _validate_checkpoint_state(model, state_dict, label):
        if not isinstance(state_dict, dict):
            raise TypeError(f"Checkpoint entry {label!r} must be a state dictionary")

        expected = model.state_dict()
        missing = sorted(
            name
            for name in expected
            if name not in state_dict and name not in GENERATED_SCHEDULE_BUFFERS
        )
        unexpected = sorted(name for name in state_dict if name not in expected)
        wrong_shapes = sorted(
            f"{name}: checkpoint={tuple(state_dict[name].shape)}, model={tuple(expected[name].shape)}"
            for name in expected.keys() & state_dict.keys()
            if getattr(state_dict[name], "shape", None) != expected[name].shape
        )
        if missing or unexpected or wrong_shapes:
            raise RuntimeError(
                f"Checkpoint entry {label!r} is incompatible: "
                f"missing={missing}, unexpected={unexpected}, wrong_shapes={wrong_shapes}"
            )

    def load_model_from_checkpoint(self):
        states = None
        if self.config.model.__contains__("model_load_path") and self.config.model.model_load_path is not None:
            self.logger(f"load model {self.config.model.model_name} from {self.config.model.model_load_path}")
            states = torch.load(self.config.model.model_load_path, map_location="cpu")
            reset_resume_epoch = (
                self.config.model.reset_resume_epoch
                if self.config.model.__contains__("reset_resume_epoch")
                else False
            )
            if reset_resume_epoch:
                self.global_epoch = 0
                self.global_step = 0
                self.next_batch_idx = 0
                self.optimizer_updates = 0
                self.topk_checkpoints = {}
            else:
                self.global_epoch = int(states.get("epoch", 0))
                self.global_step = int(states.get("step", 0))
                self.next_batch_idx = int(states.get("next_batch_idx", 0))
                self.optimizer_updates = states.get("optimizer_updates")
                self._resume_epoch_length = states.get("epoch_length")
                self.topk_checkpoints = copy.deepcopy(states.get("topk_checkpoints", {}))
                if "top" in self.topk_checkpoints:
                    self.topk_checkpoints["top"].setdefault(
                        "checkpoint_dir", str(Path(self.config.model.model_load_path).resolve().parent)
                    )

            load_ema_as_model = (
                self.config.model.load_ema_as_model
                if self.config.model.__contains__("load_ema_as_model")
                else False
            )
            model_key = "ema" if load_ema_as_model and "ema" in states else "model"
            if model_key not in states:
                raise KeyError(f"Checkpoint has no {model_key!r} state dictionary")
            selected_state = states[model_key]
            self._validate_checkpoint_state(self.net, selected_state, model_key)
            # Schedule buffers are deterministic functions of the config and
            # are intentionally absent from released checkpoints.
            complete_state = dict(selected_state)
            current_state = self.net.state_dict()
            for name in GENERATED_SCHEDULE_BUFFERS:
                if name not in complete_state:
                    complete_state[name] = current_state[name]
            self.net.load_state_dict(complete_state, strict=True)

            if self.use_ema and "ema" in states:
                self._validate_checkpoint_state(self.net, states["ema"], "ema")
                merged_shadow = dict(self.ema.shadow)
                for name, value in states["ema"].items():
                    if name in merged_shadow:
                        merged_shadow[name] = value.clone()
                self.ema.shadow = merged_shadow
                self.ema.reset_device(self.net)
            elif self.use_ema:
                # Model-only artifacts may already contain materialized EMA
                # parameters.  Mirror the loaded model into the shadow so test
                # time EMA application never restores random initialization.
                self.ema.register(self.net)
                self.ema.reset_device(self.net)

            if self.config.args.train:
                optim_path = (
                    self.config.model.optim_sche_load_path
                    if self.config.model.__contains__("optim_sche_load_path")
                    else None
                )
                if optim_path is not None:
                    try:
                        optimizer_scheduler_states = torch.load(optim_path, map_location="cpu")
                        model_id = states.get("checkpoint_id")
                        optimizer_id = optimizer_scheduler_states.get("checkpoint_id")
                        if model_id is not None and optimizer_id is not None and model_id != optimizer_id:
                            raise ValueError("Model and optimizer files are from different checkpoint writes")
                        for field in ("epoch", "step", "optimizer_updates", "next_batch_idx"):
                            if field in states and field in optimizer_scheduler_states:
                                if states[field] != optimizer_scheduler_states[field]:
                                    raise ValueError(f"Checkpoint pair disagrees on {field}")
                        if len(optimizer_scheduler_states["optimizer"]) != len(self.optimizer):
                            raise ValueError("Optimizer count differs from the checkpoint")
                        for i in range(len(self.optimizer)):
                            self.optimizer[i].load_state_dict(optimizer_scheduler_states["optimizer"][i])
                        if self.scheduler is not None:
                            for i in range(len(self.optimizer)):
                                self.scheduler[i].load_state_dict(optimizer_scheduler_states["scheduler"][i])
                    except Exception as exc:
                        raise RuntimeError(f"Could not restore the requested optimizer checkpoint: {optim_path}") from exc
                else:
                    warnings.warn("Model weights and counters were restored without optimizer moments; "
                                  "pass --resume_optim for a training resume.", RuntimeWarning)
        elif self.config.args.train and getattr(self.config.model, "optim_sche_load_path", None):
            raise ValueError("--resume_optim requires --resume_model")
        return states

    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage="train", write=True):
        (x, x_name), (x_cond, x_cond_name) = batch
        x = x.to(self.config.training.device[0], non_blocking=True)
        x_cond = x_cond.to(self.config.training.device[0], non_blocking=True)

        # Keep the training computation inside the wrapped forward call.  In
        # distributed runs ``net`` is DistributedDataParallel; calling methods
        # on ``net.module`` here would bypass DDP's forward preparation and can
        # leave gradients unsynchronized between ranks.
        loss, additional_info = net(x, x_cond)

        if write and self.is_main_process:
            self.writer.add_scalar(f"loss/{stage}", loss, step)
            self.writer.add_scalar(f"loss_base/{stage}", additional_info["loss_base"], step)
            self.writer.add_scalar(f"loss_shortcut/{stage}", additional_info["loss_shortcut"], step)
        return loss

    @torch.no_grad()
    def sample(self, net, batch, sample_path, stage="train"):
        sample_path = make_dir(os.path.join(sample_path, f"{stage}_sample"))
        (x, x_name), (x_cond, x_cond_name) = batch
        batch_size = x.shape[0] if x.shape[0] < 4 else 4

        x = x[:batch_size].to(self.config.training.device[0])
        x_cond = x_cond[:batch_size].to(self.config.training.device[0])
        grid_size = 4
        steps = self.config.model.OMIT.params.sample_step

        sample = net.sample_shortcut(
            x_cond,
            num_steps=steps,
            clip_denoised=self.config.testing.clip_denoised,
            eta=0.0,
            progress=self.is_main_process,
        ).to("cpu")
        image_grid = get_image_grid(sample, grid_size, to_normal=self.config.data.dataset_config.to_normal)
        from PIL import Image

        Image.fromarray(image_grid).save(os.path.join(sample_path, f"shortcut_{steps}_steps.png"))
        if stage != "test":
            self.writer.add_image(f"{stage}_shortcut_{steps}_steps", image_grid, self.global_step, dataformats="HWC")

        image_grid = get_image_grid(x_cond.to("cpu"), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        Image.fromarray(image_grid).save(os.path.join(sample_path, "condition.png"))
        image_grid = get_image_grid(x.to("cpu"), grid_size, to_normal=self.config.data.dataset_config.to_normal)
        Image.fromarray(image_grid).save(os.path.join(sample_path, "ground_truth.png"))

    @torch.no_grad()
    def sample_to_eval(self, net, test_loader, sample_path):
        condition_path = make_dir(os.path.join(sample_path, "condition"))
        gt_path = make_dir(os.path.join(sample_path, "ground_truth"))
        to_normal = self.config.data.dataset_config.to_normal
        sample_steps = (
            self.config.testing.shortcut_sample_steps
            if self.config.testing.__contains__("shortcut_sample_steps")
            else [self.config.model.OMIT.params.sample_step]
        )

        for steps in sample_steps:
            result_path = make_dir(os.path.join(sample_path, f"steps_{steps}"))
            pbar = tqdm(
                test_loader,
                total=len(test_loader),
                smoothing=0.01,
                desc=f"sample_to_eval {steps} steps",
                disable=not self.is_main_process,
            )
            for test_batch in pbar:
                (x, x_name), (x_cond, x_cond_name) = test_batch
                x = x.to(self.config.training.device[0])
                x_cond = x_cond.to(self.config.training.device[0])
                sample = net.sample_shortcut(
                    x_cond,
                    num_steps=int(steps),
                    clip_denoised=self.config.testing.clip_denoised,
                    eta=0.0,
                    progress=False,
                )
                for i in range(x.shape[0]):
                    if int(steps) == int(sample_steps[0]):
                        save_single_image(x_cond[i], condition_path, f"{x_cond_name[i]}.png", to_normal=to_normal)
                        save_single_image(x[i], gt_path, f"{x_name[i]}.png", to_normal=to_normal)
                    save_single_image(sample[i], result_path, f"{x_name[i]}.png", to_normal=to_normal)
