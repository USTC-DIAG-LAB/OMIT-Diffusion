import os
import argparse
import tempfile
import uuid
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Sampler
from PIL import Image
from datetime import datetime
from torchvision.utils import make_grid
from Register import Registers
from datasets import custom as _registered_datasets  # noqa: F401 - registration side effect


def remove_file(fpath):
    if os.path.exists(fpath):
        os.remove(fpath)


def make_dir(dir):
    os.makedirs(dir, exist_ok=True)
    return dir


def _path_component(value):
    value = str(value)
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Expected a single path component, got {value!r}")
    return value


def make_save_dirs(args, prefix, suffix=None):
    """Create one unique run, or explicitly reopen --resume_run_dir.

    In DDP rank zero chooses the run once and broadcasts that exact directory.
    Loading weights alone starts a new run; it never implicitly reuses their folder.
    """
    resume_dir = getattr(args, "resume_run_dir", None)
    if resume_dir and not getattr(args, "resume_model", None):
        raise ValueError("--resume_run_dir requires --resume_model")
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    result_path = None
    if rank == 0:
        if resume_dir:
            run_path = Path(resume_dir).expanduser().resolve()
            if not run_path.is_dir():
                raise FileNotFoundError(f"Resume run directory does not exist: {run_path}")
        else:
            task = _path_component(Path(args.config).stem)
            root = Path(args.result_path) / _path_component(prefix)
            if suffix:
                root /= _path_component(suffix)
            root /= task
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
            label = getattr(args, "run_name", None)
            if label:
                run_id = _path_component(label) + "-" + run_id
            run_path = root / run_id
            run_path.mkdir(parents=True, exist_ok=False)
            run_path = run_path.resolve()
        result_path = str(run_path)
    if distributed:
        shared_path = [result_path]
        dist.broadcast_object_list(shared_path, src=0)
        result_path = shared_path[0]
    image_path = make_dir(os.path.join(result_path, "image"))
    log_path = make_dir(os.path.join(result_path, "log"))
    checkpoint_path = make_dir(os.path.join(result_path, "checkpoint"))
    sample_path = make_dir(os.path.join(result_path, "samples"))
    sample_to_eval_path = make_dir(os.path.join(result_path, "sample_to_eval"))
    return result_path, image_path, checkpoint_path, log_path, sample_path, sample_to_eval_path


def config_to_plain(value):
    """Convert runtime namespaces/devices/tuples to safe YAML primitives."""
    if isinstance(value, argparse.Namespace):
        return {key: config_to_plain(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): config_to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [config_to_plain(item) for item in value]
    if isinstance(value, (Path, torch.device)):
        return str(value)
    return value


def atomic_torch_save(value, path):
    """Replace a single checkpoint only after its new bytes are complete."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class DistributedEvaluationSampler(Sampler):
    """Assign whole batches to ranks without padding or duplicate examples.

    Validation loss requires complete model batches, so drop_last=True discards
    only the final global incomplete batch. Image export uses drop_last=False.
    Validation must call the unwrapped model: ranks may have different counts.
    """
    def __init__(self, dataset, batch_size, rank, world_size, drop_last=False):
        self.size = len(dataset)
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.num_batches = self.size // batch_size if drop_last else (self.size + batch_size - 1) // batch_size

    def __iter__(self):
        for batch in range(self.rank, self.num_batches, self.world_size):
            yield from range(batch * self.batch_size, min((batch + 1) * self.batch_size, self.size))

    def __len__(self):
        return sum(min(self.batch_size, self.size - batch * self.batch_size)
                   for batch in range(self.rank, self.num_batches, self.world_size))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Parameter') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_optimizer(optim_config, parameters):
    if optim_config.optimizer == 'Adam':
        return torch.optim.Adam(parameters, lr=optim_config.lr, weight_decay=optim_config.weight_decay,
                                betas=(optim_config.beta1, 0.999))
    elif optim_config.optimizer == 'RMSProp':
        return torch.optim.RMSprop(parameters, lr=optim_config.lr, weight_decay=optim_config.weight_decay)
    elif optim_config.optimizer == 'SGD':
        return torch.optim.SGD(parameters, lr=optim_config.lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(optim_config.optimizer))


def get_dataset(data_config, stage):
    return Registers.datasets[data_config.dataset_type](data_config.dataset_config, stage=stage)


@torch.no_grad()
def save_single_image(image, save_path, file_name, to_normal=True):
    image = image.detach().clone()
    if to_normal:
        image = image.mul_(0.5).add_(0.5).clamp_(0, 1.)
    image = image.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(image)
    im.save(os.path.join(save_path, file_name))


@torch.no_grad()
def get_image_grid(batch, grid_size=4, to_normal=True):
    batch = batch.detach().clone()
    image_grid = make_grid(batch, nrow=grid_size)
    if to_normal:
        image_grid = image_grid.mul_(0.5).add_(0.5).clamp_(0, 1.)
    image_grid = image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return image_grid
