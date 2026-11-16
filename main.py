"""Training and preview entry point for the paired OMIT model."""

import argparse
import os
import yaml
import copy
import torch
import random
import numpy as np
from pathlib import Path

from utils import dict2namespace, get_runner
import torch.multiprocessing as mp
import torch.distributed as dist


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


def parse_args_and_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the task YAML config')
    parser.add_argument('-s', '--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('-r', '--result_path', type=str, default='results', help="The directory to save results")
    parser.add_argument('--run_name', default=None, help='Optional name for a new, isolated run directory')
    parser.add_argument('--resume_run_dir', default=None, help='Explicit existing run directory to reuse with --resume_model')

    parser.add_argument('-t', '--train', action='store_true', default=False, help='train the model')
    parser.add_argument('--sample_to_eval', action='store_true', default=False, help='sample for evaluation')
    parser.add_argument('--sample_at_start', action='store_true', default=False, help='Generate a preview before the first training batch')

    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids, 0,1,2,3 cpu=-1')
    parser.add_argument('--port', type=str, default='12355', help='DDP master port')

    parser.add_argument('--resume_model', type=str, default=None, help='model checkpoint')
    parser.add_argument('--resume_optim', type=str, default=None, help='optimizer checkpoint')

    parser.add_argument('--max_epoch', type=positive_int, default=None, help='Total epoch limit, including completed epochs')
    parser.add_argument('--max_steps', type=positive_int, default=None, help='Total microbatch limit, including completed microbatches')
    parser.add_argument('--num_workers', type=nonnegative_int, default=None, help='DataLoader workers; use 0 for debugging')

    args = parser.parse_args(argv)

    with open(args.config, 'r') as f:
        dict_config = yaml.load(f, Loader=yaml.FullLoader)

    namespace_config = dict2namespace(dict_config)
    namespace_config.args = args

    if args.resume_model is not None:
        namespace_config.model.model_load_path = args.resume_model
    if args.resume_optim is not None:
        namespace_config.model.optim_sche_load_path = args.resume_optim
    if args.max_epoch is not None:
        namespace_config.training.n_epochs = args.max_epoch
    if args.max_steps is not None:
        namespace_config.training.n_steps = args.max_steps
    if args.num_workers is not None:
        namespace_config.training.num_workers = args.num_workers

    has_model = bool(getattr(namespace_config.model, 'model_load_path', None))
    # The output-directory helper receives args, so expose a checkpoint supplied
    # through YAML there too; CLI and YAML resume paths follow the same contract.
    if has_model and args.resume_model is None:
        args.resume_model = namespace_config.model.model_load_path
    if args.resume_run_dir and not has_model:
        parser.error('--resume_run_dir requires --resume_model or model.model_load_path')
    if args.resume_run_dir and args.run_name:
        parser.error('--run_name and --resume_run_dir are mutually exclusive')
    if getattr(namespace_config.model, 'optim_sche_load_path', None) and not has_model:
        parser.error('--resume_optim requires --resume_model or model.model_load_path')
    if not args.train and not has_model:
        parser.error('Inference requires --resume_model or model.model_load_path; '
                     'provide a trained checkpoint for the selected task')
    for field, option in (('model_load_path', '--resume_model'),
                          ('optim_sche_load_path', '--resume_optim')):
        value = getattr(namespace_config.model, field, None)
        if value:
            checkpoint_path = Path(value).expanduser()
            if not checkpoint_path.is_file():
                parser.error(f'{option} / model.{field} is not a file: {checkpoint_path}')
            setattr(namespace_config.model, field, str(checkpoint_path))
    if has_model:
        args.resume_model = namespace_config.model.model_load_path
    return namespace_config


def set_random_seed(SEED=1234):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def DDP_run_fn(rank, world_size, config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = config.args.port
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # Keep rank 0 reproducible while giving every DDP worker an independent
    # RNG stream.  This is required for per-sample random diffusion timesteps;
    # using the same seed on every rank otherwise repeats the timestep/noise
    # draws at matching local batch positions across GPUs.
    set_random_seed(config.args.seed + rank)

    local_rank = rank
    config.training.device = [torch.device("cuda:%d" % local_rank)]
    print('using device:', config.training.device)
    config.training.local_rank = local_rank
    runner = None
    try:
        runner = get_runner(config.runner, config)
        if config.args.train:
            runner.train()
        else:
            with torch.no_grad():
                runner.test()
    finally:
        if runner is not None:
            runner.writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def CPU_singleGPU_launcher(config):
    set_random_seed(config.args.seed)
    runner = get_runner(config.runner, config)
    try:
        if config.args.train:
            runner.train()
        else:
            with torch.no_grad():
                runner.test()
    finally:
        runner.writer.close()


def DDP_launcher(world_size, run_fn, config):
    mp.spawn(run_fn,
             args=(world_size, copy.deepcopy(config)),
             nprocs=world_size,
             join=True)


def main():
    nconfig = parse_args_and_config()
    args = nconfig.args

    gpu_ids = args.gpu_ids
    if gpu_ids == "-1": # Use CPU
        nconfig.training.use_DDP = False
        nconfig.training.device = [torch.device("cpu")]
        CPU_singleGPU_launcher(nconfig)
    else:
        gpu_list = gpu_ids.split(",")
        if any(not value.isdigit() for value in gpu_list) or len(set(gpu_list)) != len(gpu_list):
            raise ValueError('--gpu_ids must be -1 or distinct comma-separated nonnegative GPU indices')
        if len(gpu_list) > 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; use --gpu_ids -1 to run on CPU')
        if len(gpu_list) > 1:
            nconfig.training.use_DDP = True
            DDP_launcher(world_size=len(gpu_list), run_fn=DDP_run_fn, config=nconfig)
        else:
            nconfig.training.use_DDP = False
            nconfig.training.device = [torch.device(f"cuda:{gpu_list[0]}")]
            CPU_singleGPU_launcher(nconfig)
    return


if __name__ == "__main__":
    main()
