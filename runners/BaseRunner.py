import argparse
import datetime
import math
import os
import tempfile
import time
import traceback
import uuid
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from itertools import islice
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm


class NullWriter:
    def add_scalar(self, *args, **kwargs): pass
    def add_image(self, *args, **kwargs): pass
    def close(self): pass


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = lambda *args, **kwargs: NullWriter()

from runners.base.EMA import EMA
from runners.utils import (
    DistributedEvaluationSampler,
    atomic_torch_save,
    config_to_plain,
    get_dataset,
    make_dir,
    make_save_dirs,
    remove_file,
)


class BaseRunner(ABC):
    def __init__(self, config):
        self.config = config
        self.is_main_process = not config.training.use_DDP or config.training.local_rank == 0
        self.global_epoch = 0
        self.global_step = 0  # Successfully processed microbatches, not optimizer updates.
        self.optimizer_updates = 0
        self.next_batch_idx = 0
        self._resume_epoch_length = None
        # A failed optimizer/scheduler/EMA update can leave a partially mutated
        # model. Such a state must never be advertised as a resumable snapshot.
        self._update_in_progress = False

        paths = make_save_dirs(config.args, config.data.dataset_name, config.model.model_name)
        config.result = argparse.Namespace(**dict(zip(
            ('result_path', 'image_path', 'ckpt_path', 'log_path', 'sample_path', 'sample_to_eval_path'), paths
        )))
        self.logger('save results to ' + config.result.result_path)
        self.writer = SummaryWriter(config.result.log_path) if self.is_main_process else NullWriter()
        self.net, self.optimizer, self.scheduler = self.initialize_model_optimizer_scheduler(
            config, is_test=not config.args.train
        )
        self.print_model_summary(self.net)

        self.use_ema = bool(getattr(getattr(config.model, 'EMA', None), 'use_ema', False))
        if self.use_ema:
            self.ema = EMA(config.model.EMA.ema_decay)
            self.update_ema_interval = int(config.model.EMA.update_ema_interval)
            self.start_ema_step = int(config.model.EMA.start_ema_step)
            if self.update_ema_interval < 1:
                raise ValueError('EMA.update_ema_interval must be at least 1')
            self.ema.register(self.net)
        self.load_model_from_checkpoint()

        if config.training.use_DDP:
            if config.training.device[0].type == 'cuda':
                self.net = DDP(self.net, device_ids=[config.training.local_rank],
                               output_device=config.training.local_rank)
            else:
                self.net = DDP(self.net)
            # DDP synchronizes model parameters, not this separate EMA mapping.
            # Broadcast the existing shadow to preserve a loaded EMA checkpoint.
            if self.use_ema:
                self.ema.reset_device(self.model)
                for value in self.ema.shadow.values():
                    dist.broadcast(value, src=0)
        self._commit_progress()
        self.save_config()

    @property
    def model(self):
        return self.net.module if self.config.training.use_DDP else self.net

    def logger(self, msg, **kwargs):
        if self.is_main_process:
            print(msg, **kwargs)

    def save_config(self):
        if not self.is_main_process:
            return
        folder = Path(self.config.result.ckpt_path)
        name = 'config.yaml'
        if (folder / name).exists():
            name = 'config_resume_' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8] + '.yaml'
        descriptor, temporary = tempfile.mkstemp(prefix='.config-', suffix='.tmp', dir=folder)
        try:
            with os.fdopen(descriptor, 'w') as handle:
                yaml.safe_dump(config_to_plain(self.config), handle, sort_keys=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, folder / name)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def initialize_model_optimizer_scheduler(self, config, is_test=False):
        net = self.initialize_model(config)
        optimizer, scheduler = [], None
        if not is_test:
            optimizer, scheduler = self.initialize_optimizer_scheduler(net, config)
        return net, optimizer, scheduler

    @abstractmethod
    def load_model_from_checkpoint(self):
        pass

    def _commit_progress(self, *, epoch=None, cursor=None):
        # One assignment keeps exception recovery from observing a mixture of
        # old and new counters if interrupted at an update or epoch boundary.
        self._committed_progress = (
            self.global_epoch if epoch is None else epoch,
            self.global_step,
            self.next_batch_idx if cursor is None else cursor,
            self.optimizer_updates,
        )

    def get_checkpoint_states(self, stage='epoch_end'):
        if self._update_in_progress:
            raise RuntimeError(
                'Cannot checkpoint an incomplete optimizer/scheduler/EMA update; '
                'reload a matching checkpoint pair from an earlier successful save'
            )
        if self.use_ema and self.ema.backup:
            raise RuntimeError('Cannot checkpoint while EMA parameters are installed')
        interrupted = stage == 'exception'
        epoch_complete = stage == 'epoch_end'
        if interrupted:
            epoch, step, cursor, updates = self._committed_progress
        else:
            epoch = self.global_epoch + int(epoch_complete)
            step = self.global_step
            cursor = 0 if epoch_complete else self.next_batch_idx
            updates = self.optimizer_updates
        checkpoint_id = uuid.uuid4().hex
        metadata = {
            'checkpoint_id': checkpoint_id,
            'epoch': epoch,
            'step': step,
            'next_batch_idx': cursor,
            'optimizer_updates': int(updates or 0),
        }
        model_states = dict(metadata, model=self.model.state_dict(),
                            epoch_length=getattr(self, 'epoch_length', None))
        if self.use_ema:
            model_states['ema'] = self.ema.shadow
        optimizer_states = dict(metadata,
                                optimizer=[optimizer.state_dict() for optimizer in self.optimizer],
                                scheduler=[] if self.scheduler is None else [scheduler.state_dict() for scheduler in self.scheduler])
        return model_states, optimizer_states

    def step_ema(self):
        self.ema.update(self.model, with_decay=self.global_step >= self.start_ema_step)

    def apply_ema(self):
        if self.use_ema:
            self.ema.apply_shadow(self.model)

    def restore_ema(self):
        if self.use_ema:
            self.ema.restore(self.model)

    @contextmanager
    def evaluation_scope(self):
        was_training = self.net.training
        if self.use_ema and self.ema.backup:
            raise RuntimeError('EMA parameters are already installed')
        try:
            # Include installation in the protected region: an interruption
            # halfway through swapping parameters must restore those swapped.
            self.apply_ema()
            self.net.eval()
            with torch.no_grad():
                yield self.model
        finally:
            self.restore_ema()
            self.net.train(was_training)

    def test_loss_step(self, test_batch, epoch, step):
        """Log a held-out loss without updating parameters or selecting weights."""
        with self.evaluation_scope() as model:
            return self.loss_fn(model, test_batch, epoch, step, stage='test_step')

    def test_loss_epoch(self, test_loader, epoch):
        totals = torch.zeros(2, dtype=torch.float64, device=self.config.training.device[0])
        # Unequal, nonpadding shards must not use DDP.forward's buffer collectives.
        with self.evaluation_scope() as model:
            for step, test_batch in enumerate(tqdm(test_loader, disable=not self.is_main_process)):
                loss = self.loss_fn(model, test_batch, epoch, step, stage='test', write=False)
                count = int(test_batch[0][0].shape[0])
                totals[0] += loss.detach().double() * count
                totals[1] += count
        if self.config.training.use_DDP:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if totals[1].item() == 0:
            raise ValueError('Test loss has no complete batches for the configured loss batch size')
        average_loss = float((totals[0] / totals[1]).item())
        if self.is_main_process:
            self.writer.add_scalar('test_epoch/loss', average_loss, epoch)
            self.writer.add_scalar('test_epoch/sample_count', int(totals[1].item()), epoch)
        return average_loss

    def sample_step(self, train_batch, test_batch):
        with self.evaluation_scope() as model:
            path = make_dir(os.path.join(self.config.result.image_path, str(self.global_step)))
            self.sample(model, train_batch, path, stage='train')
            self.sample(model, test_batch, path, stage='test')

    @abstractmethod
    def print_model_summary(self, net):
        pass

    @abstractmethod
    def initialize_model(self, config):
        pass

    @abstractmethod
    def initialize_optimizer_scheduler(self, net, config):
        pass

    @abstractmethod
    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        pass

    @abstractmethod
    def sample(self, net, batch, sample_path, stage='train'):
        pass

    @abstractmethod
    def sample_to_eval(self, net, test_loader, sample_path):
        pass

    def _write_checkpoint_pair(self, model_name, optimizer_name, states):
        folder = Path(self.config.result.ckpt_path)
        atomic_torch_save(states[0], folder / model_name)
        atomic_torch_save(states[1], folder / optimizer_name)
        # Both files include checkpoint_id, so a partially replaced pair cannot
        # silently be accepted as one consistent resume checkpoint.

    def _save_checkpoints(self, stage):
        if not self.is_main_process:
            return
        folder = Path(self.config.result.ckpt_path)
        states = self.get_checkpoint_states(stage)
        label = str(self.global_epoch + 1) if stage == 'epoch_end' else f'step_{self.global_step}'
        model_name = f'latest_model_{label}.pth'
        optim_name = f'latest_optim_sche_{label}.pth'
        self._write_checkpoint_pair(model_name, optim_name, states)
        self._write_checkpoint_pair('last_model.pth', 'last_optim_sche.pth', states)
        # Delete older files only after all replacements succeeded.
        if not getattr(self.config.training, 'keep_all_epoch_checkpoints', False):
            for pattern, keep in [('latest_model_*.pth', model_name), ('latest_optim_sche_*.pth', optim_name)]:
                for path in folder.glob(pattern):
                    if path.name != keep:
                        remove_file(str(path))

    def _restore_legacy_update_count(self, accumulation):
        if self.optimizer_updates is not None:
            return
        observed = [float(state['step']) for optimizer in self.optimizer for state in optimizer.state.values()
                    if 'step' in state]
        if observed:
            self.optimizer_updates = int(max(observed))
        else:
            self.optimizer_updates = self.global_epoch * math.ceil(self.epoch_length / accumulation)
            self.optimizer_updates += math.ceil(self.next_batch_idx / accumulation)
        warnings.warn(f'Legacy checkpoint has no optimizer_updates; using {self.optimizer_updates}. '
                      'Legacy resumes do not preserve exact random or mid-epoch state.', RuntimeWarning)

    def train(self):
        if self._update_in_progress:
            raise RuntimeError(
                'This runner contains an incomplete optimizer/scheduler/EMA update; '
                'create a new runner from an earlier successful checkpoint'
            )
        if len(self.optimizer) != 1:
            raise ValueError('This OMIT runner requires exactly one optimizer')
        accumulation = int(self.config.training.accumulate_grad_batches)
        max_steps = int(self.config.training.n_steps)
        if accumulation < 1 or max_steps < 0:
            raise ValueError('accumulate_grad_batches must be positive and n_steps nonnegative')
        for name in ('sample_interval', 'test_loss_interval', 'save_interval'):
            if getattr(self.config.training, name) <= 0:
                raise ValueError(f'training.{name} must be positive')
        train_dataset = get_dataset(self.config.data, 'train')
        test_dataset = get_dataset(self.config.data, 'test')
        test_loss_batch_size = self.config.data.test.loss_batch_size
        workers = int(getattr(self.config.training, 'num_workers', 8))
        generator = torch.Generator()
        train_sampler = None
        test_sampler = None
        if self.config.training.use_DDP:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, shuffle=self.config.data.train.shuffle, seed=self.config.args.seed)
            test_sampler = DistributedEvaluationSampler(
                test_dataset, test_loss_batch_size, dist.get_rank(), dist.get_world_size(), drop_last=True)
        train_loader = DataLoader(train_dataset, batch_size=self.config.data.train.batch_size,
                                  shuffle=self.config.data.train.shuffle if train_sampler is None else False,
                                  sampler=train_sampler, num_workers=workers, drop_last=True, generator=generator)
        test_loader = DataLoader(test_dataset, batch_size=test_loss_batch_size,
                                 shuffle=False, sampler=test_sampler, num_workers=workers, drop_last=True)
        self.epoch_length = len(train_loader)
        if self.epoch_length == 0:
            raise ValueError('Training dataset must contain at least one complete batch')
        if len(test_dataset) < test_loss_batch_size:
            raise ValueError('Test dataset must contain at least one complete loss batch')
        if self.next_batch_idx and self._resume_epoch_length not in (None, self.epoch_length):
            raise ValueError('Cannot resume a batch cursor after the training loader length changed; reset the resume epoch')
        if not 0 <= self.next_batch_idx <= self.epoch_length:
            raise ValueError('Checkpoint next_batch_idx is outside this training epoch')
        self._restore_legacy_update_count(accumulation)
        self._commit_progress()
        if len(test_dataset) % test_loss_batch_size:
            self.logger('Test loss omits the final incomplete global batch; full image evaluation does not.')
        self.logger(f'{self.epoch_length} batches per epoch; max_steps={max_steps} counts processed microbatches')

        try:
            if self.config.args.sample_at_start:
                if self.is_main_process:
                    generator.manual_seed(self.config.args.seed + self.global_epoch)
                    self.sample_step(next(iter(train_loader)), next(iter(test_loader)))
                if self.config.training.use_DDP:
                    dist.barrier()
            for epoch in range(self.global_epoch, self.config.training.n_epochs):
                if self.global_step >= max_steps:
                    break
                self.global_epoch = epoch
                start_index = self.next_batch_idx
                self._commit_progress()
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                generator.manual_seed(self.config.args.seed + epoch)
                end_index = min(self.epoch_length, start_index + max_steps - self.global_step)
                optimizer = self.optimizer[0]
                optimizer.zero_grad()
                start_time = time.time()
                batches = islice(enumerate(train_loader), start_index, end_index)
                pbar = tqdm(batches, total=end_index - start_index, disable=not self.is_main_process)
                for batch_index, train_batch in pbar:
                    self.net.train()
                    group_start = start_index + ((batch_index - start_index) // accumulation) * accumulation
                    group_size = min(accumulation, end_index - group_start)
                    should_step = batch_index + 1 == group_start + group_size
                    loss = self.loss_fn(self.net, train_batch, epoch, self.global_step + 1, stage='train')
                    (loss / group_size).backward()
                    self.global_step += 1
                    self.next_batch_idx = batch_index + 1
                    if should_step:
                        # Keep this flag set on any exception, including
                        # KeyboardInterrupt. Clearing it in finally would mark
                        # partially updated parameters/moments/EMA as safe.
                        self._update_in_progress = True
                        optimizer.step()
                        optimizer.zero_grad()
                        next_updates = self.optimizer_updates + 1
                        if self.scheduler is not None:
                            self.scheduler[0].step(loss)
                        if self.use_ema and next_updates % self.update_ema_interval == 0:
                            self.step_ema()
                        device = self.config.training.device[0]
                        if device.type == 'cuda':
                            # CUDA updates are asynchronous. Surface their
                            # errors while the incomplete-update guard is set.
                            torch.cuda.synchronize(device)
                        self.optimizer_updates = next_updates
                        self._commit_progress()
                        self._update_in_progress = False
                    log_loss = loss.detach().clone()
                    if self.config.training.use_DDP:
                        dist.reduce(log_loss, dst=0, op=dist.ReduceOp.SUM)
                        log_loss /= dist.get_world_size()
                    pbar.set_description(f'Epoch {epoch + 1}; step {self.global_step}; loss {log_loss.item():.4f}')
                    if self.global_step % 50 == 0:
                        test_batch = next(iter(test_loader), None)
                        if test_batch is not None:
                            self.test_loss_step(test_batch, epoch, self.global_step)
                    sample_period = max(1, int(self.config.training.sample_interval * self.epoch_length))
                    if self.is_main_process and self.global_step % sample_period == 0:
                        self.sample_step(train_batch, next(iter(test_loader)))

                epoch_complete = self.next_batch_idx == self.epoch_length
                if epoch_complete and ((epoch + 1) % self.config.training.test_loss_interval == 0
                                       or epoch + 1 == self.config.training.n_epochs):
                    self.test_loss_epoch(test_loader, epoch)
                reached_budget = self.global_step >= max_steps
                should_save = reached_budget or epoch + 1 == self.config.training.n_epochs or (
                    epoch_complete and (epoch + 1) % self.config.training.save_interval == 0)
                if should_save:
                    self._save_checkpoints('epoch_end' if epoch_complete else 'mid_epoch')
                self.logger('training time: ' + str(datetime.timedelta(seconds=int(time.time() - start_time))))
                if self.config.training.use_DDP:
                    dist.barrier()
                if epoch_complete:
                    self._commit_progress(epoch=epoch + 1, cursor=0)
                    self.next_batch_idx = 0
                if reached_budget:
                    break
        except (Exception, KeyboardInterrupt):
            if self.is_main_process:
                if self._update_in_progress:
                    self.logger(
                        'Skipped interrupted checkpoint: an optimizer/scheduler/EMA update did not complete. '
                        'The in-memory state may be partially updated. Existing checkpoints remain unchanged; '
                        'reload a matching model/optimizer pair from an earlier successful save, '
                        'or restart training if no such save exists.'
                    )
                else:
                    try:
                        states = self.get_checkpoint_states('exception')
                        self._write_checkpoint_pair('interrupted_model.pth', 'interrupted_optim_sche.pth', states)
                        self.logger('Saved interrupted checkpoint at the last completed optimizer group; last_model remains unchanged.')
                    except Exception:
                        self.logger('Could not save the interrupted checkpoint:')
                        traceback.print_exc()
            raise

    def test(self):
        # Preview is a rank-zero operation. Full export shards complete batches
        # without padding, so no two ranks write the same sample name.
        if not self.config.args.sample_to_eval and not self.is_main_process:
            return
        dataset = get_dataset(self.config.data, 'test')
        sampler = None
        if self.config.training.use_DDP and self.config.args.sample_to_eval:
            sampler = DistributedEvaluationSampler(dataset, self.config.data.test.batch_size,
                                                    dist.get_rank(), dist.get_world_size(), drop_last=False)
        loader = DataLoader(dataset, batch_size=self.config.data.test.batch_size,
                            shuffle=False, sampler=sampler, num_workers=int(getattr(self.config.training, 'num_workers', 1)),
                            drop_last=False)
        with self.evaluation_scope() as model:
            if self.config.args.sample_to_eval:
                self.sample_to_eval(model, loader, self.config.result.sample_to_eval_path)
            else:
                batch = next(iter(loader), None)
                if batch is None:
                    raise ValueError('Inference dataset is empty')
                self.sample(model, batch, os.path.join(self.config.result.sample_path, '0'), stage='test')
