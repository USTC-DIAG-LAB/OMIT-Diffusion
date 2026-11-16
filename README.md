# OMIT-Diffusion

PyTorch implementation of **OMIT-Diffusion: An Efficient Shortcut Diffusion
Bridge Model for One-Step Medical Image-to-Image Translation**.

OMIT-Diffusion performs paired CT-to-MRI and MRI-to-CT translation for brain,
pelvis, and thorax. A transition-interval-conditioned UNet combines base and
shortcut supervision to support one-step image generation.

## Installation

The provided environment uses Python 3.9.16, PyTorch 1.12.1 with CUDA 11.3,
and torchvision 0.13.1.

```bash
conda env create -f environment.yml
conda activate OMIT
```

Alternatively, install the pinned dependencies in a compatible Python environment:

```bash
python -m pip install -r requirements.txt
```

Run the commands below from the repository root.

## Datasets and preparation

The experiments follow Section IV-A of the paper. Brain and pelvis data are
from the **SynthRAD2023 Challenge**, and thorax data are from the
**SynthRAD2025 Grand Challenge**. Both translation directions are evaluated
for each region, giving six tasks in total.

Paired CT/MRI volumes are randomly split into **80% training and 20% held-out
evaluation data**, with a fixed split seed of **42**. The split is performed
at the volume level before slicing, so all slices from one volume pair belong
to the same partition. Both translation directions use the same partition
for a given region. The brain and pelvis datasets each contain 180 paired
volumes. The slice-pair counts reported in the paper are:

| Region | Dataset | Training slice pairs | Test slice pairs |
| --- | --- | ---: | ---: |
| Brain | SynthRAD2023 | 24,191 | 6,911 |
| Pelvis | SynthRAD2023 | 16,812 | 4,203 |
| Thorax | SynthRAD2025 | 13,870 | 3,468 |

The 80/20 ratio applies to volumes; the resulting slice counts need not have
exactly that ratio. Preserve the original volume membership when reproducing
the reported results.

There are **two partitions**, `train` and `test`, throughout the configurations,
dataset loader, and runner. The `test` partition is the held-out set described
in the paper; it is not used for parameter updates, hyperparameter tuning, or
checkpoint selection. Training and test volumes must remain disjoint.

### Image format

Prepare spatially aligned, **8-bit CT/MRI slice pairs** using the experiment's
original intensity preprocessing. The loader converts the images to RGB,
resizes each slice to **256 x 256**, and maps intensities to **[-1, 1]** for
training. For evaluation, predictions and ground-truth images are rescaled
to **[0, 1]**.

Registration, CT/MRI intensity preprocessing, and volume slicing are performed
before loading. The loader does not implement CT windowing or MRI-specific
intensity preprocessing and does not accept raw high-bit-depth medical images.

Arrange the prepared slices as follows, replacing `<region>` with `brain`,
`pelvis`, or `thorax`:

```text
dataset/<region>/train/CT/
dataset/<region>/train/MRI/
dataset/<region>/test/CT/
dataset/<region>/test/MRI/
```

Update `data.dataset_config` in the corresponding task YAML if your data are
stored elsewhere. Set `train_ct_dir`, `train_mri_dir`, `test_ct_dir`, and
`test_mri_dir` to the two paired partitions. Existing held-out folders can be
used directly through the `test_*_dir` fields, regardless of their physical
directory names; preserve their original volume membership. The loader checks
directory separation; volume membership is established when preparing the split.

Files are paired by their filename stem after removing a terminal `_ct`, `_mr`,
or `_mri` suffix, ignoring the case of the suffix. For example,
`case001_slice010_ct.png` pairs with `case001_slice010_mri.png`. Each partition
must contain unique, complete pairs.

## Tasks and pretrained models

Each task uses one configuration for training, inference, and evaluation.
The pretrained checkpoint in each directory below is named `inference_model.pth`.

| Task | Configuration | Checkpoint directory |
| --- | --- | --- |
| Brain CT to MRI | `configs/brain_ct2mri.yaml` | `checkpoints/brain/ct2mri/` |
| Brain MRI to CT | `configs/brain_mri2ct.yaml` | `checkpoints/brain/mri2ct/` |
| Pelvis CT to MRI | `configs/pelvis_ct2mri.yaml` | `checkpoints/pelvis/ct2mri/` |
| Pelvis MRI to CT | `configs/pelvis_mri2ct.yaml` | `checkpoints/pelvis/mri2ct/` |
| Thorax CT to MRI | `configs/thorax_ct2mri.yaml` | `checkpoints/thorax/ct2mri/` |
| Thorax MRI to CT | `configs/thorax_mri2ct.yaml` | `checkpoints/thorax/mri2ct/` |

Weights are stored with Git LFS. With Git LFS installed, download the weight
files after cloning the repository:

```bash
git lfs install
git lfs pull
```

Match the region, translation direction, and model configuration to the
selected checkpoint. The examples below use brain CT-to-MRI.

## Inference

Generate and export images for the complete held-out set:

```bash
python scripts/inference.py -c configs/brain_ct2mri.yaml \
  --resume_model checkpoints/brain/ct2mri/inference_model.pth \
  --gpu_ids 0 --sample_to_eval
```

The output directory is printed at startup. Omit `--sample_to_eval` to generate
a preview grid of up to four images. Full export uses
`testing.shortcut_sample_steps`; preview sampling uses
`model.OMIT.params.sample_step`. Both default to **one step**. When available
and enabled, EMA weights are used; a model-only checkpoint uses its loaded
model weights.

## Evaluation

Evaluate the complete held-out set using MAE, MSE, PSNR, and SSIM:

```bash
python scripts/evaluate.py --config configs/brain_ct2mri.yaml \
  --checkpoint checkpoints/brain/ct2mri/inference_model.pth \
  --device cuda:0 --weights auto --num_steps 1
```

`--weights auto` selects EMA weights when available, then model weights, then
a raw state dictionary. Use `--weights model` or `--weights ema` to select
explicitly. If `--checkpoint` is omitted, the evaluator uses the configuration
filename to locate `checkpoints/<region>/<direction>/inference_model.pth`.

To save predictions and choose the metrics output file:

```bash
python scripts/evaluate.py --config configs/brain_ct2mri.yaml \
  --checkpoint checkpoints/brain/ct2mri/inference_model.pth \
  --device cuda:0 --num_steps 1 \
  --save_predictions artifacts/predictions \
  --output artifacts/evaluation.json
```

Metrics are arithmetic means of per-slice values, computed on predictions
and ground-truth images mapped to `[0, 1]`. SSIM uses an 11x11 Gaussian window
with sigma 1.5 and valid convolution. Evaluation and image export use
`testing.clip_denoised`. The JSON records the metric settings and uses `test`
as the stage name for the held-out set. Without `--output`, each invocation
creates a unique JSON file in `artifacts/`.

The evaluator's elapsed time includes loading, transfer, sampling, metrics,
and optional image writing. This end-to-end time is distinct from isolated
GPU inference latency.

## Training

The paper trains OMIT-Diffusion for **80 epochs** with a learning rate of
**2e-6**, EMA decay **0.995**, and a shortcut-supervision fraction of **1/4**.
The shortcut training grid has **128 intervals**; this is separate from the
one-step inference setting.

```bash
python scripts/train.py -c configs/brain_ct2mri.yaml \
  --gpu_ids 0 --run_name paper
```

The supplied configurations use a microbatch size of 4. Gradient accumulation
is controlled by `training.accumulate_grad_batches`. The CLI `--seed` controls
training randomness; it does not create the volume split described above.
Evaluate the final checkpoint after the prescribed training schedule.
Checkpoints are saved by the epoch/step schedule, without ranking test losses.

During training, test losses and preview images are recorded for monitoring
only. They do not update parameters, change the learning rate, stop training,
or select checkpoints. `training.test_loss_interval` controls epoch-level loss
logging. The same test partition uses `data.test.loss_batch_size: 4` for these
loss computations and `data.test.batch_size` for inference and image metrics
(16 for brain/pelvis, 8 for thorax). Loss logging requires complete batches
and may omit the final incomplete batch; standalone evaluation and image
export include every test slice.

Each new run writes to:

```text
results/<dataset_name>/<model_name>/<config_stem>/<run_id>/
```

The full path is printed at startup, with model and optimizer files in the
`checkpoint/` subdirectory. Use `--result_path` to change the output root,
`--gpu_ids 0,1` for distributed training, and `--help` for other options.

### Resume training

Use the model and optimizer files from the same checkpoint save:

```bash
OMIT_RUN_DIR=/absolute/path/to/previous/run
python scripts/train.py -c configs/brain_ct2mri.yaml --gpu_ids 0 \
  --resume_model "$OMIT_RUN_DIR/checkpoint/last_model.pth" \
  --resume_optim "$OMIT_RUN_DIR/checkpoint/last_optim_sche.pth" \
  --resume_run_dir "$OMIT_RUN_DIR"
```

Omit `--resume_run_dir` to continue into a new output directory. The
`--max_steps` and `--max_epoch` limits include completed work. RNG states and
pending gradients are not serialized, so resumed training is not guaranteed
to reproduce an uninterrupted trajectory exactly.

## License

See [LICENSE](LICENSE) for the current permissions. Third-party components
and datasets remain subject to their respective licenses and terms. Upstream
attributions and permission texts are included in
[THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) and [licenses/](licenses/).
