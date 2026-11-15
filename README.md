# OMIT-Diffusion

PyTorch implementation of **OMIT-Diffusion: An Efficient Shortcut Diffusion
Bridge Model for One-Step Medical Image-to-Image Translation**.

OMIT-Diffusion performs paired CT-to-MRI and MRI-to-CT translation for brain,
pelvis, and thorax. A transition-interval-conditioned UNet learns base and
shortcut objectives and supports one-step image generation.

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

The CUDA 11.3 wheel index is included in both installation files. The full
original dependency export is available in `environment-legacy.yml`.
Run the following commands from the repository root.

## Data preparation

The project uses three independent physical data splits: `train`, `val`, and
`test`. Set the data paths in the task YAML to match your files, using
patient/volume-level disjoint splits for the experiment being reproduced. The
test split must not point to the validation directories.

Prepare spatially aligned, **8-bit CT/MRI image pairs** using the original
intensity preprocessing. The dataset converts images to RGB, resizes them to
the configured square image size, and maps values to `[-1, 1]` when
`to_normal: true`. CT windowing, MRI intensity preprocessing, registration,
and volume slicing must be performed before loading the images. Raw high-bit-depth
integer or floating-point images are rejected before RGB conversion, with the
input path reported in the error. The loader rejects any train, val, or test
paths for the same modality that resolve to the same directory. Patient-level
disjointness must follow the original experiment's split list; it is not inferred
from slice filenames.

The default data paths follow this layout:

```text
dataset/<region>/train/CT/
dataset/<region>/train/MRI/
dataset/<region>/val/CT/
dataset/<region>/val/MRI/
dataset/<region>/test/CT/
dataset/<region>/test/MRI/
```

Files are paired by their filename stem after removing a terminal `_ct`, `_mr`,
or `_mri` suffix, ignoring the case of the suffix. For example,
`case001_slice010_ct.png` pairs with `case001_slice010_mri.png`. Each split must
contain unique, complete pairs. Training flips, when enabled in the config,
are applied to both images in each pair.

## Tasks and checkpoints

| Task | Config | Checkpoint directory |
| --- | --- | --- |
| Brain CT to MRI | `configs/brain_ct2mri.yaml` | `checkpoints/brain/ct2mri/` |
| Brain MRI to CT | `configs/brain_mri2ct.yaml` | `checkpoints/brain/mri2ct/` |
| Pelvis CT to MRI | `configs/pelvis_ct2mri.yaml` | `checkpoints/pelvis/ct2mri/` |
| Pelvis MRI to CT | `configs/pelvis_mri2ct.yaml` | `checkpoints/pelvis/mri2ct/` |
| Thorax CT to MRI | `configs/thorax_ct2mri.yaml` | `checkpoints/thorax/ct2mri/` |
| Thorax MRI to CT | `configs/thorax_mri2ct.yaml` | `checkpoints/thorax/mri2ct/` |

The checkpoints currently included in these directories are named
`inference_model.pth`. The inference/export runner requires the file to be
passed explicitly with `--resume_model`. The evaluation script can either use
`--checkpoint` explicitly or, when it is omitted, automatically load
`checkpoints/<region>/<direction>/inference_model.pth`. Match the region,
translation direction, and model configuration to the selected weights.

Training resume and the inference/export runner accept checkpoint dictionaries
containing `model` and optional `ema` entries. The evaluation script also
accepts a raw state dictionary. Historical checkpoints without generated
schedule buffers are supported. Learned parameter names and shapes must match
the configured model.

### Storing weights in Git

The `.gitattributes` file routes `.pth`, `.pt`, and `.ckpt` files under
`checkpoints/` through Git LFS. Install Git LFS before adding weight files:

```bash
git lfs install
git add .gitattributes checkpoints/
```

When cloning a repository containing LFS weights, run `git lfs pull` to obtain
the model files if they were not downloaded during checkout. GitHub Releases
can also be used to distribute checkpoints independently of the source tree.

## Training

```bash
python scripts/train.py -c configs/brain_ct2mri.yaml \
  --gpu_ids 0 --run_name baseline
```

Each new run uses its own output directory:

```text
results/<dataset_name>/<model_name>/<config_stem>/<run_id>/
```

The full output path is printed at startup. Model and optimizer checkpoints
are stored in its `checkpoint/` subdirectory. Use `--result_path` to change
the output root.

| Option | Purpose |
| --- | --- |
| `--gpu_ids 0,1` | Distributed training on the selected GPUs |
| `--gpu_ids -1` | CPU execution |
| `--num_workers 0` | Load data in the main process |
| `--max_steps 100` | Limit the cumulative number of processed microbatches |
| `--max_epoch 80` | Limit the total number of epochs |
| `--sample_at_start` | Generate a preview before training |
| `--save_top` | Save the best available full-validation loss checkpoint |

The default batch size of 4 realizes the 3:1 base/shortcut split. Gradient
accumulation is set by `training.accumulate_grad_batches`. Loss validation
requires complete batches and omits a final incomplete batch; image evaluation
and full export retain every sample. `sample_interval` and `validation_interval`
control preview and full-validation frequency in epochs. Best-checkpoint
selection occurs only when full validation runs. All six task configurations
run preview and full validation every epoch. Use `--save_top` to retain the
best validation checkpoint as well as the latest checkpoint.

### Resume training

Use the model and optimizer files from the same checkpoint save:

```bash
OMIT_RUN_DIR=/absolute/path/to/previous/run
python scripts/train.py -c configs/brain_ct2mri.yaml --gpu_ids 0 \
  --resume_model "$OMIT_RUN_DIR/checkpoint/last_model.pth" \
  --resume_optim "$OMIT_RUN_DIR/checkpoint/last_optim_sche.pth" \
  --resume_run_dir "$OMIT_RUN_DIR"
```

Omit `--resume_run_dir` to continue into a new output directory. Both
`--max_steps` and `--max_epoch` are cumulative limits, including completed work.
For weight initialization or fine-tuning, omit `--resume_optim` and set
`model.reset_resume_epoch: true` when the counters should restart.

If a failure occurs outside an optimizer update, the runner saves a separate
`interrupted_model.pth` / `interrupted_optim_sche.pth` pair at the last completed
update group. If an optimizer, scheduler, or EMA update fails, no new
interrupted pair is written: reload an existing matching pair from a
successful save. Pending gradients and RNG states are not serialized, so
resuming is not guaranteed to reproduce an uninterrupted random trajectory.

## Inference

A trained checkpoint is required. Missing weights or nonexistent checkpoint
paths are rejected before constructing the model. Replace the example path
with the actual filename for the task:

```bash
OMIT_CHECKPOINT="$PWD/checkpoints/brain/ct2mri/inference_model.pth"
python scripts/inference.py -c configs/brain_ct2mri.yaml \
  --resume_model "$OMIT_CHECKPOINT" --gpu_ids 0
```

This command generates a preview grid of up to four images. To export all
validation images individually:

```bash
python scripts/inference.py -c configs/brain_ct2mri.yaml \
  --resume_model "$OMIT_CHECKPOINT" --gpu_ids 0 --sample_to_eval
```

Preview sampling uses `model.OMIT.params.sample_step`; full export uses
`testing.shortcut_sample_steps`. Both default to one step. Inference reads the
independent test directories. When enabled, EMA weights are used for sampling;
a model-only checkpoint uses its loaded model weights.

## Evaluation

```bash
python scripts/evaluate.py --config configs/brain_ct2mri.yaml \
  --checkpoint "$OMIT_CHECKPOINT" --device cuda:0 \
  --weights auto --num_steps 1
```

`--weights auto` selects EMA when available, then model weights, then a raw
state dictionary. Use `--weights model` or `--weights ema` to select explicitly.
If `--checkpoint` is omitted, the evaluator looks for
`checkpoints/<region>/<direction>/inference_model.pth`.

Optional image export and a custom metrics output path:

```bash
python scripts/evaluate.py --config configs/brain_ct2mri.yaml \
  --checkpoint "$OMIT_CHECKPOINT" --device cuda:0 \
  --batch_size 16 --num_steps 1 \
  --save_predictions artifacts/predictions \
  --output artifacts/evaluation.json
```

The evaluator reports the arithmetic mean of per-slice MAE, MSE, PSNR, and
SSIM on images mapped to `[0, 1]`. SSIM uses an 11x11 Gaussian window with
sigma 1.5 and valid convolution. Without `--output`, each invocation creates
a unique JSON file in `artifacts/`. Evaluation and inference both use
`testing.clip_denoised`; the evaluation JSON records its actual value in
`metric_protocol.clip_denoised`.

Reported elapsed time includes data loading, transfer, sampling, metrics, and
optional image writing. It measures the end-to-end evaluation loop, rather
than isolated GPU inference latency.

## Model configuration

The provided configurations use pixel concatenation, FP32 computation,
transition-interval embeddings, and attention in the UNet middle block.
Sampling steps must be power-of-two divisors of `shortcut.denoise_timesteps`.
Supported objectives are `grad` and `ysubx`, with shortcut training enabled.
The model does not support the base-only switch, noise prediction, FP16,
SpatialTransformer conditioning, or class/codebook outputs.

## License

See [LICENSE](LICENSE) for the current permissions. Third-party components
and datasets remain subject to their respective licenses and terms. Upstream
attributions and permission texts are included in
[THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) and [licenses/](licenses/).
