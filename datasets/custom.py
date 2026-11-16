"""Paired CT/MRI slice datasets for the six supported translation tasks."""

import re
from numbers import Integral
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from Register import Registers


IMAGE_EXTENSIONS = frozenset({
    '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp',
    '.ppm', '.pgm', '.pbm',
})


def _pair_ct_mri_paths(ct_dir, mri_dir):
    """Pair image files by ID, ignoring modality suffixes and non-image files."""
    def pair_key(path):
        return re.sub(r'_(?:ct|mr|mri)$', '', path.stem, flags=re.IGNORECASE)

    def index_directory(directory):
        indexed = {}
        for path in Path(directory).iterdir():
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            key = pair_key(path)
            if key in indexed:
                raise ValueError(f"Duplicate paired-image identifier {key!r} in {directory}")
            indexed[key] = path
        return indexed

    ct_by_key = index_directory(ct_dir)
    mri_by_key = index_directory(mri_dir)
    if set(ct_by_key) != set(mri_by_key):
        ct_only = sorted(set(ct_by_key) - set(mri_by_key))[:5]
        mri_only = sorted(set(mri_by_key) - set(ct_by_key))[:5]
        raise ValueError(
            f"Unpaired CT/MRI identifiers in {ct_dir} and {mri_dir}; "
            f"CT-only examples={ct_only}, MRI-only examples={mri_only}"
        )
    pairs = [(ct_by_key[key], mri_by_key[key]) for key in sorted(ct_by_key)]
    if not pairs:
        raise ValueError(f"No paired CT/MRI image files found in {ct_dir} and {mri_dir}")
    return pairs


class _CTMRIPairedDirsDataset(Dataset):
    """Paired slices with separate ``train`` and ``test`` directories.

    Each partition requires a complete CT/MRI directory pair. Test data are
    held out from training; there is no additional dataset partition.

    Input images must be 8-bit slices with the experiment's intensity preprocessing.
    RGB conversion, resizing, tensor conversion, and optional normalization
    retain the original training behavior.
    """

    target_modality = None

    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        if stage not in ('train', 'test'):
            raise ValueError(f"Unknown dataset stage {stage!r}; expected train or test")
        max_samples = getattr(dataset_config, f'{stage}_max_samples', None)
        if max_samples is not None and (
            isinstance(max_samples, bool)
            or not isinstance(max_samples, Integral)
            or max_samples <= 0
        ):
            raise ValueError(f"{stage}_max_samples must be a positive integer or None")

        split_dirs = {}
        for split_name in ('train', 'test'):
            for modality in ('ct', 'mri'):
                field = f'{split_name}_{modality}_dir'
                value = getattr(dataset_config, field, None)
                if value:
                    split_dirs[(split_name, modality)] = Path(value).expanduser().resolve()

        for modality in ('ct', 'mri'):
            train_path = split_dirs.get(('train', modality))
            if train_path is None:
                continue
            test_path = split_dirs.get(('test', modality))
            if test_path == train_path:
                raise ValueError(
                    f'test_{modality}_dir and train_{modality}_dir '
                    f'refer to the same directory: {test_path}. Training and '
                    'held-out evaluation directories must be separate.'
                )

        missing = [
            f'{stage}_{modality}_dir'
            for modality in ('ct', 'mri')
            if (stage, modality) not in split_dirs
        ]
        if missing:
            raise ValueError(
                f'Missing dataset path(s) for stage {stage!r}: {missing}'
            )

        ct_dir = split_dirs[(stage, 'ct')]
        mri_dir = split_dirs[(stage, 'mri')]
        self.pairs = _pair_ct_mri_paths(ct_dir, mri_dir)
        if max_samples is not None:
            self.pairs = self.pairs[:int(max_samples)]
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.flip = bool(dataset_config.flip) if stage == 'train' else False
        self.to_normal = bool(dataset_config.to_normal)
        self._length = len(self.pairs)
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self._length * 2 if self.flip else self._length

    def _load(self, path, do_flip):
        with Image.open(path) as source:
            if source.mode in ('I', 'F') or source.mode.startswith('I;'):
                raise ValueError(
                    f"High-bit-depth image {str(path)!r} (mode={source.mode!r}) cannot "
                    "be converted to RGB without changing its intensity values. "
                    "Apply the experiment's original CT/MRI intensity preprocessing "
                    "and save an 8-bit slice before loading it; do not pass raw "
                    "16-bit, integer or floating-point medical images directly."
                )
            with source.convert('RGB') as image:
                if do_flip:
                    with image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) as flipped:
                        tensor = self.transform(flipped)
                else:
                    tensor = self.transform(image)
        if self.to_normal:
            tensor = (tensor - 0.5) * 2.0
            tensor.clamp_(-1.0, 1.0)
        return tensor

    def __getitem__(self, index):
        do_flip = index >= self._length
        index %= self._length
        ct_path, mri_path = self.pairs[index]
        ct = self._load(ct_path, do_flip)
        mri = self._load(mri_path, do_flip)
        name = ct_path.stem
        if self.target_modality == 'MRI':
            return (mri, name), (ct, name)
        return (ct, name), (mri, name)


@Registers.datasets.register_with_name('ct_mri_paired_dirs_ct2mri')
class CTMRIPairedDirsCT2MRIDataset(_CTMRIPairedDirsDataset):
    target_modality = 'MRI'


@Registers.datasets.register_with_name('ct_mri_paired_dirs_mri2ct')
class CTMRIPairedDirsMRI2CTDataset(_CTMRIPairedDirsDataset):
    target_modality = 'CT'


# Backward-compatible alias used by early CT-to-MRI experiment configs.
if 'ct_mri_paired_dirs' not in Registers.datasets:
    Registers.datasets.register_with_name('ct_mri_paired_dirs')(CTMRIPairedDirsCT2MRIDataset)
