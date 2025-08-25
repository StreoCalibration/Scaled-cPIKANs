import torch
from scipy.stats import qmc

def affine_scale(x: torch.Tensor, domain_min: torch.Tensor, domain_max: torch.Tensor) -> torch.Tensor:
    """
    텐서를 물리적 도메인 [domain_min, domain_max]에서 표준 도메인 [-1, 1]으로 스케일링합니다.

    Args:
        x (torch.Tensor): 물리적 도메인에 있는 입력 텐서.
        domain_min (torch.Tensor): 물리적 도메인의 최솟값.
        domain_max (torch.Tensor): 물리적 도메인의 최댓값.

    Returns:
        torch.Tensor: [-1, 1] 도메인으로 스케일링된 텐서.
    """
    return 2.0 * (x - domain_min) / (domain_max - domain_min) - 1.0

def affine_unscale(x_scaled: torch.Tensor, domain_min: torch.Tensor, domain_max: torch.Tensor) -> torch.Tensor:
    """
    텐서를 표준 도메인 [-1, 1]에서 다시 물리적 도메인 [domain_min, domain_max]으로 역스케일링합니다.

    Args:
        x_scaled (torch.Tensor): [-1, 1] 도메인에 있는 입력 텐서.
        domain_min (torch.Tensor): 물리적 도메인의 최솟값.
        domain_max (torch.Tensor): 물리적 도메인의 최댓값.

    Returns:
        torch.Tensor: 물리적 도메인으로 역스케일링된 텐서.
    """
    return (x_scaled + 1.0) / 2.0 * (domain_max - domain_min) + domain_min


class LatinHypercubeSampler:
    """
    라틴 하이퍼큐브 샘플링(LHS)을 사용하여 포인트를 생성하는 샘플러.

    LHS는 순수 무작위 샘플링에 비해 도메인 전체에 걸쳐 샘플 포인트가
    더 균일하게 분포되도록 보장하는 계층화된 샘플링 기법입니다.

    Args:
        n_points (int): 생성할 샘플 포인트의 수.
        domain_min (list[float] or torch.Tensor): 각 차원에 대한 샘플링 도메인의 하한.
        domain_max (list[float] or torch.Tensor): 각 차원에 대한 샘플링 도메인의 상한.
        dtype (torch.dtype, optional): 출력 텐서의 원하는 데이터 타입. 기본값은 torch.float32.
        device (torch.device, optional): 출력 텐서의 원하는 장치. 기본값은 'cpu'.
    """
    def __init__(self, n_points: int, domain_min: list[float], domain_max: list[float], dtype: torch.dtype = torch.float32, device: torch.device = 'cpu'):
        if len(domain_min) != len(domain_max):
            raise ValueError("domain_min과 domain_max는 같은 길이를 가져야 합니다.")

        self.n_points = n_points
        self.dimensions = len(domain_min)
        self.sampler = qmc.LatinHypercube(d=self.dimensions, seed=None) # 매번 무작위 시드 사용

        self.domain_min = torch.tensor(domain_min, dtype=dtype, device=device)
        self.domain_max = torch.tensor(domain_max, dtype=dtype, device=device)
        self.dtype = dtype
        self.device = device

    def sample(self) -> torch.Tensor:
        """
        샘플 포인트를 생성합니다.

        Returns:
            torch.Tensor: 샘플링된 포인트를 포함하는 (n_points, dimensions) 크기의 텐서.
        """
        # 단위 하이퍼큐브 [0, 1]^d에서 포인트 샘플링
        unit_samples = self.sampler.random(n=self.n_points)
        unit_samples_tensor = torch.tensor(unit_samples, dtype=self.dtype, device=self.device)

        # 샘플을 물리적 도메인으로 스케일링
        samples = self.domain_min + (self.domain_max - self.domain_min) * unit_samples_tensor

        return samples


import os
import numpy as np
from torch.utils.data import Dataset
import glob

from PIL import Image

class WaferPatchDataset(Dataset):
    """
    A PyTorch Dataset for loading wafer inspection data.

    This dataset handles both synthetic and real data by loading patches
    from bucket images and an optional ground truth height map. It supports
    on-the-fly patch extraction and data augmentation.
    """
    def __init__(self, data_dir: str, patch_size: int, use_augmentation: bool = False, real_data: bool = False, num_channels: int = 12, output_format: str = 'bmp'):
        """
        Args:
            data_dir (str): Path to the directory containing the data samples.
            patch_size (int): The size (width and height) of the patches to extract.
            use_augmentation (bool): Whether to apply data augmentation.
            real_data (bool): If True, assumes no ground truth is available.
            num_channels (int): Number of input channels (bucket images).
            output_format (str): Format of bucket images ('bmp', 'png').
        """
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.use_augmentation = use_augmentation
        self.real_data = real_data
        self.num_channels = num_channels
        self.output_format = output_format

        self.sample_paths = sorted(glob.glob(os.path.join(self.data_dir, "sample_*")))
        if not self.sample_paths:
            raise FileNotFoundError(f"No samples found in directory: {self.data_dir}")

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        sample_path = self.sample_paths[idx]

        # --- Load Bucket Images based on format ---
        if self.output_format in ['bmp', 'png']:
            img_paths = sorted(glob.glob(os.path.join(sample_path, f"bucket_*.{self.output_format}")))
            if not img_paths:
                 raise FileNotFoundError(f"No bucket images found in {sample_path} with format {self.output_format}")
            if len(img_paths) != self.num_channels:
                print(f"Warning: Expected {self.num_channels} images, but found {len(img_paths)} in {sample_path}")

            images = [np.array(Image.open(p), dtype=np.float32) for p in img_paths]
            bucket_images = np.stack(images, axis=0) # Shape: (C, H, W)
        else:
            # Kept for compatibility if ever needed, but not the default.
            bucket_images_path = os.path.join(sample_path, "bucket_images.npy")
            bucket_images = np.load(bucket_images_path).astype(np.float32)

        # Load ground truth height map (target) if available
        if not self.real_data:
            ground_truth_path = os.path.join(sample_path, "ground_truth.npy")
            ground_truth = np.load(ground_truth_path).astype(np.float32) # Shape: (H, W)
        else:
            ground_truth = None # No ground truth for real data

        # --- Patch Extraction ---
        _, H, W = bucket_images.shape
        if H < self.patch_size or W < self.patch_size:
            raise ValueError(f"Image size ({H}, {W}) is smaller than patch size ({self.patch_size}).")

        # Get random top-left corner for the patch
        top = np.random.randint(0, H - self.patch_size + 1)
        left = np.random.randint(0, W - self.patch_size + 1)

        # Extract patch from bucket images
        input_patch = bucket_images[:, top:top+self.patch_size, left:left+self.patch_size]

        # Extract patch from ground truth
        if ground_truth is not None:
            target_patch = ground_truth[top:top+self.patch_size, left:left+self.patch_size]
            # Add channel dimension to ground truth
            target_patch = np.expand_dims(target_patch, axis=0) # Shape: (1, patch_size, patch_size)
        else:
            # For real data, we might not have a target, but can return a dummy one
            target_patch = np.zeros((1, self.patch_size, self.patch_size), dtype=np.float32)

        # --- Data Augmentation ---
        if self.use_augmentation:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                input_patch = np.ascontiguousarray(input_patch[:, :, ::-1])
                target_patch = np.ascontiguousarray(target_patch[:, :, ::-1])

            # Random vertical flip
            if np.random.rand() > 0.5:
                input_patch = np.ascontiguousarray(input_patch[:, ::-1, :])
                target_patch = np.ascontiguousarray(target_patch[:, ::-1, :])

            # Random 90-degree rotation
            k = np.random.randint(0, 4)
            input_patch = np.rot90(input_patch, k, axes=(1, 2))
            target_patch = np.rot90(target_patch, k, axes=(1, 2))

        # --- Convert to Tensor ---
        input_tensor = torch.from_numpy(np.ascontiguousarray(input_patch))
        target_tensor = torch.from_numpy(np.ascontiguousarray(target_patch))

        return input_tensor, target_tensor


from PIL import Image

class PinnPatchDataset(Dataset):
    """
    A PyTorch Dataset for loading wafer data for Physics-Informed models.
    This dataset loads the entire image into memory and provides random patches.
    It is designed for training on a single, large image sample.
    """
    def __init__(self, data_dir: str, patch_size: int,
                 output_format: str = 'npy', real_data: bool = False,
                 num_channels: int = 12, epoch_length: int = 1000):
        """
        Args:
            data_dir (str): Path to the directory containing the data files
                            (e.g., bucket_images.npy, ground_truth.npy).
            patch_size (int): The size of the patches to extract.
            output_format (str): The format of the bucket images ('npy', 'bmp', 'png').
            real_data (bool): If True, assumes no ground truth height map is available.
            num_channels (int): The number of bucket images (e.g., 4 lasers * 3 buckets).
            epoch_length (int): The number of random patches to yield per "epoch".
        """
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.output_format = output_format
        self.real_data = real_data
        self.num_channels = num_channels
        self.epoch_length = epoch_length

        # --- Load all data into RAM once ---
        self._load_data()

        self.H, self.W = self.bucket_images.shape[1:]
        if self.H < self.patch_size or self.W < self.patch_size:
            raise ValueError(f"Image size ({self.H}, {self.W}) is smaller than patch size ({self.patch_size}).")

    def _load_data(self):
        """Loads the bucket images and ground truth from disk into memory."""
        if self.output_format == 'npy':
            bucket_images_path = os.path.join(self.data_dir, "bucket_images.npy")
            if not os.path.exists(bucket_images_path):
                 raise FileNotFoundError(f"bucket_images.npy not found in {self.data_dir}")
            self.bucket_images = np.load(bucket_images_path).astype(np.float32)
        elif self.output_format in ['bmp', 'png']:
            img_paths = sorted(glob.glob(os.path.join(self.data_dir, f"bucket_*.{self.output_format}")))
            if not img_paths:
                 raise FileNotFoundError(f"No bucket images found in {self.data_dir} with format {self.output_format}")
            if len(img_paths) != self.num_channels:
                print(f"Warning: Expected {self.num_channels} images, but found {len(img_paths)} in {self.data_dir}")
            images = [np.array(Image.open(p), dtype=np.float32) for p in img_paths]
            self.bucket_images = np.stack(images, axis=0) # Shape: (C, H, W)
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")

        # --- Load ground truth height map (always .npy) ---
        if not self.real_data:
            ground_truth_path = os.path.join(self.data_dir, "ground_truth.npy")
            if not os.path.exists(ground_truth_path):
                # For inference, ground truth might not exist, which is fine.
                if self.real_data:
                    self.ground_truth = None
                else:
                    raise FileNotFoundError(f"ground_truth.npy not found in {self.data_dir} but real_data is False.")
            else:
                self.ground_truth = np.load(ground_truth_path).astype(np.float32) # Shape: (H, W)
        else:
            self.ground_truth = None

        if self.ground_truth is None:
            # Use placeholder if no real ground truth is available
            img_H, img_W = self.bucket_images.shape[1:]
            self.ground_truth = np.zeros((img_H, img_W), dtype=np.float32)


    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
        # idx is ignored, we always return a random patch.

        # Get random top-left corner for the patch
        top = np.random.randint(0, self.H - self.patch_size + 1)
        left = np.random.randint(0, self.W - self.patch_size + 1)

        # Extract patches from in-memory arrays
        bucket_patch = self.bucket_images[:, top:top+self.patch_size, left:left+self.patch_size]
        gt_patch = self.ground_truth[top:top+self.patch_size, left:left+self.patch_size]

        # Generate coordinates for the patch, normalized to [0, 1]
        x = torch.linspace(left / self.W, (left + self.patch_size - 1) / self.W, self.patch_size)
        y = torch.linspace(top / self.H, (top + self.patch_size - 1) / self.H, self.patch_size)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

        # --- Reshape for model and loss ---
        bucket_patch_flat = torch.from_numpy(bucket_patch).reshape(self.num_channels, -1)
        gt_patch_flat = torch.from_numpy(gt_patch).reshape(1, -1)

        return coords, bucket_patch_flat, gt_patch_flat
