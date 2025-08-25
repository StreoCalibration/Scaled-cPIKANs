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

class WaferPatchDataset(Dataset):
    """
    A PyTorch Dataset for loading wafer inspection data.

    This dataset handles both synthetic and real data by loading patches
    from bucket images and an optional ground truth height map. It supports
    on-the-fly patch extraction and data augmentation.
    """
    def __init__(self, data_dir: str, patch_size: int, use_augmentation: bool = False, real_data: bool = False):
        """
        Args:
            data_dir (str): Path to the directory containing the data samples.
                            (e.g., 'synthetic_data/train')
            patch_size (int): The size (width and height) of the patches to extract.
            use_augmentation (bool): Whether to apply data augmentation.
            real_data (bool): If True, assumes no ground truth is available.
        """
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.use_augmentation = use_augmentation
        self.real_data = real_data

        self.sample_paths = sorted(glob.glob(os.path.join(self.data_dir, "sample_*")))
        if not self.sample_paths:
            raise FileNotFoundError(f"No samples found in directory: {self.data_dir}")

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        sample_path = self.sample_paths[idx]

        # Load bucket images (input)
        bucket_images_path = os.path.join(sample_path, "bucket_images.npy")
        bucket_images = np.load(bucket_images_path).astype(np.float32) # Shape: (12, H, W)

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


class PinnPatchDataset(Dataset):
    """
    A PyTorch Dataset for loading wafer data for Physics-Informed models.

    This dataset provides coordinate grids as input for a PINN model,
    and bucket image patches as the target for the loss function. It also
    provides the ground truth height map for pre-training.
    """
    def __init__(self, data_dir: str, patch_size: int, full_image_size: tuple[int, int], real_data: bool = False):
        """
        Args:
            data_dir (str): Path to the directory containing data samples.
            patch_size (int): The size of the patches to extract.
            full_image_size (tuple[int, int]): The size (H, W) of the original full-resolution images.
            real_data (bool): If True, assumes no ground truth height map is available.
        """
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.H, self.W = full_image_size
        self.real_data = real_data

        self.sample_paths = sorted(glob.glob(os.path.join(self.data_dir, "sample_*")))
        if not self.sample_paths:
            raise FileNotFoundError(f"No samples found in directory: {self.data_dir}")

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        sample_path = self.sample_paths[idx]

        # Load bucket images (target for loss)
        bucket_images_path = os.path.join(sample_path, "bucket_images.npy")
        bucket_images = np.load(bucket_images_path).astype(np.float32) # Shape: (C, H, W)

        # Load ground truth height map (target for pre-training)
        if not self.real_data:
            ground_truth_path = os.path.join(sample_path, "ground_truth.npy")
            ground_truth = np.load(ground_truth_path).astype(np.float32) # Shape: (H, W)
        else:
            ground_truth = np.zeros((self.H, self.W), dtype=np.float32)

        # --- Patch Extraction and Coordinate Generation ---
        img_H, img_W = bucket_images.shape[1], bucket_images.shape[2]
        if img_H < self.patch_size or img_W < self.patch_size:
            raise ValueError(f"Image size ({img_H}, {img_W}) is smaller than patch size ({self.patch_size}).")

        # Get random top-left corner for the patch
        top = np.random.randint(0, img_H - self.patch_size + 1)
        left = np.random.randint(0, img_W - self.patch_size + 1)

        # Extract patches
        bucket_patch = bucket_images[:, top:top+self.patch_size, left:left+self.patch_size]
        gt_patch = ground_truth[top:top+self.patch_size, left:left+self.patch_size]

        # Generate coordinates for the patch, normalized to [0, 1]
        # These coordinates correspond to the patch's position in the *full* image
        x = torch.linspace(left / self.W, (left + self.patch_size - 1) / self.W, self.patch_size)
        y = torch.linspace(top / self.H, (top + self.patch_size - 1) / self.H, self.patch_size)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1) # Shape: (patch_size*patch_size, 2)

        # --- Reshape for model and loss ---
        # Reshape bucket patch for comparison with model output
        # (C, H, W) -> (C, H*W)
        bucket_patch_flat = torch.from_numpy(bucket_patch).reshape(bucket_patch.shape[0], -1)

        # Reshape ground truth patch for pre-training loss
        # (H, W) -> (1, H*W)
        gt_patch_flat = torch.from_numpy(gt_patch).reshape(1, -1)

        return coords, bucket_patch_flat, gt_patch_flat
