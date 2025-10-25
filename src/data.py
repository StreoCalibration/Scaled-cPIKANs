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


class AdaptiveResidualSampler:
    """
    잔차 기반 적응형 콜로케이션 포인트 샘플러.
    
    PDE 잔차가 큰 영역에 더 많은 샘플링 포인트를 배치하여
    훈련 효율성을 향상시킵니다. 이는 "adaptive collocation" 또는
    "residual-based sampling" 기법으로 알려져 있습니다.
    
    작동 원리:
    1. 초기 샘플 포인트 세트 생성 (Latin Hypercube Sampling)
    2. 모델의 PDE 잔차를 각 포인트에서 계산
    3. 잔차가 큰 영역 주변에 새로운 포인트 추가
    4. 주기적으로 반복하여 샘플 분포 개선
    
    Args:
        n_initial_points (int): 초기 샘플 포인트 수
        n_max_points (int): 최대 샘플 포인트 수
        domain_min (list[float]): 각 차원의 도메인 최솟값
        domain_max (list[float]): 각 차원의 도메인 최댓값
        refinement_ratio (float): 각 반복에서 추가할 포인트 비율 (0~1)
        residual_threshold_percentile (float): 잔차 임계값 백분위수 (0~100)
        dtype (torch.dtype): 출력 텐서 데이터 타입
        device (torch.device): 출력 텐서 장치
    """
    
    def __init__(
        self,
        n_initial_points: int,
        n_max_points: int,
        domain_min: list[float],
        domain_max: list[float],
        refinement_ratio: float = 0.2,
        residual_threshold_percentile: float = 75.0,
        dtype: torch.dtype = torch.float32,
        device: torch.device = 'cpu'
    ):
        self.n_initial_points = n_initial_points
        self.n_max_points = n_max_points
        self.domain_min = torch.tensor(domain_min, dtype=dtype, device=device)
        self.domain_max = torch.tensor(domain_max, dtype=dtype, device=device)
        self.refinement_ratio = refinement_ratio
        self.residual_threshold_percentile = residual_threshold_percentile
        self.dtype = dtype
        self.device = device
        self.dimensions = len(domain_min)
        
        # 초기 샘플 생성
        self.lhs_sampler = LatinHypercubeSampler(
            n_points=n_initial_points,
            domain_min=domain_min,
            domain_max=domain_max,
            dtype=dtype,
            device=device
        )
        
        # 현재 샘플 포인트
        self.current_points = self.lhs_sampler.sample()
        self.residuals = None
    
    def get_current_points(self) -> torch.Tensor:
        """
        현재 샘플 포인트를 반환합니다.
        
        Returns:
            torch.Tensor: (n_points, d) 형태의 샘플 포인트
        """
        return self.current_points
    
    def update_residuals(self, residuals: torch.Tensor):
        """
        각 포인트에서의 PDE 잔차를 업데이트합니다.
        
        Args:
            residuals (torch.Tensor): (n_points, *) 형태의 잔차 텐서
                                     각 포인트에서의 PDE 잔차 값
        """
        # 잔차의 절대값 또는 노름 계산
        if residuals.ndim > 1:
            # 다차원 잔차인 경우 L2 노름 사용
            self.residuals = torch.norm(residuals, dim=tuple(range(1, residuals.ndim)), keepdim=False)
        else:
            self.residuals = torch.abs(residuals)
    
    def refine(self) -> bool:
        """
        잔차가 큰 영역에 새로운 포인트를 추가합니다.
        
        Returns:
            bool: 정제가 수행되었으면 True, 최대 포인트 수에 도달했으면 False
        """
        if self.residuals is None:
            raise ValueError("update_residuals()를 먼저 호출하여 잔차를 설정해야 합니다.")
        
        n_current = self.current_points.shape[0]
        
        # 최대 포인트 수에 도달했는지 확인
        if n_current >= self.n_max_points:
            return False
        
        # 추가할 포인트 수 계산
        n_to_add = min(
            int(n_current * self.refinement_ratio),
            self.n_max_points - n_current
        )
        
        if n_to_add == 0:
            return False
        
        # 잔차 임계값 계산
        residual_threshold = torch.quantile(
            self.residuals,
            self.residual_threshold_percentile / 100.0
        )
        
        # 높은 잔차를 가진 포인트 선택
        high_residual_mask = self.residuals >= residual_threshold
        high_residual_points = self.current_points[high_residual_mask]
        
        if len(high_residual_points) == 0:
            # 모든 잔차가 비슷한 경우, 임의로 포인트 추가
            new_points = self._sample_random_points(n_to_add)
        else:
            # 높은 잔차 포인트 주변에 새 포인트 생성
            new_points = self._sample_around_points(high_residual_points, n_to_add)
        
        # 새 포인트 추가
        self.current_points = torch.cat([self.current_points, new_points], dim=0)
        
        # 잔차 초기화 (다음 업데이트까지)
        self.residuals = None
        
        return True
    
    def _sample_random_points(self, n_points: int) -> torch.Tensor:
        """
        도메인 내에서 임의 포인트를 샘플링합니다.
        
        Args:
            n_points (int): 샘플링할 포인트 수
            
        Returns:
            torch.Tensor: (n_points, d) 형태의 새 포인트
        """
        # 균일 분포 샘플링
        random_samples = torch.rand(n_points, self.dimensions, dtype=self.dtype, device=self.device)
        scaled_samples = self.domain_min + random_samples * (self.domain_max - self.domain_min)
        return scaled_samples
    
    def _sample_around_points(self, center_points: torch.Tensor, n_points: int) -> torch.Tensor:
        """
        주어진 포인트 주변에 새 포인트를 샘플링합니다.
        
        Args:
            center_points (torch.Tensor): (n_centers, d) 형태의 중심 포인트
            n_points (int): 샘플링할 포인트 수
            
        Returns:
            torch.Tensor: (n_points, d) 형태의 새 포인트
        """
        # 각 중심 포인트 주변에 가우시안 노이즈 추가
        n_centers = center_points.shape[0]
        
        # 중심 포인트를 순환적으로 선택
        selected_centers = center_points[torch.randint(0, n_centers, (n_points,))]
        
        # 도메인 크기의 일정 비율만큼의 표준편차로 가우시안 노이즈 추가
        domain_range = self.domain_max - self.domain_min
        noise_std = 0.1 * domain_range  # 도메인 범위의 10%
        
        noise = torch.randn(n_points, self.dimensions, dtype=self.dtype, device=self.device) * noise_std
        new_points = selected_centers + noise
        
        # 도메인 경계 내로 클리핑
        new_points = torch.clamp(new_points, self.domain_min, self.domain_max)
        
        return new_points
    
    def reset(self):
        """샘플러를 초기 상태로 리셋합니다."""
        self.current_points = self.lhs_sampler.sample()
        self.residuals = None


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
    def __init__(self, data_dir: str, patch_size: int, use_augmentation: bool = False, real_data: bool = False, num_channels: int = 16, output_format: str = 'bmp'):
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
                 num_channels: int = 16, epoch_length: int = 1000):
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
