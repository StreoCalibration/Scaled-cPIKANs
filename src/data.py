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
