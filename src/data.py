import torch
from scipy.stats import qmc

def affine_scale(x: torch.Tensor, domain_min: torch.Tensor, domain_max: torch.Tensor) -> torch.Tensor:
    """
    Scales a tensor from a physical domain [domain_min, domain_max] to the
    canonical domain [-1, 1].

    Args:
        x (torch.Tensor): The input tensor in the physical domain.
        domain_min (torch.Tensor): The minimum values of the physical domain.
        domain_max (torch.Tensor): The maximum values of the physical domain.

    Returns:
        torch.Tensor: The scaled tensor in the [-1, 1] domain.
    """
    return 2.0 * (x - domain_min) / (domain_max - domain_min) - 1.0

def affine_unscale(x_scaled: torch.Tensor, domain_min: torch.Tensor, domain_max: torch.Tensor) -> torch.Tensor:
    """
    Unscales a tensor from the canonical domain [-1, 1] back to the
    physical domain [domain_min, domain_max].

    Args:
        x_scaled (torch.Tensor): The input tensor in the [-1, 1] domain.
        domain_min (torch.Tensor): The minimum values of the physical domain.
        domain_max (torch.Tensor): The maximum values of the physical domain.

    Returns:
        torch.Tensor: The unscaled tensor in the physical domain.
    """
    return (x_scaled + 1.0) / 2.0 * (domain_max - domain_min) + domain_min


class LatinHypercubeSampler:
    """
    A sampler that generates points using Latin Hypercube Sampling (LHS).

    LHS is a stratified sampling technique that ensures a more uniform spread
    of sample points across the domain compared to pure random sampling.

    Args:
        n_points (int): The number of sample points to generate.
        domain_min (list[float] or torch.Tensor): The lower bounds of the sampling domain for each dimension.
        domain_max (list[float] or torch.Tensor): The upper bounds of the sampling domain for each dimension.
        dtype (torch.dtype, optional): The desired data type of the output tensor. Defaults to torch.float32.
        device (torch.device, optional): The desired device of the output tensor. Defaults to 'cpu'.
    """
    def __init__(self, n_points: int, domain_min: list[float], domain_max: list[float], dtype: torch.dtype = torch.float32, device: torch.device = 'cpu'):
        if len(domain_min) != len(domain_max):
            raise ValueError("domain_min and domain_max must have the same length.")

        self.n_points = n_points
        self.dimensions = len(domain_min)
        self.sampler = qmc.LatinHypercube(d=self.dimensions, seed=None) # Use a random seed each time

        self.domain_min = torch.tensor(domain_min, dtype=dtype, device=device)
        self.domain_max = torch.tensor(domain_max, dtype=dtype, device=device)
        self.dtype = dtype
        self.device = device

    def sample(self) -> torch.Tensor:
        """
        Generates the sample points.

        Returns:
            torch.Tensor: A tensor of shape (n_points, dimensions) containing the sampled points.
        """
        # Sample points in the unit hypercube [0, 1]^d
        unit_samples = self.sampler.random(n=self.n_points)
        unit_samples_tensor = torch.tensor(unit_samples, dtype=self.dtype, device=self.device)

        # Scale the samples to the physical domain
        samples = self.domain_min + (self.domain_max - self.domain_min) * unit_samples_tensor

        return samples
