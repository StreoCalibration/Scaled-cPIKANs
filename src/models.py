import torch
import torch.nn as nn

class ChebyKANLayer(nn.Module):
    """
    A Chebyshev-based Kolmogorov-Arnold Network (KAN) layer.

    This layer uses learnable activation functions on the edges, parameterized by
    Chebyshev polynomials, instead of fixed activation functions on the nodes.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        cheby_order (int): The order (degree) K of the Chebyshev polynomials to use.
                           The basis will have K+1 polynomials (T_0 to T_K).
    """
    def __init__(self, in_features: int, out_features: int, cheby_order: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cheby_order = cheby_order

        # Learnable coefficients for the Chebyshev polynomials.
        # Shape: (out_features, in_features, cheby_order + 1)
        self.cheby_coeffs = nn.Parameter(torch.empty(out_features, in_features, cheby_order + 1))
        # Initialize weights using a standard method, e.g., Kaiming uniform.
        nn.init.kaiming_uniform_(self.cheby_coeffs, a=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ChebyKANLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
                              The input values must be in the range [-1, 1].

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        batch_size, in_features = x.shape
        if in_features != self.in_features:
            raise ValueError(f"Input feature dimension {in_features} does not match layer's in_features {self.in_features}")

        # Create the Chebyshev polynomial basis T_k(x) for k=0...K by building a list
        # of polynomials and then stacking them. This avoids inplace operations.
        cheby_polys = []
        cheby_polys.append(torch.ones_like(x))  # T_0(x) = 1
        if self.cheby_order > 0:
            cheby_polys.append(x)  # T_1(x) = x

        # Recurrence relation: T_{k+1}(x) = 2x * T_k(x) - T_{k-1}(x)
        for k in range(1, self.cheby_order):
            next_poly = 2 * x * cheby_polys[-1] - cheby_polys[-2]
            cheby_polys.append(next_poly)

        # Stack the polynomials to form the basis matrix.
        # Shape: (batch_size, in_features, cheby_order + 1)
        cheby_basis = torch.stack(cheby_polys, dim=-1)

        # Compute the output by contracting the basis with the coefficients.
        # phi_{j,i}(x_i) = sum_k c_{j,i,k} * T_k(x_i)
        # y_j = sum_i phi_{j,i}(x_i)
        # This can be done efficiently with einsum.
        # 'bik,oik->bo' means: sum over i and k for each b and o.
        # b: batch_size, i: in_features, k: cheby_order, o: out_features
        output = torch.einsum('bik,oik->bo', cheby_basis, self.cheby_coeffs)

        return output

class Scaled_cPIKAN(nn.Module):
    """
    A Scaled Chebyshev-based Physics-Informed Kolmogorov-Arnold Network.

    This model implements the full Scaled-cPIKAN architecture as described in the
    design document. It includes affine domain scaling, a sequence of ChebyKAN layers,
    and intermediate normalization and activation functions.

    Args:
        layers_dims (list[int]): A list defining the network architecture.
                                 e.g., [2, 32, 32, 1] for a 2D input, 1D output,
                                 and 2 hidden layers with 32 neurons each.
        cheby_order (int): The order of Chebyshev polynomials for all layers.
        domain_min (torch.Tensor): A tensor with the minimum values of the physical domain for each input dimension.
        domain_max (torch.Tensor): A tensor with the maximum values of the physical domain for each input dimension.
    """
    def __init__(self, layers_dims: list[int], cheby_order: int, domain_min: torch.Tensor, domain_max: torch.Tensor):
        super().__init__()

        if not isinstance(layers_dims, list) or len(layers_dims) < 2:
            raise ValueError("layers_dims must be a list of at least two integers.")

        self.layers_dims = layers_dims
        self.cheby_order = cheby_order

        # Register domain bounds as non-trainable buffers.
        self.register_buffer('domain_min', domain_min)
        self.register_buffer('domain_max', domain_max)

        self.network = nn.ModuleList()
        for i in range(len(layers_dims) - 1):
            in_dim = layers_dims[i]
            out_dim = layers_dims[i+1]

            self.network.append(ChebyKANLayer(in_dim, out_dim, cheby_order))

            # Add LayerNorm and tanh activation for all but the last layer.
            if i < len(layers_dims) - 2:
                self.network.append(nn.LayerNorm(out_dim))
                self.network.append(nn.Tanh())

    def _affine_scale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scales the input tensor from the physical domain [min, max] to the
        canonical domain [-1, 1] required by Chebyshev polynomials.
        """
        # Ensure domain_min and domain_max are broadcastable to x's shape.
        return 2.0 * (x - self.domain_min) / (self.domain_max - self.domain_min) - 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Scaled_cPIKAN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
                              representing points in the physical domain.

        Returns:
            torch.Tensor: Output tensor, typically the predicted PDE solution.
        """
        # First, apply the essential affine domain scaling.
        x_scaled = self._affine_scale(x)

        # Pass the scaled input through the network sequence.
        for layer in self.network:
            x_scaled = layer(x_scaled)

        return x_scaled
