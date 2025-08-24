import torch
import torch.nn as nn

class PhysicsInformedLoss(nn.Module):
    """
    A generic class to compute the physics-informed loss for a PINN.

    This class calculates the total loss as a weighted sum of the PDE residual loss,
    boundary condition loss, initial condition loss, and data-driven loss. It is
    designed to be flexible, allowing users to define the specific physics of their
    problem by passing in callable functions.
    """
    def __init__(self, pde_residual_fn, bc_fns, ic_fns=None, data_loss_fn=None, loss_weights=None):
        """
        Args:
            pde_residual_fn (callable): A function that computes the PDE residual.
                Signature: `pde_residual_fn(model, points) -> torch.Tensor`
            bc_fns (list[callable] or callable): A function or list of functions, each
                computing the error for a specific boundary condition.
                Signature: `bc_fn(model, points) -> torch.Tensor`
            ic_fns (list[callable] or callable, optional): A function or list of functions
                for initial conditions. Defaults to None.
                Signature: `ic_fn(model, points) -> torch.Tensor`
            data_loss_fn (callable, optional): A function for data-driven loss.
                Signature: `data_loss_fn(model, points, true_values) -> torch.Tensor`.
                Defaults to None.
            loss_weights (dict, optional): A dictionary of weights for each loss component
                (e.g., {'pde': 1.0, 'bc': 10.0}). Defaults to equal weights of 1.0.
        """
        super().__init__()
        self.pde_residual_fn = pde_residual_fn
        self.bc_fns = bc_fns if isinstance(bc_fns, list) else [bc_fns]
        self.ic_fns = ic_fns if isinstance(ic_fns, list) else ([ic_fns] if ic_fns else [])
        self.data_loss_fn = data_loss_fn

        if loss_weights is None:
            self.loss_weights = {'pde': 1.0, 'bc': 1.0, 'ic': 1.0, 'data': 1.0}
        else:
            self.loss_weights = loss_weights
            for key in ['pde', 'bc', 'ic', 'data']:
                if key not in self.loss_weights:
                    self.loss_weights[key] = 1.0

        self.mse_loss = nn.MSELoss()

    def forward(self, model, pde_points, bc_points_dicts, ic_points_dicts=None, data_points=None):
        """
        Calculates the total physics-informed loss.

        Args:
            model (nn.Module): The neural network model (the PINN).
            pde_points (torch.Tensor): Collocation points for the PDE residual.
            bc_points_dicts (list): A list of dictionaries for boundary points.
                                         Each dict corresponds to a bc_fn and contains
                                         the points tensor under the key 'points'.
            ic_points_dicts (list, optional): A list of dicts for initial condition points.
            data_points (tuple, optional): A tuple of (input_points, true_values) for data loss.

        Returns:
            tuple[torch.Tensor, dict]: A tuple containing the total loss and a dictionary
                                       of the individual loss components.
        """
        device = next(model.parameters()).device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}

        # 1. PDE Residual Loss
        if pde_points is not None and self.pde_residual_fn is not None:
            pde_residuals = self.pde_residual_fn(model, pde_points)
            loss_pde = self.mse_loss(pde_residuals, torch.zeros_like(pde_residuals))
            loss_dict['loss_pde'] = loss_pde
            total_loss += self.loss_weights.get('pde', 1.0) * loss_pde

        # 2. Boundary Condition Loss
        loss_bc_total = torch.tensor(0.0, device=device)
        if bc_points_dicts is not None and self.bc_fns:
            for i, bc_fn in enumerate(self.bc_fns):
                if i < len(bc_points_dicts) and bc_points_dicts[i]:
                    points = bc_points_dicts[i]['points']
                    bc_errors = bc_fn(model, points)
                    loss_bc_total += self.mse_loss(bc_errors, torch.zeros_like(bc_errors))
            loss_dict['loss_bc'] = loss_bc_total
            total_loss += self.loss_weights.get('bc', 1.0) * loss_bc_total

        # 3. Initial Condition Loss
        loss_ic_total = torch.tensor(0.0, device=device)
        if ic_points_dicts is not None and self.ic_fns:
            for i, ic_fn in enumerate(self.ic_fns):
                if i < len(ic_points_dicts) and ic_points_dicts[i]:
                    points = ic_points_dicts[i]['points']
                    ic_errors = ic_fn(model, points)
                    loss_ic_total += self.mse_loss(ic_errors, torch.zeros_like(ic_errors))
            loss_dict['loss_ic'] = loss_ic_total
            total_loss += self.loss_weights.get('ic', 1.0) * loss_ic_total

        # 4. Data-driven Loss
        if data_points is not None and self.data_loss_fn is not None:
            inputs, true_values = data_points
            loss_data = self.data_loss_fn(model, inputs, true_values)
            loss_dict['loss_data'] = loss_data
            total_loss += self.loss_weights.get('data', 1.0) * loss_data

        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict
