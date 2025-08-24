import torch
from collections import defaultdict

class Trainer:
    """
    A trainer class to handle the optimization of a PINN model.

    This class implements the two-stage training process (Adam + L-BFGS)
    described in the design document. It logs the history of different loss
    components throughout the training.
    """
    def __init__(self, model, loss_fn):
        """
        Args:
            model (torch.nn.Module): The PINN model to be trained.
            loss_fn (callable): The physics-informed loss function.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.device = next(model.parameters()).device
        self.history = defaultdict(list)
        self.lbfgs_loss_dict = {} # To store loss dict from L-BFGS closure

    def train(self,
              pde_points,
              bc_points_dicts,
              ic_points_dicts=None,
              data_points=None,
              adam_epochs=20000,
              lbfgs_epochs=10,
              adam_lr=1e-3,
              log_interval=1000):
        """
        Runs the full training process, first with Adam, then with L-BFGS.

        Args:
            pde_points (torch.Tensor): Collocation points for the PDE residual.
            bc_points_dicts (list): List of dictionaries for boundary condition points.
            ic_points_dicts (list, optional): List of dictionaries for initial condition points.
            data_points (tuple, optional): Tuple of (inputs, true_values) for data loss.
            adam_epochs (int): Number of epochs for the Adam optimizer.
            lbfgs_epochs (int): Number of epochs/steps for the L-BFGS optimizer.
            adam_lr (float): Learning rate for the Adam optimizer.
            log_interval (int): How often to print loss information.

        Returns:
            dict: A dictionary containing the history of all loss components.
        """
        print("--- Starting Stage 1: Adam Optimization ---")
        self._train_adam(pde_points, bc_points_dicts, ic_points_dicts, data_points, adam_epochs, adam_lr, log_interval)

        if lbfgs_epochs > 0:
            print("\n--- Starting Stage 2: L-BFGS Optimization ---")
            self._train_lbfgs(pde_points, bc_points_dicts, ic_points_dicts, data_points, lbfgs_epochs, log_interval)

        print("\n--- Training Finished ---")
        return self.history

    def _train_adam(self, pde_points, bc_points_dicts, ic_points_dicts, data_points, epochs, lr, log_interval):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            total_loss, loss_dict = self.loss_fn(self.model, pde_points, bc_points_dicts, ic_points_dicts, data_points)

            total_loss.backward()
            optimizer.step()

            self._log_history(loss_dict, epoch, "Adam")

            if (epoch + 1) % log_interval == 0 or epoch == epochs - 1:
                self._print_log(epoch, epochs, loss_dict, "Adam")

    def _train_lbfgs(self, pde_points, bc_points_dicts, ic_points_dicts, data_points, epochs, log_interval):
        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=20,
            history_size=100,
            line_search_fn="strong_wolfe"
        )

        self.model.train()

        def closure():
            optimizer.zero_grad()
            total_loss, loss_dict = self.loss_fn(self.model, pde_points, bc_points_dicts, ic_points_dicts, data_points)
            self.lbfgs_loss_dict = loss_dict # Save for logging
            total_loss.backward()
            return total_loss

        # L-BFGS performs multiple function evaluations in one step, so we typically
        # don't loop over it like we do with Adam.
        optimizer.step(closure)

        # Log the final state after the L-BFGS optimization step.
        final_loss, final_loss_dict = self.loss_fn(self.model, pde_points, bc_points_dicts, ic_points_dicts, data_points)
        self.lbfgs_loss_dict = final_loss_dict

        self._log_history(self.lbfgs_loss_dict, self.adam_epochs, "L-BFGS")
        self._print_log(0, 1, self.lbfgs_loss_dict, "L-BFGS")

    def _log_history(self, loss_dict, epoch, stage):
        self.history['epoch'].append(epoch)
        self.history['stage'].append(stage)
        for key, value in loss_dict.items():
            self.history[key].append(value.item())

    def _print_log(self, epoch, total_epochs, loss_dict, stage):
        log_str = f"[{stage}] Epoch [{epoch+1}/{total_epochs}]"
        for key, value in loss_dict.items():
            log_str += f" - {key}: {value.item():.4e}"
        print(log_str)

    @property
    def adam_epochs(self):
        """Returns the number of Adam epochs completed, for correct epoch counting."""
        return len([s for s in self.history['stage'] if s == "Adam"])
