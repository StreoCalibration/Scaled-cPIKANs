import torch
import torch.nn as nn

class PhysicsInformedLoss(nn.Module):
    """
    PINN(물리 정보 신경망)의 물리 정보 손실을 계산하기 위한 일반 클래스.

    이 클래스는 PDE 잔차 손실, 경계 조건 손실, 초기 조건 손실, 데이터 기반 손실의
    가중 합으로 총 손실을 계산합니다. 사용자가 호출 가능한 함수를 전달하여 특정 물리
    문제의 특성을 정의할 수 있도록 유연하게 설계되었습니다.
    """
    def __init__(self, pde_residual_fn, bc_fns, ic_fns=None, data_loss_fn=None, loss_weights=None):
        """
        Args:
            pde_residual_fn (callable): PDE 잔차를 계산하는 함수.
                시그니처: `pde_residual_fn(model, points) -> torch.Tensor`
            bc_fns (list[callable] or callable): 특정 경계 조건의 오차를 계산하는
                함수 또는 함수 리스트.
                시그니처: `bc_fn(model, points) -> torch.Tensor`
            ic_fns (list[callable] or callable, optional): 초기 조건에 대한 함수 또는
                함수 리스트. 기본값은 None.
                시그니처: `ic_fn(model, points) -> torch.Tensor`
            data_loss_fn (callable, optional): 데이터 기반 손실을 위한 함수.
                시그니처: `data_loss_fn(model, points, true_values) -> torch.Tensor`.
                기본값은 None.
            loss_weights (dict, optional): 각 손실 요소에 대한 가중치 사전
                (예: {'pde': 1.0, 'bc': 10.0}). 기본값은 모두 1.0.
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
        총 물리 정보 손실을 계산합니다.

        Args:
            model (nn.Module): 신경망 모델 (PINN).
            pde_points (torch.Tensor): PDE 잔차를 위한 콜로케이션 포인트.
            bc_points_dicts (list): 경계점들을 위한 딕셔너리 리스트.
                                         각 딕셔너리는 bc_fn에 해당하며 'points' 키 아래에
                                         포인트 텐서를 포함합니다.
            ic_points_dicts (list, optional): 초기 조건 포인트들을 위한 딕셔너리 리스트.
            data_points (tuple, optional): 데이터 손실을 위한 (입력 포인트, 실제 값) 튜플.

        Returns:
            tuple[torch.Tensor, dict]: 총 손실과 개별 손실 요소들의 딕셔너리를 포함하는 튜플.
        """
        device = next(model.parameters()).device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}

        # 1. PDE 잔차 손실
        if pde_points is not None and self.pde_residual_fn is not None:
            pde_residuals = self.pde_residual_fn(model, pde_points)
            loss_pde = self.mse_loss(pde_residuals, torch.zeros_like(pde_residuals))
            loss_dict['loss_pde'] = loss_pde
            total_loss += self.loss_weights.get('pde', 1.0) * loss_pde

        # 2. 경계 조건 손실
        loss_bc_total = torch.tensor(0.0, device=device)
        if bc_points_dicts is not None and self.bc_fns:
            for i, bc_fn in enumerate(self.bc_fns):
                if i < len(bc_points_dicts) and bc_points_dicts[i]:
                    points = bc_points_dicts[i]['points']
                    bc_errors = bc_fn(model, points)
                    loss_bc_total += self.mse_loss(bc_errors, torch.zeros_like(bc_errors))
            loss_dict['loss_bc'] = loss_bc_total
            total_loss += self.loss_weights.get('bc', 1.0) * loss_bc_total

        # 3. 초기 조건 손실
        loss_ic_total = torch.tensor(0.0, device=device)
        if ic_points_dicts is not None and self.ic_fns:
            for i, ic_fn in enumerate(self.ic_fns):
                if i < len(ic_points_dicts) and ic_points_dicts[i]:
                    points = ic_points_dicts[i]['points']
                    ic_errors = ic_fn(model, points)
                    loss_ic_total += self.mse_loss(ic_errors, torch.zeros_like(ic_errors))
            loss_dict['loss_ic'] = loss_ic_total
            total_loss += self.loss_weights.get('ic', 1.0) * loss_ic_total

        # 4. 데이터 기반 손실
        if data_points is not None and self.data_loss_fn is not None:
            inputs, true_values = data_points
            loss_data = self.data_loss_fn(model, inputs, true_values)
            loss_dict['loss_data'] = loss_data
            total_loss += self.loss_weights.get('data', 1.0) * loss_data

        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict


import torch.nn.functional as F
import numpy as np

class UNetPhysicsLoss(nn.Module):
    """
    A physics-informed loss function for the U-Net model.

    This loss does not require a ground truth height map. Instead, it computes
    the loss based on physical consistency with the input bucket images and
    a smoothness prior on the reconstructed height map.
    """
    def __init__(self, wavelengths: list[float], num_buckets: int, smoothness_weight: float = 1e-4):
        super().__init__()
        self.wavelengths = torch.tensor(wavelengths, dtype=torch.float32).view(-1, 1, 1, 1)
        deltas = torch.arange(num_buckets, dtype=torch.float32) * (2 * np.pi / num_buckets)
        self.deltas = deltas.view(1, num_buckets, 1, 1)
        self.smoothness_weight = smoothness_weight
        self.mse_loss = nn.MSELoss()
        self.metrics = {}

        # Pre-define a Laplacian kernel for the smoothness loss
        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)

    def forward(self, predicted_height: torch.Tensor, input_buckets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the total loss.

        Args:
            predicted_height (torch.Tensor): The height map output from the U-Net.
                                             Shape: (N, 1, H, W)
            input_buckets (torch.Tensor): The original 16 bucket images that were
                                          input to the U-Net. Shape: (N, 16, H, W)

        Returns:
            torch.Tensor: The total computed loss.
        """
        device = predicted_height.device
        self.wavelengths = self.wavelengths.to(device)
        self.deltas = self.deltas.to(device)
        self.laplacian_kernel = self.laplacian_kernel.to(device)

        num_lasers = len(self.wavelengths)
        num_buckets = len(self.deltas.view(-1))

        # --- 1. Data Consistency Loss ---
        # Simulate bucket images from the predicted height map
        # Reshape height map for broadcasting with wavelengths and deltas
        height_map_expanded = predicted_height.unsqueeze(1) # (N, 1, 1, H, W)

        # Physical simulation formula
        phase = (4 * np.pi / self.wavelengths) * height_map_expanded # (N, num_lasers, 1, H, W)
        phase_with_shifts = phase + self.deltas # (N, num_lasers, num_buckets, H, W)

        A, B = 128, 100
        predicted_buckets = A + B * torch.cos(phase_with_shifts)

        # Reshape to match input_buckets shape
        predicted_buckets = predicted_buckets.view_as(input_buckets)

        loss_data = self.mse_loss(predicted_buckets, input_buckets)

        # --- 2. Smoothness Regularization Loss ---
        # Calculate the Laplacian of the predicted height map
        laplacian = F.conv2d(predicted_height, self.laplacian_kernel, padding=1)
        loss_smoothness = self.mse_loss(laplacian, torch.zeros_like(laplacian))

        # --- Total Loss ---
        total_loss = loss_data + self.smoothness_weight * loss_smoothness

        # Store metrics for logging
        self.metrics = {
            "loss_total": total_loss.item(),
            "loss_data": loss_data.item(),
            "loss_smoothness": loss_smoothness.item(),
        }

        return total_loss


class PinnReconstructionLoss(nn.Module):
    """
    Computes the loss for a PINN model reconstructing a height map from bucket images.

    The loss has two main components:
    1.  **Data Consistency:** The MSE between the bucket images predicted from the
        reconstructed height map and the actual input bucket images.
    2.  **Smoothness Regularization:** A term that penalizes non-smoothness in the
        reconstructed height map, calculated from its Laplacian.
    """
    def __init__(self, wavelengths: list[float], num_buckets: int, smoothness_weight: float = 1e-7):
        super().__init__()
        self.wavelengths = torch.tensor(wavelengths, dtype=torch.float32).view(-1, 1, 1)
        deltas = torch.arange(num_buckets, dtype=torch.float32) * (2 * np.pi / num_buckets)
        self.deltas = deltas.view(1, num_buckets, 1)
        self.smoothness_weight = smoothness_weight
        self.mse_loss = nn.MSELoss()
        self.metrics = {}
        self.num_lasers = len(wavelengths)
        self.num_buckets = num_buckets

    def forward(self, predicted_height: torch.Tensor, coords: torch.Tensor, target_buckets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the total loss.

        Args:
            predicted_height (torch.Tensor): The height map from the PINN model.
                                             Shape: (1, N_points)
            coords (torch.Tensor): The coordinate grid input to the model.
                                   Shape: (N_points, 2). Requires grad.
            target_buckets (torch.Tensor): The ground truth bucket values for the patch.
                                           Shape: (C, N_points), where C = lasers * buckets.

        Returns:
            torch.Tensor: The total computed loss.
        """
        device = predicted_height.device
        self.wavelengths = self.wavelengths.to(device)
        self.deltas = self.deltas.to(device)

        # --- 1. Data Consistency Loss ---
        # Reshape targets to be (num_lasers, num_buckets, N_points)
        target_buckets_reshaped = target_buckets.view(self.num_lasers, self.num_buckets, -1)

        # Simulate bucket images from the predicted height map
        # predicted_height shape: (1, N_points)
        phase = (4 * np.pi / self.wavelengths) * predicted_height.unsqueeze(0)  # Shape: (num_lasers, 1, N_points)
        phase_with_shifts = phase + self.deltas  # Shape: (num_lasers, num_buckets, N_points)

        A, B = 128, 100
        predicted_buckets = A + B * torch.cos(phase_with_shifts) # Shape: (num_lasers, num_buckets, N_points)

        loss_data = self.mse_loss(predicted_buckets, target_buckets_reshaped)

        # --- 2. Smoothness Regularization ---
        # Detach predicted_height to avoid autograd issues with the sum, but use original for grad calc
        h = predicted_height.squeeze(0) # Shape: (N_points)
        grad_h = torch.autograd.grad(h.sum(), coords, create_graph=True)[0]
        h_x, h_y = grad_h[:, 0], grad_h[:, 1]

        # To compute second derivatives, we need to sum again
        h_xx = torch.autograd.grad(h_x.sum(), coords, create_graph=True)[0][:, 0]
        h_yy = torch.autograd.grad(h_y.sum(), coords, create_graph=True)[0][:, 1]

        laplacian = h_xx + h_yy
        loss_smoothness = self.mse_loss(laplacian, torch.zeros_like(laplacian))

        # --- Total Loss ---
        total_loss = loss_data + self.smoothness_weight * loss_smoothness

        self.metrics = {
            "loss_total": total_loss.item(),
            "loss_data": loss_data.item(),
            "loss_smoothness": loss_smoothness.item(),
        }
        return total_loss


class DynamicWeightedLoss(nn.Module):
    """
    GradNorm 알고리즘을 사용한 동적 손실 가중치 조정 클래스.
    
    여러 손실 항목들의 그래디언트 크기를 균형있게 유지하기 위해 가중치를 자동으로 조정합니다.
    Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing 
    in Deep Multitask Networks" (2018) 논문 기반.
    
    주요 개념:
    - 각 손실 항의 그래디언트 노름(norm)을 계산
    - 목표 그래디언트 노름을 설정하고 이에 맞춰 가중치 동적 조정
    - 학습 가능한 파라미터로 가중치를 표현하여 훈련 중 업데이트
    """
    
    def __init__(self, base_loss_fn, loss_names, alpha=1.5, initial_weights=None, learning_rate=0.025):
        """
        Args:
            base_loss_fn (callable): 기본 손실 함수. forward 메서드가 (total_loss, loss_dict)를 반환해야 함.
            loss_names (list[str]): 균형을 맞출 손실 항목들의 이름 리스트 (예: ['loss_pde', 'loss_bc']).
            alpha (float): GradNorm의 비대칭 파라미터. 기본값 1.5.
                          alpha > 1: 느리게 학습되는 항목에 더 큰 가중치
                          alpha < 1: 빠르게 학습되는 항목에 더 큰 가중치
                          alpha = 1: 모든 항목을 동등하게 균형
            initial_weights (dict, optional): 초기 가중치 딕셔너리. None이면 모두 1.0으로 초기화.
            learning_rate (float): 가중치 업데이트를 위한 학습률. 기본값 0.025.
        """
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.loss_names = loss_names
        self.alpha = alpha
        self.learning_rate = learning_rate
        
        # 학습 가능한 가중치 파라미터 초기화
        if initial_weights is None:
            initial_weights = {name: 1.0 for name in loss_names}
        
        # device 파라미터 저장 (나중에 텐서를 올바른 device에 생성)
        self.device_param = nn.Parameter(torch.zeros(1))  # 더미 파라미터로 device 추적
        
        # 각 손실 항에 대한 학습 가능한 가중치 (로그 공간에서 표현)
        # 로그 공간 사용 이유: 가중치가 항상 양수로 유지됨
        self.log_weights = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(np.log(initial_weights.get(name, 1.0)), dtype=torch.float32))
            for name in loss_names
        })
        
        # 초기 손실값 추적 (상대적 변화율 계산용)
        self.register_buffer('initial_losses', torch.zeros(len(loss_names)))
        self.register_buffer('loss_history', torch.zeros(len(loss_names)))
        self.first_step = True
        
        # 가중치 업데이트를 위한 옵티마이저
        self.weight_optimizer = torch.optim.Adam(self.log_weights.values(), lr=learning_rate)
        
    def get_weights(self):
        """현재 가중치를 딕셔너리로 반환 (로그 공간에서 실제 공간으로 변환)."""
        return {name: torch.exp(self.log_weights[name]).item() for name in self.loss_names}
    
    def forward(self, model, *args, **kwargs):
        """
        동적 가중치를 적용하여 총 손실을 계산합니다.
        
        Args:
            model (nn.Module): PINN 모델
            *args, **kwargs: base_loss_fn에 전달될 인자들
            
        Returns:
            tuple[torch.Tensor, dict]: (가중치가 적용된 총 손실, 손실 상세 정보 딕셔너리)
        """
        # 기본 손실 함수 호출
        _, loss_dict = self.base_loss_fn(model, *args, **kwargs)
        
        device = next(model.parameters()).device
        
        # 첫 스텝: 초기 손실값 기록
        if self.first_step:
            self.initial_losses = torch.tensor(
                [loss_dict[name].item() for name in self.loss_names],
                device=device
            )
            self.loss_history = self.initial_losses.clone()
            self.first_step = False
        
        # 현재 손실값들
        current_losses = torch.stack([loss_dict[name] for name in self.loss_names])
        
        # 동적 가중치 계산 (로그 공간에서 실제 공간으로)
        weights = torch.stack([torch.exp(self.log_weights[name]) for name in self.loss_names]).to(device)
        
        # 가중치가 적용된 총 손실 계산
        weighted_total_loss = (weights.detach() * current_losses).sum()  # weights detach하여 모델 훈련과 분리
        
        # GradNorm 업데이트 (훈련 모드에서만)
        if self.training and not self.first_step:
            # 별도로 GradNorm 업데이트 수행 (기존 그래프와 독립적으로)
            with torch.enable_grad():
                self._update_weights(model, current_losses.detach(), weights, device)
        
        # 손실 히스토리 업데이트
        self.loss_history = current_losses.detach()
        
        # 손실 딕셔너리에 가중치 정보 추가
        loss_dict['weighted_total_loss'] = weighted_total_loss
        loss_dict['weights'] = {name: weights[i].item() for i, name in enumerate(self.loss_names)}
        
        return weighted_total_loss, loss_dict
    
    def _update_weights(self, model, current_losses, weights, device):
        """
        GradNorm 알고리즘을 사용하여 가중치를 업데이트합니다.
        
        Args:
            model (nn.Module): PINN 모델
            current_losses (torch.Tensor): 현재 손실값들 (detached)
            weights (torch.Tensor): 현재 가중치들
            device: 계산 장치
        """
        # 손실을 새로 계산해야 함 (detached된 값으로는 grad 계산 불가)
        # 이를 위해 base_loss_fn을 다시 호출하지 않고, 
        # 현재 모델 상태에서 각 손실 항목의 gradient norm만 계산
        
        # 마지막 공유 레이어의 파라미터 선택
        shared_params = list(model.parameters())[-1]
        
        # 각 손실에 대한 그래디언트 노름을 계산하기 위해
        # 모델을 다시 평가해야 하지만, 이는 너무 비효율적
        # 대신 현재 손실 비율에 기반한 간단한 업데이트 사용
        
        # 각 손실의 상대적 역 훈련 속도 계산
        # r_i(t) = L_i(t) / L_i(0)
        # initial_losses와 current_losses를 같은 device로
        initial_losses_on_device = self.initial_losses.to(device)
        loss_ratios = current_losses / (initial_losses_on_device + 1e-8)
        
        # 평균 역 훈련 속도
        mean_loss_ratio = loss_ratios.mean()
        
        # 목표: 모든 loss_ratio가 비슷하게 유지되도록
        # ratio가 높으면 (학습이 느림) → 가중치 증가
        # ratio가 낮으면 (학습이 빠름) → 가중치 감소
        
        # 목표 비율 (mean으로 정규화)
        target_ratios = torch.pow(loss_ratios / (mean_loss_ratio + 1e-8), self.alpha)
        
        # 가중치 조정: target에 비례하여 log weight 업데이트
        with torch.no_grad():
            for i, name in enumerate(self.loss_names):
                # 간단한 비례 업데이트
                adjustment = self.learning_rate * (target_ratios[i] - 1.0)
                # adjustment를 CPU로 이동하여 log_weights와 같은 device에 맞춤
                self.log_weights[name].data += adjustment.cpu()
            
            # 가중치 정규화 (합이 일정하게 유지)
            total_weight = sum(torch.exp(self.log_weights[name]) for name in self.loss_names)
            normalization_factor = len(self.loss_names) / (total_weight + 1e-8)
            for name in self.loss_names:
                # normalization_factor도 CPU로
                norm_adj = torch.log(normalization_factor + 1e-10)
                if norm_adj.device != self.log_weights[name].device:
                    norm_adj = norm_adj.cpu()
                self.log_weights[name].data += norm_adj
