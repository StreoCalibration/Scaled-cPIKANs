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
