import torch
import torch.nn as nn

class AffineScaling(nn.Module):
    """
    입력 좌표를 주어진 물리적 도메인에서 [-1, 1] 표준 도메인으로 아핀 변환합니다.
    이 변환은 학습되지 않는 고정된 전처리 단계입니다.
    """
    def __init__(self, domain_min, domain_max):
        super(AffineScaling, self).__init__()
        # register_buffer를 사용하여 state_dict에 저장되지만 학습 파라미터는 아님
        self.register_buffer("domain_min", torch.tensor(domain_min, dtype=torch.float32))
        self.register_buffer("domain_max", torch.tensor(domain_max, dtype=torch.float32))
        self.register_buffer("domain_range", self.domain_max - self.domain_min)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 물리적 도메인의 입력 좌표 (batch_size, n_dims)

        Returns:
            torch.Tensor: [-1, 1]로 스케일링된 좌표 (batch_size, n_dims)
        """
        # 각 차원에 대해 독립적으로 스케일링
        # x_scaled = (x - x_min) / (x_max - x_min) # [0, 1] 범위로 정규화
        # return 2 * x_scaled - 1 # [-1, 1] 범위로 변환
        return 2 * (x - self.domain_min) / self.domain_range - 1

class ChebyKANLayer(nn.Module):
    """
    체비쇼프 다항식을 기저 함수로 사용하는 KAN(Kolmogorov-Arnold Network) 레이어.
    논문의 제 3.1장에 기술된 구현을 따릅니다.
    """
    def __init__(self, input_dim, output_dim, degree=4):
        super(ChebyKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        # 학습 가능한 파라미터: 체비쇼프 계수
        # shape: (out_features, in_features, degree + 1)
        self.cheby_coeffs = nn.Parameter(torch.randn(output_dim, input_dim, degree + 1))
        nn.init.xavier_uniform_(self.cheby_coeffs) # Xavier 초기화

        # 안정적인 훈련을 위한 Layer Normalization
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 이전 레이어의 출력 (batch_size, input_dim)

        Returns:
            torch.Tensor: 현재 레이어의 출력 (batch_size, output_dim)
        """
        # 1. LayerNorm 적용 (안정성)
        x = self.layer_norm(x)
        
        # 2. tanh 활성화 함수 적용 (입력을 [-1, 1] 범위로 유지)
        x = torch.tanh(x)

        # 3. 체비쇼프 다항식 계산 (T_0, T_1, ..., T_degree)
        # x shape: (batch_size, input_dim)
        cheby_basis = [torch.ones_like(x), x] # T_0, T_1
        for n in range(2, self.degree + 1):
            # T_n+1(x) = 2x * T_n(x) - T_n-1(x)
            cheby_basis.append(2 * x * cheby_basis[-1] - cheby_basis[-2])
        
        # shape: (batch_size, input_dim, degree + 1)
        cheby_basis = torch.stack(cheby_basis, dim=-1)

        # 4. 학습 가능한 함수(phi) 평가 및 결과 합산
        # cheby_basis: (batch, in_dim, degree+1)
        # self.cheby_coeffs: (out_dim, in_dim, degree+1)
        # einsum 'bid,oid->bo':
        # b: batch, i: input_dim, d: degree, o: output_dim
        # phi_eval = einsum('bid,oid->boi', cheby_basis, self.cheby_coeffs)
        # y = torch.sum(phi_eval, dim=2)
        # 위 두 줄을 아래 한 줄로 최적화
        y = torch.einsum('bid,oid->bo', cheby_basis, self.cheby_coeffs)
        
        return y

class Scaled_cPIKAN_Model(nn.Module):
    """
    논문에 기술된 Scaled-cPIKAN 아키텍처.
    아핀 영역 스케일링과 ChebyKAN 레이어를 통합합니다.
    """
    def __init__(self, layers, degree=4, domain_min=None, domain_max=None):
        """
        Args:
            layers (list of int): 각 레이어의 뉴런 수를 정의. 예: [2, 64, 64, 2]
            degree (int): 체비쇼프 다항식의 차수
            domain_min (list of float): 입력 물리적 도메인의 최소값. 예: [x_min, y_min]
            domain_max (list of float): 입력 물리적 도메인의 최대값. 예: [x_max, y_max]
        """
        super(Scaled_cPIKAN_Model, self).__init__()

        if domain_min is None or domain_max is None:
            raise ValueError("domain_min and domain_max must be provided for AffineScaling.")

        self.scaling = AffineScaling(domain_min, domain_max)
        
        self.kan_layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.kan_layers.append(
                ChebyKANLayer(layers[i], layers[i+1], degree=degree)
            )

    def forward(self, coords):
        """
        Args:
            coords (torch.Tensor): 입력 좌표 (batch_size, input_dim)

        Returns:
            torch.Tensor: 예측된 결과 (batch_size, output_dim)
        """
        x = self.scaling(coords)
        for layer in self.kan_layers:
            x = layer(x)
        return x