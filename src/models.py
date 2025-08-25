import torch
import torch.nn as nn

class ChebyKANLayer(nn.Module):
    """
    체비쇼프 기반 콜모고로프-아르놀트 신경망(KAN) 레이어.

    이 레이어는 노드의 고정된 활성화 함수 대신, 체비쇼프 다항식으로
    매개변수화된 학습 가능한 활성화 함수를 엣지(연결선)에 사용합니다.

    Args:
        in_features (int): 입력 특징의 수.
        out_features (int): 출력 특징의 수.
        cheby_order (int): 사용할 체비쇼프 다항식의 차수(degree) K.
                           기저는 K+1개의 다항식(T_0부터 T_K)을 가집니다.
    """
    def __init__(self, in_features: int, out_features: int, cheby_order: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cheby_order = cheby_order

        # 체비쇼프 다항식을 위한 학습 가능한 계수.
        # 크기: (out_features, in_features, cheby_order + 1)
        self.cheby_coeffs = nn.Parameter(torch.empty(out_features, in_features, cheby_order + 1))
        # 표준 방법(예: Kaiming uniform)을 사용하여 가중치 초기화.
        nn.init.kaiming_uniform_(self.cheby_coeffs, a=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ChebyKANLayer의 순방향 패스.

        Args:
            x (torch.Tensor): (batch_size, in_features) 크기의 입력 텐서.
                              입력값은 반드시 [-1, 1] 범위 내에 있어야 합니다.

        Returns:
            torch.Tensor: (batch_size, out_features) 크기의 출력 텐서.
        """
        batch_size, in_features = x.shape
        if in_features != self.in_features:
            raise ValueError(f"입력 특징 차원 {in_features}이 레이어의 in_features {self.in_features}와 일치하지 않습니다.")

        # k=0...K에 대한 체비쇼프 다항식 기저 T_k(x)를 다항식 리스트를 만들어 쌓는 방식으로 생성.
        # 이는 인플레이스(inplace) 연산을 방지합니다.
        cheby_polys = []
        cheby_polys.append(torch.ones_like(x))  # T_0(x) = 1
        if self.cheby_order > 0:
            cheby_polys.append(x)  # T_1(x) = x

        # 점화식: T_{k+1}(x) = 2x * T_k(x) - T_{k-1}(x)
        for k in range(1, self.cheby_order):
            next_poly = 2 * x * cheby_polys[-1] - cheby_polys[-2]
            cheby_polys.append(next_poly)

        # 다항식들을 쌓아 기저 행렬을 형성.
        # 크기: (batch_size, in_features, cheby_order + 1)
        cheby_basis = torch.stack(cheby_polys, dim=-1)

        # 기저와 계수를 축약하여 출력을 계산.
        # phi_{j,i}(x_i) = sum_k c_{j,i,k} * T_k(x_i)
        # y_j = sum_i phi_{j,i}(x_i)
        # 이는 einsum으로 효율적으로 계산할 수 있습니다.
        # 'bik,oik->bo'는 각 b와 o에 대해 i와 k를 합산하라는 의미.
        # b: batch_size, i: in_features, k: cheby_order, o: out_features
        output = torch.einsum('bik,oik->bo', cheby_basis, self.cheby_coeffs)

        return output

class Scaled_cPIKAN(nn.Module):
    """
    스케일링된 체비쇼프 기반 물리 정보 콜모고로프-아르놀트 신경망.

    이 모델은 설계 문서에 설명된 전체 Scaled-cPIKAN 아키텍처를 구현합니다.
    아핀 영역 스케일링, ChebyKAN 레이어 시퀀스, 그리고 중간의 정규화 및
    활성화 함수를 포함합니다.

    Args:
        layers_dims (list[int]): 신경망 아키텍처를 정의하는 리스트.
                                 예: [2, 32, 32, 1]은 2D 입력, 1D 출력,
                                 그리고 각각 32개의 뉴런을 가진 2개의 은닉층을 의미.
        cheby_order (int): 모든 레이어에 대한 체비쇼프 다항식의 차수.
        domain_min (torch.Tensor): 각 입력 차원에 대한 물리적 도메인의 최솟값을 담은 텐서.
        domain_max (torch.Tensor): 각 입력 차원에 대한 물리적 도메인의 최댓값을 담은 텐서.
    """
    def __init__(self, layers_dims: list[int], cheby_order: int, domain_min: torch.Tensor, domain_max: torch.Tensor):
        super().__init__()

        if not isinstance(layers_dims, list) or len(layers_dims) < 2:
            raise ValueError("layers_dims는 최소 두 개 이상의 정수를 담은 리스트여야 합니다.")

        self.layers_dims = layers_dims
        self.cheby_order = cheby_order

        # 도메인 경계를 학습 불가능한 버퍼로 등록.
        self.register_buffer('domain_min', domain_min)
        self.register_buffer('domain_max', domain_max)

        self.network = nn.ModuleList()
        for i in range(len(layers_dims) - 1):
            in_dim = layers_dims[i]
            out_dim = layers_dims[i+1]

            self.network.append(ChebyKANLayer(in_dim, out_dim, cheby_order))

            # 마지막 레이어를 제외한 모든 레이어에 LayerNorm과 tanh 활성화 함수 추가.
            if i < len(layers_dims) - 2:
                self.network.append(nn.LayerNorm(out_dim))
                self.network.append(nn.Tanh())

    def _affine_scale(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력 텐서를 물리적 도메인 [min, max]에서 체비쇼프 다항식이 요구하는
        표준 도메인 [-1, 1]으로 스케일링합니다.
        """
        # domain_min과 domain_max가 x의 크기에 브로드캐스트 가능하도록 보장.
        return 2.0 * (x - self.domain_min) / (self.domain_max - self.domain_min) - 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scaled_cPIKAN 모델의 순방향 패스.

        Args:
            x (torch.Tensor): 물리적 도메인 내의 점들을 나타내는
                              (batch_size, in_features) 크기의 입력 텐서.

        Returns:
            torch.Tensor: 출력 텐서, 일반적으로 예측된 PDE 해.
        """
        # 먼저, 필수적인 아핀 영역 스케일링을 적용.
        x_scaled = self._affine_scale(x)

        # 스케일링된 입력을 신경망 시퀀스에 통과.
        for layer in self.network:
            x_scaled = layer(x_scaled)

        return x_scaled


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    A standard U-Net architecture for image-to-image translation tasks.
    The model takes a (N, 12, H, W) tensor and outputs a (N, 1, H, W) tensor.
    """
    def __init__(self, n_channels=12, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
