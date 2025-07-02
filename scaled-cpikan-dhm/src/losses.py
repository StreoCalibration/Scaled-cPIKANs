
import torch
import torch.nn as nn

class HelmholtzLoss(nn.Module):
    """
    데이터 충실도(Data Fidelity) 손실.
    모델이 예측한 복소장으로부터 계산된 간섭 무늬와
    실제 측정된 간섭 무늬 사이의 오차를 계산합니다.
    """
    def __init__(self):
        super(HelmholtzLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, u_pred, measured_interferograms):
        """
        Args:
            u_pred (torch.Tensor): 모델이 예측한 복소장 (batch_size, 2) -> [u_real, u_imag]
            measured_interferograms (torch.Tensor): 실제 측정된 간섭 무늬 이미지 (4, H, W)
                                                    (여기서는 배치를 위해 변환된 형태)

        Returns:
            torch.Tensor: 계산된 손실 값
        """
        # 예측된 복소장으로부터 위상(phase) 계산
        u_real = u_pred[:, 0]
        u_imag = u_pred[:, 1]
        phase_pred = torch.atan2(u_imag, u_real)

        # 4개의 위상 천이에 대한 간섭 무늬 시뮬레이션
        phase_shifts = torch.tensor([0, torch.pi / 2, torch.pi, 3 * torch.pi / 2]).to(phase_pred.device)
        
        loss = 0
        for i, shift in enumerate(phase_shifts):
            # 이상적인 간섭 무늬: I = A + B*cos(phase + shift)
            # A=0.5, B=0.5로 가정
            interferogram_pred = 0.5 + 0.5 * torch.cos(phase_pred + shift)
            
            # measured_interferograms에서 해당 shift의 이미지 추출
            # measured_interferograms는 (batch_size, 4) 형태로 전달된다고 가정
            interferogram_target = measured_interferograms[:, i]
            
            loss += self.mse_loss(interferogram_pred, interferogram_target)
        
        return loss / len(phase_shifts)

