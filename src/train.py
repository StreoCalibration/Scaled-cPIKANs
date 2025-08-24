import torch
from collections import defaultdict

class Trainer:
    """
    PINN 모델의 최적화를 처리하는 트레이너 클래스.

    이 클래스는 설계 문서에 설명된 2단계 훈련 과정(Adam + L-BFGS)을 구현합니다.
    훈련 과정 동안 다양한 손실 요소의 기록을 로깅합니다.
    """
    def __init__(self, model, loss_fn):
        """
        Args:
            model (torch.nn.Module): 훈련할 PINN 모델.
            loss_fn (callable): 물리 정보 손실 함수.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.device = next(model.parameters()).device
        self.history = defaultdict(list)
        self.lbfgs_loss_dict = {} # L-BFGS 클로저에서 손실 딕셔너리를 저장하기 위함

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
        먼저 Adam으로, 그 다음 L-BFGS로 전체 훈련 과정을 실행합니다.

        Args:
            pde_points (torch.Tensor): PDE 잔차를 위한 콜로케이션 포인트.
            bc_points_dicts (list): 경계 조건 포인트들을 위한 딕셔너리 리스트.
            ic_points_dicts (list, optional): 초기 조건 포인트들을 위한 딕셔너리 리스트.
            data_points (tuple, optional): 데이터 손실을 위한 (입력, 실제 값) 튜플.
            adam_epochs (int): Adam 옵티마이저의 에포크 수.
            lbfgs_epochs (int): L-BFGS 옵티마이저의 에포크/스텝 수.
            adam_lr (float): Adam 옵티마이저의 학습률.
            log_interval (int): 손실 정보를 얼마나 자주 출력할지 결정하는 간격.

        Returns:
            dict: 모든 손실 요소의 기록을 담은 딕셔너리.
        """
        print("--- 1단계 시작: Adam 최적화 ---")
        self._train_adam(pde_points, bc_points_dicts, ic_points_dicts, data_points, adam_epochs, adam_lr, log_interval)

        if lbfgs_epochs > 0:
            print("\n--- 2단계 시작: L-BFGS 최적화 ---")
            self._train_lbfgs(pde_points, bc_points_dicts, ic_points_dicts, data_points, lbfgs_epochs, log_interval)

        print("\n--- 훈련 종료 ---")
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
            self.lbfgs_loss_dict = loss_dict # 로깅을 위해 저장
            total_loss.backward()
            return total_loss

        # L-BFGS는 한 스텝에서 여러 함수 평가를 수행하므로, Adam처럼 반복하지 않습니다.
        optimizer.step(closure)

        # L-BFGS 최적화 스텝 후 최종 상태를 로깅합니다.
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
        log_str = f"[{stage}] 에포크 [{epoch+1}/{total_epochs}]"
        for key, value in loss_dict.items():
            log_str += f" - {key}: {value.item():.4e}"
        print(log_str)

    @property
    def adam_epochs(self):
        """완료된 Adam 에포크 수를 반환하여 정확한 에포크 계산을 돕습니다."""
        return len([s for s in self.history['stage'] if s == "Adam"])
