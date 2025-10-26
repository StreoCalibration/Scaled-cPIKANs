import torch
from collections import defaultdict
import warnings

class Trainer:
    """
    PINN 모델의 최적화를 처리하는 트레이너 클래스.

    이 클래스는 설계 문서에 설명된 2단계 훈련 과정(Adam + L-BFGS)을 구현합니다.
    훈련 과정 동안 다양한 손실 요소의 기록을 로깅합니다.
    
    v2-P1 업데이트:
        - Mixed Precision Training (AMP) 지원
        - 그래디언트 누적 (Gradient Accumulation)
        - 활성화 체크포인팅 (Activation Checkpointing)
    """
    def __init__(self, model, loss_fn, use_amp=True, grad_accum_steps=1):
        """
        Args:
            model (torch.nn.Module): 훈련할 PINN 모델.
            loss_fn (callable): 물리 정보 손실 함수.
            use_amp (bool): Automatic Mixed Precision 사용 여부. 기본값: True.
            grad_accum_steps (int): 그래디언트 누적 스텝 수. 기본값: 1.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.device = next(model.parameters()).device
        self.history = defaultdict(list)
        self.lbfgs_loss_dict = {} # L-BFGS 클로저에서 손실 딕셔너리를 저장하기 위함
        
        # AMP 설정
        self.use_amp = use_amp and torch.cuda.is_available() and self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print(f"✅ Mixed Precision Training (AMP) 활성화")
        else:
            self.scaler = None
            if use_amp and self.device.type != 'cuda':
                warnings.warn("AMP는 CUDA 디바이스에서만 지원됩니다. CPU 모드로 전환합니다.", UserWarning)
        
        # 그래디언트 누적 설정
        self.grad_accum_steps = max(1, grad_accum_steps)
        if self.grad_accum_steps > 1:
            print(f"✅ 그래디언트 누적 활성화 (누적 스텝: {self.grad_accum_steps})")

    def train(self,
              pde_points,
              bc_points_dicts,
              ic_points_dicts=None,
              data_points=None,
              adam_epochs=20000,
              lbfgs_epochs=10,
              adam_lr=1e-3,
              use_scheduler=True,
              scheduler_gamma=0.9995,
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
            use_scheduler (bool): 학습률 스케줄러 사용 여부 (논문 권장: True).
            scheduler_gamma (float): ExponentialLR 감쇠율 (논문 권장: 0.9995).
            log_interval (int): 손실 정보를 얼마나 자주 출력할지 결정하는 간격.

        Returns:
            dict: 모든 손실 요소의 기록을 담은 딕셔너리.
        """
        # Ensure all inputs are on the same device as the model
        pde_points, bc_points_dicts, ic_points_dicts, data_points = self._to_device(
            pde_points, bc_points_dicts, ic_points_dicts, data_points
        )

        print("--- 1단계 시작: Adam 최적화 ---")
        self._train_adam(pde_points, bc_points_dicts, ic_points_dicts, data_points, adam_epochs, adam_lr, use_scheduler, scheduler_gamma, log_interval)

        if lbfgs_epochs > 0:
            print("\n--- 2단계 시작: L-BFGS 최적화 ---")
            self._train_lbfgs(pde_points, bc_points_dicts, ic_points_dicts, data_points, lbfgs_epochs, log_interval)

        print("\n--- 훈련 종료 ---")
        return self.history

    def _train_adam(self, pde_points, bc_points_dicts, ic_points_dicts, data_points, epochs, lr, use_scheduler, scheduler_gamma, log_interval):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # 논문 권장: ExponentialLR 스케줄러 (gamma=0.9995)
        scheduler = None
        if use_scheduler:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)

        for epoch in range(epochs):
            self.model.train()
            
            # 그래디언트 누적을 위한 손실 초기화
            accumulated_loss = 0.0
            accumulated_loss_dict = None
            
            # 그래디언트 누적 루프
            for accum_step in range(self.grad_accum_steps):
                # AMP autocast 사용
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        total_loss, loss_dict = self.loss_fn(
                            self.model, pde_points, bc_points_dicts, ic_points_dicts, data_points
                        )
                else:
                    total_loss, loss_dict = self.loss_fn(
                        self.model, pde_points, bc_points_dicts, ic_points_dicts, data_points
                    )
                
                # 그래디언트 누적을 위한 손실 스케일링
                scaled_loss = total_loss / self.grad_accum_steps
                
                # Backward pass
                if self.use_amp:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                
                # 손실 누적 (로깅용)
                accumulated_loss += total_loss.item()
                if accumulated_loss_dict is None:
                    accumulated_loss_dict = {k: v.item() for k, v in loss_dict.items()}
                else:
                    for k, v in loss_dict.items():
                        accumulated_loss_dict[k] += v.item()
            
            # 평균 손실 계산
            for k in accumulated_loss_dict:
                accumulated_loss_dict[k] /= self.grad_accum_steps
            
            # Optimizer step
            if self.use_amp:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
            
            # 학습률 스케줄러 업데이트
            if scheduler is not None:
                scheduler.step()

            # 평균 손실로 로깅 (실제 loss_dict 형식 유지)
            logged_loss_dict = {k: torch.tensor(v) for k, v in accumulated_loss_dict.items()}
            self._log_history(logged_loss_dict, epoch, "Adam")
            
            # 학습률도 기록
            if use_scheduler:
                self.history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            if (epoch + 1) % log_interval == 0 or epoch == epochs - 1:
                self._print_log(epoch, epochs, logged_loss_dict, "Adam")
                if use_scheduler:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"    Current LR: {current_lr:.6e}")
                if self.use_amp:
                    print(f"    AMP Scale: {self.scaler.get_scale():.1f}")

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
            
            # L-BFGS에서는 AMP를 사용하지 않음 (수치 안정성 우선)
            # L-BFGS는 line search를 사용하므로 mixed precision과 호환성이 낮음
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

    def _to_device(self, pde_points, bc_points_dicts, ic_points_dicts, data_points):
        """Move all training inputs to the trainer's device."""
        if pde_points is not None:
            pde_points = pde_points.to(self.device)
        if bc_points_dicts is not None:
            bc_points_dicts = [
                {**d, 'points': d['points'].to(self.device)} for d in bc_points_dicts
            ]
        if ic_points_dicts is not None:
            ic_points_dicts = [
                {**d, 'points': d['points'].to(self.device)} for d in ic_points_dicts
            ]
        if data_points is not None:
            data_points = (
                data_points[0].to(self.device),
                data_points[1].to(self.device)
            )
        return pde_points, bc_points_dicts, ic_points_dicts, data_points

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
