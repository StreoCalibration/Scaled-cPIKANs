
import os
import torch
import torch.optim as optim
from tqdm import tqdm

class Trainer:
    """
    모델 훈련 루프를 관리하는 클래스.
    """
    def __init__(self, model, loss_fn, config):
        """
        Trainer를 초기화합니다.

        Args:
            model (nn.Module): 훈련할 PyTorch 모델
            loss_fn (nn.Module): 손실 함수
            config (dict): 훈련 설정 (learning_rate, epochs 등)
        """
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    def train(self, data_loader):
        """
        주어진 데이터로 모델을 훈련합니다.

        Args:
            data_loader (DataLoader): 훈련 데이터 로더
        """
        epochs = self.config['epochs']
        self.model.train() # 모델을 훈련 모드로 설정

        for epoch in range(epochs):
            running_loss = 0.0
            # tqdm을 사용하여 진행률 표시
            progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for coords, interferograms_target in progress_bar:
                coords = coords.to(self.device)
                interferograms_target = interferograms_target.to(self.device)

                # Forward pass
                u_pred = self.model(coords)
                loss = self.loss_fn(u_pred, interferograms_target)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                # 진행률 바에 현재 손실 표시
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_loss = running_loss / len(data_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")
        
        print("Training finished.")
        self._save_checkpoint()

    def _save_checkpoint(self):
        """훈련된 모델 가중치를 저장합니다."""
        path = self.config['checkpoint_path']
        if not os.path.exists(path):
            os.makedirs(path)
        
        checkpoint_file = os.path.join(path, "model_checkpoint.pth")
        torch.save(self.model.state_dict(), checkpoint_file)
        print(f"Model checkpoint saved to {checkpoint_file}")

