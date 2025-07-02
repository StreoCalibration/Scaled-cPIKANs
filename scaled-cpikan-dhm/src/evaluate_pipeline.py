import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from src.models import Scaled_cPIKAN_Model
from src.data_pipeline import load_processed_data # 추가

def evaluate_and_visualize(model, ground_truth, config, image_dims):
    """
    훈련된 모델을 평가하고 결과를 시각화합니다.
    """
    H, W = image_dims
    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, H)
    xx, yy = np.meshgrid(x, y)
    coords = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # 평가 모드로 설정

    with torch.no_grad():
        u_pred_full = model(torch.from_numpy(coords).to(device))
        u_pred_full = u_pred_full.cpu().numpy()

    # 예측된 복소장에서 위상 복원
    phase_pred = np.arctan2(u_pred_full[:, 1], u_pred_full[:, 0])
    phase_pred_img = phase_pred.reshape(H, W)

    # 원본 위상 (Ground Truth로부터 계산)
    wavelength_nm = config['data']['synthetic']['optics']['wavelength_nm']
    phase_true = (2 * np.pi / wavelength_nm) * ground_truth

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(phase_true, cmap='twilight')
    axes[0].set_title('True Phase')
    im = axes[1].imshow(phase_pred_img, cmap='twilight')
    axes[1].set_title('Predicted Phase after Training')
    fig.colorbar(im, ax=axes[1], label='Phase (radians)')
    plt.suptitle('Phase Reconstruction Result')
    plt.show()

def load_trained_model(config):
    """
    저장된 모델 가중치를 로드합니다.
    """
    model_config = config['model']
    model = Scaled_cPIKAN_Model(
        layers=model_config['layers'],
        degree=model_config['degree'],
        domain_min=model_config['domain']['min'],
        domain_max=model_config['domain']['max']
    )
    checkpoint_path = os.path.join(config['training']['checkpoint_path'], "model_checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    print(f"Model loaded from {checkpoint_path}")
    return model

def run_evaluation(config, trained_model=None):
    """
    평가 파이프라인을 실행합니다.
    """
    # 저장된 처리 데이터와 원본 ground truth를 불러옴
    _, _, image_dims = load_processed_data(config['data']['processed_data_path'])
    
    gt_path = os.path.join(config['data']['synthetic']['save_path'], 'synthetic_ground_truth.npy')
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found at {gt_path}. Please run the data generation/training pipeline first.")
    ground_truth = np.load(gt_path)

    if trained_model is None:
        trained_model = load_trained_model(config)
    
    evaluate_and_visualize(trained_model, ground_truth, config, image_dims)