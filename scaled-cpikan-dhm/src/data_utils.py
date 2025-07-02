
import os
import numpy as np
import yaml
from src.phantom_generator import PhantomGenerator

class DataManager:
    """
    config 파일에 정의된 데이터 모드(mode)에 따라 
    가상(synthetic) 또는 실제(real) 데이터를 로드하고 전처리합니다.
    """
    def __init__(self, config_path):
        """
        DataManager를 초기화합니다.

        Args:
            config_path (str): 설정 파일(microlens_config.yaml)의 경로.
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.data_config = self.config['data']

    def load_data(self):
        """
        설정된 모드에 따라 데이터를 로드합니다.

        Returns:
            tuple: (images, ground_truth)
                   - images: 로드된 간섭 무늬 이미지 (가상 또는 실제)
                   - ground_truth: 3D 원본 형상 (가상 데이터의 경우에만 존재)
        """
        mode = self.data_config['mode']
        print(f"Data mode set to: '{mode}'")

        if mode == 'synthetic':
            return self._load_synthetic_data()
        elif mode == 'real':
            return self._load_real_data()
        else:
            raise ValueError(f"Invalid data mode: {mode}. Choose 'synthetic' or 'real'.")

    def _load_synthetic_data(self):
        """PhantomGenerator를 사용하여 가상 데이터를 생성하고 로드합니다."""
        synthetic_config = self.data_config['synthetic']
        generator = PhantomGenerator(synthetic_config)
        interferograms, ground_truth = generator.generate()
        
        # 생성된 데이터를 저장할지 여부 확인
        save_path = synthetic_config.get('save_path')
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.save(os.path.join(save_path, 'synthetic_interferograms.npy'), interferograms)
            np.save(os.path.join(save_path, 'synthetic_ground_truth.npy'), ground_truth)
            print(f"Saved synthetic data to {save_path}")

        return interferograms, ground_truth

    def _load_real_data(self):
        """
        실제 측정된 이미지 파일을 로드합니다.
        (현재는 플레이스홀더로, 가짜 이미지 파일을 생성하여 로드를 시뮬레이션합니다.)
        """
        real_config = self.data_config['real']
        data_path = real_config['path']
        print(f"Loading real data from: {data_path}")

        # --- 플레이스홀더: 실제 데이터가 없으므로 가짜 데이터 생성 ---
        if not os.path.exists(data_path):
            print(f"Real data directory not found. Creating dummy data at {data_path}")
            os.makedirs(data_path)
            # 4개의 512x512 가짜 간섭 무늬 이미지 생성
            dummy_images = (np.random.rand(4, 512, 512) * 255).astype(np.uint8)
            for i in range(4):
                # 실제 파일처럼 보이도록 이미지 파일명 저장 (예: frame_0.png)
                # 여기서는 numpy 배열로 간단히 저장합니다.
                np.save(os.path.join(data_path, f'real_interferogram_{i}.npy'), dummy_images[i])
        # ----------------------------------------------------------

        # 실제 데이터 로드 로직
        image_files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
        if not image_files:
            raise FileNotFoundError(f"No real data files (.npy) found in {data_path}")
        
        images = []
        for file_name in sorted(image_files):
            image = np.load(os.path.join(data_path, file_name))
            images.append(image)
        
        images = np.array(images)
        # 실제 데이터는 Ground-Truth가 없으므로 None을 반환
        return images, None

