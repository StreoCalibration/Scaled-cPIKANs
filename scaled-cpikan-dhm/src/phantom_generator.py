
import numpy as np

class PhantomGenerator:
    """
    디지털 팬텀(가상 데이터)을 생성하는 클래스.
    microlens_config.yaml 파일의 'synthetic' 섹션을 기반으로 동작합니다.
    """
    def __init__(self, config):
        """
        PhantomGenerator를 초기화합니다.

        Args:
            config (dict): 설정 파일에서 'synthetic'에 해당하는 부분.
        """
        self.config = config
        self.optics = config['optics']
        self.geometry = config['geometry']
        self.noise_config = config.get('noise', {})
        # 시뮬레이션을 위한 기본 이미지 크기 설정
        self.image_size = (512, 512)

    def generate(self):
        """
        가상 데이터(간섭 무늬)와 Ground-Truth 3D 형상을 생성합니다.

        Returns:
            tuple: (interferograms, ground_truth)
                   - interferograms (np.array): 시뮬레이션된 간섭 무늬 이미지 세트 (4D array: phase_shifts, height, width)
                   - ground_truth (np.array): 검증에 사용할 3D 원본 형상 (2D array: height, width)
        """
        print("Generating digital phantom (microlens array)...")

        # 1. Ground-Truth 3D 형상 생성 (마이크로렌즈 어레이)
        ground_truth = self._create_microlens_array()

        # 2. DHM 측정 시뮬레이션 (간섭 무늬 생성)
        # 위상차(phase)는 3D 형상에 비례한다고 가정
        phase = (2 * np.pi / self.optics['wavelength_nm']) * ground_truth
        
        # 4가지 위상 천이(0, pi/2, pi, 3pi/2)를 시뮬레이션
        phase_shifts = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        interferograms = []
        for shift in phase_shifts:
            # 이상적인 간섭 무늬: I = A + B*cos(phase + shift)
            # 간단히 A=0.5, B=0.5로 가정
            interference = 0.5 + 0.5 * np.cos(phase + shift)
            interferograms.append(interference)
        
        interferograms = np.array(interferograms)

        # 3. 현실적인 노이즈 추가
        if self.noise_config.get('add_noise', False):
            interferograms = self._add_noise(interferograms)
            print("Added realistic noise to interferograms.")

        print("Phantom generation complete.")
        return interferograms, ground_truth

    def _create_microlens_array(self):
        """마이크로렌즈 어레이의 3D 형상을 numpy로 생성합니다."""
        pitch = self.geometry['lens_pitch_um']
        focal_length = self.geometry['focal_length_um']
        
        # 렌즈의 곡률반경 R 계산 (렌즈 제작자 공식의 근사)
        # n1=렌즈 굴절률, n2=매질 굴절률
        n1 = self.optics['refractive_index_lens']
        n2 = self.optics['refractive_index_medium']
        R = (n1 - n2) * focal_length

        x = np.arange(0, self.image_size[1])
        y = np.arange(0, self.image_size[0])
        xx, yy = np.meshgrid(x, y)

        # 주기적인 렌즈 패턴을 만들기 위해 좌표를 pitch로 나눈 나머지를 사용
        xx_mod = xx % pitch
        yy_mod = yy % pitch
        
        # 렌즈 중심으로부터의 거리
        r_sq = (xx_mod - pitch/2)**2 + (yy_mod - pitch/2)**2
        
        # 구면 방정식 z = R - sqrt(R^2 - r^2) 을 이용하여 렌즈의 높이(sag) 계산
        # 렌즈 영역 밖은 평평하게 처리
        sag = np.zeros_like(r_sq)
        mask = r_sq < (pitch/2)**2
        sag[mask] = R - np.sqrt(R**2 - r_sq[mask])
        
        return sag

    def _add_noise(self, images):
        """이미지에 샷 노이즈와 판독 노이즈를 추가합니다."""
        # 샷 노이즈 (푸아송 노이즈의 근사)
        shot_noise_level = self.noise_config.get('shot_noise_level', 0.0)
        noisy_images = images + np.random.randn(*images.shape) * shot_noise_level * np.sqrt(images)
        
        # 판독 노이즈 (가우시안 노이즈)
        readout_noise_std = self.noise_config.get('readout_noise_std', 0.0)
        noisy_images += np.random.randn(*images.shape) * readout_noise_std
        
        # PZT 오차는 위상 천이 단계에 적용해야 하지만, 여기서는 단순화하여 생략
        # pzt_error_std = self.noise_config.get('pzt_error_std', 0.0)

        return np.clip(noisy_images, 0, 1)

