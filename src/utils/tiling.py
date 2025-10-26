"""
슬라이딩 윈도우 기반 대규모 이미지 추론 유틸리티

이 모듈은 9344×7000급 초대형 이미지를 작은 타일로 분할하고,
각 타일에 대해 추론을 수행한 후, 오버랩 블렌딩으로 결과를 재조립하는
기능을 제공합니다.

주요 기능:
    - tile_image(): 이미지를 오버랩된 타일로 분할
    - blend_tiles(): Hanning 윈도우 블렌딩으로 타일 재조립
    - infer_with_tiling(): 전체 타일 기반 추론 파이프라인

참고:
    U-Net (Ronneberger et al., 2015)의 슬라이딩 윈도우 전략 기반
    TODO_v2.md v2-P1 작업 항목
"""

import torch
import numpy as np
from typing import Tuple, List, Optional, Callable
import warnings


def tile_image(
    image: np.ndarray,
    tile_size: int = 512,
    overlap: int = 128
) -> Tuple[List[Tuple[np.ndarray, Tuple[int, int, int, int]]], Tuple[int, ...]]:
    """
    이미지를 오버랩된 타일로 분할합니다.
    
    Args:
        image (np.ndarray): 입력 이미지, 형태 (C, H, W) 또는 (H, W).
        tile_size (int): 타일 크기 (정사각형). 기본값: 512.
        overlap (int): 타일 간 오버랩 픽셀 수. 기본값: 128.
    
    Returns:
        tiles (List[Tuple[np.ndarray, Tuple[int, int, int, int]]]): 
            각 타일과 해당 위치 정보 (y_start, y_end, x_start, x_end)의 리스트.
        original_shape (Tuple[int, ...]): 원본 이미지 형태.
    
    Raises:
        ValueError: tile_size가 overlap보다 작거나 같을 때.
        ValueError: 이미지 차원이 2D 또는 3D가 아닐 때.
    
    예제:
        >>> img = np.random.rand(3, 1024, 1024)
        >>> tiles, shape = tile_image(img, tile_size=512, overlap=128)
        >>> print(f"생성된 타일 수: {len(tiles)}")
    """
    if tile_size <= overlap:
        raise ValueError(f"tile_size({tile_size})는 overlap({overlap})보다 커야 합니다.")
    
    original_shape = image.shape
    
    # 2D 또는 3D 이미지 지원
    if image.ndim == 2:
        # 그레이스케일 (H, W) -> (1, H, W)
        image = image[np.newaxis, ...]
        is_2d = True
    elif image.ndim == 3:
        # 컬러 (C, H, W)
        is_2d = False
    else:
        raise ValueError(f"이미지는 2D (H, W) 또는 3D (C, H, W) 형태여야 합니다. 현재: {image.shape}")
    
    channels, height, width = image.shape
    stride = tile_size - overlap
    
    tiles = []
    
    # y 방향 타일링
    y_positions = list(range(0, height, stride))
    # 마지막 타일이 이미지 경계를 넘어가면 조정
    if y_positions[-1] + tile_size > height:
        y_positions[-1] = max(0, height - tile_size)
    
    # x 방향 타일링
    x_positions = list(range(0, width, stride))
    if x_positions[-1] + tile_size > width:
        x_positions[-1] = max(0, width - tile_size)
    
    # 타일 추출
    for y_start in y_positions:
        y_end = min(y_start + tile_size, height)
        for x_start in x_positions:
            x_end = min(x_start + tile_size, width)
            
            # 타일 크기가 tile_size보다 작으면 제로 패딩
            tile = image[:, y_start:y_end, x_start:x_end]
            
            if tile.shape[1] < tile_size or tile.shape[2] < tile_size:
                # 패딩 필요
                padded_tile = np.zeros((channels, tile_size, tile_size), dtype=image.dtype)
                padded_tile[:, :tile.shape[1], :tile.shape[2]] = tile
                tile = padded_tile
            
            tiles.append((tile, (y_start, y_end, x_start, x_end)))
    
    return tiles, original_shape


def create_hanning_window(size: int, overlap: int, device: str = 'cpu') -> torch.Tensor:
    """
    2D Hanning 윈도우를 생성합니다 (블렌딩용).
    
    오버랩 영역에서는 윈도우 값이 0에서 1로 부드럽게 변화하며,
    중앙 영역에서는 1의 값을 가집니다.
    
    Args:
        size (int): 타일 크기.
        overlap (int): 오버랩 픽셀 수.
        device (str): 텐서를 생성할 디바이스. 기본값: 'cpu'.
    
    Returns:
        torch.Tensor: 2D Hanning 윈도우, 형태 (size, size).
    
    예제:
        >>> window = create_hanning_window(512, 128, device='cuda')
        >>> print(window.shape)
        torch.Size([512, 512])
    """
    # 1D Hanning 윈도우 생성
    window_1d = np.ones(size)
    
    # 오버랩 영역에 Hanning 함수 적용
    if overlap > 0:
        hanning = np.hanning(2 * overlap)
        # 왼쪽/위쪽 오버랩
        window_1d[:overlap] = hanning[:overlap]
        # 오른쪽/아래쪽 오버랩
        window_1d[-overlap:] = hanning[overlap:]
    
    # 2D 윈도우 생성 (outer product)
    window_2d = np.outer(window_1d, window_1d)
    
    return torch.from_numpy(window_2d).float().to(device)


def blend_tiles(
    tiles: List[Tuple[np.ndarray, Tuple[int, int, int, int]]],
    original_shape: Tuple[int, ...],
    tile_size: int = 512,
    overlap: int = 128,
    device: str = 'cpu'
) -> np.ndarray:
    """
    타일들을 Hanning 윈도우 블렌딩으로 재조립합니다.
    
    오버랩 영역에서는 가중 평균을 통해 부드러운 전환을 보장하여
    seam artifact를 제거합니다.
    
    Args:
        tiles (List[Tuple[np.ndarray, Tuple[int, int, int, int]]]): 
            타일 리스트 (각 타일과 위치 정보).
        original_shape (Tuple[int, ...]): 원본 이미지 형태 (C, H, W) 또는 (H, W).
        tile_size (int): 타일 크기. 기본값: 512.
        overlap (int): 오버랩 픽셀 수. 기본값: 128.
        device (str): 계산에 사용할 디바이스. 기본값: 'cpu'.
    
    Returns:
        np.ndarray: 재조립된 이미지, 형태는 original_shape와 동일.
    
    예제:
        >>> tiles, shape = tile_image(img, 512, 128)
        >>> # ... 각 타일에 대해 추론 수행 ...
        >>> result = blend_tiles(tiles, shape, 512, 128)
    """
    # 원본 형태 분석
    if len(original_shape) == 2:
        # 2D 이미지
        height, width = original_shape
        channels = 1
        is_2d = True
    elif len(original_shape) == 3:
        # 3D 이미지
        channels, height, width = original_shape
        is_2d = False
    else:
        raise ValueError(f"지원하지 않는 이미지 형태: {original_shape}")
    
    # 첫 번째 타일에서 채널 수 확인 (추론 결과의 채널 수)
    first_tile_data, _ = tiles[0]
    if first_tile_data.ndim == 2:
        output_channels = 1
        first_tile_data = first_tile_data[np.newaxis, ...]
    else:
        output_channels = first_tile_data.shape[0]
    
    # 출력 이미지 및 가중치 맵 초기화
    output = torch.zeros((output_channels, height, width), dtype=torch.float32, device=device)
    weight_map = torch.zeros((height, width), dtype=torch.float32, device=device)
    
    # Hanning 윈도우 생성
    hanning_window = create_hanning_window(tile_size, overlap, device=device)
    
    # 타일 블렌딩
    for tile_data, (y_start, y_end, x_start, x_end) in tiles:
        # 타일을 텐서로 변환
        if tile_data.ndim == 2:
            tile_data = tile_data[np.newaxis, ...]
        tile_tensor = torch.from_numpy(tile_data).float().to(device)
        
        # 실제 타일 크기 계산 (경계에서는 작을 수 있음)
        actual_h = y_end - y_start
        actual_w = x_end - x_start
        
        # 윈도우 마스크 추출 (실제 크기에 맞춤)
        window_mask = hanning_window[:actual_h, :actual_w]
        
        # 타일의 실제 부분만 사용
        tile_crop = tile_tensor[:, :actual_h, :actual_w]
        
        # 가중 합산
        output[:, y_start:y_end, x_start:x_end] += tile_crop * window_mask
        weight_map[y_start:y_end, x_start:x_end] += window_mask
    
    # 가중치로 정규화 (0으로 나누기 방지)
    weight_map = torch.clamp(weight_map, min=1e-8)
    output = output / weight_map.unsqueeze(0)
    
    # NumPy로 변환
    result = output.cpu().numpy()
    
    # 2D 이미지였으면 차원 제거
    if is_2d:
        result = result.squeeze(0)
    
    # 원본 채널 수와 다르면 조정 (추론 결과가 다른 채널 수를 가질 수 있음)
    # 이 경우 그대로 반환
    return result


def infer_with_tiling(
    image: np.ndarray,
    model: torch.nn.Module,
    tile_size: int = 512,
    overlap: int = 128,
    device: str = 'cuda',
    batch_size: int = 1,
    verbose: bool = True
) -> np.ndarray:
    """
    타일 기반 추론을 수행하는 전체 파이프라인.
    
    대규모 이미지를 작은 타일로 분할하고, 각 타일에 대해 모델 추론을 수행한 후,
    Hanning 윈도우 블렌딩으로 결과를 재조립합니다.
    
    Args:
        image (np.ndarray): 입력 이미지, 형태 (C, H, W).
        model (torch.nn.Module): 추론에 사용할 PyTorch 모델.
        tile_size (int): 타일 크기. 기본값: 512.
        overlap (int): 오버랩 픽셀 수. 기본값: 128.
        device (str): 추론 디바이스. 기본값: 'cuda'.
        batch_size (int): 배치 크기 (여러 타일 병렬 처리). 기본값: 1.
        verbose (bool): 진행 상황 출력 여부. 기본값: True.
    
    Returns:
        np.ndarray: 추론 결과, 형태 (C_out, H, W).
    
    예제:
        >>> model = UNet(n_channels=16, n_classes=1).to('cuda')
        >>> model.eval()
        >>> large_img = np.random.rand(16, 9344, 7000)
        >>> result = infer_with_tiling(large_img, model, tile_size=512, overlap=128)
    
    Notes:
        - 모델은 반드시 eval() 모드여야 합니다.
        - 배치 처리로 추론 속도를 향상시킬 수 있습니다.
        - GPU 메모리가 부족하면 tile_size나 batch_size를 줄이세요.
    """
    if verbose:
        print(f"🔍 타일 기반 추론 시작")
        print(f"   이미지 크기: {image.shape}")
        print(f"   타일 크기: {tile_size}, 오버랩: {overlap}")
        print(f"   디바이스: {device}")
    
    # 모델을 평가 모드로 전환
    model.eval()
    model.to(device)
    
    # 타일 분할
    tiles, original_shape = tile_image(image, tile_size=tile_size, overlap=overlap)
    
    if verbose:
        print(f"   생성된 타일 수: {len(tiles)}")
    
    # 추론 수행
    predicted_tiles = []
    
    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i:i + batch_size]
            
            # 배치 준비
            batch_inputs = torch.stack([
                torch.from_numpy(tile_data).float()
                for tile_data, _ in batch_tiles
            ]).to(device)
            
            # 추론
            batch_outputs = model(batch_inputs)
            
            # 결과 저장
            for j, output in enumerate(batch_outputs):
                tile_idx = i + j
                _, position = tiles[tile_idx]
                predicted_tiles.append((output.cpu().numpy(), position))
            
            if verbose and (i + batch_size) % 10 == 0:
                print(f"   진행: {min(i + batch_size, len(tiles))}/{len(tiles)} 타일 완료")
    
    if verbose:
        print(f"   추론 완료. 블렌딩 시작...")
    
    # 블렌딩
    # 추론 결과의 실제 형태 사용 (입력과 다를 수 있음)
    if len(predicted_tiles) > 0:
        first_pred, _ = predicted_tiles[0]
        if first_pred.ndim == 2:
            output_shape = (original_shape[1], original_shape[2])  # (H, W)
        else:
            output_shape = (first_pred.shape[0], original_shape[1], original_shape[2])  # (C_out, H, W)
    else:
        output_shape = original_shape
    
    result = blend_tiles(
        predicted_tiles,
        original_shape=output_shape,
        tile_size=tile_size,
        overlap=overlap,
        device=device
    )
    
    if verbose:
        print(f"✅ 타일 기반 추론 완료. 결과 크기: {result.shape}")
    
    return result


# 편의 함수: 메모리 사용량 추정
def estimate_memory_usage(
    image_shape: Tuple[int, int, int],
    tile_size: int = 512,
    overlap: int = 128,
    model_params: int = 0,
    batch_size: int = 1,
    dtype: str = 'float32'
) -> dict:
    """
    타일 기반 추론의 예상 메모리 사용량을 추정합니다.
    
    Args:
        image_shape (Tuple[int, int, int]): 이미지 형태 (C, H, W).
        tile_size (int): 타일 크기. 기본값: 512.
        overlap (int): 오버랩 픽셀 수. 기본값: 128.
        model_params (int): 모델 파라미터 수. 기본값: 0.
        batch_size (int): 배치 크기. 기본값: 1.
        dtype (str): 데이터 타입. 'float32' 또는 'float16'. 기본값: 'float32'.
    
    Returns:
        dict: 메모리 사용량 정보 (MB 단위).
            - 'input_tile_mb': 입력 타일 메모리
            - 'output_tile_mb': 출력 타일 메모리
            - 'model_mb': 모델 메모리
            - 'total_mb': 총 예상 메모리
    
    예제:
        >>> memory = estimate_memory_usage((16, 9344, 7000), tile_size=512, overlap=128)
        >>> print(f"예상 메모리: {memory['total_mb']:.2f} MB")
    """
    bytes_per_element = 4 if dtype == 'float32' else 2
    
    channels, height, width = image_shape
    
    # 타일 수 계산
    stride = tile_size - overlap
    num_tiles_y = int(np.ceil(height / stride))
    num_tiles_x = int(np.ceil(width / stride))
    num_tiles = num_tiles_y * num_tiles_x
    
    # 입력 타일 메모리 (배치)
    input_tile_mb = (channels * tile_size * tile_size * bytes_per_element * batch_size) / (1024 ** 2)
    
    # 출력 타일 메모리 (가정: 1채널 출력)
    output_tile_mb = (1 * tile_size * tile_size * bytes_per_element * batch_size) / (1024 ** 2)
    
    # 모델 메모리
    model_mb = (model_params * 4) / (1024 ** 2)  # 파라미터는 보통 float32
    
    # 블렌딩용 전체 출력 버퍼
    blend_mb = (1 * height * width * bytes_per_element) / (1024 ** 2)
    
    total_mb = input_tile_mb + output_tile_mb + model_mb + blend_mb
    
    return {
        'num_tiles': num_tiles,
        'input_tile_mb': input_tile_mb,
        'output_tile_mb': output_tile_mb,
        'model_mb': model_mb,
        'blend_mb': blend_mb,
        'total_mb': total_mb
    }
