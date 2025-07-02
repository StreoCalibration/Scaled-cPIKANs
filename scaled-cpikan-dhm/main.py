

import yaml
import argparse
import torch # Add this line

from src.data_pipeline import DataManager, prepare_dataloader, save_processed_data, load_processed_data
from src.train_pipeline import run_training
from src.evaluate_pipeline import evaluate_and_visualize, load_trained_model

def main():
    parser = argparse.ArgumentParser(description="Run cPIKAN training and evaluation pipeline.")
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['data', 'train', 'evaluate', 'all'],
                        help="Mode to run: 'data' (generate/process data), 'train' (train model), 'evaluate' (evaluate model), 'all' (run all steps).")
    args = parser.parse_args()

    config_path = 'F:/Source/Test/scaled-cpikan-dhm/configs/microlens_config.yaml'

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.mode == 'data' or args.mode == 'all':
        print("--- Running DATA preparation mode ---")
        data_manager = DataManager(config_path)
        data_manager.data_config['mode'] = 'synthetic' 
        synthetic_images, ground_truth = data_manager.load_data()
        
        data_loader, image_dims = prepare_dataloader(synthetic_images, ground_truth, batch_size=1024)
        
        # DataLoader에서 Tensor를 추출하여 저장
        coords_list = []
        interferograms_target_list = []
        for coords, interferograms_target in data_loader:
            coords_list.append(coords)
            interferograms_target_list.append(interferograms_target)
        
        coords_all = torch.cat(coords_list, dim=0)
        interferograms_target_all = torch.cat(interferograms_target_list, dim=0)

        save_processed_data(config['data']['processed_data_path'], coords_all, interferograms_target_all, image_dims)
        print("DATA preparation complete.")

    if args.mode == 'train' or args.mode == 'all':
        print("\n--- Running TRAINING mode ---")
        # 훈련은 저장된 데이터를 불러와서 진행
        trained_model = run_training(config)
        print("TRAINING complete.")

    if args.mode == 'evaluate' or args.mode == 'all':
        print("\n--- Running EVALUATION mode ---")
        # 평가도 저장된 데이터를 불러와서 진행
        # 훈련 모드에서 모델을 반환받았다면 그 모델을 사용하고, 아니면 로드
        if args.mode == 'all': # 'all' 모드에서는 훈련된 모델을 바로 사용
            evaluate_and_visualize(trained_model, ground_truth, config, image_dims)
        else: # 'evaluate' 단독 모드에서는 모델을 로드
            loaded_model = load_trained_model(config)
            # ground_truth와 image_dims는 processed_data_path에서 로드
            _, _, image_dims_eval = load_processed_data(config['data']['processed_data_path'])
            # ground_truth는 data_pipeline에서 생성된 것을 사용
            data_manager_eval = DataManager(config_path=config_path) # config_path를 전달
            data_manager_eval.data_config['mode'] = 'synthetic' 
            _, ground_truth_eval = data_manager_eval.load_data()
            evaluate_and_visualize(loaded_model, ground_truth_eval, config, image_dims_eval)
        print("EVALUATION complete.")

if __name__ == '__main__':
    main()
