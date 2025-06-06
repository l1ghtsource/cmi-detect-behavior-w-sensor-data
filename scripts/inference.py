import pandas as pd
import polars as pl
import numpy as np
from scipy.stats import mode
import gc

import torch
from torch.utils.data import DataLoader

from configs.config import cfg
from data.ts_datasets import TS_CMIDataset, TS_Demo_CMIDataset
from models.ts_models import TS_MSModel, TS_IMUModel, TS_Demo_MSModel, TS_Demo_IMUModel
from utils.data_preproc import fast_seq_agg, le

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train = pd.read_csv(cfg.train_path)
train, label_encoder = le(train)
del train
gc.collect()

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    test_df = sequence.to_pandas()
    
    if not demographics.is_empty():
        test_demographics = demographics.to_pandas()
        test_df = test_df.merge(test_demographics, how='left', on='subject')

    processed_df_for_dataset = fast_seq_agg(test_df) 

    TSDataset = TS_Demo_CMIDataset if cfg.use_demo else TS_CMIDataset

    test_dataset = TSDataset(
        dataframe=processed_df_for_dataset,
        seq_len=cfg.seq_len, 
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0) 
    
    num_folds = 5
    all_fold_predicted_indices = [] 
    all_fold_logits = []

    for i in range(num_folds):
        if cfg.imu_only:
            TSModel = TS_Demo_IMUModel if cfg.use_demo else TS_IMUModel
        else:
            TSModel = TS_Demo_MSModel if cfg.use_demo else TS_MSModel

        if cfg.imu_only:
            m_params = {
                'imu_features': len(cfg.imu_cols),
                'num_classes': cfg.num_classes,
                'hidden_dim': 128
            }
            if cfg.use_demo:
                m_params['demo_features'] = len(cfg.demo_cols)
        else:
            m_params = {
                'imu_features': len(cfg.imu_cols),
                'thm_features': len(cfg.thm_cols),
                'tof_features': len(cfg.tof_cols),
                'num_classes': cfg.num_classes,
                'hidden_dim': 128
            }
            if cfg.use_demo:
                m_params['demo_features'] = len(cfg.demo_cols)

        model = TSModel(**m_params).to(device)

        model_path = f'{cfg.weights_pathes}/model_fold{i}.pt'
        model.load_state_dict(torch.load(model_path, map_location=device))

        if cfg.use_ema:
            model_path = f'{cfg.weights_pathes}/model_ema_fold{i}.pt'
            ema_state_dict = torch.load(model_path, map_location=device)
            for name, param in model.named_parameters():
                if name in ema_state_dict:
                    param.data = ema_state_dict[name]
            
        model.eval()
        
        current_fold_batch_logits = []
        with torch.no_grad():
            for batch in test_loader:
                imu_inputs = batch['imu'].to(device)
                
                if cfg.imu_only:
                    if cfg.use_demo:
                        demo_inputs = batch['demographics'].to(device)
                        outputs = model(imu_inputs, demo_inputs)
                    else:
                        outputs = model(imu_inputs)
                else:
                    thm_inputs = batch['thm'].to(device)
                    tof_inputs = batch['tof'].to(device)
                    if cfg.use_demo:
                        demo_inputs = batch['demographics'].to(device)
                        outputs = model(imu_inputs, thm_inputs, tof_inputs, demo_inputs)
                    else:
                        outputs = model(imu_inputs, thm_inputs, tof_inputs)
                        
                current_fold_batch_logits.append(outputs.cpu().numpy())
        
        concatenated_fold_logits = np.concatenate(current_fold_batch_logits, axis=0)
        
        if cfg.is_soft:
            all_fold_logits.append(concatenated_fold_logits)
        else:
            predicted_indices_for_fold = np.argmax(concatenated_fold_logits, axis=1)
            all_fold_predicted_indices.append(predicted_indices_for_fold)
    
    if cfg.is_soft:
        # soft voting: average the logits and then take the argmax
        averaged_logits = np.mean(np.array(all_fold_logits), axis=0)
        majority_vote_indices = np.argmax(averaged_logits, axis=1)
    else:
        # hard voting: take the mode of the predicted indices
        predictions_from_all_folds_array = np.array(all_fold_predicted_indices)
        majority_vote_indices, _ = mode(predictions_from_all_folds_array, axis=0, keepdims=False)
    
    final_predicted_labels_orig = label_encoder.inverse_transform(majority_vote_indices)

    return final_predicted_labels_orig[0]