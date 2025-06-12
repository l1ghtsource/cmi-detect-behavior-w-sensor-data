import pandas as pd
import polars as pl
import numpy as np
from scipy.stats import mode

import torch
from torch.utils.data import DataLoader
from entmax import entmax_bisect

from configs.config import cfg
from utils.getters import (
    get_ts_dataset, 
    get_ts_model_and_params, 
    forward_model, 
    get_prefix
)
from utils.data_preproc import fast_seq_agg, le, get_rev_mapping
from utils.tta import apply_tta

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train = pd.read_csv(cfg.train_path)
train = le(train)
train_seq = fast_seq_agg(train)

reverse_mapping = get_rev_mapping()

TSDataset = get_ts_dataset()

train_dataset = TSDataset(
    dataframe=train_seq,
    seq_len=cfg.seq_len,
    target_col=cfg.target,
    aux_target_col=cfg.aux_target,
    aux2_target_col=cfg.aux2_target,
    train=True
)

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    test_df = sequence.to_pandas()

    use_imu_only = False
    if 'tof_5_v63' in test_df.columns:
        tof_values = test_df['tof_5_v63']
        if tof_values.isna().all() or (tof_values == None).all():
            use_imu_only = True
    else:
        use_imu_only = True
    
    if not demographics.is_empty():
        test_demographics = demographics.to_pandas()
        test_df = test_df.merge(test_demographics, how='left', on='subject')

    processed_df_for_dataset = fast_seq_agg(test_df) 

    prefix = get_prefix()

    test_dataset = TSDataset(
        dataframe=processed_df_for_dataset,
        seq_len=cfg.seq_len,
        train=False,
        norm_stats=train_dataset.norm_stats
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0) 

    use_tta = len(cfg.tta_strategies) > 0 # strats != {}
    
    all_fold_predicted_indices = [] 
    all_fold_logits = []

    for i in range(cfg.n_splits):
        TSModel, m_params = get_ts_model_and_params(imu_only=use_imu_only)
        model = TSModel(**m_params).to(device)

        model_path = f'{cfg.weights_path}/{prefix}model_fold{i}.pt'
        model.load_state_dict(torch.load(model_path, map_location=device))

        if cfg.use_ema:
            model_path = f'{cfg.weights_path}/{prefix}model_ema_fold{i}.pt'
            ema_state_dict = torch.load(model_path, map_location=device)
            for name, param in model.named_parameters():
                if name in ema_state_dict:
                    param.data = ema_state_dict[name]
            
        model.eval()
        
        current_fold_batch_logits = []
        with torch.no_grad():
            for batch in test_loader:
                for key in batch.keys():
                    batch[key] = batch[key].to(device)
                if not use_tta:
                    outputs, _ = forward_model(model, batch, imu_only=use_imu_only)   
                    current_fold_batch_logits.append(outputs.cpu().numpy())
                else:
                    for key in batch.keys():
                        batch[key] = batch[key].cpu()
                    augmented_batches = apply_tta(batch, cfg.tta_strategies)
                    for key in batch.keys():
                        batch[key] = batch[key].to(device)
                    batch_tta_logits = []
                    for aug_batch in augmented_batches:
                        for key in aug_batch.keys():
                            aug_batch[key] = aug_batch[key].to(device)
                        outputs, _ = forward_model(model, aug_batch, imu_only=use_imu_only)
                        batch_tta_logits.append(outputs.cpu().numpy())
                    avg_tta_logits = np.mean(batch_tta_logits, axis=0)
                    current_fold_batch_logits.append(avg_tta_logits)
        
        concatenated_fold_logits = np.concatenate(current_fold_batch_logits, axis=0)
        
        if cfg.is_soft:
            all_fold_logits.append(concatenated_fold_logits)
        else:
            if cfg.use_entmax:
                logits_tensor = torch.tensor(concatenated_fold_logits, device=device)
                entmax_probs = entmax_bisect(logits_tensor, alpha=cfg.entmax_alpha, dim=1)
                predicted_indices_for_fold = torch.argmax(entmax_probs, dim=1).cpu().numpy()
            else:
                predicted_indices_for_fold = np.argmax(concatenated_fold_logits, axis=1)
            all_fold_predicted_indices.append(predicted_indices_for_fold)
    
    if cfg.is_soft:
        # soft voting: average the logits and then apply entmax or argmax
        averaged_logits = np.mean(np.array(all_fold_logits), axis=0)
        
        if cfg.use_entmax:
            averaged_logits_tensor = torch.tensor(averaged_logits, device=device)
            entmax_probs = entmax_bisect(averaged_logits_tensor, alpha=cfg.entmax_alpha, dim=1)
            majority_vote_indices = torch.argmax(entmax_probs, dim=1).cpu().numpy()
        else:
            majority_vote_indices = np.argmax(averaged_logits, axis=1)
    else:
        # hard voting: take the mode of the predicted indices
        predictions_from_all_folds_array = np.array(all_fold_predicted_indices)
        majority_vote_indices, _ = mode(predictions_from_all_folds_array, axis=0, keepdims=False)

    final_predicted_labels_orig = [reverse_mapping[class_idx] for class_idx in majority_vote_indices]

    return final_predicted_labels_orig[0]