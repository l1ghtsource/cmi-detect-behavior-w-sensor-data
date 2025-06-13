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

# TODO: multigpu inference

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train = pd.read_csv(cfg.train_path)
train = le(train)
train_seq = fast_seq_agg(train)

reverse_mapping, aux1_reverse_mapping, aux2_reverse_mapping = get_rev_mapping()

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

    prefix = get_prefix(use_imu_only)

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
    all_fold_predicted_indices_aux2 = []
    all_fold_logits_aux2 = []

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
        current_fold_batch_logits_aux2 = []
        with torch.no_grad():
            for batch in test_loader:
                for key in batch.keys():
                    batch[key] = batch[key].to(device)
                if not use_tta:
                    outputs, aux2_outputs = forward_model(model, batch, imu_only=use_imu_only)
                    current_fold_batch_logits.append(outputs.cpu().numpy())
                    current_fold_batch_logits_aux2.append(aux2_outputs.cpu().numpy())
                else:
                    for key in batch.keys():
                        batch[key] = batch[key].cpu()
                    augmented_batches = apply_tta(batch, cfg.tta_strategies)
                    for key in batch.keys():
                        batch[key] = batch[key].to(device)
                    batch_tta_logits = []
                    batch_tta_logits_aux2 = []
                    for aug_batch in augmented_batches:
                        for key in aug_batch.keys():
                            aug_batch[key] = aug_batch[key].to(device)
                        outputs, aux2_outputs = forward_model(model, aug_batch, imu_only=use_imu_only)
                        batch_tta_logits.append(outputs.cpu().numpy())
                        batch_tta_logits_aux2.append(aux2_outputs.cpu().numpy())
                    avg_tta_logits = np.mean(batch_tta_logits, axis=0)
                    avg_tta_logits_aux2 = np.mean(batch_tta_logits_aux2, axis=0)
                    current_fold_batch_logits.append(avg_tta_logits)
                    current_fold_batch_logits_aux2.append(avg_tta_logits_aux2)

        concatenated_fold_logits = np.concatenate(current_fold_batch_logits, axis=0)
        concatenated_fold_logits_aux2 = np.concatenate(current_fold_batch_logits_aux2, axis=0)

        if cfg.is_soft:
            all_fold_logits.append(concatenated_fold_logits)
            all_fold_logits_aux2.append(concatenated_fold_logits_aux2)
        else:
            if cfg.use_entmax:
                logits_tensor = torch.tensor(concatenated_fold_logits, device=device)
                entmax_probs = entmax_bisect(logits_tensor, alpha=cfg.entmax_alpha, dim=1)
                predicted_indices_for_fold = torch.argmax(entmax_probs, dim=1).cpu().numpy()
                logits_tensor_aux2 = torch.tensor(concatenated_fold_logits_aux2, device=device)
                entmax_probs_aux2 = entmax_bisect(logits_tensor_aux2, alpha=cfg.entmax_alpha, dim=1)
                predicted_indices_for_fold_aux2 = torch.argmax(entmax_probs_aux2, dim=1).cpu().numpy()
            else:
                predicted_indices_for_fold = np.argmax(concatenated_fold_logits, axis=1)
                predicted_indices_for_fold_aux2 = np.argmax(concatenated_fold_logits_aux2, axis=1)

            all_fold_predicted_indices.append(predicted_indices_for_fold)
            all_fold_predicted_indices_aux2.append(predicted_indices_for_fold_aux2)

    if cfg.is_soft:
        # soft voting: average the logits and then apply entmax or argmax
        averaged_logits = np.mean(np.array(all_fold_logits), axis=0)
        averaged_logits_aux2 = np.mean(np.array(all_fold_logits_aux2), axis=0)
        if cfg.use_entmax:
            averaged_logits_tensor = torch.tensor(averaged_logits, device=device)
            entmax_probs = entmax_bisect(averaged_logits_tensor, alpha=cfg.entmax_alpha, dim=1)
            majority_vote_indices = torch.argmax(entmax_probs, dim=1).cpu().numpy()
            averaged_logits_tensor_aux2 = torch.tensor(averaged_logits_aux2, device=device)
            entmax_probs_aux2 = entmax_bisect(averaged_logits_tensor_aux2, alpha=cfg.entmax_alpha, dim=1)
            majority_vote_indices_aux2 = torch.argmax(entmax_probs_aux2, dim=1).cpu().numpy()
        else:
            majority_vote_indices = np.argmax(averaged_logits, axis=1)
            majority_vote_indices_aux2 = np.argmax(averaged_logits_aux2, axis=1)
    else:
        # hard voting: take the mode of the predicted indices
        predictions_from_all_folds_array = np.array(all_fold_predicted_indices)
        majority_vote_indices, _ = mode(predictions_from_all_folds_array, axis=0, keepdims=False)
        predictions_from_all_folds_array_aux2 = np.array(all_fold_predicted_indices_aux2)
        majority_vote_indices_aux2, _ = mode(predictions_from_all_folds_array_aux2, axis=0, keepdims=False)

    if cfg.override_non_target:
        final_preds = []
        for output_idx, aux2_idx in zip(majority_vote_indices, majority_vote_indices_aux2):
            if output_idx <= 7 and aux2_idx == 0:
                final_preds.append(8) # override with first non-target class
            else:
                final_preds.append(output_idx)
        final_labels = [reverse_mapping[class_idx] for class_idx in final_preds]
    else:
        final_labels = [reverse_mapping[class_idx] for class_idx in majority_vote_indices]

    return final_labels[0]