import pandas as pd
import polars as pl
import numpy as np
from scipy.stats import mode
import glob
import os

import torch
from torch.utils.data import DataLoader
from entmax import entmax_bisect

from configs.config import cfg
from models.hybrid_model import (
    HybridModel_SingleSensor_v1,
    MultiSensor_HybridModel_v1
)
from utils.getters import (
    get_ts_dataset, 
    forward_model, 
)
from utils.data_preproc import (
    fast_seq_agg, 
    le, 
    convert_to_world_coordinates, 
    apply_symmetry,
    remove_gravity_from_acc,
    fe,
    get_rev_mapping
)
from utils.tta import apply_tta

# TODO: multigpu inference

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

reverse_mapping, aux1_reverse_mapping, aux2_reverse_mapping = get_rev_mapping()
TSDataset = get_ts_dataset()

loaded_models = {
    'imu_only': {},
    'imu+tof+thm': {}
}

use_tta = len(cfg.tta_strategies) > 0

for w_key in ['imu_only', 'imu+tof+thm']:
    if w_key not in cfg.weights_pathes:
        continue
        
    for weights_path, params in cfg.weights_pathes[w_key].items():
        print(f'loading models for {w_key}: {weights_path}')
        
        if w_key == 'imu_only':
            TSModel = HybridModel_SingleSensor_v1
        else:
            TSModel = MultiSensor_HybridModel_v1
        
        ckpt_files = sorted(
            glob.glob(os.path.join(weights_path, '*.pt'))
        )
        if not ckpt_files:
            print(f'  no checkpoints found in {weights_path}')
            continue

        fold_models = []
        for ckpt in ckpt_files:
            print(f'  loading checkpoint: {os.path.basename(ckpt)}')

            is_ema = ckpt.endswith('_ema.pt') or '_ema_' in os.path.basename(ckpt)
            model = TSModel(**params['model_params']).to(device)

            state_dict = torch.load(ckpt, map_location=device)
            model.load_state_dict(state_dict, strict=False)

            if not is_ema and cfg.use_ema:
                ema_ckpt = ckpt.replace('.pt', '_ema.pt')
                if os.path.isfile(ema_ckpt):
                    ema_state = torch.load(ema_ckpt, map_location=device)
                    for name, p in model.named_parameters():
                        if name in ema_state:
                            p.data.copy_(ema_state[name])

            model.eval()
            fold_models.append(model)

        loaded_models.setdefault(w_key, {})[weights_path] = {
            'models': fold_models,
            'params': params,
            'weight': params['weight']
        }

def preprocess_single_row(test_df, use_imu_only):
    if cfg.use_world_coords:
        test_df = convert_to_world_coordinates(test_df)

    if cfg.only_remove_g:
        test_df = remove_gravity_from_acc(test_df)

    if cfg.use_hand_symm and use_imu_only:
        right_handed_mask = test_df['handedness'] == 1
        test_df.loc[right_handed_mask, cfg.imu_cols] = apply_symmetry(test_df.loc[right_handed_mask, cfg.imu_cols])

    if cfg.apply_fe:
        test_df = fe(test_df)

    return fast_seq_agg(test_df)

def create_single_batch(processed_df):
    test_dataset = TSDataset(
        dataframe=processed_df,
        seq_len=cfg.seq_len,
        train=False,
    )
    
    batch = test_dataset[0]
    
    for key in batch.keys():
        batch[key] = batch[key].unsqueeze(0).to(device)
    
    return batch

def predict_single_batch(model, batch, use_imu_only):
    if not use_tta:
        outputs, aux2_outputs, *_ = forward_model(model, batch, imu_only=use_imu_only)
        return outputs.cpu().numpy(), aux2_outputs.cpu().numpy()
    else:
        for key in batch.keys():
            batch[key] = batch[key].cpu()
        augmented_batches = apply_tta(batch, cfg.tta_strategies)
        
        batch_tta_logits = []
        batch_tta_logits_aux2 = []
        
        for aug_batch in augmented_batches:
            for key in aug_batch.keys():
                aug_batch[key] = aug_batch[key].to(device)
            outputs, aux2_outputs, *_ = forward_model(model, batch, imu_only=use_imu_only)
            batch_tta_logits.append(outputs.cpu().numpy())
            batch_tta_logits_aux2.append(aux2_outputs.cpu().numpy())
        
        avg_tta_logits = np.mean(batch_tta_logits, axis=0)
        avg_tta_logits_aux2 = np.mean(batch_tta_logits_aux2, axis=0)
        
        return avg_tta_logits, avg_tta_logits_aux2

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

    processed_df = preprocess_single_row(test_df, use_imu_only)
    
    batch = create_single_batch(processed_df)
    
    all_model_averaged_logits = []
    all_model_averaged_logits_aux2 = []
    all_model_predictions = []
    all_model_predictions_aux2 = []
    model_weights = []

    w_key = 'imu_only' if use_imu_only else 'imu+tof+thm'
    
    for weights_path, model_data in loaded_models[w_key].items():
        model_weights.append(model_data['weight'])
        
        fold_logits = []
        fold_logits_aux2 = []

        with torch.no_grad():
            for i, model in enumerate(model_data['models']):
                logits, logits_aux2 = predict_single_batch(model, batch, use_imu_only)
                if cfg.use_entmax: # apply entmax
                    logits = entmax_bisect(torch.tensor(logits, device=device), alpha=cfg.entmax_alpha, dim=1).cpu().numpy()
                    logits_aux2 = entmax_bisect(torch.tensor(logits_aux2, device=device), alpha=cfg.entmax_alpha, dim=1).cpu().numpy()
                fold_logits.append(logits)
                fold_logits_aux2.append(logits_aux2)

        model_averaged_logits = np.mean(np.array(fold_logits), axis=0)
        model_averaged_logits_aux2 = np.mean(np.array(fold_logits_aux2), axis=0)
        
        all_model_averaged_logits.append(model_averaged_logits)
        all_model_averaged_logits_aux2.append(model_averaged_logits_aux2)
        
        if not cfg.is_soft:
            model_predictions = np.argmax(model_averaged_logits, axis=1)
            model_predictions_aux2 = np.argmax(model_averaged_logits_aux2, axis=1)
            all_model_predictions.append(model_predictions)
            all_model_predictions_aux2.append(model_predictions_aux2)

    if cfg.is_soft:
        model_weights = np.array(model_weights)
        
        all_model_averaged_logits = np.array(all_model_averaged_logits)
        all_model_averaged_logits_aux2 = np.array(all_model_averaged_logits_aux2)
        
        final_averaged_logits = np.sum(all_model_averaged_logits * model_weights[:, np.newaxis, np.newaxis], axis=0)
        final_averaged_logits_aux2 = np.sum(all_model_averaged_logits_aux2 * model_weights[:, np.newaxis, np.newaxis], axis=0)

        majority_vote_indices = np.argmax(final_averaged_logits, axis=1)
        majority_vote_indices_aux2 = np.argmax(final_averaged_logits_aux2, axis=1)
    else:
        all_models_predictions_array = np.array(all_model_predictions)
        majority_vote_indices, _ = mode(all_models_predictions_array, axis=0, keepdims=False)
        all_models_predictions_array_aux2 = np.array(all_model_predictions_aux2)
        majority_vote_indices_aux2, _ = mode(all_models_predictions_array_aux2, axis=0, keepdims=False)

    if cfg.override_non_target:
        final_preds = []
        for output_idx, aux2_idx in zip(majority_vote_indices, majority_vote_indices_aux2):
            if output_idx <= 7 and aux2_idx == 0:
                final_preds.append(8)
            else:
                final_preds.append(output_idx)
        final_labels = [reverse_mapping[class_idx] for class_idx in final_preds]
    else:
        final_labels = [reverse_mapping[class_idx] for class_idx in majority_vote_indices]

    return final_labels[0]