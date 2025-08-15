import pandas as pd
import polars as pl
import numpy as np
from scipy.stats import mode
from scipy.spatial.transform import Rotation as R
import glob
import os
import concurrent.futures

import torch
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

if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    num_gpus = torch.cuda.device_count()
    devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]
    print(f'using {num_gpus} gpus: {devices}')
else:
    devices = [torch.device('cpu')]
    print('uhhhh cpu :(')

reverse_mapping, aux1_reverse_mapping, aux2_reverse_mapping = get_rev_mapping()
TSDataset = get_ts_dataset()

loaded_models = {
    'imu_only': {},
    'imu+tof+thm': {}
}

use_tta = len(cfg.tta_strategies) > 0

model_group_idx = 0
for w_key in ['imu_only', 'imu+tof+thm']:
    if w_key not in cfg.weights_pathes:
        continue
        
    for weights_path, params in cfg.weights_pathes[w_key].items():
        group_device = devices[model_group_idx % len(devices)]
        
        print(f'loading models for {w_key}: {weights_path}')
        print(f'  -> all models in this group will be assigned to {group_device=}')
        
        if w_key == 'imu_only':
            TSModel = HybridModel_SingleSensor_v1
        else:
            TSModel = MultiSensor_HybridModel_v1
        
        ckpt_files = sorted(
            glob.glob(os.path.join(weights_path, '*.pt'))
        )
        if not ckpt_files:
            print(f'  no checkpoints found in {weights_path}')
            model_group_idx += 1
            continue

        fold_models = []
        for ckpt in ckpt_files:
            print(f'  loading checkpoint: {os.path.basename(ckpt)}')

            is_ema = ckpt.endswith('_ema.pt') or '_ema_' in os.path.basename(ckpt)
            model = TSModel(**params['model_params']).to(group_device)

            state_dict = torch.load(ckpt, map_location=group_device)
            model.load_state_dict(state_dict, strict=False)

            if not is_ema and cfg.use_ema:
                ema_ckpt = ckpt.replace('.pt', '_ema.pt')
                if os.path.isfile(ema_ckpt):
                    ema_state = torch.load(ema_ckpt, map_location=group_device)
                    for name, p in model.named_parameters():
                        if name in ema_state:
                            p.data.copy_(ema_state[name])

            model.eval()
            fold_models.append((model, group_device))

        loaded_models.setdefault(w_key, {})[weights_path] = {
            'models': fold_models,
            'params': params,
            'weight': params['weight']
        }
        model_group_idx += 1

def apply_sequence_tta(df, tta_strategies, use_imu_only):
    augmented_sequences = []
    augmented_sequences.append(df.copy())
    
    if not tta_strategies:
        return augmented_sequences
    
    sequence_ids = df['sequence_id'].unique() if 'sequence_id' in df.columns else [0]
    
    print(f'{tta_strategies=}')
    for strategy in tta_strategies:
        try:
            if strategy == 'jitter' and use_imu_only:
                df_jitter = df.copy()
                for seq_id in sequence_ids:
                    mask = df_jitter['sequence_id'] == seq_id if 'sequence_id' in df_jitter.columns else slice(None)
                    
                    acc_cols = [col for col in df_jitter.columns if col.startswith('acc_')]
                    rot_cols = [col for col in df_jitter.columns if col.startswith('rot_')]
                    
                    jitter_strength = 0.03
                    
                    for col in acc_cols + rot_cols:
                        if col in df_jitter.columns:
                            data_std = df_jitter.loc[mask, col].std()
                            if pd.isna(data_std) or data_std == 0:
                                continue
                            noise = np.random.normal(0, jitter_strength * data_std, len(df_jitter.loc[mask, col]))
                            df_jitter.loc[mask, col] += noise
                
                augmented_sequences.append(df_jitter)
            
            elif strategy == 'rotation_z' and use_imu_only:
                df_rot = df.copy()
                for seq_id in sequence_ids:
                    mask = df_rot['sequence_id'] == seq_id if 'sequence_id' in df_rot.columns else slice(None)
                    
                    max_angle_deg = 30
                    angle = np.random.uniform(-max_angle_deg, max_angle_deg) * np.pi / 180
                    cos_angle = np.cos(angle)
                    sin_angle = np.sin(angle)
                    
                    if 'acc_x' in df_rot.columns and 'acc_y' in df_rot.columns:
                        acc_x_orig = df_rot.loc[mask, 'acc_x'].values.copy()
                        acc_y_orig = df_rot.loc[mask, 'acc_y'].values.copy()
                        
                        df_rot.loc[mask, 'acc_x'] = cos_angle * acc_x_orig - sin_angle * acc_y_orig
                        df_rot.loc[mask, 'acc_y'] = sin_angle * acc_x_orig + cos_angle * acc_y_orig
                    
                    if all(col in df_rot.columns for col in ['rot_w', 'rot_x', 'rot_y', 'rot_z']):
                        quat_data = df_rot.loc[mask, ['rot_w', 'rot_x', 'rot_y', 'rot_z']].values
                        
                        if np.any(np.isnan(quat_data)) or quat_data.shape[0] == 0:
                            continue
                        
                        quat_scipy_format = quat_data[:, [1, 2, 3, 0]]
                        
                        z_rotation = R.from_euler('z', angle)
                        
                        original_rotations = R.from_quat(quat_scipy_format)
                        rotated_rotations = z_rotation * original_rotations
                        quat_rotated_scipy = rotated_rotations.as_quat()
                        
                        df_rot.loc[mask, 'rot_w'] = quat_rotated_scipy[:, 3]
                        df_rot.loc[mask, 'rot_x'] = quat_rotated_scipy[:, 0]
                        df_rot.loc[mask, 'rot_y'] = quat_rotated_scipy[:, 1]
                        df_rot.loc[mask, 'rot_z'] = quat_rotated_scipy[:, 2]
                
                augmented_sequences.append(df_rot)
                
        except Exception as e:
            print('uhh its bad...')

    return augmented_sequences

def preprocess_single_row(test_df, use_imu_only):
    if cfg.use_world_coords:
        test_df = convert_to_world_coordinates(test_df)

    if cfg.only_remove_g:
        test_df = remove_gravity_from_acc(test_df)

    if cfg.use_hand_symm and use_imu_only:
        right_handed_mask = test_df['handedness'] == 1
        cols = test_df.columns.tolist()
        test_df.loc[right_handed_mask, cols] = apply_symmetry(test_df.loc[right_handed_mask, cols])

    if cfg.apply_fe:
        test_df = fe(test_df)

    return fast_seq_agg(test_df)

def create_single_batch(processed_df, device):
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
    if use_imu_only:
        outputs, aux2_outputs, orient_outputs, ext1, ext2, ext3, ext4, ext5, ext6 = forward_model(
            model, batch, imu_only=use_imu_only
        )
        
        if cfg.ext_weights_imu:
            w0, w1, w2, w3, w4, w5, w6 = cfg.ext_weights_imu
            outputs = w0 * outputs + w1 * ext1 + w2 * ext2 + w3 * ext3 + w4 * ext4 + w5 * ext5 + w6 * ext6
    else:
        outputs, aux2_outputs, orient_outputs, ext1, ext2, ext3, ext4 = forward_model(
            model, batch, imu_only=use_imu_only
        )
        
        if cfg.ext_weights_all:
            w0, w1, w2, w3, w4 = cfg.ext_weights_all
            outputs = w0 * outputs + w1 * ext1 + w2 * ext2 + w3 * ext3 + w4 * ext4
    
    return outputs.detach(), aux2_outputs.detach(), orient_outputs.detach()

def process_model_group(models_on_device, preprocessed_batches_on_device, use_imu_only):
    group_device = models_on_device[0][1] 
    
    fold_logits_sum_list = []
    fold_logits_aux2_sum_list = []
    fold_logits_orient_sum_list = []
    
    if use_imu_only:
        use_entmax = cfg.use_imu_entmax
        entmax_alpha = cfg.imu_entmax_alpha
    else:
        use_entmax = cfg.use_imu_tof_thm_entmax
        entmax_alpha = cfg.imu_tof_thm_entmax_alpha
    
    for model, _ in models_on_device:
        tta_logits_gpu = []
        tta_logits_aux2_gpu = []
        tta_logits_orient_gpu = []
        
        for batch in preprocessed_batches_on_device:
            logits_gpu, logits_aux2_gpu, logits_orient_gpu = predict_single_batch(model, batch, use_imu_only)
            
            if use_entmax:
                logits_gpu = entmax_bisect(logits_gpu, alpha=entmax_alpha, dim=1)
                logits_aux2_gpu = entmax_bisect(logits_aux2_gpu, alpha=entmax_alpha, dim=1)
                logits_orient_gpu = entmax_bisect(logits_orient_gpu, alpha=entmax_alpha, dim=1)
            
            tta_logits_gpu.append(logits_gpu)
            tta_logits_aux2_gpu.append(logits_aux2_gpu)
            tta_logits_orient_gpu.append(logits_orient_gpu)
            
        sum_tta_logits = torch.sum(torch.stack(tta_logits_gpu), dim=0)
        sum_tta_logits_aux2 = torch.sum(torch.stack(tta_logits_aux2_gpu), dim=0)
        sum_tta_logits_orient = torch.sum(torch.stack(tta_logits_orient_gpu), dim=0)
        
        fold_logits_sum_list.append(sum_tta_logits)
        fold_logits_aux2_sum_list.append(sum_tta_logits_aux2)
        fold_logits_orient_sum_list.append(sum_tta_logits_orient)
        
    sum_fold_logits = torch.sum(torch.stack(fold_logits_sum_list), dim=0)
    sum_fold_logits_aux2 = torch.sum(torch.stack(fold_logits_aux2_sum_list), dim=0)
    sum_fold_logits_orient = torch.sum(torch.stack(fold_logits_orient_sum_list), dim=0)
    
    num_folds = len(models_on_device)
    num_tta = len(preprocessed_batches_on_device)
    total_elements = num_folds * num_tta
    
    model_averaged_logits = sum_fold_logits.cpu().numpy() / total_elements
    model_averaged_logits_aux2 = sum_fold_logits_aux2.cpu().numpy() / total_elements
    model_averaged_logits_orient = sum_fold_logits_orient.cpu().numpy() / total_elements
    
    return model_averaged_logits, model_averaged_logits_aux2, model_averaged_logits_orient

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

    if use_tta:
        augmented_sequences = apply_sequence_tta(test_df, cfg.tta_strategies, use_imu_only)
    else:
        augmented_sequences = [test_df]
    
    preprocessed_batches_by_device = {dev: [] for dev in devices}
    for aug_df in augmented_sequences:
        processed_df = preprocess_single_row(aug_df.copy(), use_imu_only)
        for dev in devices:
            batch = create_single_batch(processed_df, dev)
            preprocessed_batches_by_device[dev].append(batch)
    
    all_model_averaged_logits = []
    all_model_averaged_logits_aux2 = []
    all_model_averaged_logits_orient = []
    model_weights = []

    w_key = 'imu_only' if use_imu_only else 'imu+tof+thm'
    
    tasks_to_run = []
    for weights_path, model_data in loaded_models[w_key].items():
        group_device = model_data['models'][0][1]
        
        task_data = {
            'models_on_device': model_data['models'],
            'preprocessed_batches_on_device': preprocessed_batches_by_device[group_device],
            'use_imu_only': use_imu_only,
            'weight': model_data['weight']
        }
        tasks_to_run.append(task_data)
        
    with torch.no_grad(), concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_model_group, 
                task['models_on_device'], 
                task['preprocessed_batches_on_device'], 
                task['use_imu_only']
            ): task 
            for task in tasks_to_run
        }
        
        for future in concurrent.futures.as_completed(futures):
            task_data = futures[future]
            logits, logits_aux2, logits_orient = future.result()
            
            model_weights.append(task_data['weight'])
            all_model_averaged_logits.append(logits)
            all_model_averaged_logits_aux2.append(logits_aux2)
            all_model_averaged_logits_orient.append(logits_orient)
    
    if cfg.is_soft:
        model_weights = np.array(model_weights)
        all_model_averaged_logits = np.array(all_model_averaged_logits)
        all_model_averaged_logits_aux2 = np.array(all_model_averaged_logits_aux2)
        all_model_averaged_logits_orient = np.array(all_model_averaged_logits_orient)
        
        final_averaged_logits = np.sum(all_model_averaged_logits * model_weights[:, np.newaxis, np.newaxis], axis=0)
        final_averaged_logits_aux2 = np.sum(all_model_averaged_logits_aux2 * model_weights[:, np.newaxis, np.newaxis], axis=0)
        final_averaged_logits_orient = np.sum(all_model_averaged_logits_orient * model_weights[:, np.newaxis, np.newaxis], axis=0)
        print(f'{final_averaged_logits=}')

        majority_vote_indices = np.argmax(final_averaged_logits, axis=1)
        majority_vote_indices_aux2 = np.argmax(final_averaged_logits_aux2, axis=1)
        majority_vote_indices_orient = np.argmax(final_averaged_logits_orient, axis=1)
    else:
        all_model_predictions = [np.argmax(logits, axis=1) for logits in all_model_averaged_logits]
        all_model_predictions_aux2 = [np.argmax(logits_aux2, axis=1) for logits_aux2 in all_model_averaged_logits_aux2]
        all_model_predictions_orient = [np.argmax(logits_orient, axis=1) for logits_orient in all_model_averaged_logits_orient]
        all_models_predictions_array = np.array(all_model_predictions)
        majority_vote_indices, _ = mode(all_models_predictions_array, axis=0, keepdims=False)
        all_models_predictions_array_aux2 = np.array(all_model_predictions_aux2)
        majority_vote_indices_aux2, _ = mode(all_models_predictions_array_aux2, axis=0, keepdims=False)
        all_models_predictions_array_orient = np.array(all_model_predictions_orient)
        majority_vote_indices_orient, _ = mode(all_models_predictions_array_orient, axis=0, keepdims=False)

    if cfg.override_non_target:
        print('override non target (using aux2 head)')
        final_preds = []
        for output_idx, aux2_idx in zip(majority_vote_indices, majority_vote_indices_aux2):
            if output_idx <= 7 and aux2_idx == 0:
                final_preds.append(8)
            else:
                final_preds.append(output_idx)
        final_labels = [reverse_mapping[class_idx] for class_idx in final_preds]
    else:
        final_labels = [reverse_mapping[class_idx] for class_idx in majority_vote_indices]

    if cfg.orient_postproc:
        print('seated straight -> non target')
        final_preds = []
        for output_idx, aux2_idx, orient_idx in zip(majority_vote_indices, majority_vote_indices_aux2, majority_vote_indices_orient):
            if output_idx <= 7 and orient_idx == 0: # and aux2_idx == 0
                final_preds.append(8)
            else:
                final_preds.append(output_idx)
        final_labels = [reverse_mapping[class_idx] for class_idx in final_preds]
    else:
        final_labels = [reverse_mapping[class_idx] for class_idx in majority_vote_indices]

    return final_labels[0]