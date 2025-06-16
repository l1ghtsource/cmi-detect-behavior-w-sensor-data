import pandas as pd
import polars as pl
import numpy as np
from scipy.stats import mode

import torch
from torch.utils.data import DataLoader
from entmax import entmax_bisect

from configs.config import cfg
from models.timemil import (
    MultiSensor_TimeMIL_v1, 
    MultiSensor_TimeMIL_v2,
    TimeMIL_SingleSensor_Multibranch_v1,
    TimeMIL_SingleSensor_Singlebranch_v1
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
    get_rev_mapping
)
from utils.tta import apply_tta

# TODO: multigpu inference

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train = pd.read_csv(cfg.train_path)

if cfg.use_world_coords:
    train = convert_to_world_coordinates(train)

if cfg.use_hand_symm:
    right_handed_mask = train['handedness'] == 1
    train.loc[right_handed_mask, cfg.imu_cols] = apply_symmetry(train.loc[right_handed_mask, cfg.imu_cols])

train = le(train)
train_seq = fast_seq_agg(train)

reverse_mapping, aux1_reverse_mapping, aux2_reverse_mapping = get_rev_mapping()

TSDataset = get_ts_dataset()

train_dataset = TSDataset(
    dataframe=train_seq,
    seq_len=cfg.seq_len,
    main_target=cfg.main_target,
    orientation_aux_target=cfg.orientation_aux_target,
    seq_type_aux_target=cfg.seq_type_aux_target,
    behavior_aux_target=cfg.behavior_aux_target,
    phase_aux_target=cfg.phase_aux_target,
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

    if cfg.use_world_coords:
        test_df = convert_to_world_coordinates(test_df)

    if cfg.use_hand_symm and use_imu_only:
        right_handed_mask = test_df['handedness'] == 1
        test_df.loc[right_handed_mask, cfg.imu_cols] = apply_symmetry(test_df.loc[right_handed_mask, cfg.imu_cols])

    processed_df_for_dataset = fast_seq_agg(test_df) 

    test_dataset = TSDataset(
        dataframe=processed_df_for_dataset,
        seq_len=cfg.seq_len,
        train=False,
        norm_stats=train_dataset.norm_stats
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0) 

    use_tta = len(cfg.tta_strategies) > 0 # strats != {}
    
    all_model_averaged_logits = []
    all_model_averaged_logits_aux2 = []
    all_model_predictions = []
    all_model_predictions_aux2 = []

    w_key = 'imu_only' if use_imu_only else 'imu+tof+thm'
    for weights_path, params in cfg.weights_pathes[w_key]:
        print(f'{params=}')
        if use_imu_only:
            TSModel = TimeMIL_SingleSensor_Singlebranch_v1 if params['timemil_singlebranch'] else TimeMIL_SingleSensor_Multibranch_v1
        else:
            TSModel = MultiSensor_TimeMIL_v1 if params['timemil_ver'] == '1' else MultiSensor_TimeMIL_v2

        model = TSModel(**params['model_params']).to(device)

        current_model_fold_logits = []
        current_model_fold_logits_aux2 = []
        current_model_fold_predictions = []
        current_model_fold_predictions_aux2 = []

        for i in range(cfg.n_splits):
            print(f'model: {weights_path}, fold={i+1}')
            model_path = f'{weights_path}/{params["prefix"]}model_fold{i}.pt'
            model.load_state_dict(torch.load(model_path, map_location=device))

            if cfg.use_ema:
                model_path = f'{weights_path}/{params["prefix"]}model_ema_fold{i}.pt'
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

            current_model_fold_logits.append(concatenated_fold_logits)
            current_model_fold_logits_aux2.append(concatenated_fold_logits_aux2)
            
            if not cfg.is_soft:
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
                
                current_model_fold_predictions.append(predicted_indices_for_fold)
                current_model_fold_predictions_aux2.append(predicted_indices_for_fold_aux2)

        model_averaged_logits = np.mean(np.array(current_model_fold_logits), axis=0)
        model_averaged_logits_aux2 = np.mean(np.array(current_model_fold_logits_aux2), axis=0)
        
        all_model_averaged_logits.append(model_averaged_logits)
        all_model_averaged_logits_aux2.append(model_averaged_logits_aux2)
        
        if not cfg.is_soft:
            current_model_predictions_array = np.array(current_model_fold_predictions)
            model_averaged_predictions, _ = mode(current_model_predictions_array, axis=0, keepdims=False)
            current_model_predictions_array_aux2 = np.array(current_model_fold_predictions_aux2)
            model_averaged_predictions_aux2, _ = mode(current_model_predictions_array_aux2, axis=0, keepdims=False)
            
            all_model_predictions.append(model_averaged_predictions)
            all_model_predictions_aux2.append(model_averaged_predictions_aux2)

    if cfg.is_soft:
        final_averaged_logits = np.mean(np.array(all_model_averaged_logits), axis=0)
        final_averaged_logits_aux2 = np.mean(np.array(all_model_averaged_logits_aux2), axis=0)
        
        if cfg.use_entmax:
            final_logits_tensor = torch.tensor(final_averaged_logits, device=device)
            entmax_probs = entmax_bisect(final_logits_tensor, alpha=cfg.entmax_alpha, dim=1)
            majority_vote_indices = torch.argmax(entmax_probs, dim=1).cpu().numpy()
            final_logits_tensor_aux2 = torch.tensor(final_averaged_logits_aux2, device=device)
            entmax_probs_aux2 = entmax_bisect(final_logits_tensor_aux2, alpha=cfg.entmax_alpha, dim=1)
            majority_vote_indices_aux2 = torch.argmax(entmax_probs_aux2, dim=1).cpu().numpy()
        else:
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