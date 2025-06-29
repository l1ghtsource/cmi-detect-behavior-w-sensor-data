import pandas as pd
import polars as pl
import numpy as np
from scipy.stats import mode

import openvino as ov
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
    remove_gravity_from_acc,
    fe,
    get_rev_mapping
)
from utils.tta import apply_tta

# TODO: multigpu inference

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train = pd.read_csv(cfg.train_path)

if cfg.use_world_coords:
    train = convert_to_world_coordinates(train)

if cfg.only_remove_g: # can't be used w/ use_world_coords
    train = remove_gravity_from_acc(train)

if cfg.use_hand_symm:
    right_handed_mask = train['handedness'] == 1
    train.loc[right_handed_mask, cfg.imu_cols] = apply_symmetry(train.loc[right_handed_mask, cfg.imu_cols])

if cfg.apply_fe:
    train = fe(train)

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

    if cfg.only_remove_g: # can't be used w/ use_world_coords
        test_df = remove_gravity_from_acc(test_df)

    if cfg.use_hand_symm and use_imu_only:
        right_handed_mask = test_df['handedness'] == 1
        test_df.loc[right_handed_mask, cfg.imu_cols] = apply_symmetry(test_df.loc[right_handed_mask, cfg.imu_cols])

    if cfg.apply_fe:
        test_df = fe(test_df)

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
    model_weights = []

    core = ov.Core()

    w_key = 'imu_only' if use_imu_only else 'imu+tof+thm'
    for weights_path, params in cfg.weights_pathes[w_key].items():
        print(f'{params=}')
        model_weights.append(params['weight'])

        current_model_fold_logits = []
        current_model_fold_logits_aux2 = []
        current_model_fold_predictions = []
        current_model_fold_predictions_aux2 = []

        for i in range(cfg.n_splits):
            print(f'model: {weights_path}, fold={i+1}')
            
            ov_model_path = f'{weights_path}/{params["prefix"]}model_fold{i}_openvino.xml'
            compiled_model = core.compile_model(ov_model_path, device_name='GPU')
            
            current_fold_batch_logits = []
            current_fold_batch_logits_aux2 = []
            
            for batch in test_loader:
                inputs = prepare_openvino_inputs(batch, use_imu_only)
                
                if not use_tta:
                    outputs = compiled_model(inputs)
                    main_output = outputs[compiled_model.output(0)]
                    aux2_output = outputs[compiled_model.output(1)]
                    
                    current_fold_batch_logits.append(main_output)
                    current_fold_batch_logits_aux2.append(aux2_output)
                else:
                    augmented_batches = apply_tta(batch, cfg.tta_strategies)
                    batch_tta_logits = []
                    batch_tta_logits_aux2 = []
                    
                    for aug_batch in augmented_batches:
                        aug_inputs = prepare_openvino_inputs(aug_batch, use_imu_only)
                        outputs = compiled_model(aug_inputs)
                        main_output = outputs[compiled_model.output(0)]
                        aux2_output = outputs[compiled_model.output(1)]
                        
                        batch_tta_logits.append(main_output)
                        batch_tta_logits_aux2.append(aux2_output)
                    
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
                    logits_tensor = torch.tensor(concatenated_fold_logits)
                    entmax_probs = entmax_bisect(logits_tensor, alpha=cfg.entmax_alpha, dim=1)
                    predicted_indices_for_fold = torch.argmax(entmax_probs, dim=1).numpy()
                    logits_tensor_aux2 = torch.tensor(concatenated_fold_logits_aux2)
                    entmax_probs_aux2 = entmax_bisect(logits_tensor_aux2, alpha=cfg.entmax_alpha, dim=1)
                    predicted_indices_for_fold_aux2 = torch.argmax(entmax_probs_aux2, dim=1).numpy()
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
        model_weights = np.array(model_weights)
        
        all_model_averaged_logits = np.array(all_model_averaged_logits)
        all_model_averaged_logits_aux2 = np.array(all_model_averaged_logits_aux2)
        
        final_averaged_logits = np.sum(all_model_averaged_logits * model_weights[:, np.newaxis, np.newaxis], axis=0)
        final_averaged_logits_aux2 = np.sum(all_model_averaged_logits_aux2 * model_weights[:, np.newaxis, np.newaxis], axis=0)
        
        if cfg.use_entmax:
            final_logits_tensor = torch.tensor(final_averaged_logits)
            entmax_probs = entmax_bisect(final_logits_tensor, alpha=cfg.entmax_alpha, dim=1)
            majority_vote_indices = torch.argmax(entmax_probs, dim=1).numpy()
            final_logits_tensor_aux2 = torch.tensor(final_averaged_logits_aux2)
            entmax_probs_aux2 = entmax_bisect(final_logits_tensor_aux2, alpha=cfg.entmax_alpha, dim=1)
            majority_vote_indices_aux2 = torch.argmax(entmax_probs_aux2, dim=1).numpy()
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

def prepare_openvino_inputs(batch, use_imu_only):
    inputs = {}
    
    if use_imu_only:
        inputs['imu_data'] = batch['imu_data'].numpy()
        inputs['pad_mask'] = batch['pad_mask'].numpy()
    else:
        inputs['imu_data'] = batch['imu_data'].numpy()
        inputs['tof_data'] = batch['tof_data'].numpy()
        inputs['thm_data'] = batch['thm_data'].numpy()
        inputs['pad_mask'] = batch['pad_mask'].numpy()
    
    return inputs