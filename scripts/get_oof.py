import os, json, gc
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold

from configs.config import cfg
from utils.seed import seed_everything
from utils.data_preproc import (
    fast_seq_agg, le, convert_to_world_coordinates,
    remove_gravity_from_acc, apply_symmetry, fe
)
from utils.getters import (
    get_ts_dataset, get_ts_model_and_params,
    forward_model, get_prefix
)
from utils.metrics import comp_metric

seed_everything()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train = pd.read_csv(cfg.train_path)
train_demographics = pd.read_csv(cfg.train_demographics_path)
train = train.merge(train_demographics, how='left', on='subject')

weight_pathes = {
    0: ...,
    1: ...,
    2: ...,
    3: ...,
    4: ...
}

if cfg.use_world_coords:
    train = convert_to_world_coordinates(train)
if cfg.only_remove_g:
    train = remove_gravity_from_acc(train)
if cfg.use_hand_symm and cfg.imu_only:
    rh_mask = train['handedness'] == 1
    train.loc[rh_mask, cfg.imu_cols] = apply_symmetry(train.loc[rh_mask, cfg.imu_cols])
if cfg.apply_fe:
    train = fe(train)

train = le(train)
train_seq = fast_seq_agg(train)

def save_oof():
    prefix = get_prefix(cfg.imu_only)
    sgkf = StratifiedGroupKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)

    oof_preds = np.zeros((len(train_seq), cfg.main_num_classes), dtype=np.float32)
    oof_targets = train_seq[cfg.main_target].values

    for fold, (_, val_idx) in enumerate(sgkf.split(
            train_seq, train_seq[cfg.main_target].values, train_seq[cfg.group].values)):
        print(f'{fold=}')

        TSDataset = get_ts_dataset()
        val_dataset = TSDataset(
            dataframe=train_seq.iloc[val_idx].reset_index(drop=True),
            seq_len=cfg.seq_len,
            main_target=cfg.main_target,
            orientation_aux_target=cfg.orientation_aux_target,
            seq_type_aux_target=cfg.seq_type_aux_target,
            behavior_aux_target=cfg.behavior_aux_target,
            phase_aux_target=cfg.phase_aux_target,
            train=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=cfg.bs, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True
        )

        TSModel, m_params = get_ts_model_and_params(imu_only=cfg.imu_only)
        model = TSModel(**m_params).to(device)

        weight_path = weight_pathes[fold]
        model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
        model.eval()

        preds_fold = []
        with torch.no_grad():
            for batch in tqdm(val_loader, leave=False):
                for k in batch:
                    batch[k] = batch[k].to(device)
                logits, *_ = forward_model(model, batch, imu_only=cfg.imu_only)
                preds_fold.append(logits.cpu().numpy())
        oof_preds[val_idx] = np.concatenate(preds_fold, axis=0)

        del model, val_dataset, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    oof_labels = np.argmax(oof_preds, axis=1)
    oof_avg_f1, oof_bin_f1, oof_macro_f1 = comp_metric(oof_targets, oof_labels)
    print(f'OOF avg F1={oof_avg_f1:.4f} | bin F1={oof_bin_f1:.4f} | macro F1={oof_macro_f1:.4f}')

    os.makedirs(cfg.oof_dir, exist_ok=True)
    np.save(os.path.join(cfg.oof_dir, f'{prefix}oof_preds.npy'), oof_preds)
    np.save(os.path.join(cfg.oof_dir, f'{prefix}oof_labels.npy'), oof_labels)
    np.save(os.path.join(cfg.oof_dir, f'{prefix}oof_targets.npy'), oof_targets)

    oof_info = {
        'oof_avg_f1': float(oof_avg_f1),
        'oof_binary_f1': float(oof_bin_f1),
        'oof_macro_f1': float(oof_macro_f1),
    }
    with open(os.path.join(cfg.oof_dir, f'{prefix}oof_info.json'), 'w') as f:
        json.dump(oof_info, f, indent=2)