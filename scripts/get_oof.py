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
    sgkf   = StratifiedGroupKFold(
        n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed
    )

    N, C = len(train_seq), cfg.main_num_classes
    oof_main = np.zeros((N, C), dtype=np.float32)

    extractor_names = [f'ext{i}' for i in range(1, 7)]
    oof_ext = {name: np.zeros((N, C), dtype=np.float32)
               for name in extractor_names}

    oof_targets = train_seq[cfg.main_target].values

    for fold, (_, val_idx) in enumerate(
            sgkf.split(train_seq,
                       train_seq[cfg.main_target].values,
                       train_seq[cfg.group].values)):
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

        model.load_state_dict(torch.load(weight_pathes[fold],
                                         map_location=device), strict=False)
        model.eval()

        preds_main, preds_ext = [], {n: [] for n in extractor_names}

        with torch.no_grad():
            for batch in tqdm(val_loader, leave=False):
                for k in batch:
                    batch[k] = batch[k].to(device)

                (logits,
                 _seq_type, _orient,
                 ext1, ext2, ext3, ext4, ext5, ext6) = forward_model(
                                                    model, batch, imu_only=cfg.imu_only)

                preds_main.append(logits.cpu().numpy())
                for name, tensor in zip(extractor_names,
                                        [ext1, ext2, ext3, ext4, ext5, ext6]):
                    preds_ext[name].append(tensor.cpu().numpy())

        oof_main[val_idx] = np.concatenate(preds_main, axis=0)
        for name in extractor_names:
            oof_ext[name][val_idx] = np.concatenate(preds_ext[name], axis=0)

        del model, val_dataset, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _metric(probs):
        return comp_metric(oof_targets, np.argmax(probs, axis=1))

    main_avg_f1, main_bin_f1, main_mac_f1 = _metric(oof_main)
    print(f'MAIN  : avg={main_avg_f1:.4f}  bin={main_bin_f1:.4f}  macro={main_mac_f1:.4f}')

    ext_metrics = {}
    for name in extractor_names:
        m_avg, m_bin, m_mac = _metric(oof_ext[name])
        ext_metrics[name] = {'oof_avg_f1': float(m_avg),
                             'oof_binary_f1': float(m_bin),
                             'oof_macro_f1': float(m_mac)}
        print(f'{name.upper():<5}: avg={m_avg:.4f}  bin={m_bin:.4f}  macro={m_mac:.4f}')

    os.makedirs(cfg.oof_dir, exist_ok=True)
    np.save(os.path.join(cfg.oof_dir, f'{prefix}oof_preds_main.npy'),   oof_main)
    np.save(os.path.join(cfg.oof_dir, f'{prefix}oof_labels_main.npy'),
            np.argmax(oof_main, axis=1))

    for name in extractor_names:
        np.save(os.path.join(cfg.oof_dir, f'{prefix}oof_preds_{name}.npy'),   oof_ext[name])
        np.save(os.path.join(cfg.oof_dir, f'{prefix}oof_labels_{name}.npy'),
                np.argmax(oof_ext[name], axis=1))

    np.save(os.path.join(cfg.oof_dir, f'{prefix}oof_targets.npy'), oof_targets)

    oof_info = {
        'main_output': {'oof_avg_f1': main_avg_f1,
                        'oof_binary_f1': main_bin_f1,
                        'oof_macro_f1': main_mac_f1},
        'extractor_outputs': ext_metrics
    }
    with open(os.path.join(cfg.oof_dir, f'{prefix}oof_info.json'), 'w') as f:
        json.dump(oof_info, f, indent=2)