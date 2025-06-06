import os
import gc

import pandas as pd
import numpy as np

from tqdm import tqdm

from sklearn.model_selection import StratifiedGroupKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from configs.config import cfg
from data.ts_datasets import TS_CMIDataset, TS_Demo_CMIDataset
from models.ts_models import TS_MSModel, TS_IMUModel, TS_Demo_MSModel, TS_Demo_IMUModel
from modules.ema import EMA
from utils.data_preproc import fast_seq_agg, le
from utils.metrics import just_stupid_macro_f1_haha

# --- load data ---

train = pd.read_csv(cfg.train_path)
train_demographics = pd.read_csv(cfg.train_demographics_path)

test = pd.read_csv(cfg.test_path)
test_demographics = pd.read_csv(cfg.test_demographics_path)

# --- join demographics stats ---

train = train.merge(train_demographics, how='left', on='subject')
test = train.merge(test_demographics, how='left', on='subject')

# -- convert to seq ---

train_seq = fast_seq_agg(train)
test_seq = fast_seq_agg(test)

del train, test
gc.collect()

train_seq, label_encoder = le(train_seq)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(train_loader, model, optimizer, criterion, device, scheduler, ema=None):
    model.train()
        
    total_loss = 0
    total_samples = 0
    all_targets = []
    all_preds = []
    
    loop = tqdm(train_loader, desc='train', leave=False)
    
    for batch in loop:
        optimizer.zero_grad()
        
        imu_inputs = batch['imu'].to(device)
        targets = batch['target'].to(device)
        
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

        loss = criterion(outputs, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        if ema is not None:
            ema.update()

        total_loss += loss.item() * targets.size(0)
        total_samples += targets.size(0)
        
        preds = torch.argmax(outputs, dim=1)
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        
        loop.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / total_samples
    m = just_stupid_macro_f1_haha(all_targets, all_preds)
    return avg_loss, m

def valid_epoch(val_loader, model, criterion, device, ema=None):
    model.eval()

    if cfg.use_ema and ema is not None:
        ema.apply_shadow()

    total_loss = 0
    total_samples = 0
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        loop = tqdm(val_loader, desc='val', leave=False)
        for batch in loop:
            imu_inputs = batch['imu'].to(device)
            targets = batch['target'].to(device)
            
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

            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            loop.set_postfix(loss=loss.item())

    if cfg.use_ema and ema is not None:
        ema.restore()
    
    avg_loss = total_loss / total_samples
    f1_score_val = just_stupid_macro_f1_haha(all_targets, all_preds)
    return avg_loss, f1_score_val, all_targets, all_preds

def run_training_with_stratified_group_kfold():
    os.makedirs(cfg.model_dir, exist_ok=True)
    
    sgkf = StratifiedGroupKFold(n_splits=cfg.n_splits, shuffle=False)
    targets = train_seq[cfg.target].values
    groups = train_seq[cfg.group].values
    
    oof_preds = np.zeros((len(train_seq), cfg.num_classes))
    oof_targets = train_seq[cfg.target].values
    
    best_models = []
    best_f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_seq, targets, groups)):
        print(f'fold {fold+1}/{cfg.n_splits}')
        # if fold != cfg.curr_fold:
        #     continue
        
        train_subset = train_seq.iloc[train_idx].reset_index(drop=True)
        val_subset = train_seq.iloc[val_idx].reset_index(drop=True)
        
        TSDataset = TS_Demo_CMIDataset if cfg.use_demo else TS_CMIDataset
        
        train_dataset = TSDataset(
            dataframe=train_subset,
            seq_len=cfg.seq_len,
            target_col=cfg.target
        )
        val_dataset = TSDataset(
            dataframe=val_subset,
            seq_len=cfg.seq_len,
            target_col=cfg.target
        )
        
        train_loader = DataLoader(train_dataset, batch_size=cfg.bs, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=cfg.bs, shuffle=False, num_workers=4)

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

        ema = EMA(model, decay=cfg.ema_decay) if cfg.use_ema else None

        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

        num_training_steps = cfg.n_epochs * len(train_loader)
        num_warmup_steps = int(cfg.num_warmup_steps_ratio * num_training_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        best_val_score = -np.inf
        patience_counter = 0
        best_model_path = os.path.join(cfg.model_dir, f'model_fold{fold}.pt')
        best_ema_path = os.path.join(cfg.model_dir, f'model_ema_fold{fold}.pt') if cfg.use_ema else None
        
        for epoch in range(cfg.n_epochs):
            print(f'{epoch=}')
            
            train_loss, train_f1 = train_epoch(train_loader, model, optimizer, criterion, device, scheduler, ema)
            val_loss, val_f1, _, _ = valid_epoch(val_loader, model, criterion, device, ema)
            
            print(f'{train_loss=}, {train_f1=}')
            print(f'{val_loss=}, {val_f1=}')
            
            if val_f1 > best_val_score:
                best_val_score = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                if cfg.use_ema and ema is not None:
                    ema_state_dict = {}
                    for name in ema.shadow:
                        ema_state_dict[name] = ema.shadow[name]
                    torch.save(ema_state_dict, best_ema_path)
            else:
                patience_counter += 1
            
            if patience_counter >= cfg.patience:
                print('early stopping')
                break

        model.load_state_dict(torch.load(best_model_path))

        if cfg.use_ema and best_ema_path and os.path.exists(best_ema_path):
            ema_state_dict = torch.load(best_ema_path, map_location=device)
            for name, param in model.named_parameters():
                if name in ema_state_dict:
                    param.data = ema_state_dict[name]

        best_models.append(model)
        best_f1_scores.append(best_val_score)
        
        model.eval()
        all_preds = []
        with torch.no_grad():
            val_loader = DataLoader(val_dataset, batch_size=cfg.bs, shuffle=False, num_workers=4)
            for batch in val_loader:
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
                        
                all_preds.append(outputs.cpu().numpy())
        all_preds = np.concatenate(all_preds, axis=0)
        oof_preds[val_idx] = all_preds
    
    oof_pred_labels = np.argmax(oof_preds, axis=1)
    oof_m = just_stupid_macro_f1_haha(oof_targets, oof_pred_labels)
    print(f'{oof_m=}')
    
    return best_models, oof_preds