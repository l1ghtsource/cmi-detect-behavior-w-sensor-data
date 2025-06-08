import os
import gc
import json

import pandas as pd
import numpy as np

from tqdm import tqdm

from sklearn.model_selection import StratifiedGroupKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

import wandb

from configs.config import cfg
from modules.ema import EMA
from modules.mixup import MixupLoss, mixup_batch
from utils.getters import get_ts_dataset, get_ts_model_and_params, forward_model
from utils.data_preproc import fast_seq_agg, le
from utils.metrics import just_stupid_macro_f1_haha
from utils.seed import seed_everything

# --- set seed ---

seed_everything(cfg.seed)

# --- wandb ---

if cfg.do_wandb_log:
    os.environ['WANDB_API_KEY'] = cfg.wandb_api_key

# --- load data ---

train = pd.read_csv(cfg.train_path)
train_demographics = pd.read_csv(cfg.train_demographics_path)

test = pd.read_csv(cfg.test_path)
test_demographics = pd.read_csv(cfg.test_demographics_path)

# --- join demographics stats ---

train = train.merge(train_demographics, how='left', on='subject')
test = test.merge(test_demographics, how='left', on='subject')

# -- convert to seq ---

train_seq = fast_seq_agg(train)
test_seq = fast_seq_agg(test)

del train, test
gc.collect()

train_seq, label_encoder, label_encoder_aux = le(train_seq)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(train_loader, model, optimizer, criterion, device, scheduler, ema=None, current_step=0, num_warmup_steps=0, fold=None):
    model.train()
        
    total_loss = 0
    total_samples = 0
    all_targets = []
    all_preds = []
    
    if cfg.use_mixup:
        mixup_criterion = MixupLoss(criterion, cfg.num_classes, cfg.mixup_alpha)
    
    loop = tqdm(train_loader, desc='train', leave=False)

    for batch in loop:
        optimizer.zero_grad()
        
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        
        curr_proba = np.random.random()
        is_warmup_phase = current_step < num_warmup_steps
        if cfg.use_mixup and curr_proba > cfg.mixup_proba and not is_warmup_phase:
            mixed_batch, targets_a, targets_b, lam = mixup_batch(batch, cfg.mixup_alpha, device)
            outputs = forward_model(model, mixed_batch, imu_only=cfg.imu_only)
            loss = mixup_criterion(outputs, targets_a, targets_b, lam)
            targets = batch['target']
        else:
            outputs = forward_model(model, batch, imu_only=cfg.imu_only)
            targets = batch['target']
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
        
        if cfg.do_wandb_log: # every batch
            wandb.log({
                f'fold_{fold}/train_batch_loss': loss.item(),
                f'fold_{fold}/learning_rate': scheduler.get_last_lr()[0],
                f'fold_{fold}/current_step': current_step
            })
        
        loop.set_postfix(loss=loss.item())
        current_step += 1
    
    avg_loss = total_loss / total_samples
    m = just_stupid_macro_f1_haha(all_targets, all_preds)

    return avg_loss, m, current_step

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
            for key in batch.keys():
                batch[key] = batch[key].to(device)
    
            outputs = forward_model(model, batch, imu_only=cfg.imu_only)
            targets = batch['target']
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
    os.makedirs(cfg.oof_dir, exist_ok=True)

    sgkf = StratifiedGroupKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
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
        
        if cfg.do_wandb_log:
            prefix1 = 'decomposewhar_' if cfg.use_dwhar else ''
            prefix2 = 'imu_only_' if cfg.imu_only else ''
            run_name = f'{prefix1}{prefix2}fold_{fold}'
            
            wandb.init(
                project=cfg.wandb_project,
                name=run_name,
                config={
                    'fold': fold,
                    'seq_len': cfg.seq_len,
                    'batch_size': cfg.bs,
                    'learning_rate': cfg.lr,
                    'n_epochs': cfg.n_epochs,
                    'num_classes': cfg.num_classes,
                    'use_mixup': cfg.use_mixup,
                    'use_ema': cfg.use_ema,
                    'imu_only': cfg.imu_only,
                    'seed': cfg.seed,
                    'weight_decay': cfg.weight_decay,
                    'label_smoothing': cfg.label_smoothing,
                    'patience': cfg.patience
                },
                tags=[f'fold_{fold}', prefix1.rstrip('_'), prefix2.rstrip('_')] if prefix1 or prefix2 else [f'fold_{fold}']
            )
        
        train_subset = train_seq.iloc[train_idx].reset_index(drop=True)
        val_subset = train_seq.iloc[val_idx].reset_index(drop=True)
        
        TSDataset = get_ts_dataset()
        
        train_dataset = TSDataset(
            dataframe=train_subset,
            seq_len=cfg.seq_len,
            target_col=cfg.target,
            aux_target_col=cfg.aux_target,
            train=True
        )
        val_dataset = TSDataset(
            dataframe=val_subset,
            seq_len=cfg.seq_len,
            target_col=cfg.target,
            aux_target_col=cfg.aux_target,
            train=False
        )
        
        train_loader = DataLoader(train_dataset, batch_size=cfg.bs, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=cfg.bs, shuffle=False, num_workers=4)

        TSModel, m_params = get_ts_model_and_params(imu_only=cfg.imu_only)
        model = TSModel(**m_params).to(device)

        if cfg.do_wandb_log:
            wandb.watch(model, log='all', log_freq=10)

        ema = EMA(model, decay=cfg.ema_decay) if cfg.use_ema else None

        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

        num_training_steps = cfg.n_epochs * len(train_loader)
        num_warmup_steps = int(cfg.num_warmup_steps_ratio * num_training_steps)
        current_step = 0

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        best_val_score = -np.inf
        patience_counter = 0
        prefix1 = 'decomposewhar_' if cfg.use_dwhar else ''
        prefix2 = 'imu_only_' if cfg.imu_only else ''
        best_model_path = os.path.join(cfg.model_dir, f'{prefix1}{prefix2}model_fold{fold}.pt')
        best_ema_path = os.path.join(cfg.model_dir, f'{prefix1}{prefix2}model_ema_fold{fold}.pt') if cfg.use_ema else None
        
        for epoch in range(cfg.n_epochs):
            print(f'{epoch=}')
            
            train_loss, train_f1, current_step = train_epoch(
                train_loader, model, optimizer, criterion, device, scheduler, 
                ema, current_step, num_warmup_steps, fold
            )
            val_loss, val_f1, _, _ = valid_epoch(val_loader, model, criterion, device, ema)
            
            print(f'{train_loss=}, {train_f1=}')
            print(f'{val_loss=}, {val_f1=}')
            
            if cfg.do_wandb_log:
                wandb.log({
                    f'fold_{fold}/epoch': epoch,
                    f'fold_{fold}/train_loss': train_loss,
                    f'fold_{fold}/train_f1': train_f1,
                    f'fold_{fold}/val_loss': val_loss,
                    f'fold_{fold}/val_f1': val_f1,
                    f'fold_{fold}/best_val_f1': best_val_score,
                    f'fold_{fold}/patience_counter': patience_counter
                })
            
            if val_f1 > best_val_score:
                best_val_score = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                if cfg.use_ema and ema is not None:
                    ema_state_dict = {}
                    for name in ema.shadow:
                        ema_state_dict[name] = ema.shadow[name]
                    torch.save(ema_state_dict, best_ema_path)
                
                if cfg.do_wandb_log:
                    wandb.log({f'fold_{fold}/new_best_val_f1': best_val_score})
            else:
                patience_counter += 1
            
            if patience_counter >= cfg.patience:
                print('early stopping')
                if cfg.do_wandb_log:
                    wandb.log({f'fold_{fold}/early_stopped_epoch': epoch})
                break

        model.load_state_dict(torch.load(best_model_path))

        if cfg.use_ema and best_ema_path and os.path.exists(best_ema_path):
            ema_state_dict = torch.load(best_ema_path, map_location=device)
            for name, param in model.named_parameters():
                if name in ema_state_dict:
                    param.data = ema_state_dict[name]

        best_models.append(model)
        best_f1_scores.append(best_val_score)
        
        if cfg.do_wandb_log:
            wandb.log({
                f'fold_{fold}/final_val_f1': best_val_score,
                f'fold_{fold}/fold_completed': True
            })
        
        model.eval()
        all_preds = []
        with torch.no_grad():
            val_loader = DataLoader(val_dataset, batch_size=cfg.bs, shuffle=False, num_workers=4)
            for batch in val_loader:
                for key in batch.keys():
                    batch[key] = batch[key].to(device)

                outputs = forward_model(model, batch, imu_only=cfg.imu_only)
                all_preds.append(outputs.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        oof_preds[val_idx] = all_preds
        
        if cfg.do_wandb_log:
            wandb.finish()
    
    oof_pred_labels = np.argmax(oof_preds, axis=1)
    oof_m = just_stupid_macro_f1_haha(oof_targets, oof_pred_labels)
    print(f'{oof_m=}')
    
    if cfg.do_wandb_log:
        wandb.init(
            project=cfg.wandb_project,
            name=f'{prefix1}{prefix2}final_results',
            config={'final_results': True}
        )
        
        wandb.log({
            'oof_macro_f1': oof_m,
            'mean_cv_f1': np.mean(best_f1_scores),
            'std_cv_f1': np.std(best_f1_scores),
            'fold_f1_scores': {f'fold_{i}': score for i, score in enumerate(best_f1_scores)}
        })
        
        wandb.finish()
    
    oof_preds_path = os.path.join(cfg.oof_dir, f'{prefix1}{prefix2}oof_preds.npy')
    oof_targets_path = os.path.join(cfg.oof_dir, f'{prefix1}{prefix2}oof_targets.npy')
    oof_pred_labels_path = os.path.join(cfg.oof_dir, f'{prefix1}{prefix2}oof_pred_labels.npy')

    np.save(oof_preds_path, oof_preds)
    np.save(oof_targets_path, oof_targets)
    np.save(oof_pred_labels_path, oof_pred_labels)

    oof_info = {
        'oof_macro_f1': oof_m,
        'best_f1_scores_per_fold': best_f1_scores,
        'mean_cv_f1': np.mean(best_f1_scores),
        'std_cv_f1': np.std(best_f1_scores)
    }
    
    oof_info_path = os.path.join(cfg.oof_dir, f'{prefix1}{prefix2}oof_info.json')
    with open(oof_info_path, 'w') as f:
        json.dump(oof_info, f, indent=2)

    return best_models, oof_preds