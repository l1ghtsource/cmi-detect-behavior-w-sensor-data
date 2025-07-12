import os
import json

import pandas as pd
import numpy as np

from tqdm import tqdm

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup
)

import wandb

from configs.config import cfg
from modules.ema import EMA
from modules.mixup import MixupLoss, mixup_batch
from optimizers.lookahead import Lookahead
from optimizers.sam import SAM
from utils.getters import (
    get_optimizer, 
    get_ts_dataset, 
    get_ts_model_and_params,
    forward_model,
    get_prefix
)
from utils.data_preproc import (
    fast_seq_agg, 
    le, 
    convert_to_world_coordinates, 
    remove_gravity_from_acc,
    apply_symmetry,
    fe
)
from utils.metrics import just_stupid_macro_f1_haha, comp_metric
from utils.seed import seed_everything

# --- set seed ---

seed_everything()
g = torch.Generator(device='cpu').manual_seed(cfg.seed)

# --- wandb ---

if cfg.do_wandb_log:
    os.environ['WANDB_API_KEY'] = cfg.wandb_api_key

# --- load data ---

train = pd.read_csv(cfg.train_path)
train_demographics = pd.read_csv(cfg.train_demographics_path)

# --- join demographics stats ---

train = train.merge(train_demographics, how='left', on='subject')

# --- preproc & convert to seq ---

if cfg.use_world_coords:
    train = convert_to_world_coordinates(train)

if cfg.only_remove_g: # can't be used w/ use_world_coords
    train = remove_gravity_from_acc(train)

if cfg.use_hand_symm and cfg.imu_only:
    right_handed_mask = train['handedness'] == 1
    train.loc[right_handed_mask, cfg.imu_cols] = apply_symmetry(train.loc[right_handed_mask, cfg.imu_cols])

if cfg.apply_fe:
    train = fe(train)

train = le(train)
train_seq = fast_seq_agg(train)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(train_loader, model, optimizer, main_criterion, hybrid_criterions, seq_type_criterion, orientation_criterion, device, scheduler, ema=None, current_step=0, num_warmup_steps=0, fold=None):
    model.train()
        
    total_loss = 0
    total_samples = 0
    all_targets = []
    all_preds = []
    
    if cfg.use_mixup:
        mixup_criterion = MixupLoss(main_criterion, hybrid_criterions, seq_type_criterion, orientation_criterion)
    
    loop = tqdm(train_loader, desc='train', leave=False)

    for batch in loop:
        optimizer.zero_grad()
        
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        
        curr_proba = np.random.random()
        is_warmup_phase = current_step < num_warmup_steps
        if cfg.use_mixup and curr_proba > cfg.mixup_proba and not is_warmup_phase:
            mixed_batch, targets_a, targets_b, seq_type_targets_a, seq_type_targets_b, orientation_targets_a, orientation_targets_b, lam = mixup_batch(batch, cfg.mixup_alpha, device)
            outputs, seq_type_outputs, orientation_outputs, ext1_out1, ext2_out1, ext3_out1, ext4_out1, ext5_out1, ext6_out1 = forward_model(model, mixed_batch, imu_only=cfg.imu_only)
            main_loss, ext1_out1_loss, ext2_out1_loss, ext3_out1_loss, ext4_out1_loss, ext5_out1_loss, ext6_out1_loss, seq_type_loss, orientation_loss = mixup_criterion(
                outputs, seq_type_outputs, orientation_outputs, 
                ext1_out1, ext2_out1, ext3_out1, ext4_out1, ext5_out1, ext6_out1,
                targets_a, targets_b, 
                seq_type_targets_a, seq_type_targets_b, 
                orientation_targets_a, orientation_targets_b, lam
            )      
            ext_out1_loss = (ext1_out1_loss + ext2_out1_loss + ext3_out1_loss + ext4_out1_loss +  ext5_out1_loss + ext6_out1_loss) / 6  
            loss = cfg.main_weight * main_loss + cfg.main_weight * ext_out1_loss + cfg.seq_type_aux_weight * seq_type_loss + cfg.orientation_aux_weight * orientation_loss
            targets = batch['main_target']
            seq_type_aux_targets = batch['seq_type_aux_target']
            orientation_aux_targets = batch['orientation_aux_target']
        else:
            outputs, seq_type_outputs, orientation_outputs, ext1_out1, ext2_out1, ext3_out1, ext4_out1, ext5_out1, ext6_out1 = forward_model(model, batch, imu_only=cfg.imu_only)
            targets = batch['main_target']
            seq_type_aux_targets = batch['seq_type_aux_target']
            orientation_aux_targets = batch['orientation_aux_target']
            main_loss = main_criterion(outputs, targets)
            ext1_out1_loss = hybrid_criterions[0](ext1_out1, targets)
            ext2_out1_loss = hybrid_criterions[1](ext2_out1, targets)
            ext3_out1_loss = hybrid_criterions[2](ext3_out1, targets)
            ext4_out1_loss = hybrid_criterions[3](ext4_out1, targets)
            ext5_out1_loss = hybrid_criterions[4](ext5_out1, targets)
            ext6_out1_loss = hybrid_criterions[5](ext6_out1, targets)
            ext_out1_loss = (ext1_out1_loss + ext2_out1_loss + ext3_out1_loss + ext4_out1_loss +  ext5_out1_loss + ext6_out1_loss) / 6  
            seq_type_loss = seq_type_criterion(seq_type_outputs, seq_type_aux_targets)
            orientation_loss = orientation_criterion(orientation_outputs, orientation_aux_targets)
            loss = cfg.main_weight * main_loss + cfg.main_weight * ext_out1_loss + cfg.seq_type_aux_weight * seq_type_loss + cfg.orientation_aux_weight * orientation_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_norm)

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
                f'fold_{fold}/train_main_loss': main_loss.item(),
                f'fold_{fold}/train_ext1_out1_loss': ext1_out1_loss.item(),
                f'fold_{fold}/train_ext2_out1_loss': ext2_out1_loss.item(),
                f'fold_{fold}/train_ext3_out1_loss': ext3_out1_loss.item(),
                f'fold_{fold}/train_ext4_out1_loss': ext4_out1_loss.item(),
                f'fold_{fold}/train_ext5_out1_loss': ext5_out1_loss.item(),
                f'fold_{fold}/train_ext6_out1_loss': ext6_out1_loss.item(),
                f'fold_{fold}/train_seq_type_loss': seq_type_loss.item(),
                f'fold_{fold}/train_orientation_loss': orientation_loss.item(),
                f'fold_{fold}/learning_rate': scheduler.get_last_lr()[0],
                f'fold_{fold}/current_step': current_step
            })
        
        loop.set_postfix(loss=loss.item())
        current_step += 1
    
    avg_loss = total_loss / total_samples
    avg_m, bm, mm = comp_metric(all_targets, all_preds)

    return avg_loss, avg_m, bm, mm, current_step

def valid_epoch(val_loader, model, main_criterion, hybrid_criterions, seq_type_criterion, orientation_criterion, device, ema=None):
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
    
            outputs, seq_type_outputs, orientation_outputs, ext1_out1, ext2_out1, ext3_out1, ext4_out1, ext5_out1, ext6_out1 = forward_model(model, batch, imu_only=cfg.imu_only)
            targets = batch['main_target']
            seq_type_aux_targets = batch['seq_type_aux_target']
            orientation_aux_targets = batch['orientation_aux_target']
            main_loss = main_criterion(outputs, targets)
            ext1_out1_loss = hybrid_criterions[0](ext1_out1, targets)
            ext2_out1_loss = hybrid_criterions[1](ext2_out1, targets)
            ext3_out1_loss = hybrid_criterions[2](ext3_out1, targets)
            ext4_out1_loss = hybrid_criterions[3](ext4_out1, targets)
            ext5_out1_loss = hybrid_criterions[4](ext5_out1, targets)
            ext6_out1_loss = hybrid_criterions[5](ext6_out1, targets)
            ext_out1_loss = (ext1_out1_loss + ext2_out1_loss + ext3_out1_loss + ext4_out1_loss +  ext5_out1_loss + ext6_out1_loss) / 6  
            seq_type_loss = seq_type_criterion(seq_type_outputs, seq_type_aux_targets)
            orientation_loss = orientation_criterion(orientation_outputs, orientation_aux_targets)
            loss = cfg.main_weight * main_loss + cfg.main_weight * ext_out1_loss + cfg.seq_type_aux_weight * seq_type_loss + cfg.orientation_aux_weight * orientation_loss
            
            total_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            loop.set_postfix(loss=loss.item())

    if cfg.use_ema and ema is not None:
        ema.restore()
    
    avg_loss = total_loss / total_samples
    avg_m, bm, mm = comp_metric(all_targets, all_preds)
    return avg_loss, avg_m, bm, mm, all_targets, all_preds

def average_model_weights(checkpoints, top_k):
    if len(checkpoints) < top_k:
        top_k = len(checkpoints)

    top_checkpoints = checkpoints[:top_k]
    averaged_state_dict = {}
    
    first_state_dict = torch.load(top_checkpoints[0]['model_path'], map_location=device)
    
    for key in first_state_dict.keys():
        averaged_state_dict[key] = torch.zeros_like(first_state_dict[key], dtype=torch.float)
    
    for checkpoint in top_checkpoints:
        state_dict = torch.load(checkpoint['model_path'], map_location=device)
        for key in state_dict.keys():
            averaged_state_dict[key] += state_dict[key].float() / top_k
    
    return averaged_state_dict

def get_oof_predictions(model, val_loader):
    model.eval()

    all_preds = []
    with torch.no_grad():
        for batch in val_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            outputs, _, _, _, _, _, _, _, _ = forward_model(model, batch, imu_only=cfg.imu_only)
            all_preds.append(outputs.cpu().numpy())

    return np.concatenate(all_preds, axis=0)

def run_training_with_stratified_group_kfold():
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.oof_dir, exist_ok=True)

    if cfg.use_main_target_weighting:
        class_weights_main = compute_class_weight(class_weight='balanced', classes=np.arange(cfg.main_num_classes), y=train_seq[cfg.main_target].values)
        class_weights_main_tensor = torch.tensor(class_weights_main, dtype=torch.float).to(device)

    if cfg.use_seq_type_aux_target_weighting:
        class_weights_seq_type = compute_class_weight(class_weight='balanced', classes=np.arange(cfg.seq_type_aux_num_classes), y=train_seq[cfg.seq_type_aux_target].values)
        class_weights_seq_type_tensor = torch.tensor(class_weights_seq_type, dtype=torch.float).to(device)

    prefix = get_prefix(cfg.imu_only)

    sgkf = StratifiedGroupKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    targets = train_seq[cfg.main_target].values
    groups = train_seq[cfg.group].values
    
    oof_preds_top1 = np.zeros((len(train_seq), cfg.main_num_classes))
    oof_preds_top3_avg = np.zeros((len(train_seq), cfg.main_num_classes))
    oof_preds_top5_avg = np.zeros((len(train_seq), cfg.main_num_classes))
    oof_targets = train_seq[cfg.main_target].values
    
    best_models = []
    best_f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_seq, targets, groups)):
        print(f'fold {fold+1}/{cfg.n_splits}')
        # if fold != cfg.curr_fold:
        #     continue
        
        if cfg.do_wandb_log:
            run_name = f'{prefix}fold_{fold}'
            
            wandb.init(
                project=cfg.wandb_project,
                name=run_name,
                config={
                    'fold': fold,
                    'seq_len': cfg.seq_len,
                    'batch_size': cfg.bs,
                    'learning_rate': cfg.lr,
                    'n_epochs': cfg.n_epochs,
                    'use_mixup': cfg.use_mixup,
                    'use_ema': cfg.use_ema,
                    'imu_only': cfg.imu_only,
                    'seed': cfg.seed,
                    'weight_decay': cfg.weight_decay,
                    'label_smoothing': cfg.label_smoothing,
                    'patience': cfg.patience,
                    'use_lookahead': cfg.use_lookahead,
                    'optimizer': cfg.optim_type,
                },
                tags=[f'fold_{fold}']
            )
        
        train_subset = train_seq.iloc[train_idx].reset_index(drop=True)
        val_subset = train_seq.iloc[val_idx].reset_index(drop=True)
        
        TSDataset = get_ts_dataset()
        
        train_dataset = TSDataset(
            dataframe=train_subset,
            seq_len=cfg.seq_len,
            main_target=cfg.main_target,
            orientation_aux_target=cfg.orientation_aux_target,
            seq_type_aux_target=cfg.seq_type_aux_target,
            behavior_aux_target=cfg.behavior_aux_target,
            phase_aux_target=cfg.phase_aux_target,
            train=True
        )
        val_dataset = TSDataset(
            dataframe=val_subset,
            seq_len=cfg.seq_len,
            main_target=cfg.main_target,
            orientation_aux_target=cfg.orientation_aux_target,
            seq_type_aux_target=cfg.seq_type_aux_target,
            behavior_aux_target=cfg.behavior_aux_target,
            phase_aux_target=cfg.phase_aux_target,
            train=False,
            norm_stats=train_dataset.norm_stats
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.bs, 
            shuffle=True, 
            pin_memory=True, 
            persistent_workers=True, 
            prefetch_factor=4, 
            num_workers=4,
            generator=g,
            worker_init_fn=lambda worker_id: np.random.seed(cfg.seed + worker_id)
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=cfg.bs, 
            shuffle=False, 
            pin_memory=True, 
            persistent_workers=True, 
            prefetch_factor=4, 
            num_workers=4,
            generator=g,
            worker_init_fn=lambda worker_id: np.random.seed(cfg.seed + worker_id)           
        )

        TSModel, m_params = get_ts_model_and_params(imu_only=cfg.imu_only)
        model = TSModel(**m_params).to(device)

        fucking_kaggle_p100 = True
        if not fucking_kaggle_p100:
            model = torch.compile(model, mode='max-autotune')

        # if cfg.do_wandb_log:
        #     wandb.watch(model, log='all', log_freq=10)

        ema = EMA(model, decay=cfg.ema_decay) if cfg.use_ema else None

        optimizer = get_optimizer(model=model, lr=cfg.lr, lr_muon=cfg.lr_muon, weight_decay=cfg.weight_decay)
        if cfg.use_lookahead:
            optimizer = Lookahead(optimizer)    
        if cfg.use_sam:
            optimizer = SAM(optimizer)    

        if cfg.use_main_target_weighting:
            main_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing, weight=class_weights_main_tensor)
        else:
            main_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

        if cfg.use_seq_type_aux_target_weighting:
            seq_type_criterion = nn.CrossEntropyLoss(weight=class_weights_seq_type_tensor)
        else:
            seq_type_criterion = nn.CrossEntropyLoss()

        orientation_criterion = nn.CrossEntropyLoss()

        hybrid_criterions = [nn.CrossEntropyLoss() for _ in range(6)]

        num_training_steps = cfg.n_epochs * len(train_loader)
        num_warmup_steps = int(cfg.num_warmup_steps_ratio * num_training_steps)
        current_step = 0

        scheduler_params = {
            'optimizer': optimizer,
            'num_warmup_steps': num_warmup_steps,
            'num_training_steps': num_training_steps
        }

        if cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(**scheduler_params)
        elif cfg.scheduler == 'cosine_cycle':
            scheduler = get_cosine_schedule_with_warmup(**scheduler_params, num_cycles=4)
        elif cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(**scheduler_params)  
        
        best_val_score = -np.inf
        patience_counter = 0
        fold_checkpoints = []
        
        for epoch in range(cfg.n_epochs):
            print(f'{epoch=}')
            
            train_loss, avg_m_train, bm_train, mm_train, current_step = train_epoch(
                train_loader, model, optimizer, main_criterion, hybrid_criterions, seq_type_criterion, orientation_criterion, device, scheduler, 
                ema, current_step, num_warmup_steps, fold
            )
            val_loss, avg_m_val, bm_val, mm_val, _, _ = valid_epoch(val_loader, model, main_criterion, hybrid_criterions, seq_type_criterion, orientation_criterion, device, ema)
            
            print(f'{train_loss=}, {avg_m_train=}, {bm_train=}, {mm_train=},')
            print(f'{val_loss=}, {avg_m_val=}, {bm_val=}, {mm_val=}')

            if cfg.do_wandb_log:
                wandb.log({
                    f'fold_{fold}/epoch': epoch,
                    f'fold_{fold}/train_loss': train_loss,
                    f'fold_{fold}/train_avg_f1': avg_m_train,
                    f'fold_{fold}/train_binary_f1': bm_train,
                    f'fold_{fold}/train_macro_f1': mm_train,
                    f'fold_{fold}/val_loss': val_loss,
                    f'fold_{fold}/val_avg_f1': avg_m_val,
                    f'fold_{fold}/val_binary_f1': bm_val,
                    f'fold_{fold}/val_macro_f1': mm_val,
                    f'fold_{fold}/best_val_avg_f1': best_val_score,
                    f'fold_{fold}/patience_counter': patience_counter
                })

            model_path = os.path.join(cfg.model_dir, f'{prefix}model_fold{fold}_val_f1_{avg_m_val:.4f}_epoch{epoch:03d}.pt')
            ema_path = os.path.join(cfg.model_dir, f'{prefix}model_ema_fold{fold}_val_f1_{avg_m_val:.4f}_epoch{epoch:03d}.pt') if cfg.use_ema else None

            torch.save(model.state_dict(), model_path)
            if cfg.use_ema and ema is not None:
                ema_state_dict = {}
                for name in ema.shadow:
                    ema_state_dict[name] = ema.shadow[name]
                torch.save(ema_state_dict, ema_path)

            fold_checkpoints.append({
                'score': avg_m_val,
                'epoch': epoch,
                'model_path': model_path,
                'ema_path': ema_path
            })

            fold_checkpoints.sort(key=lambda x: x['score'], reverse=True)
            
            if len(fold_checkpoints) > 5:
                to_remove = fold_checkpoints[5:]
                fold_checkpoints = fold_checkpoints[:5]
                
                for checkpoint in to_remove:
                    if os.path.exists(checkpoint['model_path']):
                        os.remove(checkpoint['model_path'])
                    if checkpoint['ema_path'] and os.path.exists(checkpoint['ema_path']):
                        os.remove(checkpoint['ema_path'])

            if avg_m_val > best_val_score:
                best_val_score = avg_m_val
                patience_counter = 0
                if cfg.do_wandb_log:
                    wandb.log({f'fold_{fold}/new_best_val_avg_f1': best_val_score})
            else:
                patience_counter += 1
            
            if patience_counter >= cfg.patience:
                print('early stopping')
                if cfg.do_wandb_log:
                    wandb.log({f'fold_{fold}/early_stopped_epoch': epoch})
                break

        best_checkpoint = fold_checkpoints[0]
        model.load_state_dict(torch.load(best_checkpoint['model_path']))
        if cfg.use_ema and best_checkpoint['ema_path'] and os.path.exists(best_checkpoint['ema_path']):
            ema_state_dict = torch.load(best_checkpoint['ema_path'], map_location=device)
            for name, param in model.named_parameters():
                if name in ema_state_dict:
                    param.data = ema_state_dict[name]
        
        preds_top1 = get_oof_predictions(model, val_loader)
        oof_preds_top1[val_idx] = preds_top1
        
        averaged_weights_top3 = average_model_weights(fold_checkpoints, 3)
        model.load_state_dict(averaged_weights_top3)
        preds_top3_avg = get_oof_predictions(model, val_loader)
        oof_preds_top3_avg[val_idx] = preds_top3_avg

        averaged_weights_top5 = average_model_weights(fold_checkpoints, 5)
        model.load_state_dict(averaged_weights_top5)
        preds_top5_avg = get_oof_predictions(model, val_loader)
        oof_preds_top5_avg[val_idx] = preds_top5_avg

        model.load_state_dict(torch.load(best_checkpoint['model_path']))
        if cfg.use_ema and best_checkpoint['ema_path'] and os.path.exists(best_checkpoint['ema_path']):
            ema_state_dict = torch.load(best_checkpoint['ema_path'], map_location=device)
            for name, param in model.named_parameters():
                if name in ema_state_dict:
                    param.data = ema_state_dict[name]

        best_models.append(model)
        best_f1_scores.append(best_val_score)
        
        if cfg.do_wandb_log:
            wandb.log({
                f'fold_{fold}/final_val_avg_f1': best_val_score,
                f'fold_{fold}/fold_completed': True
            })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if cfg.do_wandb_log:
            wandb.finish()
    
    oof_pred_labels_top1 = np.argmax(oof_preds_top1, axis=1)
    oof_pred_labels_top3_avg = np.argmax(oof_preds_top3_avg, axis=1)
    oof_pred_labels_top5_avg = np.argmax(oof_preds_top5_avg, axis=1)
    
    oof_m, oof_bm, oof_mm = comp_metric(oof_targets, oof_pred_labels_top1)
    oof_m_top3_avg, oof_bm_top3_avg, oof_mm_top3_avg = comp_metric(oof_targets, oof_pred_labels_top3_avg)
    oof_m_top5_avg, oof_bm_top5_avg, oof_mm_top5_avg = comp_metric(oof_targets, oof_pred_labels_top5_avg)
    
    print(f'use top1 models: {oof_m=}, {oof_bm=}, {oof_mm=}')
    print(f'use top3 avg models: {oof_m_top3_avg=}, {oof_bm_top3_avg=}, {oof_mm_top3_avg=}')
    print(f'use top5 avg models: {oof_m_top5_avg=}, {oof_bm_top5_avg=}, {oof_mm_top5_avg=}')
    
    if cfg.do_wandb_log:
        wandb.init(
            project=cfg.wandb_project,
            name=f'{prefix}final_results',
            config={'final_results': True}
        )
        
        wandb.log({
            'oof_avg_f1': oof_m,
            'oof_binary_f1': oof_bm,
            'oof_macro_f1': oof_mm,
            'oof_avg_f1_top3_avg': oof_m_top3_avg,
            'oof_binary_f1_top3_avg': oof_bm_top3_avg,
            'oof_macro_f1_top3_avg': oof_mm_top3_avg,
            'oof_avg_f1_top5_avg': oof_m_top5_avg,
            'oof_binary_f1_top5_avg': oof_bm_top5_avg,
            'oof_macro_f1_top5_avg': oof_mm_top5_avg,
            'mean_cv_avg_f1': np.mean(best_f1_scores),
            'std_cv_avg_f1': np.std(best_f1_scores),
            'fold_avg_f1_scores': {f'fold_{i}': score for i, score in enumerate(best_f1_scores)}
        })
        
        wandb.finish()
    
    oof_preds_top1_path = os.path.join(cfg.oof_dir, f'{prefix}oof_preds_top1.npy')
    oof_pred_labels_top1_path = os.path.join(cfg.oof_dir, f'{prefix}oof_pred_labels_top1.npy')
    
    oof_preds_top3_avg_path = os.path.join(cfg.oof_dir, f'{prefix}oof_preds_top3_avg.npy')
    oof_pred_labels_top3_avg_path = os.path.join(cfg.oof_dir, f'{prefix}oof_pred_labels_top3_avg.npy')
    
    oof_preds_top5_avg_path = os.path.join(cfg.oof_dir, f'{prefix}oof_preds_top5_avg.npy')
    oof_pred_labels_top5_avg_path = os.path.join(cfg.oof_dir, f'{prefix}oof_pred_labels_top5_avg.npy')
    
    oof_targets_path = os.path.join(cfg.oof_dir, f'{prefix}oof_targets.npy')

    np.save(oof_preds_top1_path, oof_preds_top1)
    np.save(oof_pred_labels_top1_path, oof_pred_labels_top1)
    
    np.save(oof_preds_top3_avg_path, oof_preds_top3_avg)
    np.save(oof_pred_labels_top3_avg_path, oof_pred_labels_top3_avg)
    
    np.save(oof_preds_top5_avg_path, oof_preds_top5_avg)
    np.save(oof_pred_labels_top5_avg_path, oof_pred_labels_top5_avg)
    
    np.save(oof_targets_path, oof_targets)

    oof_info = {
        'top1_model': {
            'oof_avg_f1': oof_m,
            'oof_binary_f1': oof_bm,
            'oof_macro_f1': oof_mm
        },
        'top3_avg_model': {
            'oof_avg_f1': oof_m_top3_avg,
            'oof_binary_f1': oof_bm_top3_avg,
            'oof_macro_f1': oof_mm_top3_avg
        },
        'top5_avg_model': {
            'oof_avg_f1': oof_m_top5_avg,
            'oof_binary_f1': oof_bm_top5_avg,
            'oof_macro_f1': oof_mm_top5_avg
        },
        'cv_statistics': {
            'best_avg_f1_scores_per_fold': best_f1_scores,
            'mean_cv_avg_f1': np.mean(best_f1_scores),
            'std_cv_avg_f1': np.std(best_f1_scores)
        }
    }
    
    oof_info_path = os.path.join(cfg.oof_dir, f'{prefix}oof_info.json')
    with open(oof_info_path, 'w') as f:
        json.dump(oof_info, f, indent=2)

    return best_models, oof_preds_top1