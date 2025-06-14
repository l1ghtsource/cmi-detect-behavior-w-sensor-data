from torch.optim import AdamW
from optimizers.adan import Adan
from optimizers.adamp import AdamP
from optimizers.madgrad import MADGRAD
from optimizers.adafisher import AdaFisherW
from optimizers.ranger import Ranger

from data.ts_datasets import (
    TS_CMIDataset,
    TS_CMIDataset_DecomposeWHAR,
    TS_CMIDataset_DecomposeWHAR_Megasensor
)
from models.ts_models import TS_MSModel, TS_IMUModel
from models.decompose_whar import (
    MultiSensor_DecomposeWHAR_v1, DecomposeWHAR_SingleSensor_v1, 
    MultiSensor_DecomposeWHAR_v2, DecomposeWHAR_SingleSensor_v2
)
from models.timemil import (
    MultiSensor_TimeMIL_v1, TimeMIL_SingleSensor_v1,
    MultiSensor_TimeMIL_v2,
)
from configs.config import cfg

def get_optimizer(params):
    if cfg.optim_type == 'adamw':
        return AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optim_type == 'adan':
        return Adan(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optim_type == 'adamp':
        return AdamP(params, lr=cfg.lr, weight_decay=cfg.weight_decay) 
    elif cfg.optim_type == 'madgrad':
        return MADGRAD(params, lr=cfg.lr, weight_decay=cfg.weight_decay) 
    elif cfg.optim_type == 'adafisherw':
        return AdaFisherW(params, lr=cfg.lr, weight_decay=cfg.weight_decay) 
    elif cfg.optim_type == 'ranger':
        print('use ranger w/o weight decay pls')
        return Ranger(params, lr=cfg.lr, weight_decay=cfg.weight_decay) 
    else:
        raise Exception('stick your finger in your ass')

def get_ts_dataset():
    if cfg.selected_model in ('timemil', 'decomposewhar'): # datasex is compatitable for decopmosewhar and timemil !! so i'm happy today
        return TS_CMIDataset_DecomposeWHAR_Megasensor if cfg.use_megasensor else TS_CMIDataset_DecomposeWHAR
    return TS_CMIDataset

def get_ts_model_and_params(imu_only):
    if cfg.selected_model == 'decomposewhar':
        if cfg.use_megasensor or imu_only: # all data in one sensor OR only imu sensor
            dwhar_model = DecomposeWHAR_SingleSensor_v1 if cfg.dwhar_ver == '1' else DecomposeWHAR_SingleSensor_v2
            return dwhar_model, {
                'M': cfg.imu_vars + cfg.thm_vars + cfg.tof_vars if cfg.use_megasensor else cfg.imu_vars,
                'L': cfg.seq_len,
                'num_classes': cfg.main_num_classes,
                'D': cfg.ddim,
                'S': cfg.stride
            }
        else:
            dwhar_model = MultiSensor_DecomposeWHAR_v1 if cfg.dwhar_ver == '1' else MultiSensor_DecomposeWHAR_v2
            return dwhar_model, { # multi sensor model
                'num_imu': cfg.imu_num_sensor,
                'num_thm': cfg.thm_num_sensor,
                'num_tof': cfg.tof_num_sensor,
                'imu_vars': cfg.imu_vars,
                'thm_vars': cfg.thm_vars,
                'tof_vars': cfg.tof_vars,
                'L': cfg.seq_len,
                'num_classes': cfg.main_num_classes,
                'D': cfg.ddim,
                'S': cfg.stride,
                # 'use_cross_sensor': cfg.use_cross_sensor
            }
    elif cfg.selected_model == 'timemil':
        if imu_only:
            timemil_model = TimeMIL_SingleSensor_v1
            return timemil_model, { # only imu sensor
                'n_classes': cfg.main_num_classes,
                'mDim': cfg.timemil_dim, 
                'max_seq_len': cfg.seq_len,
                'dropout': cfg.timemil_dropout
            }
        else:
            timemil_model = MultiSensor_TimeMIL_v1 if cfg.timemil_ver == '1' else MultiSensor_TimeMIL_v2
            return timemil_model, { # multi sensor model
                'n_classes': cfg.main_num_classes,
                'mDim': cfg.timemil_dim, 
                'max_seq_len': cfg.seq_len,
                'dropout': cfg.timemil_dropout
            }
    elif cfg.selected_model == 'baseline':
        if imu_only: # only imu sensor
            model_cls = TS_IMUModel
            params = {
                'imu_features': len(cfg.imu_cols),
                'num_classes': cfg.main_num_classes,
                'hidden_dim': 128
            }
            return model_cls, params
        else: # multi sensor model
            model_cls = TS_MSModel
            params = {
                'imu_features': len(cfg.imu_cols),
                'thm_features': len(cfg.thm_cols),
                'tof_features': len(cfg.tof_cols),
                'num_classes': cfg.main_num_classes,
                'hidden_dim': 128
            }
            return model_cls, params

def forward_model(model, batch, imu_only):
    inputs = []
    if cfg.selected_model == 'decomposewhar' and cfg.use_megasensor:
        inputs.append(batch['megasensor'])
    else:
        inputs.append(batch['imu'])
        if not imu_only:
            inputs.append(batch['thm'])
            inputs.append(batch['tof'])
    if cfg.use_demo:
        inputs.append(batch['demography_bin'])
        inputs.append(batch['demography_cont'])
    if cfg.use_time_pos:
        inputs.append(batch['time_pos'])
    if cfg.use_diff:
        inputs.append(batch['imu_diff'])
        inputs.append(batch['thm_diff'])
        inputs.append(batch['tof_diff'])
    if cfg.use_pad_mask:
        inputs.append(batch['pad_mask'])
    return model(*inputs)
            
# haha what a shit
def get_prefix(imu_only):
    prefix_parts = []
    
    model = cfg.selected_model
    prefix_parts.append(model)

    if model == 'decomposewhar':
        prefix_parts.append(f'ver{cfg.dwhar_ver}')
    elif model == 'timemil':
        prefix_parts.append(f'ver{cfg.timemil_ver}')

    if imu_only:
        prefix_parts.append('imu_only')
    
    prefix_parts.append(f'seq_len{cfg.seq_len}')
    
    if cfg.use_demo:
        prefix_parts.append('use_demo')

    if cfg.use_stats_vectors:
        prefix_parts.append('use_stats_vectors')

    if cfg.use_diff:
        prefix_parts.append('use_diff')

    if cfg.use_time_pos:
        prefix_parts.append('use_time_pos')

    if cfg.use_pad_mask:
        prefix_parts.append('use_pad_mask')

    if cfg.use_lookahead:
        prefix_parts.append('lookahead')

    if cfg.use_sam:
        prefix_parts.append('sam')

    if cfg.use_main_target_weighting:
        prefix_parts.append('main_target_weighting')

    if cfg.use_seq_type_aux_target_weighting:
        prefix_parts.append('seq_type_aux_target_weighting')
    
    prefix_parts.append(cfg.optim_type)

    prefix_parts.append(f'ls{cfg.label_smoothing}')
    
    if cfg.use_ema:
        prefix_parts.append(f'ema_{cfg.ema_decay}')

    if cfg.denoise_data != 'none':
        prefix_parts.append(cfg.denoise_data)
    
    return '_'.join(prefix_parts) + '_'