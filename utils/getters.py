import torch.nn as nn
from torch.optim import AdamW, Adam
from optimizers.adan import Adan
from optimizers.adamp import AdamP
from optimizers.madgrad import MADGRAD
from optimizers.adafisher import AdaFisherW
from optimizers.ranger import Ranger
from optimizers.muon import SingleDeviceMuonWithAuxAdam

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
    MultiSensor_TimeMIL_v1, TimeMIL_SingleSensor_Singlebranch_v1,
    MultiSensor_TimeMIL_v2, TimeMIL_SingleSensor_Multibranch_v1
)
from models.husformer import (
    MultiSensor_HUSFORMER_v1, MultiSensor_HUSFORMER_v2,
    SingleSensor_HUSFORMER_v1
)
from models.medformer import (
    MultiSensor_Medformer_v1, Medformer_SingleSensor_v1
)
from models.harmamba import (
    MultiSensor_HARMamba_v1, HARMamba_SingleSensor_v1
)
from models.modern_tcn import (
    MultiSensor_ModernTCN_v1, ModernTCN_SingleSensor_v1
)
from models.multi_bigru import (
    MultiResidualBiGRU_SingleSensor_v1
)
from models.se_unet import (
    SE_Unet_SingleSensor_v1, MultiSensor_SE_Unet_v1
)
from models.squeezeformer import (
    Squeezeformer_MultiSensor_v1, Squeezeformer_SingleSensor_v1
)
from models.panns_clf import (
    PANNsCLF_SingleSensor_v1
)
from models.convtran import (
    ConvTran_SingleSensor_v1, ConvTran_SingleSensor_MultiScale_v1, 
    ConvTran_SingleSensor_Residual_v1, ConvTran_SingleSensor_Inception_v1,
    ConvTran_SingleSensor_NoTranLol_v1,
    ConvTran_SingleSensor_SE_v1,
    TimeCNN_SingleSensor_v1
)
from models.filternet import (
    FilterNet_SingleSensor_v1
)
from models.basic_cnn1ds import (
    CNN1D_SingleSensor_v1
)
from models.public_model import (
    Public_SingleSensor_v1, Public_SingleSensor_v2
)
from models.wavenet import (
    WaveNet_SingleSensor_v1
)
from models.hybrid_model import (
    HybridModel_SingleSensor_v1,
    MultiSensor_HybridModel_v1
)
from models.imunet import (
    IMUNet_SingleSensor_v1
)
from configs.config import cfg

def get_muon_param_groups(model, lr_muon=0.02, lr_adam=3e-4, weight_decay=0.01):
    linear_weights = []
    linear_biases = []
    other_params = []
    
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for _, param in module.named_parameters():
                if param.ndim >= 2:
                    linear_weights.append(param)
                else:
                    linear_biases.append(param)
    
    all_linear_params = set(linear_weights + linear_biases)
    for _, param in model.named_parameters():
        if param not in all_linear_params:
            other_params.append(param)
    
    adamw_params = other_params + linear_biases
    
    param_groups = [
        dict(params=linear_weights, use_muon=True,
             lr=lr_muon, weight_decay=weight_decay),
        dict(params=adamw_params, use_muon=False,
             lr=lr_adam, betas=(0.9, 0.95), weight_decay=weight_decay),
    ]
    
    return param_groups

class SafeMuonWithAuxAdam(SingleDeviceMuonWithAuxAdam):
    def step(self, closure=None):
        for group in self.param_groups:
            original_params = group["params"]
            valid_params = [p for p in original_params if p.grad is not None]
            
            if len(valid_params) != len(original_params):
                print(f"Warning: {len(original_params) - len(valid_params)} params have grad=None")
            
            group["params"] = valid_params
        
        result = super().step(closure)
        
        for group in self.param_groups:
            pass
            
        return result

def get_optimizer(model, lr=cfg.lr, lr_muon=cfg.lr_muon, weight_decay=cfg.weight_decay):
    if cfg.optim_type == 'adamw':
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if cfg.optim_type == 'adam':
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif cfg.optim_type == 'adan':
        return Adan(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif cfg.optim_type == 'adamp':
        return AdamP(model.parameters(), lr=lr, weight_decay=weight_decay) 
    elif cfg.optim_type == 'madgrad':
        return MADGRAD(model.parameters(), lr=lr, weight_decay=weight_decay) 
    elif cfg.optim_type == 'adafisherw':
        return AdaFisherW(model.parameters(), lr=lr, weight_decay=weight_decay) 
    elif cfg.optim_type == 'ranger':
        print('use ranger w/o weight decay pls')
        return Ranger(model.parameters(), lr=lr, weight_decay=weight_decay) 
    elif cfg.optim_type == 'muonwauxadam':
        param_groups = get_muon_param_groups(model, lr_muon=lr_muon, lr_adam=lr, weight_decay=weight_decay)
        return SafeMuonWithAuxAdam(param_groups)
    else:
        raise Exception('stick your finger in your ass')

def get_ts_dataset():
    if cfg.selected_model not in ('baseline'): # datasex is compatitable for decopmosewhar and timemil !! so i'm happy today
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
            timemil_model = TimeMIL_SingleSensor_Singlebranch_v1 if cfg.timemil_singlebranch else TimeMIL_SingleSensor_Multibranch_v1
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
    elif cfg.selected_model == 'harmamba':
        if imu_only: # only imu sensor
            model_cls = HARMamba_SingleSensor_v1
            params = {
                'num_classes': cfg.main_num_classes,
            }
            return model_cls, params
        else: # multi sensor model
            model_cls = MultiSensor_HARMamba_v1
            params = {
                'num_classes': cfg.main_num_classes,
            }
            return model_cls, params
    elif cfg.selected_model == 'husformer':
        if imu_only: # only imu sensor
            model_cls = SingleSensor_HUSFORMER_v1
            params = {
                'output_dim': cfg.main_num_classes,
            }
            return model_cls, params
        else: # multi sensor model
            model_cls = MultiSensor_HUSFORMER_v2
            params = {
                'output_dim': cfg.main_num_classes,
            }
            return model_cls, params
    elif cfg.selected_model == 'medformer':
        if imu_only: # only imu sensor
            model_cls = Medformer_SingleSensor_v1
            params = {
                'num_classes': cfg.main_num_classes,
            }
            return model_cls, params
        else: # multi sensor model
            model_cls = MultiSensor_Medformer_v1
            params = {
                'num_classes': cfg.main_num_classes,
            }
            return model_cls, params
    elif cfg.selected_model == 'moderntcn':
        if imu_only: # only imu sensor
            model_cls = ModernTCN_SingleSensor_v1
            params = {
                'class_num': cfg.main_num_classes,
            }
            return model_cls, params
        else: # multi sensor model
            model_cls = MultiSensor_ModernTCN_v1
            params = {
                'class_num': cfg.main_num_classes,
            }
            return model_cls, params
    elif cfg.selected_model == 'multubigru':
        if imu_only: # only imu sensor
            model_cls = MultiResidualBiGRU_SingleSensor_v1
            params = {
                'num_classes': cfg.main_num_classes,
            }
            return model_cls, params
        else: # multi sensor model
            model_cls = ...
            params = {
                'num_classes': cfg.main_num_classes,
            }
            return model_cls, params
    elif cfg.selected_model == 'se_unet':
        if imu_only: # only imu sensor
            model_cls = SE_Unet_SingleSensor_v1
            params = {
                'num_classes': cfg.main_num_classes,
            }
            return model_cls, params
        else: # multi sensor model
            model_cls = MultiSensor_SE_Unet_v1
            params = {
                'num_classes': cfg.main_num_classes,
            }
            return model_cls, params
    elif cfg.selected_model == 'squeezeformer':
        if imu_only: # only imu sensor
            model_cls = Squeezeformer_SingleSensor_v1
            params = {
                'num_classes': cfg.main_num_classes,
            }
            return model_cls, params
        else: # multi sensor model
            model_cls = Squeezeformer_MultiSensor_v1
            params = {
                'num_classes': cfg.main_num_classes,
            }
            return model_cls, params
    elif cfg.selected_model == 'panns':
        if imu_only: # only imu sensor
            model_cls = PANNsCLF_SingleSensor_v1
            params = {
                'num_classes': cfg.main_num_classes,
            }
            return model_cls, params
        else: # multi sensor model
            # TODO: add panns multisensor model
            return None
    elif cfg.selected_model == 'convtran':
        if imu_only: # only imu sensor
            if cfg.convtran_type == 'default':
                model_cls = ConvTran_SingleSensor_v1
            elif cfg.convtran_type == 'multiscale':
                model_cls = ConvTran_SingleSensor_MultiScale_v1
            elif cfg.convtran_type == 'residual':
                model_cls = ConvTran_SingleSensor_Residual_v1
            elif cfg.convtran_type == 'inception':
                model_cls = ConvTran_SingleSensor_Inception_v1
            elif cfg.convtran_type == 'se':
                model_cls = ConvTran_SingleSensor_SE_v1
            elif cfg.convtran_type == 'notran':
                model_cls = ConvTran_SingleSensor_NoTranLol_v1
            params = {
                'num_classes': cfg.main_num_classes,
            }
            return model_cls, params
        else: # multi sensor model
            # TODO: add convtran multisensor model
            return None    
    elif cfg.selected_model == 'timecnn':
        if imu_only: # only imu sensor
            model_cls = TimeCNN_SingleSensor_v1
            params = {
                'num_classes': cfg.main_num_classes,
            }
            return model_cls, params
        else: # multi sensor model
            # TODO: add convtran multisensor model
            return None   
    elif cfg.selected_model == 'filternet':
        if imu_only: # only imu sensor
            model_cls = FilterNet_SingleSensor_v1
            params = {
                'num_classes': cfg.main_num_classes,
            }
            return model_cls, params
        else: # multi sensor model
            # TODO: add filternet multisensor model
            return None   
    elif cfg.selected_model == 'cnn1d':
        if imu_only: # only imu sensor
            model_cls = CNN1D_SingleSensor_v1
            params = {
                'num_classes': cfg.main_num_classes,
            }
            return model_cls, params
        else: # multi sensor model
            # TODO: add cnn1d multisensor model
            return None   
    elif cfg.selected_model == 'public':
        if imu_only: # only imu sensor
            model_cls = Public_SingleSensor_v1
            params = {
                'num_classes': cfg.main_num_classes,
            }
            return model_cls, params
        else: # multi sensor model
            # TODO: add public multisensor model
            return None 
    elif cfg.selected_model == 'public2':
        if imu_only: # only imu sensor
            model_cls = Public_SingleSensor_v2
            params = {
                'num_classes': cfg.main_num_classes,
            }
            return model_cls, params
        else: # multi sensor model
            # TODO: add public2 multisensor model
            return None 
    elif cfg.selected_model == 'wavenet':
        if imu_only: # only imu sensor
            model_cls = WaveNet_SingleSensor_v1
            params = {
                'num_classes': cfg.main_num_classes,
            }
            return model_cls, params
        else: # multi sensor model
            # TODO: add wavenet multisensor model
            return None 
    elif cfg.selected_model == 'hybrid':
        if imu_only: # only imu sensor
            model_cls = HybridModel_SingleSensor_v1
            params = {
                'num_classes': cfg.main_num_classes,
            }
            return model_cls, params
        else: # multi sensor model
            model_cls = MultiSensor_HybridModel_v1
            params = {
                'num_classes': cfg.main_num_classes,
            }
            return model_cls, params 
    elif cfg.selected_model == 'imunet':
        if imu_only: # only imu sensor
            model_cls = IMUNet_SingleSensor_v1
            params = {
                'num_classes': cfg.main_num_classes,
            }
            return model_cls, params
        else: # multi sensor model
            # TODO: add imunet multisensor model
            return None 
    elif cfg.selected_model == 'baseline':
        if imu_only: # only imu sensor
            model_cls = TS_IMUModel
            params = {
                'imu_features': len(cfg.imu_cols),
                'num_classes': cfg.main_num_classes,
                'hidden_dim': 256
            }
            return model_cls, params
        else: # multi sensor model
            model_cls = TS_MSModel
            params = {
                'imu_features': len(cfg.imu_cols),
                'thm_features': len(cfg.thm_cols),
                'tof_features': len(cfg.tof_cols),
                'num_classes': cfg.main_num_classes,
                'hidden_dim': 256
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
        prefix_parts.append(f'{cfg.timemil_extractor}')
    elif model == 'convtran':
        prefix_parts.append(f'{cfg.convtran_type}')
    elif model == 'cnn1d':
        prefix_parts.append(f'{cfg.cnn1d_extractor}')
        prefix_parts.append(f'{cfg.cnn1d_pooling}')
        if cfg.cnn1d_use_neck:
            prefix_parts.append('use_neck')

    if imu_only:
        prefix_parts.append('imu_only')
    
    prefix_parts.append(f'seq_len{cfg.seq_len}')
    
    if cfg.use_demo:
        prefix_parts.append('use_demo')

    if cfg.use_stats_vectors:
        prefix_parts.append('use_stats_vectors')

    if cfg.use_pad_mask:
        prefix_parts.append('use_pad_mask')

    if cfg.use_world_coords:
        prefix_parts.append('use_world_coords')

    if cfg.only_remove_g:
        prefix_parts.append('only_remove_g')

    if cfg.use_hand_symm:
        prefix_parts.append('use_hand_symm')

    if cfg.use_lookahead:
        prefix_parts.append('lookahead')

    if cfg.use_sam:
        prefix_parts.append('sam')

    if cfg.use_main_target_weighting:
        prefix_parts.append('main_target_weighting')

    if cfg.use_seq_type_aux_target_weighting:
        prefix_parts.append('seq_type_aux_target_weighting')

    if cfg.fe_mag_ang:
        prefix_parts.append('fe_mag_ang')
    
    if cfg.fe_col_diff:
        prefix_parts.append('fe_col_diff')

    if cfg.fe_time_pos:
        prefix_parts.append('fe_time_pos')

    if cfg.lag_lead_cum:
        prefix_parts.append('lag_lead_cum')

    if cfg.use_windows:
        prefix_parts.append('use_windows')

    if cfg.fe_col_prod:
        prefix_parts.append('fe_col_prod')

    if cfg.fe_angles:
        prefix_parts.append('fe_angles')

    if cfg.fe_euler:
        prefix_parts.append('fe_euler')

    if cfg.fe_freq_wavelet:
        prefix_parts.append('fe_freq_wavelet')

    if cfg.fe_gravity:
        prefix_parts.append('fe_gravity')

    if cfg.kaggle_fe:
        prefix_parts.append('kaggle_fe')

    if cfg.fe_relative_quat:
        prefix_parts.append('fe_relative_quat')

    if cfg.use_quat6d:
        prefix_parts.append('use_quat6d')

    if cfg.use_dct:
        prefix_parts.append('use_dct')
    
    prefix_parts.append(cfg.optim_type)

    prefix_parts.append(f'ls{cfg.label_smoothing}')
    
    if cfg.use_ema:
        prefix_parts.append(f'ema_{cfg.ema_decay}')

    if cfg.denoise_data != 'none':
        prefix_parts.append(cfg.denoise_data)

    if cfg.reverse_seq:
        prefix_parts.append('reverse_seq')

    if cfg.is_zebra:
        prefix_parts.append('zebra')
    
    return '_'.join(prefix_parts) + '_'