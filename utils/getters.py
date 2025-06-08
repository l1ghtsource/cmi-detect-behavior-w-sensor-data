from data.ts_datasets import (
    TS_CMIDataset,
    TS_Demo_CMIDataset,
    TS_CMIDataset_DecomposeWHAR,
    TS_CMIDataset_DecomposeWHAR_Megasensor
)
from models.ts_models import TS_MSModel, TS_IMUModel, TS_Demo_MSModel, TS_Demo_IMUModel
from models.decompose_whar import (
    MultiSensor_DecomposeWHAR_v1, DecomposeWHAR_SingleSensor_v1, 
    MultiSensor_DecomposeWHAR_v2, DecomposeWHAR_SingleSensor_v2
)
from models.timemil import MultiSensor_TimeMIL_v1, TimeMIL_SingleSensor_v1
from configs.config import cfg

def get_ts_dataset():
    if cfg.use_dwhar or cfg.use_timemil: # datasex is compatitable for decopmosewhar and timemil !! so i'm happy today
        return TS_CMIDataset_DecomposeWHAR_Megasensor if cfg.use_megasensor else TS_CMIDataset_DecomposeWHAR
    return TS_Demo_CMIDataset if cfg.use_demo else TS_CMIDataset

def get_ts_model_and_params(imu_only):
    if cfg.use_dwhar: # decomposewhar
        if cfg.use_megasensor or imu_only: # all data in one sensor OR only imu sensor
            return DecomposeWHAR_SingleSensor_v2, {
                'M': cfg.imu_vars + cfg.thm_vars + cfg.tof_vars if cfg.use_megasensor else cfg.imu_vars,
                'L': cfg.seq_len,
                'num_classes': cfg.num_classes,
                'D': cfg.ddim,
                'S': cfg.stride
            }
        else:
            return MultiSensor_DecomposeWHAR_v2, { # multi sensor model
                'num_imu': cfg.imu_num_sensor,
                'num_thm': cfg.thm_num_sensor,
                'num_tof': cfg.tof_num_sensor,
                'imu_vars': cfg.imu_vars,
                'thm_vars': cfg.thm_vars,
                'tof_vars': cfg.tof_vars,
                'L': cfg.seq_len,
                'num_classes': cfg.num_classes,
                'D': cfg.ddim,
                'S': cfg.stride,
                # 'use_cross_sensor': cfg.use_cross_sensor
            }
    elif cfg.use_timemil: # timemil
        if imu_only:
            return TimeMIL_SingleSensor_v1, { # only imu sensor
                'n_classes': cfg.num_classes,
                'mDim': cfg.timemil_dim, 
                'max_seq_len': cfg.seq_len,
                'dropout': cfg.timemil_dropout
            }
        else:
            return MultiSensor_TimeMIL_v1, { # multi sensor model
                'n_classes': cfg.num_classes,
                'mDim': cfg.timemil_dim, 
                'max_seq_len': cfg.seq_len,
                'dropout': cfg.timemil_dropout
            }
    else: # classic cnn-lstm model
        if imu_only: # only imu sensor
            model_cls = TS_Demo_IMUModel if cfg.use_demo else TS_IMUModel
            params = {
                'imu_features': len(cfg.imu_cols),
                'num_classes': cfg.num_classes,
                'hidden_dim': 128
            }
            if cfg.use_demo: # w/ demography
                params['demo_features'] = len(cfg.demo_cols)
            return model_cls, params
        else: # multi sensor model
            model_cls = TS_Demo_MSModel if cfg.use_demo else TS_MSModel
            params = {
                'imu_features': len(cfg.imu_cols),
                'thm_features': len(cfg.thm_cols),
                'tof_features': len(cfg.tof_cols),
                'num_classes': cfg.num_classes,
                'hidden_dim': 128
            }
            if cfg.use_demo: # w/ demography
                params['demo_features'] = len(cfg.demo_cols)
            return model_cls, params

def forward_model(model, batch, imu_only):
    if cfg.use_dwhar:
        if cfg.use_megasensor:
            return model(batch['megasensor'])
        elif imu_only:
            return model(batch['imu'])
        else:
            return model(batch['imu'], batch['thm'], batch['tof'])
    elif cfg.use_timemil:
        if imu_only:
            return model(batch['imu'])
        else:
            return model(batch['imu'], batch['thm'], batch['tof'])
    else:
        if imu_only:
            return model(batch['imu'], batch['demographics']) if cfg.use_demo else model(batch['imu'])
        else:
            if cfg.use_demo:
                return model(batch['imu'], batch['thm'], batch['tof'], batch['demographics'])
            else:
                return model(batch['imu'], batch['thm'], batch['tof'])