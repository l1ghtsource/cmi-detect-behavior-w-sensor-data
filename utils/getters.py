from data.ts_datasets import (
    TS_CMIDataset,
    TS_Demo_CMIDataset,
    TS_CMIDataset_DecomposeWHAR,
    TS_CMIDataset_DecomposeWHAR_Megasensor
)
from models.ts_models import TS_MSModel, TS_IMUModel, TS_Demo_MSModel, TS_Demo_IMUModel
from models.decompose_whar import MultiSensorDecomposeWHAR, OneSensorDecomposeWHAR
from configs.config import cfg

def get_ts_dataset():
    if cfg.use_dwhar:
        return TS_CMIDataset_DecomposeWHAR_Megasensor if cfg.use_megasensor else TS_CMIDataset_DecomposeWHAR
    return TS_Demo_CMIDataset if cfg.use_demo else TS_CMIDataset

def get_ts_model_and_params(imu_only):
    if cfg.use_dwhar:
        if cfg.use_megasensor or imu_only:
            return OneSensorDecomposeWHAR, {
                'M': len(cfg.imu_cols) + len(cfg.thm_cols) + len(cfg.tof_cols) if cfg.use_megasensor else len(cfg.imu_cols),
                'L': cfg.seq_len,
                'num_classes': cfg.num_classes,
                'D': cfg.ddim,
                'S': cfg.stride
            }
        else:
            return MultiSensorDecomposeWHAR, {
                'imu_num_sensor': cfg.imu_num_sensor,
                'thm_num_sensor': cfg.thm_num_sensor,
                'tof_num_sensor': cfg.tof_num_sensor,
                'imu_M': len(cfg.imu_cols),
                'thm_M': len(cfg.thm_cols),
                'tof_M': len(cfg.tof_cols),
                'L': cfg.seq_len,
                'num_classes': cfg.num_classes,
                'D': cfg.ddim,
                'S': cfg.stride,
                'use_cross_sensor': cfg.use_cross_sensor
            }

    if imu_only:
        model_cls = TS_Demo_IMUModel if cfg.use_demo else TS_IMUModel
        params = {
            'imu_features': len(cfg.imu_cols),
            'num_classes': cfg.num_classes,
            'hidden_dim': 128
        }
        if cfg.use_demo:
            params['demo_features'] = len(cfg.demo_cols)
        return model_cls, params
    else:
        model_cls = TS_Demo_MSModel if cfg.use_demo else TS_MSModel
        params = {
            'imu_features': len(cfg.imu_cols),
            'thm_features': len(cfg.thm_cols),
            'tof_features': len(cfg.tof_cols),
            'num_classes': cfg.num_classes,
            'hidden_dim': 128
        }
        if cfg.use_demo:
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
    else:
        if imu_only:
            return model(batch['imu'], batch['demographics']) if cfg.use_demo else model(batch['imu'])
        else:
            if cfg.use_demo:
                return model(batch['imu'], batch['thm'], batch['tof'], batch['demographics'])
            else:
                return model(batch['imu'], batch['thm'], batch['tof'])