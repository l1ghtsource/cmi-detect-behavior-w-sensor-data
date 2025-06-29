import openvino as ov
import torch

from configs.config import cfg
from models.timemil import (
    MultiSensor_TimeMIL_v1, 
    MultiSensor_TimeMIL_v2,
    TimeMIL_SingleSensor_Multibranch_v1,
    TimeMIL_SingleSensor_Singlebranch_v1
)

def convert_models_to_openvino():
    for sensor_type in ['imu_only', 'imu+tof+thm']:
        for weights_path, params in cfg.weights_pathes[sensor_type].items():
            if sensor_type == 'imu_only':
                TSModel = TimeMIL_SingleSensor_Singlebranch_v1 if params['timemil_singlebranch'] else TimeMIL_SingleSensor_Multibranch_v1
            else:
                TSModel = MultiSensor_TimeMIL_v1 if params['timemil_ver'] == '1' else MultiSensor_TimeMIL_v2
            
            model = TSModel(**params['model_params'])
            
            example_input = create_example_input(sensor_type)
            
            for i in range(cfg.n_splits):
                model_path = f'{weights_path}/{params["prefix"]}model_fold{i}.pt'
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                
                ov_model = ov.convert_model(model, example_input=example_input)
                ov_model_path = f'{weights_path}/{params["prefix"]}model_fold{i}_openvino.xml'
                ov.save_model(ov_model, ov_model_path)
                print(f'done !!! {model_path} -> {ov_model_path}')

def create_example_input(sensor_type):
    if sensor_type == 'imu_only':
        return {
            'imu_data': torch.randn(1, cfg.seq_len, len(cfg.imu_cols)),
            'pad_mask': torch.ones(cfg.seq_len,)
        }
    else:
        return {
            'imu_data': torch.randn(1, cfg.seq_len, len(cfg.imu_cols)),
            'tof_data': torch.randn(5, cfg.seq_len, len(cfg.tof_cols)),
            'thm_data': torch.randn(5, cfg.seq_len, len(cfg.thm_cols)),
            'pad_mask': torch.ones(cfg.seq_len,)
        }
