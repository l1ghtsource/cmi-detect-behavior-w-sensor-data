import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from configs.config import cfg

def convert_to_world_coordinates(df):
    quats = df[['rot_w', 'rot_x', 'rot_y', 'rot_z']].to_numpy()
    accs = df[['acc_x', 'acc_y', 'acc_z']].to_numpy()

    df_world = df.copy()
    valid_quat_mask = ~np.isnan(quats).any(axis=1)

    valid_quats = quats[valid_quat_mask][:, [1, 2, 3, 0]]
    rots = R.from_quat(valid_quats)
    
    acc_world = rots.apply(accs[valid_quat_mask])

    g = np.array([0, 0, 9.81])
    acc_world_linear = acc_world - g

    df_world.loc[valid_quat_mask, ['acc_x', 'acc_y', 'acc_z']] = acc_world_linear

    return df_world

# idea from https://www.kaggle.com/code/rktqwe/lb-0-77-linear-accel-tf-bilstm-gru-attention/notebook (remove only g from acc)
def remove_gravity_from_acc(df):
    acc_values = df[['acc_x', 'acc_y', 'acc_z']].values
    quat_values = df[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values

    df_copy = df.copy()

    num_samples = acc_values.shape[0]
    linear_accel = np.zeros_like(acc_values)
    
    gravity_world = np.array([0, 0, 9.81])

    for i in range(num_samples):
        if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
            linear_accel[i, :] = acc_values[i, :] 
            continue
        try:
            rotation = R.from_quat(quat_values[i])
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
        except ValueError as e:
             print(f'{e=}')
             linear_accel[i, :] = acc_values[i, :]

    df_copy[['acc_x', 'acc_y', 'acc_z']] = linear_accel
             
    return df_copy

# https://stackoverflow.com/questions/32438252/efficient-way-to-apply-mirror-effect-on-quaternion-rotation
def apply_symmetry(data): # TODO: test it?? its can be wrong..
    transformed = data.copy()
    transformed['acc_x'] = -transformed['acc_x']
    transformed['acc_z'] = -transformed['acc_z']
    transformed['rot_x'] = -transformed['rot_x']
    transformed['rot_z'] = -transformed['rot_z']
    return transformed

# def rolling_agg(dt: pd.DataFrame, step: int, aggfunc: str, cols: list, back: bool = False) -> pd.DataFrame:
#     if back:
#         rolling = dt[cols][::-1].rolling(step, min_periods=1)
#         suffix = f"_back_rolling_{step}_{aggfunc}"
#     else:
#         rolling = dt[cols].rolling(step, min_periods=1)
#         suffix = f"_rolling_{step}_{aggfunc}"

#     if aggfunc.startswith("quantile"):
#         quantile = int(aggfunc.split("_")[1]) / 100
#         return rolling.quantile(quantile).add_suffix(suffix)
#     else:
#         return rolling.agg(aggfunc).add_suffix(suffix)

def fe(df):
    df['acc_mag'] = np.sqrt(df['acc_x'] ** 2 + df['acc_y'] ** 2 + df['acc_z'] ** 2) # TODO: add rot mag and rot mag jerk
    df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))
    df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)

    seq_len = len(df)
    df['time_position'] = np.arange(seq_len) / seq_len
    df['time_from_start'] = np.arange(seq_len)
    df['time_to_end'] = seq_len - np.arange(seq_len)

    # for shift in [1, 2, 3, -1, -2]:
    #     if shift > 0:
    #         suffix_name = f"_lag_{shift}"
    #         fill_data = df[cfg.imu_cols].iloc[:shift]
    #     else:
    #         suffix_name = f"_lead_{abs(shift)}"
    #         fill_data = df[cfg.imu_cols].iloc[shift:]
        
    #     df = df.join(
    #         df[cfg.imu_cols]
    #         .shift(shift)
    #         .fillna(fill_data)
    #         .add_suffix(suffix_name)
    #     )
    
    # jerk_cols = []
    # for col in cfg.imu_cols:
    #     jerk_col = f'{col}_jerk'
    #     df[jerk_col] = df[col].diff().fillna(0)
    #     jerk_cols.append(jerk_col)
    
    # window_sizes = [3, 5, 10, 15]
    # aggfuncs = ["mean", "std", "max", "min", "quantile_75", "quantile_25"]
    
    # for window in window_sizes:
    #     for aggfunc in aggfuncs:
    #         df = df.join(
    #             rolling_agg(df, window, aggfunc, cfg.imu_cols)
    #         )
    
    # for window in [3, 5, 10]:
    #     for aggfunc in ["mean", "std", "max"]:
    #         df = df.join(
    #             rolling_agg(df, window, aggfunc, cfg.imu_cols, back=True)
    #         )
    
    # for window in [3, 5, 10]:
    #     for aggfunc in ["mean", "std", "max"]:
    #         df = df.join(
    #             rolling_agg(df, window, aggfunc, jerk_cols)
    #         )
    
    # sign_change_cols = []
    # for col in jerk_cols:
    #     sign_change_col = f'{col}_sign_change'
    #     sign_change = (
    #         df[col].apply(np.sign)
    #         .diff()
    #         .apply(np.abs)
    #         .divide(2)
    #         .fillna(0)
    #     )
    #     df[sign_change_col] = sign_change
    #     sign_change_cols.append(sign_change_col)
    
    # for window in [3, 5, 10]:
    #     df = df.join(
    #         rolling_agg(df, window, "sum", sign_change_cols)
    #     )

    return df

def fast_seq_agg(df):
    sc = cfg.static_cols
    seq_cols = [c for c in df.columns if c not in sc + ['sequence_counter', 'row_id']]
    static_cols = [c for c in sc if c in df.columns]

    df = df.sort_values(['sequence_id', 'sequence_counter']).reset_index(drop=True)

    seq_id_codes, _ = pd.factorize(df['sequence_id'])
    _, seq_start_idxs = np.unique(seq_id_codes, return_index=True)

    res = {'sequence_id': df['sequence_id'].values[seq_start_idxs]}

    for c in static_cols:
        res[c] = df[c].values[seq_start_idxs]

    for c in seq_cols:
        res[c] = np.split(df[c].values, seq_start_idxs[1:])

    res_df = pd.DataFrame(res)
    
    for col in cfg.tof_cols:
        if col in res_df.columns:
            res_df[col] = res_df[col].apply(lambda x: np.where(x == -1, 255, x))

    return res_df

def le(df):
    mapper_main = {
        "Above ear - pull hair": 0,
        "Cheek - pinch skin": 1,
        "Eyebrow - pull hair": 2,
        "Eyelash - pull hair": 3, 
        "Forehead - pull hairline": 4,
        "Forehead - scratch": 5,
        "Neck - pinch skin": 6, 
        "Neck - scratch": 7,
        "Drink from bottle/cup": 8,
        "Feel around in tray and pull out an object": 9,
        "Glasses on/off": 10,
        "Pinch knee/leg skin": 11, 
        "Pull air toward your face": 12,
        "Scratch knee/leg skin": 13,
        "Text on phone": 14,
        "Wave hello": 15,
        "Write name in air": 16,
        "Write name on leg": 17,
    }

    mapper_main_collapsed = {
        0: 0,
        1: 1,
        2: 2,
        3: 3, 
        4: 4,
        5: 5,
        6: 6, 
        7: 7,
        8: 8,
        9: 8,
        10: 8,
        11: 8, 
        12: 8,
        13: 8,
        14: 8,
        15: 8,
        16: 8,
        17: 8,
    }

    mapper_orientation = {
        'Seated Straight': 0,
        'Seated Lean Non Dom - FACE DOWN': 1,
        'Lie on Back': 2,
        'Lie on Side - Non Dominant': 3
    }

    mapper_seq_type = {
        'Non-Target': 0,
        'Target': 1,
    }

    mapper_behaviour = {
        'Hand at target location': 1,
        'Moves hand to target location': 2,
        'Performs gesture': 3,
        'Relaxes and moves hand to target location': 4,
    }

    mapper_phase = {
        'Transition': 1,
        'Gesture': 2,
    }

    df[cfg.main_target] = df[cfg.main_target].map(mapper_main)
    df[cfg.main_clpsd_target] = df[cfg.main_target].map(mapper_main_collapsed)
    df[cfg.orientation_aux_target] = df[cfg.orientation_aux_target].map(mapper_orientation)
    df[cfg.seq_type_aux_target] = df[cfg.seq_type_aux_target].map(mapper_seq_type)
    df[cfg.behavior_aux_target] = df[cfg.behavior_aux_target].map(mapper_behaviour)
    df[cfg.phase_aux_target] = df[cfg.phase_aux_target].map(mapper_phase)

    return df

def get_rev_mapping():
    main_ae_zhok = {
        "Above ear - pull hair": 0,
        "Cheek - pinch skin": 1,
        "Eyebrow - pull hair": 2,
        "Eyelash - pull hair": 3, 
        "Forehead - pull hairline": 4,
        "Forehead - scratch": 5,
        "Neck - pinch skin": 6, 
        "Neck - scratch": 7,
        "Drink from bottle/cup": 8,
        "Feel around in tray and pull out an object": 9,
        "Glasses on/off": 10,
        "Pinch knee/leg skin": 11, 
        "Pull air toward your face": 12,
        "Scratch knee/leg skin": 13,
        "Text on phone": 14,
        "Wave hello": 15,
        "Write name in air": 16,
        "Write name on leg": 17,
    }

    orientation_ae_zhok = {
        'Seated Straight': 0,
        'Seated Lean Non Dom - FACE DOWN': 1,
        'Lie on Back': 2,
        'Lie on Side - Non Dominant': 3
    }

    seq_type_ae_zhok = {
        'Non-Target': 0,
        'Target': 1,
    }

    return {y: x for x, y in main_ae_zhok.items()}, {y: x for x, y in orientation_ae_zhok.items()}, {y: x for x, y in seq_type_ae_zhok.items()}