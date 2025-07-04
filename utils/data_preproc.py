import pandas as pd
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from configs.config import cfg

# take it from https://arxiv.org/pdf/1812.07035
def add_quat6d(df: pd.DataFrame,
               w_col: str = 'rot_w',
               x_col: str = 'rot_x',
               y_col: str = 'rot_y',
               z_col: str = 'rot_z',
               prefix: str = 'quat6d_',
               eps: float = 1e-8) -> pd.DataFrame:
    q = df[[w_col, x_col, y_col, z_col]].to_numpy(float)
    mask_valid = (~np.isnan(q).any(axis=1)) & (np.linalg.norm(q, axis=1) > eps)
    quat6d = np.zeros((len(df), 6), dtype=np.float32)

    if mask_valid.any():
        # (w, x, y, z) -> (x, y, z, w)
        q_scipy = q[mask_valid][:, [1, 2, 3, 0]]
        rot_mats = R.from_quat(q_scipy).as_matrix()          # (M, 3, 3)
        first_two_cols = rot_mats[:, :, :2].reshape(-1, 6)   # (M, 6)
        quat6d[mask_valid] = first_two_cols

    for i in range(6):
        df[f'{prefix}{i}'] = quat6d[:, i]

    df.drop(columns=[w_col, x_col, y_col, z_col], inplace=True)

    return df

def remove_gravity_from_acc_df(acc_data, rot_data):
    if isinstance(acc_data, pd.DataFrame):
        acc_values = acc_data[['acc_x', 'acc_y', 'acc_z']].values
    else:
        acc_values = acc_data

    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

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
        except ValueError:
             linear_accel[i, :] = acc_values[i, :]
             
    return linear_accel

def calculate_angular_velocity_from_quat(rot_data, time_delta=1/200): # Assuming 200Hz sampling rate
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = quat_values.shape[0]
    angular_vel = np.zeros((num_samples, 3))

    for i in range(num_samples - 1):
        q_t = quat_values[i]
        q_t_plus_dt = quat_values[i+1]

        if np.all(np.isnan(q_t)) or np.all(np.isclose(q_t, 0)) or \
           np.all(np.isnan(q_t_plus_dt)) or np.all(np.isclose(q_t_plus_dt, 0)):
            continue

        try:
            rot_t = R.from_quat(q_t)
            rot_t_plus_dt = R.from_quat(q_t_plus_dt)
            delta_rot = rot_t.inv() * rot_t_plus_dt
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except ValueError:
            pass
            
    return angular_vel

def calculate_angular_distance(rot_data):
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = quat_values.shape[0]
    angular_dist = np.zeros(num_samples)

    for i in range(num_samples - 1):
        q1 = quat_values[i]
        q2 = quat_values[i+1]

        if np.all(np.isnan(q1)) or np.all(np.isclose(q1, 0)) or \
           np.all(np.isnan(q2)) or np.all(np.isclose(q2, 0)):
            angular_dist[i] = 0
            continue
        try:
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)
            relative_rotation = r1.inv() * r2
            angle = np.linalg.norm(relative_rotation.as_rotvec())
            angular_dist[i] = angle
        except ValueError:
            angular_dist[i] = 0
            pass
            
    return angular_dist

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

    df_world[['acc_x_world', 'acc_y_world', 'acc_z_world']] = np.nan
    df_world.loc[valid_quat_mask, ['acc_x_world', 'acc_y_world', 'acc_z_world']] = acc_world_linear

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

    df_copy[['acc_x_remove_g', 'acc_y_remove_g', 'acc_z_remove_g']] = np.nan
    df_copy[['acc_x_remove_g', 'acc_y_remove_g', 'acc_z_remove_g']] = linear_accel
             
    return df_copy

# https://stackoverflow.com/questions/32438252/efficient-way-to-apply-mirror-effect-on-quaternion-rotation
def apply_symmetry(data): # TODO: test it?? its can be wrong..
    transformed = data.copy()
    transformed['acc_x'] = -transformed['acc_x']
    transformed['acc_z'] = -transformed['acc_z']
    transformed['rot_x'] = -transformed['rot_x']
    transformed['rot_z'] = -transformed['rot_z']
    return transformed

def fe(df):
    if cfg.kaggle_fe:
        df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
        df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))
        
        df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
        df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)
        
        def get_linear_accel(df):
            res = remove_gravity_from_acc_df(
                df[['acc_x', 'acc_y', 'acc_z']],
                df[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
            )
            res = pd.DataFrame(res, columns=['linear_acc_x', 'linear_acc_y', 'linear_acc_z'], index=df.index)
            return res
        
        linear_accel_df = df.groupby('sequence_id').apply(get_linear_accel, include_groups=False)
        linear_accel_df = linear_accel_df.droplevel('sequence_id')
        df = df.join(linear_accel_df)
        
        df['linear_acc_mag'] = np.sqrt(df['linear_acc_x']**2 + df['linear_acc_y']**2 + df['linear_acc_z']**2)
        df['linear_acc_mag_jerk'] = df.groupby('sequence_id')['linear_acc_mag'].diff().fillna(0)

        def calc_angular_velocity(df):
            res = calculate_angular_velocity_from_quat( df[['rot_x', 'rot_y', 'rot_z', 'rot_w']] )
            res = pd.DataFrame(res, columns=['angular_vel_x', 'angular_vel_y', 'angular_vel_z'], index=df.index)
            return res
        
        angular_velocity_df = df.groupby('sequence_id').apply(calc_angular_velocity, include_groups=False)
        angular_velocity_df = angular_velocity_df.droplevel('sequence_id')
        df = df.join(angular_velocity_df)

        # df['angular_jerk_x'] = df.groupby('sequence_id')['angular_vel_x'].diff().fillna(0)
        # df['angular_jerk_y'] = df.groupby('sequence_id')['angular_vel_y'].diff().fillna(0)
        # df['angular_jerk_z'] = df.groupby('sequence_id')['angular_vel_z'].diff().fillna(0)

        # df['angular_snap_x'] = df.groupby('sequence_id')['angular_jerk_x'].diff().fillna(0)
        # df['angular_snap_y'] = df.groupby('sequence_id')['angular_jerk_y'].diff().fillna(0)
        # df['angular_snap_z'] = df.groupby('sequence_id')['angular_jerk_z'].diff().fillna(0)

        def calc_angular_distance(df):
            res = calculate_angular_distance(df[['rot_x', 'rot_y', 'rot_z', 'rot_w']])
            res = pd.DataFrame(res, columns=['angular_distance'], index=df.index)
            return res
        
        angular_distance_df = df.groupby('sequence_id').apply(calc_angular_distance, include_groups=False)
        angular_distance_df = angular_distance_df.droplevel('sequence_id')
        df = df.join(angular_distance_df)

    if cfg.fe_mag_ang: # don't use w/ kaggle_fe
        df['acc_mag'] = np.sqrt(df['acc_x'] ** 2 + df['acc_y'] ** 2 + df['acc_z'] ** 2)
        df['rot_mag'] = np.sqrt(df['rot_x'] ** 2 + df['rot_y'] ** 2 + df['rot_z'] ** 2)
        df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))

    if cfg.fe_col_diff:
        df['XY_acc'] = df['acc_x'] - df['acc_y']
        df['XZ_acc'] = df['acc_x'] - df['acc_z']
        df['YZ_acc'] = df['acc_y'] - df['acc_z']
        # df['XY_rot'] = df['rot_x'] - df['rot_y']
        # df['XZ_rot'] = df['rot_x'] - df['rot_z']
        # df['YZ_rot'] = df['rot_y'] - df['rot_z']

    if cfg.lag_lead_cum: # haha cum
        for c in ['acc_x', 'acc_y', 'acc_z']:
            df[f'{c}_lag_diff'] = df.groupby('sequence_id')[c].diff() # add 2, 3
            df[f'{c}_lead_diff'] = df.groupby('sequence_id')[c].diff(-1) # add -2, -3
            # df[f'{c}_lag_diff3'] = df.groupby('sequence_id')[c].diff(3)
            # df[f'{c}_lead_diff3'] = df.groupby('sequence_id')[c].diff(-3)
            # df[f'{c}_lag_diff5'] = df.groupby('sequence_id')[c].diff(5)
            # df[f'{c}_lead_diff5'] = df.groupby('sequence_id')[c].diff(-5)
            df[f'{c}_cumsum'] = df.groupby('sequence_id')[c].cumsum()
            df[f'{c}_cumsum'] = df.groupby('sequence_id')[f'{c}_cumsum'].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-6)
            )

    if cfg.fe_col_prod:
        for acc_col in ['acc_x', 'acc_y', 'acc_z']:
            for rot_col in ['rot_x', 'rot_y', 'rot_z']:
                df[f'{acc_col}_times_{rot_col}'] = df[acc_col] * df[rot_col]

    if cfg.fe_angles:
        df['acc_angle_xy'] = np.arctan2(df['acc_y'], df['acc_x'])
        df['acc_angle_xz'] = np.arctan2(df['acc_z'], df['acc_x'])
        df['acc_angle_yz'] = np.arctan2(df['acc_z'], df['acc_y'])
        df['acc_direction_change'] = np.abs(np.diff(df['acc_angle_xy'], prepend=df['acc_angle_xy'].iloc[0]))

    if cfg.fe_euler:
        df['roll'] = np.arctan2(
            2 * (df['rot_w'] * df['rot_x'] + df['rot_y'] * df['rot_z']), 
            1 - 2 * (df['rot_x'] ** 2 + df['rot_y'] ** 2)
        )
        df['pitch'] = np.arcsin(
            np.clip(2 * (df['rot_w'] * df['rot_y'] - df['rot_z'] * df['rot_x']), -1, 1)
        )
        df['yaw'] = np.arctan2(
            2 * (df['rot_w'] * df['rot_z'] + df['rot_x'] * df['rot_y']), 
            1 - 2 * (df['rot_y'] ** 2 + df['rot_z'] ** 2)
        )

    if cfg.fe_gravity:
        df['gravity_x'] = 2 * (df['rot_x'] * df['rot_z'] - df['rot_w'] * df['rot_y'])
        df['gravity_y'] = 2 * (df['rot_w'] * df['rot_x'] + df['rot_y'] * df['rot_z'])
        df['gravity_z'] = df['rot_w'] ** 2 - df['rot_x'] ** 2 - df['rot_y'] ** 2 + df['rot_z'] ** 2
        
        df['acc_vertical'] = (df['acc_x'] * df['gravity_x'] + 
                            df['acc_y'] * df['gravity_y'] + 
                            df['acc_z'] * df['gravity_z'])
        
        df['acc_horizontal_x'] = df['acc_x'] - df['acc_vertical'] * df['gravity_x']
        df['acc_horizontal_y'] = df['acc_y'] - df['acc_vertical'] * df['gravity_y']
        df['acc_horizontal_z'] = df['acc_z'] - df['acc_vertical'] * df['gravity_z']
        
        df['acc_horizontal_mag'] = np.sqrt(df['acc_horizontal_x'] ** 2 + 
                                        df['acc_horizontal_y'] ** 2 + 
                                        df['acc_horizontal_z'] ** 2)
        
        df.drop(columns=['acc_horizontal_x', 'acc_horizontal_y', 'acc_horizontal_z'], inplace=True)

    if cfg.fe_time_pos:
        seq_len = df.groupby('sequence_id')['sequence_id'].transform('count')
        df['time_from_start'] = df.groupby('sequence_id').cumcount() / seq_len
        df['time_to_end'] = 1 - df['time_from_start']
        df['sin_time_position'] = np.sin(df['time_from_start'] * seq_len * np.pi)

    if cfg.use_windows:
        window_sizes = [3, 5, 10]
        aggfuncs = ['mean', 'std', 'max', 'min']
        cols = ['acc_x', 'acc_y', 'acc_z']
        
        for window in window_sizes:
            for aggfunc in aggfuncs:
                result = df.groupby('sequence_id')[cols].rolling(
                    window, min_periods=1
                ).agg(aggfunc).reset_index(level=0, drop=True)
                for col in cols:
                    df[f'{col}_rolling_{window}_{aggfunc}'] = result[col]
                    df[f'{col}_back_rolling_{window}_{aggfunc}'] = (
                        df.groupby('sequence_id')[col]
                        .apply(lambda x: x[::-1].rolling(window, min_periods=1).agg(aggfunc)[::-1])
                        .reset_index(level=0, drop=True)
                    )

    if cfg.fe_relative_quat:
        EPS = 1e-8

        def _is_valid_q(q: np.ndarray) -> bool:
            return (not np.isnan(q).any()) and (np.linalg.norm(q) > EPS)

        def rel_quat_block(group: pd.DataFrame) -> pd.DataFrame:
            first_valid = None
            for w, x, y, z in group[['rot_w', 'rot_x', 'rot_y', 'rot_z']].to_numpy():
                q = np.array([w, x, y, z])
                if _is_valid_q(q):
                    first_valid = Quaternion(w, x, y, z).inverse
                    break

            if first_valid is None:
                zeros = np.zeros((len(group), 5))
                cols = ['rel_dqw', 'rel_dqx', 'rel_dqy', 'rel_dqz', 'rel_angle']
                return pd.DataFrame(zeros, columns=cols, index=group.index)

            rel_q_out = np.zeros((len(group), 5))

            for i, (w, x, y, z) in enumerate(
                group[['rot_w', 'rot_x', 'rot_y', 'rot_z']].to_numpy()
            ):
                q_cur = np.array([w, x, y, z])

                if _is_valid_q(q_cur):
                    rel_q = Quaternion(w, x, y, z) * first_valid
                    rel_q_out[i, :4] = [rel_q.w, rel_q.x, rel_q.y, rel_q.z]
                    rel_q_out[i, 4] = 2 * np.arccos(np.clip(rel_q.w, -1.0, 1.0))
                else:
                    rel_q_out[i, :] = 0.0

            return pd.DataFrame(
                rel_q_out,
                columns=['rel_dqw', 'rel_dqx', 'rel_dqy', 'rel_dqz', 'rel_angle'],
                index=group.index,
            )

        rel_df = (
            df.groupby('sequence_id', group_keys=False)
            .apply(rel_quat_block)
        )
        df = df.join(rel_df)

    if cfg.use_quat6d:
        df = add_quat6d(df)

    df[cfg.imu_cols] = df[cfg.imu_cols].ffill().bfill().fillna(0).values.astype('float32')
    
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