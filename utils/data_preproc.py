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
    transformed['acc_y'] = -transformed['acc_y']
    transformed['rot_x'] = -transformed['rot_x']
    transformed['rot_y'] = -transformed['rot_y']
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

        def compute_detrended_velocity(group):
            acc_cols = ['linear_acc_x', 'linear_acc_y', 'linear_acc_z']
            vel = group[acc_cols].cumsum() * 0.1
            vel.columns = ['linear_vel_x', 'linear_vel_y', 'linear_vel_z']
            t = np.arange(len(group))
            for col in vel.columns:
                trend = np.polyval(np.polyfit(t, vel[col], 2), t)
                vel[col] -= trend
            return vel
        
        linear_vel_df = df.groupby('sequence_id').apply(compute_detrended_velocity, include_groups=False)
        linear_vel_df = linear_vel_df.droplevel('sequence_id')
        df = df.join(linear_vel_df)

        # def compute_detrended_position(group):
        #     vel_cols = ['linear_vel_x', 'linear_vel_y', 'linear_vel_z']
        #     pos = group[vel_cols].cumsum() * 0.1
        #     pos.columns = ['pos_x', 'pos_y', 'pos_z']
        #     t = np.arange(len(group))
        #     for col in pos.columns:
        #         trend = np.polyval(np.polyfit(t, pos[col], 2), t)
        #         pos[col] -= trend
        #     return pos
        
        # position_df = df.groupby('sequence_id').apply(compute_detrended_position, include_groups=False)
        # position_df = position_df.droplevel('sequence_id')
        # df = df.join(position_df)

        # def compute_cumulative_trajectory_length(group):
        #     pos_diff = group[['pos_x', 'pos_y', 'pos_z']].diff().fillna(0)
        #     dist = np.sqrt((pos_diff ** 2).sum(axis=1))
        #     cumulative_length = dist.cumsum()
        #     return cumulative_length

        # df['cumulative_trajectory_length'] = df.groupby('sequence_id').apply(
        #     compute_cumulative_trajectory_length, include_groups=False
        # ).reset_index(level=0, drop=True)
        # df['cumulative_trajectory_length'] = df.groupby('sequence_id')[f'cumulative_trajectory_length'].transform(
        #     lambda x: (x - x.mean()) / (x.std() + 1e-6)
        # )

        # epsilon = 1e-6
        # v = df[['linear_vel_x', 'linear_vel_y', 'linear_vel_z']].values
        # a = df[['linear_acc_x', 'linear_acc_y', 'linear_acc_z']].values
        # cross = np.cross(v, a)
        # cross_mag = np.linalg.norm(cross, axis=1)
        # v_mag = np.linalg.norm(v, axis=1).clip(min=epsilon)
        # df['trajectory_curvature'] = cross_mag / (v_mag ** 3)
        # dot = np.einsum('ij,ij->i', v, a)
        # df['tangential_accel'] = dot / v_mag.clip(min=epsilon)

        # for col in (
        #     'acc_mag', 'acc_mag_jerk', 
        #     'rot_angle', 'rot_angle_vel', 
        #     'linear_acc_mag', 'linear_acc_mag_jerk', 
        #     'angular_distance', 
        #     'angular_vel_x', 'angular_vel_y', 'angular_vel_z'
        # ):
        #     for x in (-2, -1, 1, 2):
        #         df[f'{col}_shift_{x}'] = df.groupby('sequence_id')[col].shift(x)

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

    if cfg.use_thm_neighbour_diff:
        # 2 -> 5 -> 4 -> 3 -> 1 -> 2 ring
        df['thm_25'] = df['thm_2'] - df['thm_5']
        df['thm_54'] = df['thm_5'] - df['thm_4']
        df['thm_43'] = df['thm_4'] - df['thm_3']
        df['thm_31'] = df['thm_3'] - df['thm_1']
        df['thm_12'] = df['thm_1'] - df['thm_2']

    if cfg.use_thm_diff:
        for i in range(1, 6):
            for j in [1]: # in (-3, -2, -1, 1, 2, 3):
                df[f'thm_{i}_diff_{j}'] = df.groupby('sequence_id')[f'thm_{i}'].diff(j)

    # def compute_tof_features_for_sequence(group):
    #     seq_len = len(group)
        
    #     features = {}
        
    #     for sensor_id in range(1, 6):
    #         sensor_cols = [col for col in cfg.tof_cols if col.startswith(f'tof_{sensor_id}_')]
            
    #         if not sensor_cols:
    #             continue
                
    #         sensor_data = group[sensor_cols].values # (seq_len, 64)
    #         sensor_images = sensor_data.reshape(seq_len, 8, 8)
            
    #         stats_mean, stats_std, stats_min, stats_max, stats_range = [], [], [], [], []
    #         com_x, com_y = [], []
    #         grad_x_mean, grad_y_mean = [], []
    #         neg_count = []
    #         contact_area_50, contact_area_100, contact_area_200 = [], [], []
            
    #         for t in range(seq_len):
    #             img = sensor_images[t]
                
    #             img_clean = np.where(img == -1, np.nan, img)

    #             if cfg.use_tof_stats:
    #                 mean_val = np.nanmean(img_clean)
    #                 std_val = np.nanstd(img_clean)
    #                 min_val = np.nanmin(img_clean)
    #                 max_val = np.nanmax(img_clean)
    #                 range_val = max_val - min_val if not (np.isnan(max_val) or np.isnan(min_val)) else 0
                    
    #                 stats_mean.append(mean_val if not np.isnan(mean_val) else 0)
    #                 stats_std.append(std_val if not np.isnan(std_val) else 0)
    #                 stats_min.append(min_val if not np.isnan(min_val) else 0)
    #                 stats_max.append(max_val if not np.isnan(max_val) else 0)
    #                 stats_range.append(range_val if not np.isnan(range_val) else 0)
                
    #             if cfg.use_tof_com:
    #                 if not np.isnan(min_val):
    #                     min_idx = np.unravel_index(np.nanargmin(img_clean), img_clean.shape)
    #                     com_x.append(min_idx[1])
    #                     com_y.append(min_idx[0])
    #                 else:
    #                     com_x.append(4)
    #                     com_y.append(4)
                
    #             if cfg.use_tof_grad:
    #                 img_filled = img_clean.copy()
    #                 nan_mask = np.isnan(img_filled)
    #                 if np.any(nan_mask):
    #                     img_filled[nan_mask] = mean_val if not np.isnan(mean_val) else 0
                    
    #                 grad_x = np.abs(np.diff(img_filled, axis=1))  # (8, 7)
    #                 grad_y = np.abs(np.diff(img_filled, axis=0))  # (7, 8)
                    
    #                 grad_x_mean.append(np.mean(grad_x))
    #                 grad_y_mean.append(np.mean(grad_y))
                
    #             if cfg.use_tof_neg_count:
    #                 neg_count.append(np.sum(img == -1))

    #             if cfg.use_tof_contact_area:
    #                 contact_area_50.append(np.sum((img_clean < 50) & (~np.isnan(img_clean))))
    #                 contact_area_100.append(np.sum((img_clean < 100) & (~np.isnan(img_clean))))
    #                 contact_area_200.append(np.sum((img_clean < 200) & (~np.isnan(img_clean))))
                
    #         if cfg.use_tof_stats:
    #             features[f'tof_{sensor_id}_mean'] = stats_mean
    #             features[f'tof_{sensor_id}_std'] = stats_std
    #             features[f'tof_{sensor_id}_min'] = stats_min
    #             features[f'tof_{sensor_id}_max'] = stats_max
    #             features[f'tof_{sensor_id}_range'] = stats_range
            
    #         if cfg.use_tof_com:
    #             features[f'tof_{sensor_id}_com_x'] = com_x
    #             features[f'tof_{sensor_id}_com_y'] = com_y

    #         if cfg.use_tof_grad:
    #             features[f'tof_{sensor_id}_grad_x'] = grad_x_mean
    #             features[f'tof_{sensor_id}_grad_y'] = grad_y_mean
            
    #         if cfg.use_tof_neg_count:
    #             features[f'tof_{sensor_id}_neg_count'] = neg_count

    #         if cfg.use_tof_contact_area:
    #             features[f'tof_{sensor_id}_contact_area_50'] = contact_area_50
    #             features[f'tof_{sensor_id}_contact_area_100'] = contact_area_100
    #             features[f'tof_{sensor_id}_contact_area_200'] = contact_area_200
        
    #     return pd.DataFrame(features, index=group.index)
    
    # tof_features = df.groupby('sequence_id', group_keys=False).apply(compute_tof_features_for_sequence)
    
    # if cfg.use_tof_neighbour_diff:
    #     sensor_pairs = [(2, 1), (1, 5), (5, 4), (4, 3), (3, 2)]
    #     for s1, s2 in sensor_pairs:
    #         if f'tof_{s1}_mean' in tof_features.columns and f'tof_{s2}_mean' in tof_features.columns:
    #             tof_features[f'tof_mean_diff_{s1}_{s2}'] = (
    #                 tof_features[f'tof_{s1}_mean'] - tof_features[f'tof_{s2}_mean']
    #             )

    # df = df.join(tof_features)

    df[cfg.imu_cols] = df[cfg.imu_cols].ffill().bfill().fillna(0).values.astype('float32')
    df[cfg.thm_cols] = df[cfg.thm_cols].ffill().bfill().fillna(0).values.astype('float32')
    df[cfg.tof_cols] = df[cfg.tof_cols].ffill().bfill().fillna(0).values.astype('float32')

    return df

def _tof_norm(arr: np.ndarray) -> np.ndarray:
    arr = np.array([np.nan if v is None else v for v in arr], dtype=float)
    arr[arr == -1] = 255
    return arr / 255

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
            res_df[col] = res_df[col].apply(_tof_norm)

    return res_df

def kalman_1d(z: np.ndarray,
              Q: float = 1e-5,
              R: float = 1e-2,
              x0: float | None = None,
              P0: float = 1.0) -> np.ndarray:
    n = len(z)
    x_hat = np.empty(n)
    x_est = z[0] if x0 is None else x0
    P = P0
    for k in range(n):
        x_pred = x_est
        P = P + Q
        K = P / (P + R)
        x_est = x_pred + K * (z[k] - x_pred)
        P = (1.0 - K) * P
        x_hat[k] = x_est
    return x_hat

def apply_kalman_to_sequences(df: pd.DataFrame,
                              sensor_cols: list[str] = ['acc_x', 'acc_y', 'acc_z'],
                              Q: float = 1e-5,
                              R: float = 1e-2) -> pd.DataFrame:
    out = df.copy()
    for col in sensor_cols:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda seq: kalman_1d(np.asarray(seq, dtype=float), Q=Q, R=R)
            )
    return out

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