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

def apply_symmetry(data):
    transformed = data.copy()
    transformed['acc_z'] = -transformed['acc_z']
    transformed['acc_y'] = -transformed['acc_y']
    transformed['rot_y'] = -transformed['rot_y']
    transformed['rot_z'] = -transformed['rot_z']
    return transformed

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