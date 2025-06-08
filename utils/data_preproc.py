import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from configs.config import cfg

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

    return pd.DataFrame(res)

def le(train_seq):
    label_encoder = LabelEncoder()
    label_encoder_aux = LabelEncoder()

    train_seq[cfg.target] = label_encoder.fit_transform(train_seq[cfg.target])
    train_seq[cfg.aux_target] = label_encoder_aux.fit_transform(train_seq[cfg.aux_target])

    return train_seq, label_encoder, label_encoder_aux

def get_means(train_seq):
    feature_means = {}

    for col in cfg.imu_cols + cfg.thm_cols + cfg.tof_cols:
        all_values = np.concatenate(train_seq[col].values)
        feature_means[col] = np.nanmean(all_values)

    return feature_means