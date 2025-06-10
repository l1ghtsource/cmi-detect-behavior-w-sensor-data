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

def le(df):
    mapper_target = {
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

    mapper_aux = {
        'Non-Target': 0,
        'Target': 1
    }

    df[cfg.target] = df[cfg.target].map(mapper_target)
    df[cfg.aux_target] = df[cfg.aux_target].map(mapper_aux)

    return df

def get_rev_mapping():
    ae_zhok = {
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

    return {y: x for x, y in ae_zhok.items()}