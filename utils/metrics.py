import numpy as np
from sklearn.metrics import f1_score

def just_stupid_macro_f1_haha(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def comp_metric(y_true, y_pred):
    bscore = f1_score(
        np.where(y_true <= 7, 1, 0),
        np.where(y_pred <= 7, 1, 0),
        zero_division=0.0,
    )

    mscore = f1_score(
        np.where(y_true <= 7, y_true, 99),
        np.where(y_pred <= 7, y_pred, 99),
        average="macro", 
        zero_division=0.0,
    )

    return (bscore + mscore) / 2, bscore, mscore