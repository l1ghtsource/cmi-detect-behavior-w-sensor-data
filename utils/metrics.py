from sklearn.metrics import f1_score

def just_stupid_macro_f1_haha(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')