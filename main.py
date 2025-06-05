from configs.config import cfg
import warnings
warnings.filterwarnings('ignore')

if cfg.is_train:
    from scripts.train import run_training_with_stratified_group_kfold
    run_training_with_stratified_group_kfold()
elif cfg.is_infer:
    from scripts.inference import predict
    pass