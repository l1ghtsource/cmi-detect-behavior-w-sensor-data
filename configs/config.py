from types import SimpleNamespace
import pprint

# --- cfg ---
cfg = SimpleNamespace(**{})

# --- pathes ---
cfg.train_path = '/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv'
cfg.train_demographics_path = '/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv'
cfg.test_path = '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv'
cfg.test_demographics_path = '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv'

# --- cols and data info ---
cfg.demo_cols = ['adult_child', 'age', 'sex', 'handedness', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']
cfg.imu_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
cfg.thm_cols = [f'thm_{i}' for i in range(1, 6)]
cfg.tof_cols = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]
cfg.num_tof_sensors = 5
cfg.tof_vector_length = 64
cfg.static_cols = [
    'sequence_id', 'sequence_type', 'gesture', 'orientation', 'behavior', 'subject',
    'adult_child', 'age', 'sex', 'handedness',
    'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm'
]

# --- train/infer flags ---
cfg.is_train = True
cfg.is_infer = False

# --- important vars ---
cfg.target = 'gesture'
cfg.aux_target = 'sequence_type'
cfg.group = 'subject'
cfg.seq_len = 120
cfg.num_classes = 18
cfg.n_splits = 5
cfg.curr_fold = 0
cfg.seed = 42

# --- im ds cfg ---
cfg.im_size = 160
cfg.transform_type = 'cwt'
cfg.use_grads = True
cfg.window_tof = True

# --- ts ds cfg ---
cfg.norm_ts = True
cfg.use_demo = False
cfg.imu_only = True

# --- save dir ---
cfg.model_dir = 'weights'
cfg.oof_dir = 'oofs'

# --- decomposewhar !! ---
cfg.use_dwhar = False
cfg.use_cross_sensor = False
cfg.use_megasensor = False # can't be used w/ imu_only=True
cfg.kernel_size = 3
cfg.emb_kernel_size = (cfg.seq_len // (cfg.kernel_size * 2)) * 2
cfg.stride = cfg.emb_kernel_size // 2
cfg.ddim = 64
cfg.reduction_ratio = 1
cfg.num_layers = 2
cfg.num_a_layers = 1
cfg.num_m_layers = 1
cfg.imu_num_sensor = 1
cfg.thm_num_sensor = 5
cfg.tof_num_sensor = 5
cfg.imu_vars = 7
cfg.thm_vars = 1
cfg.tof_vars = 8 * 8

# --- timemil ---
cfg.use_timemil = True
cfg.timemil_dim = 64
cfg.timemil_dropout = 0.0

# --- im model ---
cfg.encoder_name = 'timm/test_convnext.r160_in1k' # timm/test_vit.r160_in1k
cfg.encoder_hidden_dim = 64
cfg.im_pretrained = True

# --- train params ---
cfg.bs = 128
cfg.n_epochs = 50
cfg.patience = 5
cfg.lr = 1e-3
cfg.weight_decay = 3e-4
cfg.num_warmup_steps_ratio = 0.03
cfg.label_smoothing = 0.05
cfg.max_norm = 2.0
cfg.use_lookahead = True
cfg.optim_type = 'adamw' # ['adamw', 'adan', 'adamp', 'madgrad']

# --- ts augs ---
cfg.max_augmentations_per_sample = 2
cfg.jitter_proba = 0.6
cfg.jitter_sensors = ['imu', 'tof', 'thm']
cfg.magnitude_warp_proba = 0.4
cfg.magnitude_warp_sensors = ['imu', 'thm']
cfg.time_warp_proba = 0.3
cfg.time_warp_sensors = ['imu', 'tof', 'thm']
cfg.scaling_proba = 0.1
cfg.scaling_sensors = ['imu', 'thm']

# --- mixup ---
cfg.use_mixup = True
cfg.mixup_proba = 0.7
cfg.mixup_alpha = 0.4

# --- ema ---
cfg.use_ema = False
cfg.ema_decay = 0.999

# --- inference params ---
cfg.weights_path = '/kaggle/input/dwhar-models'
cfg.is_soft = True
cfg.use_entmax = False
cfg.entmax_alpha = 1.25 # 1.05 SMALL ALPHA IS A KEY FOR ENTMAX ??? who knows..
cfg.tta_strategies = {}
# cfg.tta_strategies = {
#     'jitter': {
#         'sigma': 0.01,
#         'sensors': ['imu', 'thm']
#     },
#     'scaling': {
#         'sigma': 0.03,
#         'sensors': ['imu', 'thm']
#     }
# }

# --- logging ---
cfg.do_wandb_log = True
cfg.wandb_project = 'cmi-kaggle-comp-2025'
cfg.wandb_api_key = '...'

pprint.pprint(vars(cfg))