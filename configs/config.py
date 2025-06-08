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
cfg.seq_len = 110
cfg.num_classes = 18
cfg.n_splits = 5
cfg.curr_fold = 0
cfg.seed = 42

# --- ts ds cfg ---
cfg.norm_ts = True
cfg.use_demo = False
cfg.imu_only = True

# --- im ds cfg ---
cfg.morlet_sd_spread = 6
cfg.morlet_sd_factor = 2.5

# --- save dir ---
cfg.model_dir = 'weights'

# --- decomposewhar !! ---
cfg.use_dwhar = True
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

# --- train params ---
cfg.bs = 256
cfg.n_epochs = 50
cfg.patience = 5
cfg.lr = 1e-3
cfg.weight_decay = 1e-2
cfg.num_warmup_steps_ratio = 0.03
cfg.label_smoothing = 0.05

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
cfg.ema_decay = 0.998

# --- inference params ---
cfg.weights_path = None
cfg.is_soft = True
cfg.use_entmax = False

pprint.pprint(vars(cfg))