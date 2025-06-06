from types import SimpleNamespace

# --- cfg ---

cfg = SimpleNamespace(**{})

cfg.train_path = '/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv'
cfg.train_demographics_path = '/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv'
cfg.test_path = '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv'
cfg.test_demographics_path = '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv'

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

cfg.is_train = True
cfg.is_infer = False

cfg.seed = 42

cfg.morlet_sd_spread = 6
cfg.morlet_sd_factor = 2.5

cfg.target = 'gesture'
cfg.group = 'subject'
cfg.seq_len = 110
cfg.num_classes = 18
cfg.n_splits = 5
cfg.curr_fold = 0

cfg.use_demo = False
cfg.imu_only = False

cfg.model_dir = 'weights'

cfg.bs = 256 # 128
cfg.n_epochs = 50 # 100
cfg.patience = 7 # 10
cfg.lr = 1e-4 # 2e-4, 5e-5, 1e-3
cfg.weight_decay = 1e-2 # 1e-3, 1e-4
cfg.num_warmup_steps_ratio = 0.03 # 0.05, 0.02
cfg.label_smoothing = 0.05 # 0.02, 0.03

cfg.use_ema = True
cfg.ema_decay = 0.998 # 0.999

cfg.weights_pathes = '/kaggle/input/cmi-model/pytorch/default/6/weights'
cfg.is_soft = True
cfg.use_entmax = False