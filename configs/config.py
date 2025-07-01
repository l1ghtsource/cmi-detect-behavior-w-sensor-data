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
cfg.demo_bin_cols = ['adult_child', 'sex', 'handedness']
cfg.demo_cont_cols = ['age', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']
cfg.imu_cols = [
    'acc_x', 'acc_y', 'acc_z',
    'rot_w', 'rot_x', 'rot_y', 'rot_z',
    'time_from_start', 'time_to_end', 'sin_time_position'
]
cfg.thm_cols = [f'thm_{i}' for i in range(1, 6)]
cfg.tof_cols = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]
cfg.num_tof_sensors = 5
cfg.tof_vector_length = 64
cfg.static_cols = [
    'sequence_id', 'sequence_type', 'gesture', 'orientation', 'subject',
    'adult_child', 'age', 'sex', 'handedness',
    'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm', 'gesture_clpsd'
]

# --- train/infer flags ---
cfg.is_train = True
cfg.is_infer = False

# --- important vars ---
cfg.main_target = 'gesture'
cfg.main_num_classes = 18
cfg.orientation_aux_target = 'orientation'
cfg.orientation_aux_num_classes = 4
cfg.seq_type_aux_target = 'sequence_type'
cfg.seq_type_aux_num_classes = 2
cfg.main_clpsd_target = 'gesture_clpsd'
cfg.main_clpsd_num_classes = 9
cfg.behavior_aux_target = 'behavior'
cfg.phase_aux_target = 'phase'
cfg.group = 'subject'
cfg.seq_len = 120
cfg.n_splits = 5
cfg.curr_fold = 0
cfg.seed = 42
cfg.selected_model = 'timemil' # ['timemil', 'decomposewhar', 'moderntcn', 'harmamba', 'medformer', 'husformer', 'multubigru', 'se_unet', 'squeezeformer', 'panns', 'baseline']

# --- target things ---
cfg.main_weight = 1.0
cfg.orientation_aux_weight = 0
cfg.seq_type_aux_weight = 0.5
cfg.main_clpsd_weight = 0
cfg.behavior_aux_weight = 0
cfg.phase_aux_weight = 0
cfg.use_main_target_weighting = False
cfg.use_orientation_aux_target_weighting = False
cfg.use_seq_type_aux_target_weighting = False

# --- ts ds cfg ---
cfg.norm_ts = False # normalize time-series (z-score)
cfg.denoise_data = 'none' # ['none', 'wavelet', 'savgol', 'butter', 'firwin']
cfg.use_demo = False # use demography data
cfg.use_stats_vectors = False # use some global seq stats
cfg.use_pad_mask = True # mask padding values
cfg.use_world_coords = False # sensor coord -> world coord + remove g
cfg.only_remove_g = False # only remove g in sensor coord (can't be used w/ use_world_coords)
cfg.use_hand_symm = False # mirror left -> right
cfg.apply_fe = True # some feature engineering
cfg.fe_mag_ang = False # magnitude and rot angle
cfg.fe_col_diff = False # x-y, x-z, y-z
cfg.lag_lead_cum = True # lag, lead, cumsum for sensor data
cfg.fe_time_pos = True # info about time pos in !orig! ts (before pad&trunc)
cfg.fe_col_prod = False # acc(x/y/z) * rot(x/y/z)
cfg.use_windows = False # some rolling stats 
cfg.fe_angles = False # add xy yz zx angles
cfg.fe_euler = False # euler angles from quat
cfg.fe_freq_wavelet = False # freq and wavelet features from acc
cfg.fe_gravity = False # gravity vector [vx, vy, vz]
cfg.imu_only = True # use only imu sensor
cfg.imu_add = 0 # new features

if cfg.fe_mag_ang:
    cfg.imu_add += 4
if cfg.fe_col_diff:
    cfg.imu_add += 3
if cfg.lag_lead_cum:
    cfg.imu_add += 9
if cfg.fe_col_prod:
    cfg.imu_add += 9
if cfg.fe_angles:
    cfg.imu_add += 4
if cfg.fe_euler:
    cfg.imu_add += 3
if cfg.fe_freq_wavelet:
    cfg.imu_add += 21
if cfg.fe_gravity:
    cfg.imu_add += 5

# --- im ds cfg ---
cfg.im_size = 160
cfg.transform_type = 'cwt'
cfg.use_grads = True
cfg.window_tof = True

# --- save dir ---
cfg.model_dir = 'weights'
cfg.oof_dir = 'oofs'

# --- decomposewhar !! ---
cfg.use_cross_sensor = False
cfg.use_megasensor = False # can't be used w/ imu_only=True
cfg.kernel_size = 3
cfg.emb_kernel_size = cfg.seq_len // cfg.kernel_size
cfg.stride = cfg.emb_kernel_size // 2
cfg.ddim = 256
cfg.reduction_ratio = 1
cfg.num_layers = 2
cfg.num_a_layers = 1
cfg.num_m_layers = 1
cfg.imu_num_sensor = 1
cfg.thm_num_sensor = 5
cfg.tof_num_sensor = 5
cfg.imu_vars = len(cfg.imu_cols) + cfg.imu_add
cfg.thm_vars = 1
cfg.tof_vars = 8 * 8
cfg.dwhar_ver = '1'

# --- timemil ---
cfg.timemil_dim = 256
cfg.timemil_dropout = 0.0
cfg.timemil_ver = '1'
cfg.timemil_extractor = 'inception_time' # ['inception_time', 'resnet', 'efficientnet', 'inception_resnet', 'letmecook', 'inception_time2', 'densenet']
cfg.timemil_singlebranch = True

# --- im model ---
cfg.encoder_name = 'timm/test_convnext.r160_in1k' # timm/test_vit.r160_in1k
cfg.encoder_hidden_dim = 64
cfg.im_pretrained = True

# --- train params ---
cfg.bs = 128
cfg.n_epochs = 50
cfg.patience = 10
cfg.lr = 1e-3
cfg.weight_decay = 3e-4
cfg.num_warmup_steps_ratio = 0.03
cfg.label_smoothing = 0.05
cfg.max_norm = 2.0
cfg.use_lookahead = True
cfg.use_sam = False
cfg.scheduler = 'cosine' # ['cosine', 'cosine_cycle', 'linear']
cfg.optim_type = 'adamw' # ['adamw', 'adan', 'adamp', 'madgrad', 'adafisherw', 'ranger', 'muon', 'muonwauxadam']

# --- ts augs ---
cfg.max_augmentations_per_sample = 4

cfg.jitter_proba = 0.5
cfg.jitter_sensors = ['imu', 'tof', 'thm']

cfg.magnitude_warp_proba = 0.3
cfg.magnitude_warp_sensors = ['imu', 'thm']

cfg.time_warp_proba = 0.5
cfg.time_warp_sensors = ['imu', 'tof', 'thm']

cfg.scaling_proba = 0.3
cfg.scaling_sensors = ['imu', 'thm']

cfg.window_slice_proba = 0
cfg.window_slice_sensors = ['imu', 'tof', 'thm']

cfg.window_warp_proba = 0
cfg.window_warp_sensors = ['imu', 'thm']

cfg.permutation_proba = 0
cfg.permutation_sensors = ['imu', 'tof', 'thm']

cfg.rotation_proba = 0.5
cfg.rotation_sensors = ['imu']
cfg.rotation_max_angle = 30

cfg.moda_proba = 0.4
cfg.moda_sensors = ['imu']

# --- mixup ---
cfg.use_mixup = True
cfg.mixup_proba = 0.7
cfg.mixup_alpha = 0.4

# --- ema ---
cfg.use_ema = False
cfg.ema_decay = 0.999

# --- inference params ---
cfg.weights_pathes = {
    'imu_only': {},
    'imu+tof+thm': {}
}
cfg.is_soft = True
cfg.use_entmax = False
cfg.entmax_alpha = 1.25
cfg.override_non_target = False
cfg.tta_strategies = {
    # 'jitter': {
    #     'sigma': 0.03,
    #     'sensors': ['imu', 'thm', 'tof']
    # },
    # 'scaling': {
    #     'sigma': 0.03,
    #     'sensors': ['imu', 'thm']
    # },
    # 'magnitude_warp': {
    #     'sigma': 0.07, 
    #     'knot': 3,
    #     'sensors': ['imu', 'thm']
    # },
    # 'time_warp': {
    #     'sigma': 0.05, 
    #     'knot': 3,
    #     'sensors': ['imu', 'tof', 'thm']
    # },
}

# --- logging ---
cfg.do_wandb_log = True
cfg.wandb_project = 'cmi-kaggle-comp-2025'
cfg.wandb_api_key = '...'

pprint.pprint(vars(cfg))