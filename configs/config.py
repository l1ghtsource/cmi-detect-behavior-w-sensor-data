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
    # 'quat6d_0', 'quat6d_1', 'quat6d_2', 'quat6d_3', 'quat6d_4', 'quat6d_5',
    'rot_w', 'rot_x', 'rot_y', 'rot_z',
    'acc_mag', 'rot_angle', 'acc_mag_jerk', 'rot_angle_vel',
    'linear_acc_x', 'linear_acc_y', 'linear_acc_z', 'linear_acc_mag', 'linear_acc_mag_jerk',
    'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
    'angular_distance',
    'linear_vel_x', 'linear_vel_y', 'linear_vel_z',
    # 'pos_x', 'pos_y', 'pos_z',
    # 'cumulative_trajectory_length', 'trajectory_curvature', 'tangential_accel',
    # 'roll', 'pitch', 'yaw',
    # 'gravity_x', 'gravity_y', 'gravity_z', 'acc_vertical', 'acc_horizontal_mag', 
    # 'XY_acc', 'XZ_acc', 'YZ_acc',
    'acc_x_lag_diff', 'acc_y_lag_diff', 'acc_z_lag_diff',
    'acc_x_lead_diff', 'acc_y_lead_diff', 'acc_z_lead_diff',
    'acc_x_cumsum', 'acc_y_cumsum', 'acc_z_cumsum',
    # 'acc_x_lag_diff3', 'acc_y_lag_diff3', 'acc_z_lag_diff3',
    # 'acc_x_lead_diff3', 'acc_y_lead_diff3', 'acc_z_lead_diff3',
    # 'acc_x_lag_diff5', 'acc_y_lag_diff5', 'acc_z_lag_diff5',
    # 'acc_x_lead_diff5', 'acc_y_lead_diff5', 'acc_z_lead_diff5',
    # 'time_from_start', 'time_to_end', 'sin_time_position',
    # 'acc_x_world', 'acc_y_world', 'acc_z_world',
    # 'acc_x_remove_g', 'acc_y_remove_g', 'acc_z_remove_g',
    # 'rel_dqw', 'rel_dqx', 'rel_dqy', 'rel_dqz', 'rel_angle',
    # 'acc_mag_shift_-2', 'acc_mag_shift_-1', 'acc_mag_shift_1', 'acc_mag_shift_2', 
    # 'acc_mag_jerk_shift_-2', 'acc_mag_jerk_shift_-1', 'acc_mag_jerk_shift_1', 'acc_mag_jerk_shift_2', 
    # 'rot_angle_shift_-2', 'rot_angle_shift_-1', 'rot_angle_shift_1', 'rot_angle_shift_2', 
    # 'rot_angle_vel_shift_-2', 'rot_angle_vel_shift_-1', 'rot_angle_vel_shift_1', 'rot_angle_vel_shift_2', 
    # 'linear_acc_mag_shift_-2', 'linear_acc_mag_shift_-1', 'linear_acc_mag_shift_1', 'linear_acc_mag_shift_2', 
    # 'linear_acc_mag_jerk_shift_-2', 'linear_acc_mag_jerk_shift_-1', 'linear_acc_mag_jerk_shift_1', 'linear_acc_mag_jerk_shift_2', 
    # 'angular_distance_shift_-2', 'angular_distance_shift_-1', 'angular_distance_shift_1', 'angular_distance_shift_2', 
    # 'angular_vel_x_shift_-2', 'angular_vel_x_shift_-1', 'angular_vel_x_shift_1', 'angular_vel_x_shift_2',
    # 'angular_vel_y_shift_-2', 'angular_vel_y_shift_-1', 'angular_vel_y_shift_1', 'angular_vel_y_shift_2',
    # 'angular_vel_z_shift_-2', 'angular_vel_z_shift_-1', 'angular_vel_z_shift_1', 'angular_vel_z_shift_2',
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
cfg.curr_folds = [0, 1, 2, 3, 4]
cfg.seed = 42
cfg.selected_model = 'hybrid' # ['timemil', 'decomposewhar', 'convtran', 'hybrid', 'cnn1d', 'timecnn', 'moderntcn', 'filternet', 'harmamba', 'medformer', 'husformer', 'multubigru', 'se_unet', 'squeezeformer', 'panns', 'wavenet', 'baseline', 'imunet', 'public', 'public2']

# --- target things ---
cfg.main_weight = 1.0
cfg.orientation_aux_weight = 0.5
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
cfg.fe_time_pos = False # info about time pos in !orig! ts (before pad&trunc)
cfg.fe_col_prod = False # acc(x/y/z) * rot(x/y/z)
cfg.use_windows = False # some rolling stats 
cfg.fe_angles = False # add xy yz zx angles
cfg.fe_euler = False # euler angles from quat
cfg.fe_freq_wavelet = False # freq and wavelet features from acc
cfg.fe_gravity = False # gravity vector [vx, vy, vz]
cfg.kaggle_fe = True # some fe before init ds (so augs works bad)
cfg.fe_relative_quat = False # add relative quat to first frame
cfg.use_quat6d = False # better rot repr for nn
cfg.use_thm_neighbour_diff = False # use thm neighbour sensor differences
cfg.use_thm_diff = False # use simple thm.diff() like grad
cfg.use_tof_stats = False # use mean, std, min, max, range for tof
cfg.use_tof_com = False # use center of mass (minimum distance point)
cfg.use_tof_grad = False # use x/y mean gradients
cfg.use_tof_neg_count = False # num of -1 in tof image
cfg.use_tof_contact_area = False # num of <N points in tof image
cfg.use_tof_neighbour_diff = False # use tof neighbour sensor differences
cfg.imu_only = False # use only imu sensor
cfg.imu_add = 0 # new features
cfg.use_dct = False # dct -> bp -> idct
cfg.use_gnn_fusion = False # gat instead of concat branch fusion in hybrid model
cfg.reverse_seq = False # seq -> seq[::-1]
cfg.branch_dropout_proba = 0 # replace tof&thm w/ zeros for single model usage (w/o switch on infer)

# if cfg.fe_mag_ang:
#     cfg.imu_add += 4
# if cfg.fe_col_diff:
#     cfg.imu_add += 3
# if cfg.lag_lead_cum:
#     cfg.imu_add += 9
# if cfg.fe_col_prod:
#     cfg.imu_add += 9
# if cfg.fe_angles:
#     cfg.imu_add += 4
# if cfg.fe_euler:
#     cfg.imu_add += 3
# if cfg.fe_freq_wavelet:
#     cfg.imu_add += 21
# if cfg.fe_gravity:
#     cfg.imu_add += 5
# if cfg.kaggle_fe:
#     cfg.imu_add += 13

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
cfg.dwhar_ver = '2'

# --- timemil ---
cfg.timemil_dim = 64
cfg.timemil_dropout = 0.35
cfg.timemil_ver = '1'
cfg.timemil_extractor = 'inception_time' # ['inception_time', 'lite', 'resnet', 'efficientnet', 'inception_resnet', 'letmecook', 'inception_time2', 'densenet']
cfg.timemil_singlebranch = True

# --- convtran ---
cfg.convtran_emb_size = 128
cfg.convtran_num_heads = 8
cfg.convtran_dim_ff = 256
cfg.convtran_dropout = 0.1
cfg.convtran_type = 'notran' # ['default', 'notran', 'multiscale', 'residual', 'inception', 'se']

# --- cnn1d ---
cfg.cnn1d_extractor = 'resnet' # ['inception_time', 'lite', 'resnet', 'efficientnet', 'densenet', 'hinception']
cfg.cnn1d_out_channels = 32
cfg.cnn1d_pooling = 'se_mean' # ['gap', 'gem', 'se_mean']
cfg.cnn1d_use_neck = True

# --- im model ---
cfg.encoder_name = 'timm/test_convnext.r160_in1k' # timm/test_vit.r160_in1k
cfg.encoder_hidden_dim = 64
cfg.im_pretrained = True

# --- train params ---
cfg.bs = 128#256
cfg.n_epochs = 100
cfg.patience = 25
cfg.lr = 1e-3
cfg.lr_muon = 1e-3
cfg.weight_decay = 3e-4
cfg.num_warmup_steps_ratio = 0.03
cfg.label_smoothing = 0.05
cfg.max_norm = 2.0
cfg.use_lookahead = False
cfg.use_sam = False
cfg.scheduler = 'cosine' # ['cosine', 'cosine_cycle', 'linear']
cfg.optim_type = 'adamw' # ['adamw', 'adam', 'adan', 'adamp', 'madgrad', 'adafisherw', 'ranger', 'muonwauxadam']
cfg.use_conf_aware_weights = False

# --- ts augs ---
cfg.max_augmentations_per_sample = 1

cfg.jitter_proba = 0.7 # its good
cfg.jitter_sensors = ['imu', 'tof', 'thm']

cfg.magnitude_warp_proba = 0.0 # useless ?
cfg.magnitude_warp_sensors = ['imu', 'thm']

cfg.time_warp_proba = 0.0 # worst
cfg.time_warp_sensors = ['imu', 'tof', 'thm']

cfg.scaling_proba = 0.0 # useless
cfg.scaling_sensors = ['imu', 'thm']

cfg.window_slice_proba = 0.0 # bad
cfg.window_slice_sensors = ['imu', 'tof', 'thm']

cfg.window_warp_proba = 0.0 # bad
cfg.window_warp_sensors = ['imu', 'thm']

cfg.permutation_proba = 0.0 # bad
cfg.permutation_sensors = ['imu', 'tof', 'thm']

cfg.rotation_proba = 0.0 # bad ??
cfg.rotation_sensors = ['imu']
cfg.rotation_max_angle = 30

cfg.moda_proba = 0.0 # bad ??
cfg.moda_sensors = ['imu']

cfg.time_mask_proba = 0.0 # useless ??
cfg.time_mask_n_features = 3
cfg.time_mask_max_width_frac = 0.2
cfg.time_mask_sensors = ['imu', 'thm', 'tof']

cfg.feature_mask_proba = 0.5 # its good
cfg.feature_mask_n_features = 1
cfg.feature_mask_sensors = ['imu', 'thm', 'tof']

# --- mixup ---
cfg.use_mixup = True
cfg.mixup_proba = 0.7 # it's reverse proba lol so real_mixup_proba = 1 - mixup_proba ^_^
cfg.mixup_alpha = 0.4
cfg.is_zebra = False
cfg.is_cutmix = False
cfg.is_wtfmix = False
cfg.is_channel_wtfmix = False

# --- ema ---
cfg.use_ema = False
cfg.ema_decay = 0.999

# --- inference params ---
cfg.weights_pathes = {
    'imu_only': {
        '/kaggle/input/cmi-rows-79-80-fulldatamodels/imu': { # full data model - 79 row
            'weight': 1,
            'model_params': {
                'num_classes': 18,
                'use_dct': False,
                'reverse_seq': False, # seq
            }
        },
        '/kaggle/input/cmi-85row-imu/kaggle/working/weights': { # full data model - 85 row (like 79, just rev seq)
            'weight': 1,
            'model_params': {
                'num_classes': 18,
                'use_dct': False,
                'reverse_seq': True, # seq[::-1]
            }
        },
    },
    'imu+tof+thm': {
        '/kaggle/input/timemil-soupchik-imu-16-06/top3_avg_models_81row_excel': { # top3 avg models - 81 row [zebra mixup]
            'weight': 1,
            'model_params': {
                'num_classes': 18,
                'use_gnn_fusion': False,
                'reverse_seq': False, # seq
            }
        },
        '/kaggle/input/cmi-another-checkpoints/top3_avg_models_76row_excel': { # top3 avg models - 76 row [normal mixup]
            'weight': 1,
            'model_params': {
                'num_classes': 18,
                'use_gnn_fusion': False,
                'reverse_seq': False, # seq
            }
        },
        '/kaggle/input/cmi-more-another-checkpoints/top3_avg_models_84row_excel': { # top3 avg models - 84 row [normal mixup]
            'weight': 1,
            'model_params': {
                'num_classes': 18,
                'use_gnn_fusion': False,
                'reverse_seq': True, # seq[::-1]
            }
        },
    }
}
cfg.ext_weights_imu = []
cfg.ext_weights_all = []
cfg.is_soft = True
cfg.use_entmax = False
cfg.entmax_alpha = 1.25
cfg.override_non_target = False
cfg.orient_postproc = False
cfg.tta_strategies = []

# --- logging ---
cfg.do_wandb_log = True
cfg.wandb_project = 'cmi-kaggle-comp-2025'
cfg.wandb_api_key = '...'

pprint.pprint(vars(cfg))