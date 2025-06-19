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
cfg.imu_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
cfg.thm_cols = [f'thm_{i}' for i in range(1, 6)]
cfg.tof_cols = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]
cfg.num_tof_sensors = 5
cfg.tof_vector_length = 64
cfg.static_cols = [
    'sequence_id', 'sequence_type', 'gesture', 'orientation', 'subject',
    'adult_child', 'age', 'sex', 'handedness',
    'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm'
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
cfg.selected_model = 'timemil' # ['timemil', 'decomposewhar', 'moderntcn', 'harmamba', 'medformer', 'husformer', 'baseline']

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
cfg.norm_ts = True
cfg.denoise_data = 'none' # ['none', 'wavelet', 'savgol', 'butter']
cfg.use_demo = False
cfg.use_stats_vectors = False
cfg.use_diff = False
cfg.use_time_pos = False
cfg.use_pad_mask = True
cfg.use_world_coords = False
cfg.use_hand_symm = False
cfg.imu_only = True

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
cfg.imu_vars = 7
cfg.thm_vars = 1
cfg.tof_vars = 8 * 8
cfg.dwhar_ver = '1'

# --- timemil ---
cfg.timemil_dim = 256
cfg.timemil_dropout = 0.0
cfg.timemil_ver = '1'
cfg.timemil_extractor = 'inception_time' # ['inception_time', 'resnet', 'efficientnet', 'inception_resnet']
cfg.timemil_singlebranch = True

# --- im model ---
cfg.encoder_name = 'timm/test_convnext.r160_in1k' # timm/test_vit.r160_in1k
cfg.encoder_hidden_dim = 64
cfg.im_pretrained = True

# --- train params ---
cfg.bs = 128
cfg.n_epochs = 50
cfg.patience = 7
cfg.lr = 1e-3
cfg.weight_decay = 3e-4
cfg.num_warmup_steps_ratio = 0.03
cfg.label_smoothing = 0.05
cfg.max_norm = 2.0
cfg.use_lookahead = True
cfg.use_sam = False
cfg.optim_type = 'adamw' # ['adamw', 'adan', 'adamp', 'madgrad', 'adafisherw', 'ranger']

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
cfg.weights_pathes = {
    'imu_only': {
        '/kaggle/input/timemil-soupchik-imu-16-06/timemil_resnet_7branches': {
            'weight': 895.7572691,
            'prefix': 'timemil_ver1_resnet_imu_only_seq_len120_use_pad_mask_lookahead_adamw_ls0.05_',
            'timemil_singlebranch': False,
            'timemil_ver': '1',
            'model_params': {
                'n_classes': 18,
                'mDim': 256, 
                'max_seq_len': 120,
                'dropout': 0.0,
                'timemil_extractor': 'resnet',
            }
        },
        '/kaggle/input/timemil-soupchik-imu-16-06/timemil_resnet_1branch': {
            'weight': 91.88802719,
            'prefix': 'timemil_ver1_resnet_imu_only_seq_len120_use_pad_mask_lookahead_adamw_ls0.05_',
            'timemil_singlebranch': True,
            'timemil_ver': '1',
            'model_params': {
                'n_classes': 18,
                'mDim': 256, 
                'max_seq_len': 120,
                'dropout': 0.0,
                'timemil_extractor': 'resnet',
            }
        },
        '/kaggle/input/timemil-soupchik-imu-16-06/timemil_effnet_7branches': {
            'weight': 406.48007812,
            'prefix': 'timemil_ver1_efficientnet_imu_only_seq_len120_use_pad_mask_lookahead_adamw_ls0.05_',
            'timemil_singlebranch': False,
            'timemil_ver': '1',
            'model_params': {
                'n_classes': 18,
                'mDim': 256, 
                'max_seq_len': 120,
                'dropout': 0.0,
                'timemil_extractor': 'efficientnet',
            }
        },
        '/kaggle/input/timemil-soupchik-imu-16-06/timemil_effnet_1branch': {
            'weight': 245.86609582,
            'prefix': 'timemil_ver1_efficientnet_imu_only_seq_len120_use_pad_mask_lookahead_adamw_ls0.05_',
            'timemil_singlebranch': True,
            'timemil_ver': '1',
            'model_params': {
                'n_classes': 18,
                'mDim': 256, 
                'max_seq_len': 120,
                'dropout': 0.0,
                'timemil_extractor': 'efficientnet',
            }
        },
        '/kaggle/input/timemil-soupchik-imu-16-06/timemil_inception_7branches': {
            'weight': 94.62517775,
            'prefix': 'timemil_ver1_inception_time_imu_only_seq_len120_use_pad_mask_lookahead_adamw_ls0.05_',
            'timemil_singlebranch': False,
            'timemil_ver': '1',
            'model_params': {
                'n_classes': 18,
                'mDim': 256, 
                'max_seq_len': 120,
                'dropout': 0.0,
                'timemil_extractor': 'inception_time',
            }
        },
        '/kaggle/input/timemil-soupchik-imu-16-06/timemil_inception_1branch': {
            'weight': 745.34457709,
            'prefix': 'timemil_ver1_inception_time_imu_only_seq_len120_use_pad_mask_lookahead_adamw_ls0.05_',
            'timemil_singlebranch': True,
            'timemil_ver': '1',
            'model_params': {
                'n_classes': 18,
                'mDim': 256, 
                'max_seq_len': 120,
                'dropout': 0.0,
                'timemil_extractor': 'inception_time',
            }
        },
    },
    'imu+tof+thm': {
        '/kaggle/input/timemil-soupchik-imu-16-06/timemil_resnet_multi_3branches': {
            'weight': 427.88332544,
            'prefix': 'timemil_ver1_resnet_seq_len120_use_pad_mask_lookahead_adamw_ls0.05_',
            'timemil_singlebranch': False,
            'timemil_ver': '1',
            'model_params': {
                'n_classes': 18,
                'mDim': 256, 
                'max_seq_len': 120,
                'dropout': 0.0,
                'timemil_extractor': 'resnet',
            }
        },
        '/kaggle/input/timemil-soupchik-imu-16-06/timemil_effnet_multi_3branches': {
            'weight': 527.91607931,
            'prefix': 'timemil_ver1_effnet_seq_len120_use_pad_mask_lookahead_adamw_ls0.05_',
            'timemil_singlebranch': False,
            'timemil_ver': '1',
            'model_params': {
                'n_classes': 18,
                'mDim': 256, 
                'max_seq_len': 120,
                'dropout': 0.0,
                'timemil_extractor': 'efficientnet',
            }
        },
        '/kaggle/input/timemil-soupchik-imu-16-06/timemil_inception_multi_3branches': {
            'weight': 693.26744422,
            'prefix': 'timemil_ver1_inception_time_seq_len120_use_pad_mask_lookahead_adamw_ls0.05_',
            'timemil_singlebranch': False,
            'timemil_ver': '1',
            'model_params': {
                'n_classes': 18,
                'mDim': 256, 
                'max_seq_len': 120,
                'dropout': 0.0,
                'timemil_extractor': 'inception_time',
            }
        },
    }
}
cfg.is_soft = True
cfg.use_entmax = False
cfg.entmax_alpha = 1.25 # 1.05 SMALL ALPHA IS A KEY FOR ENTMAX ??? who knows..
cfg.override_non_target = False
# cfg.tta_strategies = {}
cfg.tta_strategies = {
    'jitter': {
        'sigma': 0.03,
        'sensors': ['imu', 'thm', 'tof']
    },
    'scaling': {
        'sigma': 0.03,
        'sensors': ['imu', 'thm']
    },
    'magnitude_warp': {
        'sigma': 0.07, 
        'knot': 3,
        'sensors': ['imu', 'thm']
    },
    'time_warp': {
        'sigma': 0.05, 
        'knot': 3,
        'sensors': ['imu', 'tof', 'thm']
    },
}

# --- logging ---
cfg.do_wandb_log = True
cfg.wandb_project = 'cmi-kaggle-comp-2025'
cfg.wandb_api_key = '...'

pprint.pprint(vars(cfg))