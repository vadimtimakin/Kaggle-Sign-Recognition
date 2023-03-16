from omegaconf import OmegaConf

config = {
    'general': {
        'experiment_name': 'model_upd',
        'seed': 0xFACED,
        'num_classes': 250, 
    },
    'paths': {
        'path_to_csv': 'home/toefl/K/asl-signs/train.csv',
        'path_to_data': './feature_data/feature_data.npy',
        'path_to_labels': './feature_data/feature_labels.npy',
        'path_to_json': 'home/toefl/K/asl-signs/sign_to_prediction_index_map.json',
        'path_to_folder': 'home/toefl/K/asl-signs/',
        'pq_path': 'home/toefl/K/asl-signs/train_landmark_files/53618/1001379621.parquet',
        'path_to_checkpoints': './checkpoints/${general.experiment_name}',
    },
    'training': {
        'num_epochs': 100,
        'early_stopping_epochs': 100,
        'lr': 0.003,

        'mixed_precision': True,
        'gradient_accumulation': False,
        'gradient_clipping': False,
        'gradient_accumulation_steps': 2,
        'clip_value': 1,
        
        'warmup_scheduler': False,
        'warmup_epochs': 3,
        'warmup_multiplier': 1,

        'debug': False,
        'number_of_train_debug_samples': 5000,
        'number_of_val_debug_samples': 1000,
        
        'device': 'cuda',
        'save_best': True,
        'save_last': False,
        'verbose_plots': False,
        'resume_from_latest_checkpoint': False,
    },
    'dataloader_params': {
        'batch_size': 4096,
        'num_workers': 8,
        'pin_memory': False,
        'persistent_workers': True,
        'shuffle': True,
    },
    'split': {
        'n_splits': 5,
        'folds_to_train': [0, 1, 2, 3, 4],
        'already_split': False,
    },
    'model': {           
        'freeze_batchnorms': False,
        'converter_sample_input_shape': [50, 543, 2],
        'model_sample_input_shape': [1, 472],
        'params': {
            "in_features": 472,
            "first_out_features": 1024,
            "num_classes": '${general.num_classes}',
            "num_blocks": 3,
            "drop_rate": 0.4,
        },
    },
    'optimizer': {
        'name': 'Adam',
        'params': {
            'lr': '${training.lr}',
            'weight_decay': 1e-4,
        },
    },
    'scheduler': {
        'name': 'CosineAnnealingLR', 
        'interval': 'epoch',
        'params': {
            'T_max': '${training.num_epochs}',
            'eta_min': 1e-7,
        },
    },
    'loss': {
        'name': 'CrossEntropyLoss',
        'params': {
            'reduction': 'mean'
        },
    },
    'metric': {
        'name': 'accuracy_score',
        'params': {
        },
    },
    'logging': {
        'prints': False,
        'txt_file': True,
        'wandb': False,
        'wandb_username': 'toefl',
        'wandb_project_name': 'GISLR',
    }, 
}

config = OmegaConf.create(config)