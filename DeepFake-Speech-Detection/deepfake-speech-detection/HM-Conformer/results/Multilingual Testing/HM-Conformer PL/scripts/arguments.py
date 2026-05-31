import os
import itertools

def get_args():
    """
    Returns
        system_args (dict): path, log setting
        experiment_args (dict): hyper-parameters
        args (dict): system_args + experiment_args
    """
    system_args = {
        # expeirment info
        'project'       : 'Multilingual Testing',
        'name'          : 'HM-Conformer PL',
        'tags'          : [],
        'description'   : '',

        # log
        'path_log'      : '/content/deepfake-speech-detection/HM-Conformer/results',
        'neptune_user'  : '',
        'neptune_token' : '',
        'wandb_group'   : '',
        'wandb_entity'  : '',
        'wandb_api_key' : '',
        
        # datasets
        # OLD: ASVspoof dataset paths (commented out)
        # 'path_train'    : '/data/ASVspoof2019',
        # 'path_test'     : '/data/ASVspoof2021_DF',
        # 'path_test_LA'  : None,
        
        # NEW: MultilingualDataset paths
        # For local (non-Docker) usage:
        # 'path_train'    : '/Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios',
        # 'labels_path'   : '/Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios/labels.json',
        # 'dataset_root'  : '/Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios',
        # For Docker usage, uncomment and use these paths instead:
        'path_train'    : '/content/deepfake-speech-detection/HM-Conformer/dataset',
        'labels_path'   : '/content/deepfake-speech-detection/HM-Conformer/dataset/labels.json',
        'dataset_root'  : '/content/deepfake-speech-detection/HM-Conformer/dataset',
        #'train_split'   : 0.8,
        #'val_split'     : 0.1,
        #'test_split'    : 0.1,
        'train_split'   : 0,
        'val_split'     : 0,
        'test_split'    : 1,
        
        # Language filtering: Set to a language code (e.g., 'en', 'it', 'es') to filter dataset
        # Set to None to use all languages
        'selected_language': 'pl',  # Filter for English only. Change to other language codes (e.g., 'it', 'es') or None for all languages
        
        # Common augmentation paths
        'path_musan'    : '/content/deepfake-speech-detection/HM-Conformer/data/musan',
        'path_rir'      : '/content/deepfake-speech-detection/HM-Conformer/data/RIRS_NOISES/simulated_rirs',

        # others
        'num_workers': 2,  # Reduced for Colab (0-2 recommended)
        'usable_gpu': '0', # ex) '0,1'
    }

    experiment_args = {
        'TEST'              : True,  # Set to True for testing/inference only
        # experiment
        'epoch'             : 100,
        # 'batch_size'        : 240,
        'batch_size'        : 32,  # Small batch size for Colab GPU memory
        'rand_seed'         : 1,
        
        # frontend model
        'bin_size'          : 120,
        'output_size'       : 128,
        'input_layer'       : "conv2d2", 
        'pos_enc_layer_type': "rel_pos",  
        'linear_units'      : 256,
        'cnn_module_kernel' : 15,
        'dropout'           : 0.75,
        'emb_dropout'       : 0.3,
        
        # backend model
        'use_pooling'       : False,
        'input_mean_std'    : False,
        'embedding_size'    : 64,
        
        # OCSoftmax loss
        'num_class'         : 1,
        'feat_dim'          : 2,
        'r_real'            : 0.9,
        'r_fake'            : 0.2,
        'alpha'             : 20.0,
        'loss_weight'       : [0.4, 0.3, 0.2, 0.1, 0.1],
        
        
        # data processing
        'sample_rate'       : 16000, 
        'n_lfcc'            : 40, 
        'coef'              : 0.97, 
        'n_fft'             : 512, 
        'win_length'        : 320, 
        'hop'               : 160, 
        'with_delta'        : True, 
        'with_emphasis'     : True, 
        'with_energy'       : True,
        'train_crop_size'   : 16000 * 4,
        'test_crop_size'    : 16000 * 4,
        
        # data augmentation
        # 1. when Reading file
        'DA_codec_speed'    : False,         # codec: 'aac', 'flac', 'm4a', 'mp3', 'ogg', 'wav', 'wav', 'wma', speed: 'slow', 'fast'
        # 2. when __getitem__
        'DA_p'              : 0.5,
        'DA_list'           : [], # 'ACN': add_coloured_noise, 'FQM': frq_masking, 'MUS': MUSAN, 'RIR': RIR
        'DA_params'         : {
            'MUS': {'path': system_args['path_musan']},
            'RIR': {'path': system_args['path_rir']}  
        },
        # 3. when processing WaveformAugmentation which is in Framework
        #'DA_wav_aug_list'   : ['ACN'], 
        'DA_wav_aug_list'   : [],
            # 'ACN': add_colored_noise, 'GAN': gain, 'HPF': high pass filter, 'LPF': low pass filter
            # if use 'HPF' or 'LPF' training speed will be slow
        'DA_wav_aug_params' :  {
            'sr': 16000,
            'ACN': {'min_snr_in_db': 10, 'max_snr_in_db': 40, 'min_f_decay': -2.0, 'max_f_decay': 2.0, 'p': 1},
            'HPF': {'min_cutoff_freq': 20.0, 'max_cutoff_freq': 2400.0, 'p': 0.5},
            'LPF': {'min_cutoff_freq': 150.0, 'max_cutoff_freq': 7500.0, 'p': 0.5},
            'GAN': {'min_gain_in_db': -15.0, 'max_gain_in_db': 5.0, 'p': 0.5}
        },
        # 4. when extracting acoustic_feature
        'DA_frq_p'          : 1,
        #'DA_frq_mask'       : True,
        'DA_frq_mask'       : False,
        'DA_frq_mask_max'   : 20,
        
        # learning rate
        'lr'                : 1e-6,
        'lr_min'            : 1e-6,
		'weight_decay'      : 1e-4,
        'T_mult'            : 1,
        
    }

    args = {}
    for k, v in itertools.chain(system_args.items(), experiment_args.items()):
        args[k] = v
    args['path_scripts'] = os.path.dirname(os.path.realpath(__file__))
    args['path_params'] = args['path_scripts'] + '/params'

    return args, system_args, experiment_args
