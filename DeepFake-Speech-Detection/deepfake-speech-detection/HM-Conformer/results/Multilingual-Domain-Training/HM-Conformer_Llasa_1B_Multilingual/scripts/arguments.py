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
        'project'       : 'Multilingual-Domain-Training',
        'name'          : 'HM-Conformer_OuteTTS',
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
        'train_split'   : 0.8,
        'val_split'     : 0.1,
        'test_split'    : 0.1,
        
        # Language filtering: Set to a language code (e.g., 'en', 'it', 'es') to filter dataset
        # Set to None to use all languages
        'selected_language': None,  # Filter for English only. Change to other language codes (e.g., 'it', 'es') or None for all languages

        # Fake-model filtering (optional):
        # If set, keep ALL real samples, and keep ONLY fake samples where model_or_speaker == selected_fake_model.
        # Can be combined with selected_language.
        'selected_fake_model': 'OuteTTS',
        
        # Common augmentation paths
        'path_musan'    : '/content/deepfake-speech-detection/HM-Conformer/data/musan',
        'path_rir'      : '/content/deepfake-speech-detection/HM-Conformer/data/RIRS_NOISES/simulated_rirs',

        # others
        'num_workers': 4,
        'usable_gpu': '0', # ex) '0,1'
    }

    experiment_args = {
        'TEST'              : False,  # Set to True for testing/inference only
        # Which checkpoint epoch to load when TEST=True.
        # - Set to an int (e.g., 60) to force that epoch.
        # - Set to None to auto-pick the latest available epoch in path_params.
        'load_epoch'        : 10,
        # experiment
        #'epoch'             : 200,
        'epoch'             : 10,
        #'batch_size'        : 240,  # Small batch size for Colab GPU memory
        'batch_size'        : 120,  # Small batch size for Colab GPU memory
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
        'DA_codec_speed'    : True,         # codec: 'aac', 'flac', 'm4a', 'mp3', 'ogg', 'wav', 'wav', 'wma', speed: 'slow', 'fast'
        # 2. when __getitem__
        'DA_p'              : 0.5,
        'DA_list'           : [], # 'ACN': add_coloured_noise, 'FQM': frq_masking, 'MUS': MUSAN, 'RIR': RIR
        'DA_params'         : {
            'MUS': {'path': system_args['path_musan']},
            'RIR': {'path': system_args['path_rir']}  
        },
        # 3. when processing WaveformAugmentation which is in Framework
        #'DA_wav_aug_list'   : ['ACN'], 
        'DA_wav_aug_list'   : ['ACN'],
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
        'DA_frq_mask'       : True,
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
    args['path_params'] = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'results', 'Multilingual-Train', 'HM-Conformer EN', 'models'))

    # ------------------------------------------------------------
    # Environment overrides (for automation / sweeps)
    # ------------------------------------------------------------
    # Supported env vars:
    # - HM_SELECTED_LANGUAGE: language code (e.g. "en") or "all"/"none" for no filter
    # - HM_SELECTED_FAKE_MODEL: model_or_speaker name for fake samples (e.g. "Chatterbox Multilingual") or "all"/"none" for no filter
    # - HM_LOAD_EPOCH: int epoch to load (e.g. "60") or "none" to auto-pick latest
    # - HM_PATH_PARAMS: override checkpoint directory (models folder)
    # - HM_LABELS_PATH: override labels.json path
    # - HM_DATASET_ROOT: override dataset root folder
    # - HM_PATH_LOG: override log root folder
    # - HM_PROJECT / HM_NAME: override experiment naming
    # - HM_USABLE_GPU: override usable_gpu (e.g. "0" or "0,1")
    # - HM_TEST: "1"/"true" to force test-only; "0"/"false" to force training mode
    def _env_str(key: str):
        v = os.environ.get(key, None)
        if v is None:
            return None
        v = str(v).strip()
        return v if v != "" else None

    def _env_bool(key: str):
        v = _env_str(key)
        if v is None:
            return None
        if v.lower() in ("1", "true", "yes", "y", "t", "on"):
            return True
        if v.lower() in ("0", "false", "no", "n", "f", "off"):
            return False
        return None

    def _env_int_or_none(key: str):
        v = _env_str(key)
        if v is None:
            return None
        if v.lower() in ("none", "null", "auto", "latest"):
            return None
        try:
            return int(v)
        except ValueError:
            return None

    # Dataset / paths
    v = _env_str("HM_PATH_PARAMS")
    if v is not None:
        args["path_params"] = v
    v = _env_str("HM_LABELS_PATH")
    if v is not None:
        args["labels_path"] = v
    v = _env_str("HM_DATASET_ROOT")
    if v is not None:
        args["dataset_root"] = v
    v = _env_str("HM_PATH_LOG")
    if v is not None:
        args["path_log"] = v

    # Experiment naming
    v = _env_str("HM_PROJECT")
    if v is not None:
        args["project"] = v
    v = _env_str("HM_NAME")
    if v is not None:
        args["name"] = v

    # GPU selection
    v = _env_str("HM_USABLE_GPU")
    if v is not None:
        args["usable_gpu"] = v

    # Language selection
    v = _env_str("HM_SELECTED_LANGUAGE")
    if v is not None:
        if v.lower() in ("all", "none", "null"):
            args["selected_language"] = None
        else:
            args["selected_language"] = v

    # Fake-model selection (applies ONLY to fake samples; real samples are always kept)
    v = _env_str("HM_SELECTED_FAKE_MODEL")
    if v is not None:
        if v.lower() in ("all", "none", "null"):
            args["selected_fake_model"] = None
        else:
            args["selected_fake_model"] = v

    # TEST mode and checkpoint epoch
    vb = _env_bool("HM_TEST")
    if vb is not None:
        args["TEST"] = vb
    le = _env_int_or_none("HM_LOAD_EPOCH")
    if "load_epoch" in experiment_args:
        # Keep experiment_args in sync for logger/log_arguments
        experiment_args["load_epoch"] = le if _env_str("HM_LOAD_EPOCH") is not None else experiment_args["load_epoch"]
    if _env_str("HM_LOAD_EPOCH") is not None:
        args["load_epoch"] = le

    # If testing/inference only, evaluate on the full (optionally language-filtered) dataset.
    # This prevents empty train/val splits and matches the common "test-only" expectation.
    if args.get('TEST', False):
        args['train_split'] = 0.0
        args['val_split'] = 0.0
        args['test_split'] = 1.0

    # Basic config sanity check: train/val/test splits must sum to 1.0
    split_sum = float(args.get('train_split', 0.0)) + float(args.get('val_split', 0.0)) + float(args.get('test_split', 0.0))
    if abs(split_sum - 1.0) > 1e-6:
        raise ValueError(
            "train/val/test splits must sum to 1.0, got "
            f"train_split={args.get('train_split')}, "
            f"val_split={args.get('val_split')}, "
            f"test_split={args.get('test_split')} (sum={split_sum})"
        )

    return args, system_args, experiment_args
