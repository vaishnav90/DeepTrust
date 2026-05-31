import os
import sys
import random
import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader, DistributedSampler

if os.path.exists('/exp_lib'):
    sys.path.append('/exp_lib')
import egg_exp
import arguments
import data_processing
import train

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def run(process_id, args, experiment_args):
    #===================================================
    #                    Setting      
    #===================================================
    torch.cuda.empty_cache()
    
    # set reproducible
    set_seed(args['rand_seed'])
    
    # DDP 
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args['port']
    args['rank'] = process_id
    args['device'] = f'cuda:{process_id}'
    torch.distributed.init_process_group(
            backend='nccl', world_size=args['world_size'], rank=args['rank'])
    flag_parent = process_id == 0

    try:
        # Helpful runtime info (rank0 only)
        if flag_parent:
            print("\n" + "=" * 60)
            print("HM-Conformer Run Info")
            print("=" * 60)
            print(f"TEST mode:           {args.get('TEST')}")
            print(f"Selected language:   {args.get('selected_language')}")
            print(f"Selected fake model: {args.get('selected_fake_model')}")
            print(f"labels_path:         {args.get('labels_path', args.get('path_train', '') + '/labels.json')}")
            print(f"dataset_root:        {args.get('dataset_root', args.get('path_train'))}")
            print(f"path_params:         {args.get('path_params')}")
            print(f"load_epoch:          {args.get('load_epoch')}")
            print(f"usable_gpu:          {args.get('usable_gpu')}")
            print(f"world_size:          {args.get('world_size')}")
            print(f"batch_size (per GPU):{args.get('batch_size')}")
            print("=" * 60 + "\n")

        # logger
        if flag_parent:
            builder = egg_exp.log.LoggerList.Builder(args['name'], args['project'], args['tags'], args['description'], args['path_scripts'], args)
            builder.use_local_logger(args['path_log'])
            # builder.use_neptune_logger(args['neptune_user'], args['neptune_token'])
            # builder.use_wandb_logger(args['wandb_entity'], args['wandb_api_key'], args['wandb_group'])
            logger = builder.build()
            logger.log_arguments(experiment_args)
        else:
            logger = None
        
        # data loader
        # ============================================================================
        # NEW CODE: MultilingualDataset
        # ============================================================================
        multilingual_dataset = egg_exp.data.dataset.MultilingualDataset(
            labels_path=args.get('labels_path', args['path_train'] + '/labels.json'),
            dataset_root=args.get('dataset_root', args['path_train']),
            train_split=args.get('train_split', 0.8),
            val_split=args.get('val_split', 0.1),
            test_split=args.get('test_split', 0.1),
            random_seed=args['rand_seed'],
            selected_language=args.get('selected_language', None),
            selected_fake_model=args.get('selected_fake_model', None),
            print_info=flag_parent
        )

        # Always create test loader
        test_set_DF = data_processing.TestSet(
            multilingual_dataset.test_set,
            args['test_crop_size']
        )
        test_sampler = DistributedSampler(test_set_DF, shuffle=False)
        test_loader_DF = DataLoader(
            test_set_DF,
            num_workers=args['num_workers'],
            batch_size=args['batch_size'],
            pin_memory=True,
            sampler=test_sampler,
            drop_last=False
        )

        # Only create train/val loaders when actually training
        train_sampler = None
        train_loader = None
        val_loader = None
        if not args['TEST']:
            train_set = data_processing.TrainSet(
                multilingual_dataset.train_set,
                args['train_crop_size'],
                args['DA_p'],
                args['DA_list'],
                args['DA_params']
            )
            train_sampler = DistributedSampler(train_set, shuffle=True)
            train_loader = DataLoader(
                train_set,
                num_workers=args['num_workers'],
                batch_size=args['batch_size'],
                pin_memory=True,
                sampler=train_sampler,
                drop_last=True
            )

            val_set = data_processing.TestSet(
                multilingual_dataset.val_set,
                args['test_crop_size']
            )
            val_sampler = DistributedSampler(val_set, shuffle=False)
            val_loader = DataLoader(
                val_set,
                num_workers=args['num_workers'],
                batch_size=args['batch_size'],
                pin_memory=True,
                sampler=val_sampler,
                drop_last=False
            )
        # ============================================================================
        
        # Waveform augmentation
        augmentation = None
        if len(args['DA_wav_aug_list']) != 0:
            augmentation = egg_exp.data.augmentation.WaveformAugmetation(args['DA_wav_aug_list'], args['DA_wav_aug_params'])
        
        # data preprocessing
        preprocessing = egg_exp.framework.model.LFCC(args['sample_rate'], args['n_lfcc'], 
                args['coef'], args['n_fft'], args['win_length'], args['hop'], args['with_delta'], args['with_emphasis'], args['with_energy'],
                args['DA_frq_mask'], args['DA_frq_p'], args['DA_frq_mask_max'])
     
        # frontend
        frontend = egg_exp.framework.model.HM_Conformer(bin_size=args['bin_size'], output_size=args['output_size'], input_layer=args['input_layer'],
                pos_enc_layer_type=args['pos_enc_layer_type'], linear_units=args['linear_units'], cnn_module_kernel=args['cnn_module_kernel'],
                dropout=args['dropout'], emb_dropout=args['emb_dropout'], multiloss=True)

        # backend
        backends = []
        criterions = []
        for i in range(5):
            backend = egg_exp.framework.model.CLSBackend(in_dim=args['output_size'], hidden_dim=args['embedding_size'], use_pooling=args['use_pooling'], input_mean_std=args['input_mean_std'])
            backends.append(backend)
            
            # criterion
            criterion = egg_exp.framework.loss.OCSoftmax(embedding_size=args['embedding_size'], 
                num_class=args['num_class'], feat_dim=args['feat_dim'], r_real=args['r_real'], 
                r_fake=args['r_fake'], alpha=args['alpha'])
            criterions.append(criterion)
        
        # set framework
        framework = egg_exp.framework.DeepfakeDetectionFramework_DA_multiloss(
            augmentation=augmentation,
            preprocessing=preprocessing,
            frontend=frontend,
            backend=backends,
            loss=criterions,
            loss_weight=args['loss_weight'],
        )
        framework.use_distributed_data_parallel(f'cuda:{process_id}', True)

        # ===================================================
        #                    Test (test-only)
        # ===================================================
        if args['TEST']:
            if flag_parent:
                print("Loading model checkpoint...")
            framework.load_model(args)
            if flag_parent:
                print("Model loaded. Running test...")
            
            metrics = train.test(framework, test_loader_DF, get_full_metrics=True)
            
            # Print results (rank0 only)
            if flag_parent:
                print('\n' + '='*60)
                print('Test Results')
                print('='*60)
                print(f'EER:           {metrics["eer"]:.4f}%')
                print(f'Accuracy:      {metrics["accuracy"]:.4f}')
                print(f'F1 Score:      {metrics["f1_score"]:.4f}')
                print(f'Precision:     {metrics["precision"]:.4f}')
                print(f'Recall:        {metrics["recall"]:.4f}')
                if metrics.get('roc_auc') is not None:
                    print(f'ROC AUC:       {metrics["roc_auc"]:.4f}')
                print(f'Threshold:     {metrics["threshold"]:.4f}')
                print('\nConfusion Matrix:')
                print(metrics['confusion_matrix'])
                print('\nClassification Report:')
                print(metrics['classification_report'])
                print('='*60)
                # Backward-compatible print
                print('\nDF: ', metrics['eer'])
            
            # Log metrics (rank0 only)
            if logger is not None:
                logger.log_metric('DF_EER', metrics['eer'], 0)
                logger.log_metric('DF_Accuracy', metrics['accuracy'], 0)
                logger.log_metric('DF_F1', metrics['f1_score'], 0)
                logger.log_metric('DF_Precision', metrics['precision'], 0)
                logger.log_metric('DF_Recall', metrics['recall'], 0)
                if metrics.get('roc_auc') is not None:
                    logger.log_metric('DF_ROC_AUC', metrics['roc_auc'], 0)

        # ===================================================
        #                    Train
        # ===================================================
        else:
            # optimizer
            optimizer = torch.optim.Adam(framework.get_parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
            
            # lr scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=args['epoch'],
                T_mult=args['T_mult'],
                eta_min=args['lr_min']
            )

            best_eer_DF = 100
            cnt_early_stop = 0

            # load model
            pre_trained_model = os.path.join(args['path_scripts'], 'model')
            if os.path.exists(pre_trained_model):
                state_dict = {}
                for pt in os.listdir((pre_trained_model)):
                    state_dict[pt.replace('.pt', '')] = torch.load(pt)
                framework.load_state_dict(state_dict)

            for epoch in range(1, args['epoch'] + 1):
                scheduler.step(epoch)

                # train
                train_sampler.set_epoch(epoch)
                train.train(epoch, framework, optimizer, train_loader, logger)

                # validate (compute validation loss)
                train.validate(epoch, framework, val_loader, logger)

                # test_DF
                if epoch % 5 == 0:
                    cnt_early_stop += 1
                    eer = train.test(framework, test_loader_DF)

                    # logging
                    if eer < best_eer_DF:
                        cnt_early_stop = 0
                        best_eer_DF = eer
                        best_state_ft = framework.copy_state_dict()
                        if logger is not None:
                            logger.log_metric('BestEER', eer, epoch)
                            for key, v in best_state_ft.items():
                                logger.save_model(
                                    f'check_point_DF_{key}_{epoch}', v)
                    if logger is not None:
                        logger.log_metric('EER', eer, epoch)
                    if cnt_early_stop >= 6:
                        break
    finally:
        # Prevent NCCL resource leak warnings
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
                

if __name__ == '__main__':
    # get arguments
    args, system_args, experiment_args = arguments.get_args()
    
    # set reproducible
    set_seed(args['rand_seed'])

    # check gpu environment
    if args['usable_gpu'] is None: 
        args['gpu_ids'] = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args['usable_gpu']
        args['gpu_ids'] = args['usable_gpu'].split(',')
    assert 0 < len(args['gpu_ids']), 'Only GPU env are supported'
    
    args['port'] = f'10{datetime.datetime.now().microsecond % 100}'

    # set DDP
    args['world_size'] = len(args['gpu_ids'])
    args['batch_size'] = args['batch_size'] // args['world_size']
    if args['batch_size'] % args['world_size'] != 0:
        print(f'The batch size is resized to {args["batch_size"] * args["world_size"]} because the rest are discarded.')
    torch.cuda.empty_cache()
    
    # start
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.spawn(
        run, 
        nprocs=args['world_size'], 
        args=(args, experiment_args)
    )