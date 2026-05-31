#!/usr/bin/env python3
"""
Evaluate a trained checkpoint and compute metrics: Accuracy, F1, Confusion Matrix, ROC AUC.

Usage:
    python tools/evaluate_checkpoint.py \
        --checkpoint_dir "HM-Conformer/results/Multilingual Test 15k/HM-Conformer/models" \
        --epoch 80 \
        --dataset_path "/path/to/dataset_audios" \
        --labels_path "/path/to/dataset_audios/labels.json"
"""

import argparse
import sys
import os
import torch
from pathlib import Path

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        accuracy_score, f1_score, confusion_matrix, roc_auc_score,
        roc_curve, precision_recall_curve, precision_score, recall_score,
        classification_report
    )
except ImportError as e:
    print(f"Error: Required packages not installed. Install with: pip install numpy matplotlib seaborn scikit-learn")
    print(f"Missing package: {e}")
    sys.exit(1)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'exp_lib'))

import egg_exp
from hm_conformer import arguments, data_processing
from egg_exp.util.model_test import df_test

def load_checkpoint(checkpoint_dir, epoch, device='cpu'):
    """Load checkpoint files for a specific epoch."""
    checkpoint_dir = Path(checkpoint_dir)
    state_dict = {}
    
    # Load frontend
    frontend_file = checkpoint_dir / f'check_point_DF_frontend_{epoch}.pt'
    if not frontend_file.exists():
        raise FileNotFoundError(f"Frontend checkpoint not found: {frontend_file}")
    state_dict['frontend'] = torch.load(frontend_file, map_location=device)
    print(f"Loaded: {frontend_file.name}")
    
    # Load backends and losses
    for i in range(5):
        backend_file = checkpoint_dir / f'check_point_DF_backend{i}_{epoch}.pt'
        loss_file = checkpoint_dir / f'check_point_DF_loss{i}_{epoch}.pt'
        
        if backend_file.exists():
            state_dict[f'backend{i}'] = torch.load(backend_file, map_location=device)
            print(f"Loaded: {backend_file.name}")
        else:
            print(f"Warning: Backend {i} checkpoint not found: {backend_file}")
        
        if loss_file.exists():
            state_dict[f'loss{i}'] = torch.load(loss_file, map_location=device)
            print(f"Loaded: {loss_file.name}")
        else:
            print(f"Warning: Loss {i} checkpoint not found: {loss_file}")
    
    return state_dict

def build_framework(args, device='cpu'):
    """Build the framework exactly as in main.py."""
    # Data preprocessing
    preprocessing = egg_exp.framework.model.LFCC(
        args['sample_rate'], args['n_lfcc'], 
        args['coef'], args['n_fft'], args['win_length'], args['hop'],
        args['with_delta'], args['with_emphasis'], args['with_energy'],
        args['DA_frq_mask'], args['DA_frq_p'], args['DA_frq_mask_max']
    )
    
    # Frontend
    frontend = egg_exp.framework.model.HM_Conformer(
        bin_size=args['bin_size'], 
        output_size=args['output_size'], 
        input_layer=args['input_layer'],
        pos_enc_layer_type=args['pos_enc_layer_type'], 
        linear_units=args['linear_units'], 
        cnn_module_kernel=args['cnn_module_kernel'],
        dropout=args['dropout'], 
        emb_dropout=args['emb_dropout'], 
        multiloss=True
    )
    
    # Backends and losses
    backends = []
    criterions = []
    for i in range(5):
        backend = egg_exp.framework.model.CLSBackend(
            in_dim=args['output_size'], 
            hidden_dim=args['embedding_size'], 
            use_pooling=args['use_pooling'], 
            input_mean_std=args['input_mean_std']
        )
        backends.append(backend)
        
        criterion = egg_exp.framework.loss.OCSoftmax(
            embedding_size=args['embedding_size'], 
            num_class=args['num_class'], 
            feat_dim=args['feat_dim'], 
            r_real=args['r_real'], 
            r_fake=args['r_fake'], 
            alpha=args['alpha']
        )
        criterions.append(criterion)
    
    # Waveform augmentation (if used during training)
    # For evaluation, we can create a dummy augmentation that does nothing
    # or check if augmentation was used during training
    augmentation = None
    if len(args.get('DA_wav_aug_list', [])) != 0:
        augmentation = egg_exp.data.augmentation.WaveformAugmetation(
            args['DA_wav_aug_list'], 
            args['DA_wav_aug_params']
        )
    else:
        # Create a dummy augmentation that passes through (no augmentation during eval anyway)
        # We still need to use DeepfakeDetectionFramework_DA_multiloss for multiloss support
        class DummyAugmentation:
            def __call__(self, x):
                return x
        augmentation = DummyAugmentation()
    
    # Framework - always use multiloss framework since we have 5 backends/losses
    framework = egg_exp.framework.DeepfakeDetectionFramework_DA_multiloss(
        augmentation=augmentation,
        preprocessing=preprocessing,
        frontend=frontend,
        backend=backends,
        loss=criterions,
        loss_weight=args['loss_weight'],
    )
    
    # Move to device (but don't use DDP for evaluation)
    for key in framework.trainable_modules.keys():
        framework.trainable_modules[key].to(device)
    framework.device = device
    
    return framework

def compute_metrics(scores, labels, threshold=None):
    """Compute all metrics from scores and labels."""
    scores = np.array(scores)
    labels = np.array(labels, dtype=int)
    
    # If threshold is None, find optimal threshold using ROC curve
    if threshold is None:
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        # Find threshold closest to EER (where FPR = 1 - TPR)
        eer_idx = np.nanargmin(np.abs(fpr - (1 - tpr)))
        threshold = thresholds[eer_idx]
        print(f"Using EER threshold: {threshold:.4f}")
    
    # Binarize predictions
    predictions = (scores >= threshold).astype(int)
    
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(labels, scores)
    except ValueError:
        roc_auc = None
        print("Warning: Could not compute ROC AUC (possibly only one class present)")
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Classification report
    report = classification_report(labels, predictions, target_names=['Real', 'Fake'])
    
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'threshold': threshold,
        'classification_report': report,
        'scores': scores,
        'labels': labels,
        'predictions': predictions
    }
    
    return metrics

def plot_metrics(metrics, output_dir):
    """Plot confusion matrix, ROC curve, and PR curve."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Confusion Matrix
    ax1 = axes[0]
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('Confusion Matrix')
    
    # 2. ROC Curve
    ax2 = axes[1]
    scores = metrics['scores']
    labels = metrics['labels']
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    ax2.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {metrics["roc_auc"]:.3f})' if metrics['roc_auc'] else 'ROC')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curve
    ax3 = axes[2]
    precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
    ax3.plot(recall, precision, linewidth=2, label='PR Curve')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision-Recall Curve')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_metrics.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'evaluation_metrics.pdf', bbox_inches='tight')
    print(f"Metrics plots saved to: {output_dir / 'evaluation_metrics.png'}")
    plt.close()

def evaluate_checkpoint(checkpoint_dir, epoch, dataset_path, labels_path, args, device='cpu', output_dir=None):
    """Evaluate a checkpoint and compute all metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluating checkpoint from epoch {epoch}")
    print(f"{'='*60}\n")
    
    # Load checkpoint
    state_dict = load_checkpoint(checkpoint_dir, epoch, device)
    
    # Build framework
    framework = build_framework(args, device)
    
    # Load state dict (handle DDP wrapper by removing "module." prefix if present)
    print("\nLoading state dict into framework...")
    for key in state_dict.keys():
        if key not in framework.trainable_modules:
            print(f"Warning: {key} not found in framework, skipping...")
            continue
            
        model_state = state_dict[key]
        target_model = framework.trainable_modules[key]
        target_state = target_model.state_dict()
        
        # Check if loaded state has "module." prefix (from DDP wrapper)
        if len(model_state.keys()) > 0:
            first_key = list(model_state.keys())[0]
            has_module_prefix = first_key.startswith('module.')
            
            if has_module_prefix:
                # Remove "module." prefix from all keys
                new_state = {}
                for param_name, param_value in model_state.items():
                    if param_name.startswith('module.'):
                        new_key = param_name[7:]  # Remove "module." prefix
                        if new_key in target_state:
                            new_state[new_key] = param_value
                        # else: skip this parameter
                    else:
                        if param_name in target_state:
                            new_state[param_name] = param_value
                model_state = new_state
        
        # Load into framework
        try:
            missing_keys, unexpected_keys = target_model.load_state_dict(model_state, strict=False)
            if missing_keys:
                print(f"  Warning: Missing keys in {key}: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"  Warning: Missing keys in {key}: {missing_keys}")
            if unexpected_keys:
                print(f"  Warning: Unexpected keys in {key}: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"  Warning: Unexpected keys in {key}: {unexpected_keys}")
            print(f"✓ Loaded {key}")
        except Exception as e:
            print(f"✗ Error loading {key}: {e}")
    
    framework.eval()
    print("Framework set to evaluation mode")
    
    # Load dataset
    print(f"\nLoading dataset from: {dataset_path}")
    multilingual_dataset = egg_exp.data.dataset.MultilingualDataset(
        labels_path=labels_path,
        dataset_root=dataset_path,
        train_split=args.get('train_split', 0.8),
        val_split=args.get('val_split', 0.1),
        test_split=args.get('test_split', 0.1),
        random_seed=args.get('rand_seed', 1),
        print_info=True
    )
    
    # Create test set
    test_set = data_processing.TestSet(
        multilingual_dataset.test_set,
        args['test_crop_size']
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=args.get('num_workers', 4),
        pin_memory=False
    )
    
    # Evaluate
    print("\nRunning evaluation...")
    eer, scores, labels = df_test(framework, test_loader, run_on_ddp=False, get_scores=True)
    
    print(f"\nEER: {eer:.4f}%")
    
    # Compute metrics
    metrics = compute_metrics(scores, labels)
    
    # Print results
    print(f"\n{'='*60}")
    print("Evaluation Metrics")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    if metrics['roc_auc']:
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"Threshold: {metrics['threshold']:.4f}")
    print(f"\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print(f"\nClassification Report:")
    print(metrics['classification_report'])
    
    # Plot metrics
    if output_dir is None:
        output_dir = Path(checkpoint_dir).parent / f'evaluation_epoch_{epoch}'
    else:
        output_dir = Path(output_dir)
    
    plot_metrics(metrics, output_dir)
    
    # Save metrics to file
    metrics_file = output_dir / 'metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write(f"Evaluation Results for Epoch {epoch}\n")
        f.write("="*60 + "\n\n")
        f.write(f"EER: {eer:.4f}%\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"F1 Score:  {metrics['f1_score']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        if metrics['roc_auc']:
            f.write(f"ROC AUC:   {metrics['roc_auc']:.4f}\n")
        f.write(f"Threshold: {metrics['threshold']:.4f}\n")
        f.write(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}\n")
        f.write(f"\nClassification Report:\n{metrics['classification_report']}\n")
    
    print(f"\nMetrics saved to: {metrics_file}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing checkpoint files')
    parser.add_argument('--epoch', type=int, required=True,
                       help='Epoch number of the checkpoint to evaluate')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to dataset root directory')
    parser.add_argument('--labels_path', type=str, required=True,
                       help='Path to labels.json file')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    # Get model arguments (same as training)
    model_args, _, _ = arguments.get_args()
    
    # Override dataset paths
    model_args['path_train'] = args.dataset_path
    model_args['labels_path'] = args.labels_path
    model_args['dataset_root'] = args.dataset_path
    
    # Evaluate
    evaluate_checkpoint(
        args.checkpoint_dir,
        args.epoch,
        args.dataset_path,
        args.labels_path,
        model_args,
        device=args.device,
        output_dir=args.output_dir
    )

if __name__ == '__main__':
    main()

