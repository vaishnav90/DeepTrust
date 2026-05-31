from tqdm import tqdm

import torch
import torch.distributed as dist

from egg_exp.util import df_test

def train(epoch, framework, optimizer, loader, logger):
    framework.train()
    
    count = 0
    loss_sum = 0
    loss_sum_list = [0]*5

    with tqdm(total=len(loader), ncols=90) as pbar:
        for x, label in loader:
            # to GPU
            x = x.to(torch.float32).to(framework.device)
            label = label.to(framework.device)
            
            # clear grad
            optimizer.zero_grad()
            
            # feed forward
            _, loss, loss_embs = framework(x, label)
            
            # backpropagation
            loss.backward()
            optimizer.step()
            
            # logging
            if logger is not None:
                count += 1
                loss_sum += loss.item()
                for i in range(5):
                    loss_sum_list[i] += loss_embs[i].item()

                if len(loader) * 0.02 <= count:
                    logger.log_metric('Loss', loss_sum / count)
                    loss_sum = 0
                    for i in range(5):
                        logger.log_metric(f'Loss{i}', loss_sum_list[i] / count)
                        loss_sum_list[i] = 0
                    count = 0

                desc = f'[{epoch}|(loss): {loss.item():.4f}'
                pbar.set_description(desc)
                pbar.update(1)

    _synchronize()

def validate(framework, loader):
    """Compute validation loss without backpropagation."""
    framework.eval()
    
    count = 0
    loss_sum = 0
    loss_sum_list = [0]*5
    
    with torch.no_grad():
        for x, label in loader:
            # to GPU
            x = x.to(torch.float32).to(framework.device)
            label = label.to(framework.device)
            
            # feed forward (no backpropagation)
            _, loss, loss_embs = framework(x, label)
            
            # accumulate losses
            count += 1
            loss_sum += loss.item()
            for i in range(5):
                loss_sum_list[i] += loss_embs[i].item()
    
    # Compute average losses
    avg_loss = loss_sum / count if count > 0 else 0
    avg_loss_list = [loss_sum_list[i] / count if count > 0 else 0 for i in range(5)]
    
    _synchronize()
    return avg_loss, avg_loss_list

def test(framework, loader, get_full_metrics=False):
    """
    Test the framework on a dataset.
    
    Args:
        framework: The model framework
        loader: DataLoader for test data
        get_full_metrics: If True, returns dict with EER and other metrics
    
    Returns:
        If get_full_metrics=False: float (EER)
        If get_full_metrics=True: dict with metrics
    """
    # Get EER and scores
    eer, scores, labels = df_test(framework, loader, run_on_ddp=True, get_scores=True)
    
    if get_full_metrics:
        # Compute additional metrics
        import numpy as np
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            roc_auc_score, confusion_matrix, classification_report, roc_curve
        )
        from egg_exp.util.model_test import _flatten_labels
        
        # Flatten labels and scores (handles DDP nested structure)
        labels_flat = _flatten_labels(labels)
        scores_flat = _flatten_labels(scores)
        
        scores = np.array(scores_flat, dtype=np.float64)
        labels = np.array(labels_flat, dtype=np.int32)
        
        # Ensure binary format (1D array)
        if labels.ndim > 1:
            labels = labels.flatten()
        if scores.ndim > 1:
            scores = scores.flatten()
        
        # Find threshold at EER
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        eer_idx = np.nanargmin(np.abs(fpr - (1 - tpr)))
        threshold = thresholds[eer_idx]
        
        # Binarize predictions
        predictions = (scores >= threshold).astype(int)
        
        # Compute metrics
        metrics = {
            'eer': eer,
            'accuracy': accuracy_score(labels, predictions),
            'f1_score': f1_score(labels, predictions),
            'precision': precision_score(labels, predictions),
            'recall': recall_score(labels, predictions),
            'threshold': threshold,
            'confusion_matrix': confusion_matrix(labels, predictions),
            'classification_report': classification_report(labels, predictions, target_names=['Real', 'Fake'])
        }
        
        # ROC AUC (if possible)
        try:
            metrics['roc_auc'] = roc_auc_score(labels, scores)
        except ValueError:
            metrics['roc_auc'] = None
        
        return metrics
    else:
        return eer

def _synchronize():
    torch.cuda.empty_cache()
    dist.barrier()