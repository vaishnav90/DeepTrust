from tqdm import tqdm

import torch
import torch.distributed as dist

from egg_exp.util import df_test

def _ddp_allreduce_sum(t: torch.Tensor) -> torch.Tensor:
    """All-reduce SUM if torch.distributed is initialized; otherwise return as-is."""
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t

def _ddp_reduce_losses(
    loss_sum: float,
    loss_sum_list: list,
    count: int,
    device: torch.device,
):
    """
    Reduce (sum) loss totals and count across DDP ranks, then return global averages.
    Averages are per-batch, consistent with local accumulation.
    """
    # Pack totals into one tensor: [total_loss_sum, loss0_sum..loss4_sum, count]
    packed = torch.tensor(
        [float(loss_sum), *[float(x) for x in loss_sum_list], float(count)],
        device=device,
        dtype=torch.float32,
    )
    packed = _ddp_allreduce_sum(packed)

    total_sum = float(packed[0].item())
    per_branch_sum = [float(packed[i].item()) for i in range(1, 6)]
    # Count is summed as float32; round to avoid rare truncation (e.g. 127.99999).
    total_count = int(torch.round(packed[6]).item())

    if total_count <= 0:
        return 0.0, [0.0] * 5

    avg_total = total_sum / total_count
    avg_list = [s / total_count for s in per_branch_sum]
    return avg_total, avg_list

def train(epoch, framework, optimizer, loader, logger):
    framework.train()
    
    # Counters for periodic logging
    count = 0
    loss_sum = 0
    loss_sum_list = [0]*5

    # Counters for epoch-level averages
    epoch_count = 0
    epoch_loss_sum = 0.0
    epoch_loss_sum_list = [0.0]*5

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

            # Epoch-level accumulation (irrespective of logging frequency)
            epoch_count += 1
            epoch_loss_sum += loss.item()
            for i in range(5):
                epoch_loss_sum_list[i] += loss_embs[i].item()

    _synchronize()

    avg_loss, avg_loss_list = _ddp_reduce_losses(
        loss_sum=epoch_loss_sum,
        loss_sum_list=epoch_loss_sum_list,
        count=epoch_count,
        device=framework.device,
    )

    # Log per-epoch averages so they align with validation metrics
    if logger is not None:
        # New metric names (requested)
        logger.log_metric('Train_Loss_Epoch', avg_loss, epoch)
        for i in range(5):
            logger.log_metric(f'Train_Loss{i}_Epoch', avg_loss_list[i], epoch)
        # Backward-compatible metric names (keep existing plots working)
        logger.log_metric('TrainLoss', avg_loss, epoch)
        for i in range(5):
            logger.log_metric(f'TrainLoss{i}', avg_loss_list[i], epoch)

    return avg_loss, avg_loss_list

def validate(epoch, framework, loader, logger=None):
    """Compute and log validation loss without backpropagation (DDP-safe)."""
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
    
    _synchronize()

    avg_loss, avg_loss_list = _ddp_reduce_losses(
        loss_sum=loss_sum,
        loss_sum_list=loss_sum_list,
        count=count,
        device=framework.device,
    )

    if logger is not None:
        # New metric names (requested)
        logger.log_metric('Val_Loss', avg_loss, epoch)
        for i in range(5):
            logger.log_metric(f'Val_Loss{i}', avg_loss_list[i], epoch)
        # Backward-compatible metric names (keep existing plots working)
        logger.log_metric('ValLoss', avg_loss, epoch)
        for i in range(5):
            logger.log_metric(f'ValLoss{i}', avg_loss_list[i], epoch)

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
    if dist.is_available() and dist.is_initialized():
        dist.barrier()