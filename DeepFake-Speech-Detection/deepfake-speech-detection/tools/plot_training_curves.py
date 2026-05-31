#!/usr/bin/env python3
"""
Plot training curves from log files.

Usage:
    python tools/plot_training_curves.py --results_dir "HM-Conformer/results/Multilingual Test 15k/HM-Conformer"
"""

import argparse
import re
import sys
from pathlib import Path

try:
    import matplotlib
    # Headless-friendly backend (works in servers/CI/Colab without GUI)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: matplotlib and numpy are required. Install with: pip install matplotlib numpy")
    sys.exit(1)

_NUM_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

def parse_step_metric_file(metric_file: Path, metric_name: str):
    """
    Parse step-logged metrics like:
      "Loss: 9.94"
      "Loss0: 1.23"
    Returns values list; x-axis is implicit "log step".
    """
    values = []
    with open(metric_file, 'r') as f:
        for line in f:
            match = re.search(rf'{re.escape(metric_name)}:\s*({_NUM_RE})', line)
            if match:
                values.append(float(match.group(1)))
    return values

def parse_epoch_metric_file(metric_file: Path, metric_name: str):
    """
    Parse epoch-logged metrics like:
      "[5] EER: 49.93"
      "[1] ValLoss: 7.12"
      "[3] Train_Loss_Epoch: 6.98"
    Returns (epochs, values).
    """
    epochs = []
    values = []
    with open(metric_file, 'r') as f:
        for line in f:
            match = re.search(rf'\[(\d+)\]\s*{re.escape(metric_name)}:\s*({_NUM_RE})', line)
            if match:
                epochs.append(int(match.group(1)))
                values.append(float(match.group(2)))
    return epochs, values

def first_existing(results_path: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        p = results_path / f"{name}.txt"
        if p.exists():
            return p
    return None

def parse_step_losses(results_dir):
    """Parse step-logged Loss*.txt files."""
    losses_dict = {}
    results_path = Path(results_dir)
    
    loss_file = results_path / 'Loss.txt'
    if loss_file.exists():
        losses_dict['Loss (step)'] = parse_step_metric_file(loss_file, 'Loss')
    
    for i in range(10):
        loss_file = results_path / f'Loss{i}.txt'
        if loss_file.exists():
            losses_dict[f'Loss{i} (step)'] = parse_step_metric_file(loss_file, f'Loss{i}')
    
    return losses_dict

def parse_epoch_losses(results_dir):
    """
    Parse epoch-logged train/val losses.
    Supports both new names (Train_Loss_Epoch / Val_Loss*) and old names (TrainLoss / ValLoss*).
    """
    results_path = Path(results_dir)

    train_total_file = first_existing(results_path, ['Train_Loss_Epoch', 'TrainLoss'])
    val_total_file = first_existing(results_path, ['Val_Loss', 'ValLoss'])

    train_total = None
    val_total = None
    if train_total_file is not None:
        metric = train_total_file.stem
        train_total = parse_epoch_metric_file(train_total_file, metric)
    if val_total_file is not None:
        metric = val_total_file.stem
        val_total = parse_epoch_metric_file(val_total_file, metric)

    # Per-branch (try new then old)
    train_branches = {}
    val_branches = {}
    for i in range(10):
        tf = first_existing(results_path, [f'Train_Loss{i}_Epoch', f'TrainLoss{i}'])
        vf = first_existing(results_path, [f'Val_Loss{i}', f'ValLoss{i}'])
        if tf is not None:
            train_branches[i] = parse_epoch_metric_file(tf, tf.stem)
        if vf is not None:
            val_branches[i] = parse_epoch_metric_file(vf, vf.stem)

    return train_total, val_total, train_branches, val_branches

def _approx_epoch_curve_from_step_values(step_values: list[float], epochs: list[int]):
    """
    Fallback: approximate epoch-level curve by splitting step-logged values into
    len(epochs) contiguous chunks and averaging each chunk.
    """
    if not step_values or not epochs:
        return None
    e = len(epochs)
    n = len(step_values)
    if e <= 0 or n <= 0:
        return None

    # Split into equal-ish chunks (last chunk gets the remainder).
    base = n // e
    if base <= 0:
        return None
    means = []
    start = 0
    for i in range(e):
        end = start + base if i < e - 1 else n
        chunk = step_values[start:end]
        if len(chunk) == 0:
            means.append(float('nan'))
        else:
            means.append(float(np.mean(chunk)))
        start = end
    return epochs, means

def plot_training_curves(results_dir, output_file=None):
    """Plot training curves from log files."""
    results_path = Path(results_dir)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curves', fontsize=16, fontweight='bold')
    
    # 1) Epoch-level: Train vs Val total loss
    ax_epoch = axes[0, 0]
    train_total, val_total, train_branches, val_branches = parse_epoch_losses(results_dir)

    # Fallback: if we don't have epoch-level train logs, approximate from step-level Loss*.txt
    # so users can still compare against ValLoss across epochs.
    if train_total is None and val_total is not None:
        epochs_val, _ = val_total
        step_losses = parse_step_metric_file(results_path / 'Loss.txt', 'Loss') if (results_path / 'Loss.txt').exists() else []
        approx = _approx_epoch_curve_from_step_values(step_losses, epochs_val)
        if approx is not None:
            train_total = approx

    if val_total is not None:
        epochs_val, _ = val_total
        # Approximate missing per-branch train curves from step-level Loss{i}.txt
        for i in range(10):
            if i in train_branches:
                continue
            lf = results_path / f'Loss{i}.txt'
            if not lf.exists():
                continue
            step_vals = parse_step_metric_file(lf, f'Loss{i}')
            approx = _approx_epoch_curve_from_step_values(step_vals, epochs_val)
            if approx is not None:
                train_branches[i] = approx

    any_epoch = False
    if train_total is not None:
        ep, vals = train_total
        label = 'Train loss (epoch)'
        if first_existing(results_path, ['Train_Loss_Epoch', 'TrainLoss']) is None:
            label = 'Train loss (epoch, approx from Loss.txt)'
        ax_epoch.plot(ep, vals, 'o-', linewidth=2, markersize=4, label=label)
        any_epoch = True
    if val_total is not None:
        ep, vals = val_total
        ax_epoch.plot(ep, vals, 's-', linewidth=2, markersize=4, label='Val loss (epoch)')
        any_epoch = True

    if any_epoch:
        ax_epoch.set_xlabel('Epoch')
        ax_epoch.set_ylabel('Loss')
        ax_epoch.set_title('Train vs Val Loss (per epoch)')
        ax_epoch.legend()
        ax_epoch.grid(True, alpha=0.3)
    else:
        ax_epoch.text(0.5, 0.5, "No epoch loss logs found\n(TrainLoss/ValLoss or Train_Loss_Epoch/Val_Loss)",
                      ha='center', va='center')
        ax_epoch.axis('off')

    # 2) Epoch-level: per-branch (if available)
    ax_branch = axes[0, 1]
    any_branch = False
    for i, (ep, vals) in sorted(train_branches.items()):
        ax_branch.plot(ep, vals, '-', linewidth=1.5, label=f'Train loss{i}')
        any_branch = True
    for i, (ep, vals) in sorted(val_branches.items()):
        ax_branch.plot(ep, vals, '--', linewidth=1.5, label=f'Val loss{i}')
        any_branch = True
    if any_branch:
        ax_branch.set_xlabel('Epoch')
        ax_branch.set_ylabel('Loss')
        ax_branch.set_title('Per-branch Loss (epoch)')
        ax_branch.legend(ncol=2, fontsize=8)
        ax_branch.grid(True, alpha=0.3)
    else:
        ax_branch.axis('off')
    
    # 2. Plot EER over epochs
    ax2 = axes[1, 0]
    eer_file = results_path / 'EER.txt'
    if eer_file.exists():
        epochs, eers = parse_epoch_metric_file(eer_file, 'EER')
        ax2.plot(epochs, eers, 'o-', linewidth=2, markersize=6, label='EER')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('EER (%)')
        ax2.set_title('Equal Error Rate (EER)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()  # Lower EER is better
    
    # 3. Plot Best EER over epochs
    ax3 = axes[1, 1]
    best_eer_file = results_path / 'BestEER.txt'
    if best_eer_file.exists():
        epochs, best_eers = parse_epoch_metric_file(best_eer_file, 'BestEER')
        ax3.plot(epochs, best_eers, 's-', linewidth=2, markersize=6, label='Best EER', color='green')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Best EER (%)')
        ax3.set_title('Best Equal Error Rate')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.invert_yaxis()  # Lower EER is better
        
        # Add annotation for best value
        if len(best_eers) > 0:
            min_idx = np.argmin(best_eers)
            min_epoch = epochs[min_idx]
            min_eer = best_eers[min_idx]
            ax3.annotate(f'Best: {min_eer:.2f}% @ epoch {min_epoch}',
                        xy=(min_epoch, min_eer),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    
    # Save figure
    if output_file is None:
        output_file = results_path / 'training_curves.png'
    else:
        output_file = Path(output_file)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {output_file}")
    
    # Also save as PDF
    pdf_file = output_file.with_suffix('.pdf')
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"Training curves saved to: {pdf_file}")
    
    # NOTE: plt.show() is intentionally not called by default (headless-safe).

def main():
    parser = argparse.ArgumentParser(description='Plot training curves from log files')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Path to results directory containing log files')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path for the plot (default: results_dir/training_curves.png)')
    parser.add_argument('--show', action='store_true',
                       help='Show interactive window (may fail in headless environments)')
    
    args = parser.parse_args()
    
    plot_training_curves(args.results_dir, args.output)
    if args.show:
        plt.show()

if __name__ == '__main__':
    main()

