#!/usr/bin/env python3
"""
Dataset Balancing Script

This script provides visualization and balancing capabilities for the deepfake speech detection dataset.
It loads labels.json, creates visualizations showing dataset distribution, and provides functionality
to balance the dataset by downsampling or filtering.
"""

import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_labels(labels_path: Path) -> List[Dict]:
    """Load labels.json file."""
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    print(f"Loaded {len(labels)} audio file entries from {labels_path}")
    return labels


def analyze_dataset(labels: List[Dict]) -> Dict:
    """Analyze the dataset and return statistics."""
    stats = {
        'total': len(labels),
        'by_label': Counter(),
        'by_model_or_speaker': Counter(),
        'by_language': Counter(),
        'by_label_and_language': defaultdict(lambda: Counter()),
        'by_label_and_model': defaultdict(lambda: Counter()),
    }
    
    for entry in labels:
        label = entry.get('label', 'unknown')
        model_or_speaker = entry.get('model_or_speaker', 'unknown')
        language = entry.get('language', 'unknown')
        
        stats['by_label'][label] += 1
        stats['by_model_or_speaker'][model_or_speaker] += 1
        stats['by_language'][language] += 1
        stats['by_label_and_language'][label][language] += 1
        stats['by_label_and_model'][label][model_or_speaker] += 1
    
    return stats


def plot_real_vs_fake_distribution(stats: Dict, output_dir: Path):
    """Create visualization showing real vs fake distribution."""
    by_label = stats['by_label']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart
    labels = list(by_label.keys())
    counts = [by_label[label] for label in labels]
    colors = ['#2ecc71' if l == 'real' else '#e74c3c' for l in labels]
    
    bars = ax1.bar(labels, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Label', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Audio Files', fontsize=12, fontweight='bold')
    ax1.set_title('Real vs Fake Audio Distribution', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Pie chart
    if len(labels) == 2:
        colors_pie = ['#2ecc71', '#e74c3c']
        ax2.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors_pie,
                startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax2.set_title('Real vs Fake Proportion', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = output_dir / '01_real_vs_fake_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_audios_per_model(stats: Dict, output_dir: Path, top_n: int = 30):
    """Create visualization showing number of audios per model/speaker."""
    by_model = stats['by_model_or_speaker']
    
    # Get top N models
    top_models = by_model.most_common(top_n)
    models = [m[0] for m in top_models]
    counts = [m[1] for m in top_models]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create horizontal bar chart for better readability
    y_pos = np.arange(len(models))
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    
    bars = ax.barh(y_pos, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel('Number of Audio Files', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model/Speaker', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Models/Speakers by Audio Count', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{int(count):,}',
                ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    output_path = output_dir / '02_audios_per_model.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_audios_per_language(stats: Dict, output_dir: Path):
    """Create visualization showing number of audios per language."""
    by_language = stats['by_language']
    
    languages = sorted(by_language.keys())
    counts = [by_language[lang] for lang in languages]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(languages)))
    bars = ax1.bar(range(len(languages)), counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax1.set_xticks(range(len(languages)))
    ax1.set_xticklabels(languages, rotation=45, ha='right', fontsize=10)
    ax1.set_xlabel('Language', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Audio Files', fontsize=12, fontweight='bold')
    ax1.set_title('Audio Files per Language', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=9, rotation=90)
    
    # Pie chart
    ax2.pie(counts, labels=languages, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 10})
    ax2.set_title('Language Distribution', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = output_dir / '03_audios_per_language.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_label_by_language(stats: Dict, output_dir: Path):
    """Create visualization showing real vs fake distribution per language."""
    by_label_and_language = stats['by_label_and_language']
    
    languages = sorted(set(
        lang for label_dict in by_label_and_language.values()
        for lang in label_dict.keys()
    ))
    
    real_counts = [by_label_and_language.get('real', Counter()).get(lang, 0) for lang in languages]
    fake_counts = [by_label_and_language.get('fake', Counter()).get(lang, 0) for lang in languages]
    
    x = np.arange(len(languages))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    bars1 = ax.bar(x - width/2, real_counts, width, label='Real', color='#2ecc71', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, fake_counts, width, label='Fake', color='#e74c3c', alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Language', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Audio Files', fontsize=12, fontweight='bold')
    ax.set_title('Real vs Fake Distribution by Language', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(languages, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.tight_layout()
    output_path = output_dir / '04_real_vs_fake_by_language.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_label_by_model(stats: Dict, output_dir: Path, top_n: int = 20):
    """Create visualization showing real vs fake distribution per model (top N)."""
    by_label_and_model = stats['by_label_and_model']
    
    # Get all models and their total counts
    all_models = Counter()
    for label_dict in by_label_and_model.values():
        all_models.update(label_dict)
    
    top_models = [m[0] for m in all_models.most_common(top_n)]
    
    real_counts = [by_label_and_model.get('real', Counter()).get(m, 0) for m in top_models]
    fake_counts = [by_label_and_model.get('fake', Counter()).get(m, 0) for m in top_models]
    
    x = np.arange(len(top_models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(18, 8))
    
    bars1 = ax.bar(x - width/2, real_counts, width, label='Real', color='#2ecc71', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, fake_counts, width, label='Fake', color='#e74c3c', alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Model/Speaker', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Audio Files', fontsize=12, fontweight='bold')
    ax.set_title(f'Real vs Fake Distribution by Model/Speaker (Top {top_n})', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(top_models, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / '05_real_vs_fake_by_model.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_statistics(stats: Dict):
    """Print detailed statistics to console."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    print(f"\nTotal audio files: {stats['total']:,}")
    
    print("\n--- By Label ---")
    for label, count in stats['by_label'].most_common():
        percentage = (count / stats['total']) * 100
        print(f"  {label:10s}: {count:8,} ({percentage:5.2f}%)")
    
    print("\n--- By Language (Top 10) ---")
    for lang, count in stats['by_language'].most_common(10):
        percentage = (count / stats['total']) * 100
        print(f"  {lang:10s}: {count:8,} ({percentage:5.2f}%)")
    
    print("\n--- Top 10 Models/Speakers ---")
    for model, count in stats['by_model_or_speaker'].most_common(10):
        percentage = (count / stats['total']) * 100
        print(f"  {model:40s}: {count:8,} ({percentage:5.2f}%)")
    
    print("\n--- Real vs Fake by Language ---")
    for lang in sorted(stats['by_label_and_language']['real'].keys() | 
                      stats['by_label_and_language']['fake'].keys()):
        real_count = stats['by_label_and_language']['real'].get(lang, 0)
        fake_count = stats['by_label_and_language']['fake'].get(lang, 0)
        total = real_count + fake_count
        if total > 0:
            ratio = real_count / fake_count if fake_count > 0 else float('inf')
            print(f"  {lang:10s}: Real={real_count:6,} Fake={fake_count:6,} Ratio={ratio:5.2f}:1")
    
    print("="*60 + "\n")


def create_all_visualizations(labels: List[Dict], output_dir: Path):
    """Create all visualization plots."""
    print("\nAnalyzing dataset...")
    stats = analyze_dataset(labels)
    
    print_statistics(stats)
    
    print("\nCreating visualizations...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_real_vs_fake_distribution(stats, output_dir)
    plot_audios_per_model(stats, output_dir, top_n=30)
    plot_audios_per_language(stats, output_dir)
    plot_label_by_language(stats, output_dir)
    plot_label_by_model(stats, output_dir, top_n=20)
    
    print(f"\nAll visualizations saved to: {output_dir}")


def balance_dataset(
    labels: List[Dict],
    strategy: str = 'equal',
    target_ratio: Optional[float] = None,
    min_samples_per_model: int = 10,
    random_seed: int = 42
) -> List[Dict]:
    """
    Balance the dataset according to specified strategy.
    
    Args:
        labels: List of label entries
        strategy: 'equal' (equalize real and fake), 'ratio' (maintain target ratio), 
                  'min_per_model' (ensure minimum samples per model)
        target_ratio: For 'ratio' strategy, the desired real:fake ratio
        min_samples_per_model: Minimum samples to keep per model/speaker
        random_seed: Random seed for reproducibility
    
    Returns:
        Balanced list of label entries
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Separate by label
    real_labels = [e for e in labels if e.get('label') == 'real']
    fake_labels = [e for e in labels if e.get('label') == 'fake']
    
    print(f"\nOriginal dataset:")
    print(f"  Real: {len(real_labels):,}")
    print(f"  Fake: {len(fake_labels):,}")
    print(f"  Ratio (Real:Fake): {len(real_labels)/len(fake_labels):.2f}:1" if fake_labels else "  Ratio: N/A")
    
    if strategy == 'equal':
        # Equalize real and fake
        target_count = min(len(real_labels), len(fake_labels))
        print(f"\nBalancing to equal counts: {target_count:,} each")
        
        balanced_real = random.sample(real_labels, target_count) if len(real_labels) > target_count else real_labels
        balanced_fake = random.sample(fake_labels, target_count) if len(fake_labels) > target_count else fake_labels
        
        balanced = balanced_real + balanced_fake
        random.shuffle(balanced)
        
    elif strategy == 'ratio':
        # Maintain target ratio
        if target_ratio is None:
            raise ValueError("target_ratio must be specified for 'ratio' strategy")
        
        if len(fake_labels) == 0:
            print("Warning: No fake labels found, returning original dataset")
            return labels
        
        # Calculate target counts
        target_fake = min(len(fake_labels), len(real_labels) / target_ratio)
        target_real = int(target_fake * target_ratio)
        target_real = min(target_real, len(real_labels))
        
        print(f"\nBalancing to ratio {target_ratio}:1 (Real:Fake)")
        print(f"  Target Real: {target_real:,}")
        print(f"  Target Fake: {int(target_fake):,}")
        
        balanced_real = random.sample(real_labels, target_real) if len(real_labels) > target_real else real_labels
        balanced_fake = random.sample(fake_labels, int(target_fake)) if len(fake_labels) > target_fake else fake_labels
        
        balanced = balanced_real + balanced_fake
        random.shuffle(balanced)
        
    elif strategy == 'min_per_model':
        # Ensure minimum samples per model
        print(f"\nFiltering to ensure minimum {min_samples_per_model} samples per model/speaker")
        
        # Group by model/speaker
        by_model = defaultdict(list)
        for entry in labels:
            model = entry.get('model_or_speaker', 'unknown')
            by_model[model].append(entry)
        
        balanced = []
        removed_models = []
        
        for model, entries in by_model.items():
            if len(entries) >= min_samples_per_model:
                balanced.extend(entries)
            else:
                removed_models.append((model, len(entries)))
        
        random.shuffle(balanced)
        
        if removed_models:
            print(f"\nRemoved {len(removed_models)} models with < {min_samples_per_model} samples:")
            for model, count in sorted(removed_models, key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {model}: {count} samples")
            if len(removed_models) > 10:
                print(f"  ... and {len(removed_models) - 10} more")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Update statistics
    balanced_real_count = sum(1 for e in balanced if e.get('label') == 'real')
    balanced_fake_count = sum(1 for e in balanced if e.get('label') == 'fake')
    
    print(f"\nBalanced dataset:")
    print(f"  Real: {balanced_real_count:,}")
    print(f"  Fake: {balanced_fake_count:,}")
    print(f"  Total: {len(balanced):,}")
    if balanced_fake_count > 0:
        print(f"  Ratio (Real:Fake): {balanced_real_count/balanced_fake_count:.2f}:1")
    
    return balanced


def save_balanced_labels(balanced_labels: List[Dict], output_path: Path):
    """Save balanced labels to a new JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(balanced_labels, f, ensure_ascii=False, indent=2)
    
    print(f"\nBalanced labels saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize and balance the deepfake speech detection dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create visualizations only
  python balance_dataset.py --visualize-only
  
  # Balance dataset to equal real/fake counts
  python balance_dataset.py --balance equal --output labels_balanced.json
  
  # Balance dataset to 2:1 ratio (real:fake)
  python balance_dataset.py --balance ratio --target-ratio 2.0 --output labels_balanced.json
  
  # Filter to models with at least 100 samples
  python balance_dataset.py --balance min_per_model --min-samples 100 --output labels_balanced.json
        """
    )
    
    parser.add_argument(
        '--labels',
        type=str,
        default='/Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios/labels.json',
        help='Path to labels.json file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/Users/usuario/Documents/github/deepfake-speech-detection/dataset_audios/plots',
        help='Directory to save visualization plots'
    )
    
    parser.add_argument(
        '--visualize-only',
        action='store_true',
        help='Only create visualizations, do not balance dataset'
    )
    
    parser.add_argument(
        '--balance',
        type=str,
        choices=['equal', 'ratio', 'min_per_model'],
        help='Balancing strategy to use'
    )
    
    parser.add_argument(
        '--target-ratio',
        type=float,
        help='Target ratio (real:fake) for ratio balancing strategy'
    )
    
    parser.add_argument(
        '--min-samples',
        type=int,
        default=10,
        help='Minimum samples per model for min_per_model strategy'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for balanced labels.json file'
    )
    
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Load labels
    labels_path = Path(args.labels)
    labels = load_labels(labels_path)
    
    # Create visualizations
    output_dir = Path(args.output_dir)
    create_all_visualizations(labels, output_dir)
    
    # Balance dataset if requested
    if not args.visualize_only and args.balance:
        balanced_labels = balance_dataset(
            labels,
            strategy=args.balance,
            target_ratio=args.target_ratio,
            min_samples_per_model=args.min_samples,
            random_seed=args.random_seed
        )
        
        if args.output:
            save_balanced_labels(balanced_labels, Path(args.output))
        else:
            print("\nWarning: --output not specified, balanced labels not saved")
            print("Use --output <path> to save the balanced dataset")


if __name__ == '__main__':
    main()

