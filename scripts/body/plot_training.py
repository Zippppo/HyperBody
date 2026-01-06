"""
Utility script to visualize training logs.

Usage:
    python scripts/plot_training.py --log_dir logs/body_unet_bs2_lr0.0001_ch32

    # Compare multiple experiments
    python scripts/plot_training.py \
        --log_dirs logs/exp1 logs/exp2 logs/exp3 \
        --labels "Exp1" "Exp2" "Exp3"
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_training_log(log_dir):
    """Load training log from directory."""
    log_path = Path(log_dir) / "training_log.json"
    if not log_path.exists():
        raise FileNotFoundError(f"Training log not found: {log_path}")

    with open(log_path, 'r') as f:
        return json.load(f)


def plot_single_experiment(log_dir, output_dir=None):
    """Plot training curves for a single experiment."""
    log_data = load_training_log(log_dir)
    history = log_data['training_history']
    best_metrics = log_data['best_metrics']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Training Progress: {Path(log_dir).name}", fontsize=16)

    # Plot 1: Loss curves
    ax = axes[0, 0]
    ax.plot(history['epoch'], history['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(history['epoch'], history['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Accuracy
    ax = axes[0, 1]
    ax.plot(history['epoch'], history['train_accuracy'], label='Train Acc', linewidth=2, color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Validation mIoU
    ax = axes[1, 0]
    ax.plot(history['epoch'], history['val_mIoU'], linewidth=2, color='purple')
    best_epoch = best_metrics['best_epoch']
    best_miou = best_metrics['best_val_mIoU']
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best (epoch {best_epoch})')
    ax.axhline(best_miou, color='red', linestyle='--', alpha=0.5)
    ax.scatter([best_epoch], [best_miou], color='red', s=100, zorder=5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mIoU')
    ax.set_title(f'Validation mIoU (Best: {best_miou:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Learning rate
    ax = axes[1, 1]
    ax.plot(history['epoch'], history['learning_rate'], linewidth=2, color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    if output_dir:
        output_path = Path(output_dir) / f"{Path(log_dir).name}_training_curves.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.savefig(Path(log_dir) / "training_curves.png", dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {Path(log_dir) / 'training_curves.png'}")

    plt.show()


def plot_comparison(log_dirs, labels=None, output_path=None):
    """Plot comparison of multiple experiments."""
    if labels is None:
        labels = [Path(d).name for d in log_dirs]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training Comparison", fontsize=16)

    colors = plt.cm.tab10(np.linspace(0, 1, len(log_dirs)))

    for log_dir, label, color in zip(log_dirs, labels, colors):
        log_data = load_training_log(log_dir)
        history = log_data['training_history']
        best_metrics = log_data['best_metrics']

        # Plot 1: Validation Loss
        axes[0].plot(history['epoch'], history['val_loss'],
                    label=label, linewidth=2, color=color)

        # Plot 2: Validation mIoU
        axes[1].plot(history['epoch'], history['val_mIoU'],
                    label=label, linewidth=2, color=color)
        best_epoch = best_metrics['best_epoch']
        best_miou = best_metrics['best_val_mIoU']
        axes[1].scatter([best_epoch], [best_miou], color=color, s=100,
                       marker='*', zorder=5, edgecolors='black')

        # Plot 3: Learning Rate
        axes[2].plot(history['epoch'], history['learning_rate'],
                    label=label, linewidth=2, color=color)

    # Configure axes
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Loss')
    axes[0].set_title('Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation mIoU')
    axes[1].set_title('Validation mIoU')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to: {output_path}")
    else:
        plt.savefig("training_comparison.png", dpi=150, bbox_inches='tight')
        print("Saved comparison plot to: training_comparison.png")

    plt.show()


def print_summary(log_dir):
    """Print training summary."""
    log_data = load_training_log(log_dir)
    history = log_data['training_history']
    best_metrics = log_data['best_metrics']

    print("\n" + "=" * 60)
    print(f"Training Summary: {Path(log_dir).name}")
    print("=" * 60)
    print(f"Total epochs: {len(history['epoch'])}")
    print(f"Best val mIoU: {best_metrics['best_val_mIoU']:.4f}")
    print(f"Best epoch: {best_metrics['best_epoch']}")

    if len(history['val_loss']) > 0:
        print(f"Final val loss: {history['val_loss'][-1]:.4f}")
        print(f"Final val mIoU: {history['val_mIoU'][-1]:.4f}")

    if len(history['train_loss']) > 0:
        print(f"Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"Final train acc: {history['train_accuracy'][-1]:.4f}")

    print(f"Last update: {log_data['last_update']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Visualize training logs")
    parser.add_argument("--log_dir", type=str,
                       help="Path to single experiment log directory")
    parser.add_argument("--log_dirs", nargs='+',
                       help="Paths to multiple experiment log directories for comparison")
    parser.add_argument("--labels", nargs='+',
                       help="Labels for experiments (optional)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save plots")
    parser.add_argument("--summary", action="store_true",
                       help="Print summary only")

    args = parser.parse_args()

    if args.log_dir:
        # Single experiment
        if args.summary:
            print_summary(args.log_dir)
        else:
            plot_single_experiment(args.log_dir, args.output_dir)

    elif args.log_dirs:
        # Multiple experiments comparison
        if args.summary:
            for log_dir in args.log_dirs:
                print_summary(log_dir)
        else:
            output_path = None
            if args.output_dir:
                output_path = Path(args.output_dir) / "training_comparison.png"
            plot_comparison(args.log_dirs, args.labels, output_path)

    else:
        parser.print_help()
        print("\nError: Please specify either --log_dir or --log_dirs")


if __name__ == "__main__":
    main()
