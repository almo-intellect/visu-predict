"""
Report generation module for the traffic prediction transformer.

Produces a multi-section PDF report covering training analysis, performance
metrics, attention visualisation, cross-validation results, and baseline
comparisons.  Falls back to an emergency report when the full pipeline fails.
"""

import gc
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import TrainingConfig
from .utils import get_maputo_timestamp


# =============================================================================
# Matplotlib style setup
# =============================================================================

def _apply_plot_style() -> None:
    """Apply a consistent matplotlib style with fallbacks."""
    style_candidates = ["seaborn-v0_8", "seaborn", "ggplot", "default"]
    for style in style_candidates:
        try:
            plt.style.use(style)
            return
        except OSError:
            continue
    # If nothing works, just use whatever matplotlib defaults to
    pass


# =============================================================================
# Helper: Error page (fallback for any failed section)
# =============================================================================

def _create_error_page(
    section_name: str,
    error: Exception,
    pdf: PdfPages,
) -> None:
    """Add a single-page error notice to the PDF when a section fails."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(
        0.5, 0.6,
        f"Error in section: {section_name}",
        ha="center", va="center", fontsize=18, fontweight="bold", color="red",
    )
    fig.text(
        0.5, 0.45,
        str(error),
        ha="center", va="center", fontsize=11, color="gray",
        wrap=True,
    )
    fig.text(
        0.5, 0.30,
        "This section was skipped due to the error above.",
        ha="center", va="center", fontsize=10,
    )
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# Helper: Title page
# =============================================================================

def _create_title_page(
    model: torch.nn.Module,
    config: TrainingConfig,
    timestamp: str,
    pdf: PdfPages,
) -> None:
    """Create a title page with model info, configuration highlights, and footer."""
    fig = plt.figure(figsize=(11, 8.5))

    # Title
    fig.text(
        0.5, 0.85,
        "Traffic Prediction Transformer",
        ha="center", va="center", fontsize=28, fontweight="bold",
    )
    fig.text(
        0.5, 0.78,
        "Comprehensive Training & Evaluation Report",
        ha="center", va="center", fontsize=16, color="gray",
    )

    # Model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info_lines = [
        f"Dataset: {config.dataset_name}",
        f"Total Parameters: {total_params:,}",
        f"Trainable Parameters: {trainable_params:,}",
        f"Hidden Dimension: {config.hidden_dim}",
        f"Number of Layers: {config.num_layers}",
        f"Number of Heads: {config.num_heads}",
        f"Sequence Length: {config.seq_length}  |  Prediction Length: {config.pred_length}",
        f"Learning Rate: {config.learning_rate}",
        f"Batch Size: {config.batch_size}",
        f"Optimizer: {config.optimizer_type}  |  Scheduler: {config.scheduler_type}",
    ]

    y_start = 0.62
    for i, line in enumerate(info_lines):
        fig.text(
            0.5, y_start - i * 0.04,
            line,
            ha="center", va="center", fontsize=11,
        )

    # Footer
    fig.text(
        0.5, 0.06,
        f"Generated: {timestamp}",
        ha="center", va="center", fontsize=9, color="gray",
    )

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# Helper: Training analysis
# =============================================================================

def _create_training_analysis(
    train_losses: List[float],
    val_losses: List[float],
    pdf: PdfPages,
) -> None:
    """Create training/validation loss curves and convergence insights."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    # --- Panel 1: Loss curves ---
    ax1 = fig.add_subplot(gs[0, 0])
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, label="Train Loss", linewidth=1.5)
    if val_losses:
        ax1.plot(epochs, val_losses, label="Validation Loss", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Convergence patterns (moving average improvement) ---
    ax2 = fig.add_subplot(gs[0, 1])
    window = max(5, len(train_losses) // 20)
    if len(train_losses) >= window:
        moving_avg = np.convolve(
            train_losses, np.ones(window) / window, mode="valid"
        )
        improvement = -np.diff(moving_avg)
        ax2.plot(
            range(1, len(improvement) + 1),
            improvement,
            color="green", linewidth=1.2,
        )
        ax2.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Improvement (negative = worsening)")
        ax2.set_title(f"Convergence Rate (window={window})")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Not enough epochs\nfor convergence analysis",
                 ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Convergence Rate")

    # --- Panel 3: Additional training insights ---
    ax3 = fig.add_subplot(gs[1, 0])
    insights: List[str] = []

    # Early vs late convergence
    if len(train_losses) >= 10:
        early_loss = np.mean(train_losses[:len(train_losses) // 5])
        late_loss = np.mean(train_losses[-len(train_losses) // 5:])
        reduction_pct = ((early_loss - late_loss) / early_loss) * 100 if early_loss != 0 else 0
        insights.append(f"Loss reduction: {reduction_pct:.1f}%")

    # Train-val gap
    if val_losses and len(val_losses) > 0:
        final_train = train_losses[-1]
        final_val = val_losses[-1]
        gap = abs(final_val - final_train)
        insights.append(f"Final train-val gap: {gap:.6f}")
        if final_val > final_train * 1.5:
            insights.append("WARNING: Possible overfitting detected")

    # Loss distribution
    if len(train_losses) > 1:
        loss_std = np.std(train_losses[-len(train_losses) // 4:])
        insights.append(f"Late-stage loss std: {loss_std:.6f}")

    # Stability analysis
    if len(train_losses) >= 20:
        last_segment = train_losses[-len(train_losses) // 10:]
        if len(last_segment) > 1:
            stability = np.std(last_segment) / (np.mean(last_segment) + 1e-10)
            insights.append(f"Stability (CV of tail): {stability:.4f}")
            if stability < 0.01:
                insights.append("Training is STABLE")
            elif stability < 0.05:
                insights.append("Training is MODERATELY STABLE")
            else:
                insights.append("Training shows HIGH VARIANCE")

    ax3.axis("off")
    ax3.set_title("Training Insights", fontsize=12, fontweight="bold")
    y_pos = 0.95
    for insight in insights:
        color = "red" if "WARNING" in insight else "black"
        ax3.text(0.05, y_pos, insight, fontsize=10, color=color,
                 transform=ax3.transAxes, verticalalignment="top")
        y_pos -= 0.10

    # --- Panel 4: Loss distribution histogram ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(train_losses, bins=30, alpha=0.7, label="Train Loss", color="steelblue")
    if val_losses:
        ax4.hist(val_losses, bins=30, alpha=0.5, label="Val Loss", color="coral")
    ax4.axvline(np.mean(train_losses), color="blue", linestyle="--", alpha=0.7,
                label=f"Train mean: {np.mean(train_losses):.4f}")
    ax4.axvline(np.std(train_losses) + np.mean(train_losses), color="blue",
                linestyle=":", alpha=0.5)
    ax4.set_xlabel("Loss Value")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Loss Distribution")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    fig.suptitle("Training Analysis", fontsize=16, fontweight="bold", y=0.98)
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# Helper: Performance analysis
# =============================================================================

def _create_performance_analysis(
    predictions: np.ndarray,
    actuals: np.ndarray,
    pdf: PdfPages,
) -> None:
    """Create scatter, error distribution, time-series, and residual plots."""
    predictions_flat = predictions.flatten()
    actuals_flat = actuals.flatten()

    mae = mean_absolute_error(actuals_flat, predictions_flat)
    rmse = float(np.sqrt(mean_squared_error(actuals_flat, predictions_flat)))
    r2 = r2_score(actuals_flat, predictions_flat)
    mape = float(np.mean(np.abs((actuals_flat - predictions_flat) / (actuals_flat + 1e-10)))) * 100

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # --- Scatter: actual vs predicted ---
    ax1 = fig.add_subplot(gs[0, 0])
    sample_size = min(5000, len(predictions_flat))
    indices = np.random.choice(len(predictions_flat), sample_size, replace=False)
    ax1.scatter(actuals_flat[indices], predictions_flat[indices],
                alpha=0.3, s=5, color="steelblue")
    lims = [
        min(actuals_flat[indices].min(), predictions_flat[indices].min()),
        max(actuals_flat[indices].max(), predictions_flat[indices].max()),
    ]
    ax1.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Actual vs Predicted")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Error distribution ---
    ax2 = fig.add_subplot(gs[0, 1])
    errors = predictions_flat - actuals_flat
    ax2.hist(errors, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
    ax2.axvline(np.mean(errors), color="red", linestyle="--", linewidth=1.5,
                label=f"Mean: {np.mean(errors):.4f}")
    ax2.axvline(np.mean(errors) + np.std(errors), color="orange", linestyle=":",
                label=f"Std: {np.std(errors):.4f}")
    ax2.axvline(np.mean(errors) - np.std(errors), color="orange", linestyle=":")
    ax2.set_xlabel("Prediction Error")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Error Distribution")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Sample time series ---
    ax3 = fig.add_subplot(gs[0, 2])
    ts_length = min(200, len(predictions_flat))
    ts_indices = range(ts_length)
    ax3.plot(ts_indices, actuals_flat[:ts_length], label="Actual",
             linewidth=1.2, alpha=0.8)
    ax3.plot(ts_indices, predictions_flat[:ts_length], label="Predicted",
             linewidth=1.2, alpha=0.8)
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Value")
    ax3.set_title("Sample Time Series Comparison")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # --- Residual plot ---
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(predictions_flat[indices], errors[indices],
                alpha=0.3, s=5, color="steelblue")
    ax4.axhline(y=0, color="red", linestyle="--", linewidth=1.0)
    ax4.set_xlabel("Predicted Value")
    ax4.set_ylabel("Residual")
    ax4.set_title("Residual Plot")
    ax4.grid(True, alpha=0.3)

    # --- Metrics text box ---
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis("off")
    metrics_text = (
        f"Performance Metrics\n"
        f"{'=' * 35}\n\n"
        f"MAE:   {mae:.6f}\n"
        f"RMSE:  {rmse:.6f}\n"
        f"R2:    {r2:.6f}\n"
        f"MAPE:  {mape:.2f}%\n\n"
        f"Predictions:  {len(predictions_flat):,} values\n"
        f"Error Mean:   {np.mean(errors):.6f}\n"
        f"Error Std:    {np.std(errors):.6f}"
    )
    ax5.text(
        0.15, 0.95, metrics_text,
        transform=ax5.transAxes,
        fontsize=12, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", alpha=0.8),
    )

    fig.suptitle("Performance Analysis", fontsize=16, fontweight="bold", y=0.98)
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# Helper: Attention analysis
# =============================================================================

def _create_attention_analysis(
    attention_weights: Optional[Union[np.ndarray, torch.Tensor]],
    config: TrainingConfig,
    pdf: PdfPages,
) -> None:
    """Create an average attention weight heatmap, with shape handling."""
    fig, ax = plt.subplots(figsize=(10, 8))

    if attention_weights is None:
        ax.text(0.5, 0.5, "No attention weights available",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.set_title("Attention Analysis")
        pdf.savefig(fig)
        plt.close(fig)
        return

    # Convert to numpy if needed
    if isinstance(attention_weights, torch.Tensor):
        attn = attention_weights.detach().cpu().numpy()
    else:
        attn = np.array(attention_weights)

    # Handle various shapes: (batch, heads, seq, seq), (heads, seq, seq), (seq, seq)
    while attn.ndim > 2:
        attn = attn.mean(axis=0)

    sns.heatmap(
        attn,
        cmap="viridis",
        ax=ax,
        xticklabels=False,
        yticklabels=False,
    )
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title("Average Attention Weights")

    fig.suptitle("Attention Analysis", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# Helper: Cross-validation analysis
# =============================================================================

def _create_cross_validation_analysis(
    fold_metrics: Optional[List[Dict[str, float]]],
    pdf: PdfPages,
) -> None:
    """Create box plots of metrics across cross-validation folds."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Cross-Validation Analysis", fontsize=16, fontweight="bold")

    if not fold_metrics or len(fold_metrics) == 0:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "No fold metrics available",
                    ha="center", va="center", fontsize=12, transform=ax.transAxes)
            ax.set_title("")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)
        return

    metric_names = ["mae", "rmse", "r2", "mape"]
    display_names = ["MAE", "RMSE", "R2 Score", "MAPE (%)"]

    for ax, metric_key, display_name in zip(axes.flat, metric_names, display_names):
        values = [
            fm[metric_key] for fm in fold_metrics
            if metric_key in fm
        ]
        if values:
            ax.boxplot(values, vert=True, patch_artist=True,
                       boxprops=dict(facecolor="lightblue", alpha=0.7))
            ax.scatter(
                [1] * len(values), values,
                color="steelblue", zorder=3, alpha=0.6, s=30,
            )
            ax.set_title(f"{display_name}\nMean: {np.mean(values):.4f}  Std: {np.std(values):.4f}")
            ax.set_ylabel(display_name)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"No '{metric_key}' data",
                    ha="center", va="center", fontsize=11, transform=ax.transAxes)
            ax.set_title(display_name)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# Helper: Baseline comparison
# =============================================================================

def _create_baseline_comparison(
    baseline_metrics: Optional[Dict[str, Dict[str, float]]],
    fold_metrics: Optional[List[Dict[str, float]]],
    pdf: PdfPages,
) -> None:
    """Create bar charts comparing the transformer against baseline models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle("Baseline Comparison", fontsize=16, fontweight="bold")

    if not baseline_metrics or len(baseline_metrics) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, "No baseline metrics available",
                    ha="center", va="center", fontsize=12, transform=ax.transAxes)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig)
        plt.close(fig)
        return

    # Compute transformer averages from fold metrics
    transformer_metrics: Dict[str, float] = {}
    if fold_metrics and len(fold_metrics) > 0:
        for key in ["mae", "rmse", "r2"]:
            values = [fm[key] for fm in fold_metrics if key in fm]
            if values:
                transformer_metrics[key] = float(np.mean(values))

    metric_keys = ["mae", "rmse", "r2"]
    display_names = ["MAE (lower is better)", "RMSE (lower is better)", "R2 Score (higher is better)"]

    for ax, m_key, d_name in zip(axes, metric_keys, display_names):
        model_names: List[str] = []
        model_values: List[float] = []

        # Add transformer result
        if m_key in transformer_metrics:
            model_names.append("Transformer")
            model_values.append(transformer_metrics[m_key])

        # Add baselines
        for bl_name, bl_dict in baseline_metrics.items():
            if m_key in bl_dict:
                model_names.append(bl_name)
                model_values.append(bl_dict[m_key])

        if model_names:
            colors = ["steelblue"] + ["coral"] * (len(model_names) - 1)
            bars = ax.bar(model_names, model_values, color=colors[:len(model_names)],
                          alpha=0.8, edgecolor="white")
            ax.set_title(d_name, fontsize=10)
            ax.set_ylabel(m_key.upper())
            ax.grid(True, alpha=0.3, axis="y")

            # Value labels
            for bar, val in zip(bars, model_values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.4f}", ha="center", va="bottom", fontsize=8)

            ax.tick_params(axis="x", rotation=30)
        else:
            ax.text(0.5, 0.5, f"No '{m_key}' data",
                    ha="center", va="center", fontsize=11, transform=ax.transAxes)
            ax.set_title(d_name, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# Helper: Configuration summary
# =============================================================================

def _create_config_summary(
    config: TrainingConfig,
    pdf: PdfPages,
) -> None:
    """Create a text page listing all configuration attributes."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Configuration Summary", fontsize=16, fontweight="bold", y=0.97)

    ax = fig.add_subplot(111)
    ax.axis("off")

    config_lines: List[str] = []
    for attr_name in sorted(vars(config)):
        if not attr_name.startswith("_"):
            value = getattr(config, attr_name)
            config_lines.append(f"{attr_name}: {value}")

    # Split into columns if too many lines
    text_content = "\n".join(config_lines)

    ax.text(
        0.02, 0.98, text_content,
        transform=ax.transAxes,
        fontsize=7, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.5),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# Helper: Summary statistics
# =============================================================================

def _create_summary_statistics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    train_losses: List[float],
    val_losses: List[float],
    fold_metrics: Optional[List[Dict[str, float]]],
    baseline_metrics: Optional[Dict[str, Dict[str, float]]],
    pdf: PdfPages,
) -> None:
    """Create a text page with overall metrics, training, CV, and baseline summaries."""
    predictions_flat = predictions.flatten()
    actuals_flat = actuals.flatten()

    mae = mean_absolute_error(actuals_flat, predictions_flat)
    rmse = float(np.sqrt(mean_squared_error(actuals_flat, predictions_flat)))
    r2 = r2_score(actuals_flat, predictions_flat)
    mape = float(np.mean(np.abs((actuals_flat - predictions_flat) / (actuals_flat + 1e-10)))) * 100

    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Summary Statistics", fontsize=16, fontweight="bold", y=0.97)

    ax = fig.add_subplot(111)
    ax.axis("off")

    lines: List[str] = []

    # Overall metrics
    lines.append("=" * 50)
    lines.append("OVERALL PERFORMANCE METRICS")
    lines.append("=" * 50)
    lines.append(f"  MAE:   {mae:.6f}")
    lines.append(f"  RMSE:  {rmse:.6f}")
    lines.append(f"  R2:    {r2:.6f}")
    lines.append(f"  MAPE:  {mape:.2f}%")
    lines.append("")

    # Training summary
    lines.append("=" * 50)
    lines.append("TRAINING SUMMARY")
    lines.append("=" * 50)
    lines.append(f"  Total Epochs: {len(train_losses)}")
    if train_losses:
        lines.append(f"  Final Train Loss: {train_losses[-1]:.6f}")
        lines.append(f"  Best Train Loss:  {min(train_losses):.6f}")
    if val_losses:
        lines.append(f"  Final Val Loss:   {val_losses[-1]:.6f}")
        lines.append(f"  Best Val Loss:    {min(val_losses):.6f}")
    lines.append("")

    # Cross-validation summary
    if fold_metrics and len(fold_metrics) > 0:
        lines.append("=" * 50)
        lines.append("CROSS-VALIDATION SUMMARY")
        lines.append("=" * 50)
        lines.append(f"  Number of Folds: {len(fold_metrics)}")
        for key in ["mae", "rmse", "r2", "mape"]:
            values = [fm[key] for fm in fold_metrics if key in fm]
            if values:
                lines.append(
                    f"  {key.upper():>6s}:  mean={np.mean(values):.6f}  "
                    f"std={np.std(values):.6f}  "
                    f"min={np.min(values):.6f}  max={np.max(values):.6f}"
                )
        lines.append("")

    # Baseline comparison
    if baseline_metrics and len(baseline_metrics) > 0:
        lines.append("=" * 50)
        lines.append("BASELINE COMPARISON")
        lines.append("=" * 50)
        for bl_name, bl_dict in baseline_metrics.items():
            bl_parts = [f"{k.upper()}={v:.4f}" for k, v in bl_dict.items()]
            lines.append(f"  {bl_name}: {', '.join(bl_parts)}")
        lines.append("")

    text_content = "\n".join(lines)
    ax.text(
        0.02, 0.98, text_content,
        transform=ax.transAxes,
        fontsize=9, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.5),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# Emergency report
# =============================================================================

def _create_emergency_report(
    predictions: Optional[np.ndarray],
    actuals: Optional[np.ndarray],
    train_losses: Optional[List[float]],
    val_losses: Optional[List[float]],
    pdf_path: str,
) -> None:
    """Create a minimal fallback report when the full pipeline fails."""
    try:
        with PdfPages(pdf_path) as pdf:
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle("Emergency Report", fontsize=18, fontweight="bold", color="red")

            ax = fig.add_subplot(111)
            ax.axis("off")

            lines: List[str] = [
                "The full report generation failed.",
                "Below is a minimal summary of available data.",
                "",
            ]

            # Basic metrics
            if predictions is not None and actuals is not None:
                predictions_flat = predictions.flatten()
                actuals_flat = actuals.flatten()
                mae = mean_absolute_error(actuals_flat, predictions_flat)
                rmse = float(np.sqrt(mean_squared_error(actuals_flat, predictions_flat)))
                r2 = r2_score(actuals_flat, predictions_flat)
                lines.append("BASIC METRICS:")
                lines.append(f"  MAE:  {mae:.6f}")
                lines.append(f"  RMSE: {rmse:.6f}")
                lines.append(f"  R2:   {r2:.6f}")
                lines.append(f"  Predictions shape: {predictions.shape}")
                lines.append("")

            # Final losses
            if train_losses and len(train_losses) > 0:
                lines.append(f"Final Train Loss: {train_losses[-1]:.6f}")
                lines.append(f"Total Epochs:     {len(train_losses)}")
            if val_losses and len(val_losses) > 0:
                lines.append(f"Final Val Loss:   {val_losses[-1]:.6f}")

            text_content = "\n".join(lines)
            ax.text(
                0.1, 0.85, text_content,
                transform=ax.transAxes,
                fontsize=11, verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.8", facecolor="mistyrose", alpha=0.8),
            )

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        print(f"Emergency report saved to: {pdf_path}")
    except Exception as e:
        print(f"Even emergency report generation failed: {e}")


# =============================================================================
# Main report generation function
# =============================================================================

def generate_traffic_report(
    model: torch.nn.Module,
    test_loader: DataLoader,
    scaler: Any,
    config: TrainingConfig,
    device: Union[str, torch.device],
    train_losses: List[float],
    val_losses: List[float],
    fold_metrics: Optional[List[Dict[str, float]]],
    baseline_metrics: Optional[Dict[str, Dict[str, float]]],
    results_dir: str,
    timestamp: Optional[str] = None,
    adjacency_matrix: Optional[torch.Tensor] = None,
) -> str:
    """Generate a comprehensive multi-section PDF report.

    Sections (each wrapped in its own error handler):
      1. Title page
      2. Training analysis
      3. Performance analysis
      4. Attention analysis
      5. Cross-validation analysis
      6. Baseline comparison
      7. Configuration summary
      8. Summary statistics

    Falls back to :func:`_create_emergency_report` if unrecoverable errors
    occur.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model (used for inference and parameter counts).
    test_loader : DataLoader
        DataLoader over the test set.
    scaler :
        Scaler object with an ``inverse_transform`` method.
    config : TrainingConfig
        Experiment configuration.
    device : str or torch.device
        Device on which to run inference.
    train_losses, val_losses : list of float
        Per-epoch training and validation losses.
    fold_metrics : list of dict, optional
        Per-fold evaluation metrics from cross-validation.
    baseline_metrics : dict of dict, optional
        Metrics for each baseline model.
    results_dir : str
        Directory in which to write the PDF.
    timestamp : str, optional
        Timestamp string for file naming; generated if ``None``.

    Returns
    -------
    str
        Absolute path to the generated PDF file.
    """
    if timestamp is None:
        timestamp = get_maputo_timestamp()

    # Apply consistent plot style
    _apply_plot_style()

    reports_dir = os.path.join(results_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    pdf_path = os.path.join(reports_dir, f"traffic_report_{timestamp}.pdf")

    # ------------------------------------------------------------------
    # Collect predictions
    # ------------------------------------------------------------------
    all_predictions: List[np.ndarray] = []
    all_actuals: List[np.ndarray] = []
    attention_weights_collected: Optional[np.ndarray] = None

    model.eval()
    # PATCH: forward the adjacency matrix (when provided) so GNN-enabled
    # models produce the same predictions here as in evaluate_model.
    if adjacency_matrix is not None:
        adjacency_matrix = adjacency_matrix.to(device)
    try:
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    raw_inputs = batch[0]
                    # PATCH: dict batches (feature-attention mode).
                    if isinstance(raw_inputs, dict):
                        inputs = {k: v.to(device) for k, v in raw_inputs.items()}
                    else:
                        inputs = raw_inputs.to(device)
                    targets = batch[1].to(device)
                else:
                    inputs = batch.to(device)
                    targets = None

                if adjacency_matrix is not None:
                    outputs = model(inputs, adjacency_matrix=adjacency_matrix)
                else:
                    outputs = model(inputs)

                # Attempt to capture attention weights if model exposes them
                if hasattr(model, "attention_weights") and model.attention_weights is not None:
                    attn_w = model.attention_weights
                    if isinstance(attn_w, torch.Tensor):
                        attn_w = attn_w.detach().cpu().numpy()
                    if attention_weights_collected is None:
                        attention_weights_collected = attn_w
                    else:
                        try:
                            attention_weights_collected = (
                                attention_weights_collected + attn_w
                            ) / 2.0
                        except Exception:
                            pass

                # Move to CPU and inverse-transform
                preds_np = outputs.detach().cpu().numpy()
                if targets is not None:
                    targets_np = targets.detach().cpu().numpy()
                else:
                    targets_np = np.zeros_like(preds_np)

                # Inverse-transform if scaler is available
                if scaler is not None:
                    try:
                        orig_shape = preds_np.shape
                        preds_np = scaler.inverse_transform(
                            preds_np.reshape(-1, preds_np.shape[-1])
                        ).reshape(orig_shape)
                        targets_np = scaler.inverse_transform(
                            targets_np.reshape(-1, targets_np.shape[-1])
                        ).reshape(orig_shape)
                    except Exception:
                        # Scaler may not match shape; use raw values
                        pass

                all_predictions.append(preds_np)
                all_actuals.append(targets_np)

    except Exception as e:
        warnings.warn(f"Error during prediction collection: {e}")

    # Concatenate
    if all_predictions:
        predictions = np.concatenate(all_predictions, axis=0)
        actuals = np.concatenate(all_actuals, axis=0)
    else:
        predictions = np.array([0.0])
        actuals = np.array([0.0])

    # Free GPU memory before report generation
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Generate PDF sections
    # ------------------------------------------------------------------
    try:
        with PdfPages(pdf_path) as pdf:

            # 1. Title page
            try:
                _create_title_page(model, config, timestamp, pdf)
            except Exception as e:
                warnings.warn(f"Error creating title page: {e}")
                _create_error_page("Title Page", e, pdf)

            # 2. Training analysis
            try:
                _create_training_analysis(train_losses, val_losses, pdf)
            except Exception as e:
                warnings.warn(f"Error creating training analysis: {e}")
                _create_error_page("Training Analysis", e, pdf)

            # 3. Performance analysis
            try:
                _create_performance_analysis(predictions, actuals, pdf)
            except Exception as e:
                warnings.warn(f"Error creating performance analysis: {e}")
                _create_error_page("Performance Analysis", e, pdf)

            # 4. Attention analysis
            try:
                _create_attention_analysis(attention_weights_collected, config, pdf)
            except Exception as e:
                warnings.warn(f"Error creating attention analysis: {e}")
                _create_error_page("Attention Analysis", e, pdf)

            # 5. Cross-validation analysis
            try:
                _create_cross_validation_analysis(fold_metrics, pdf)
            except Exception as e:
                warnings.warn(f"Error creating cross-validation analysis: {e}")
                _create_error_page("Cross-Validation Analysis", e, pdf)

            # 6. Baseline comparison
            try:
                _create_baseline_comparison(baseline_metrics, fold_metrics, pdf)
            except Exception as e:
                warnings.warn(f"Error creating baseline comparison: {e}")
                _create_error_page("Baseline Comparison", e, pdf)

            # 7. Configuration summary
            try:
                _create_config_summary(config, pdf)
            except Exception as e:
                warnings.warn(f"Error creating config summary: {e}")
                _create_error_page("Configuration Summary", e, pdf)

            # 8. Summary statistics
            try:
                _create_summary_statistics(
                    predictions, actuals,
                    train_losses, val_losses,
                    fold_metrics, baseline_metrics,
                    pdf,
                )
            except Exception as e:
                warnings.warn(f"Error creating summary statistics: {e}")
                _create_error_page("Summary Statistics", e, pdf)

        print(f"Traffic report saved to: {pdf_path}")

    except Exception as e:
        warnings.warn(f"Full report generation failed: {e}. Creating emergency report.")
        _create_emergency_report(predictions, actuals, train_losses, val_losses, pdf_path)

    return pdf_path
