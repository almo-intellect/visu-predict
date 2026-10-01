"""
Visualization module for the traffic prediction transformer.

Provides comprehensive plotting functions for attention weight analysis,
feature attribution, prediction explanation, hidden state inspection,
training history, and model comparison.

Layer 6 in the dependency hierarchy.
"""

import os
import random
import pickle
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

from .utils import get_maputo_timestamp
from .config import TrainingConfig


# =============================================================================
# Constants
# =============================================================================

BOUNDARY_COLORS = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']


# =============================================================================
# Internal helpers
# =============================================================================

def _get_feature_groups(config: TrainingConfig) -> Dict[str, Tuple[int, int]]:
    """Return ``{group_name: (start_idx, end_idx)}`` derived from *config*.

    Groups are built in canonical order: traffic, time, holiday, weather,
    lagged, spatial.  Each group's extent depends on which features are
    enabled.
    """
    groups: Dict[str, Tuple[int, int]] = {}
    idx = 0

    num_sensors = getattr(config, 'num_sensors', 325)
    groups['traffic'] = (idx, idx + num_sensors)
    idx += num_sensors

    if getattr(config, 'use_time_features', False):
        time_dim = 4
        groups['time'] = (idx, idx + time_dim)
        idx += time_dim

    if getattr(config, 'use_holiday_feature', False):
        groups['holiday'] = (idx, idx + 1)
        idx += 1

    if getattr(config, 'use_weather_feature', False):
        weather_type = getattr(config, 'weather_feature_type', 'all_features')
        if weather_type == 'all_features':
            w_dim = 8
        elif weather_type == 'wind':
            w_dim = 2
        else:
            w_dim = 1
        groups['weather'] = (idx, idx + w_dim)
        idx += w_dim

    if getattr(config, 'use_lagged_features', False):
        num_lags = getattr(config, 'num_lags', 1)
        lag_dim = num_sensors * num_lags
        groups['lagged'] = (idx, idx + lag_dim)
        idx += lag_dim

    if getattr(config, 'use_spatial_features', False):
        spatial_dim = getattr(config, 'spatial_feature_dim', 336)
        groups['spatial'] = (idx, idx + spatial_dim)
        idx += spatial_dim

    return groups


def _get_model_details_text(config: TrainingConfig) -> str:
    """Build a human-readable model-details string for figure annotations."""
    lines = [
        f"Dataset: {config.dataset_name}",
        f"Hidden: {config.hidden_dim}  Heads: {config.num_heads}  "
        f"Layers: {config.num_layers}",
        f"Seq: {config.seq_length}  Pred: {config.pred_length}",
        f"Decoder: {config.decoder_type}",
        f"Dropout: {config.dropout}  LR: {config.learning_rate}",
    ]
    extras: List[str] = []
    if config.use_time_features:
        extras.append("time")
    if config.use_holiday_feature:
        extras.append("holiday")
    if config.use_weather_feature:
        extras.append(f"weather({config.weather_feature_type})")
    if config.use_lagged_features:
        extras.append(f"lagged(n={config.num_lags})")
    if getattr(config, 'use_spatial_features', False):
        extras.append("spatial")
    if extras:
        lines.append("Features: " + ", ".join(extras))
    return "\n".join(lines)


def _add_model_details_subplot(
    fig: plt.Figure,
    config: TrainingConfig,
    gs_position: Any = None,
) -> None:
    """Add a text-only subplot at the bottom of *fig* with model details."""
    if gs_position is not None:
        ax = fig.add_subplot(gs_position)
    else:
        ax = fig.add_axes([0.05, 0.01, 0.9, 0.06])
    ax.axis('off')
    text = _get_model_details_text(config)
    ax.text(
        0.5, 0.5, text,
        transform=ax.transAxes, ha='center', va='center', fontsize=7,
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                  edgecolor='gray', alpha=0.8),
    )


def _draw_feature_group_boundaries(
    ax: plt.Axes,
    feature_groups: Dict[str, Tuple[int, int]],
    limit: int,
) -> None:
    """Draw dashed boundary lines for each feature group on *ax*."""
    for gidx, (gname, (gstart, gend)) in enumerate(feature_groups.items()):
        color = BOUNDARY_COLORS[gidx % len(BOUNDARY_COLORS)]
        if gstart < limit and gstart > 0:
            ax.axhline(y=gstart, color=color, linewidth=0.8,
                       linestyle='--', alpha=0.7)
            ax.axvline(x=gstart, color=color, linewidth=0.8,
                       linestyle='--', alpha=0.7)


def _collect_encoder_attention(model: nn.Module) -> List[np.ndarray]:
    """Collect attention weights stored on each encoder layer."""
    encoder = getattr(model, 'transformer',
                      getattr(model, 'encoder', None))
    weights: List[np.ndarray] = []
    if encoder is not None:
        for layer in encoder:
            w = getattr(layer, 'attn_weights', None)
            if w is not None:
                if isinstance(w, torch.Tensor):
                    weights.append(w.detach().cpu().numpy())
                else:
                    weights.append(np.asarray(w))
    return weights


def _safe_attn_map(attn_w: np.ndarray, sample_idx: int = 0) -> np.ndarray:
    """Extract a 2-D attention map from a potentially higher-dim array."""
    if attn_w.ndim == 4:
        return attn_w[sample_idx].mean(axis=0)
    if attn_w.ndim == 3:
        return attn_w[sample_idx]
    if attn_w.ndim == 2:
        return attn_w
    s = int(math.sqrt(attn_w.size))
    return attn_w.flatten()[:s * s].reshape(s, s)


# =============================================================================
# 9. create_feature_mapping
# =============================================================================

def create_feature_mapping(config: TrainingConfig) -> Dict[int, str]:
    """Map feature indices to human-readable names.

    Builds a dictionary mapping each column index in the input tensor to a
    descriptive string.  The order mirrors the concatenation order used by
    the data module: traffic -> time -> holiday -> weather -> lagged ->
    spatial.

    Parameters
    ----------
    config : TrainingConfig
        Experiment configuration.

    Returns
    -------
    Dict[int, str]
        ``{column_index: human_name}`` for every position in the
        concatenated input vector.
    """
    mapping: Dict[int, str] = {}
    idx = 0

    num_sensors = getattr(config, 'num_sensors', 325)

    # Traffic features
    for s in range(num_sensors):
        mapping[idx] = f"sensor_{s}"
        idx += 1

    # Time features
    if getattr(config, 'use_time_features', False):
        for name in ['hour', 'day_of_week', 'week_of_year', 'month']:
            mapping[idx] = f"time_{name}"
            idx += 1

    # Holiday feature
    if getattr(config, 'use_holiday_feature', False):
        mapping[idx] = "holiday"
        idx += 1

    # Weather features
    if getattr(config, 'use_weather_feature', False):
        wt = getattr(config, 'weather_feature_type', 'all_features')
        if wt == 'all_features':
            names = ['temperature', 'weather_condition', 'visibility',
                     'wind_speed', 'wind_direction', 'humidity',
                     'dew_point', 'cloud_cover']
        elif wt == 'wind':
            names = ['wind_speed', 'wind_direction']
        else:
            names = [wt]
        for n in names:
            mapping[idx] = f"weather_{n}"
            idx += 1

    # Lagged features
    if getattr(config, 'use_lagged_features', False):
        num_lags = getattr(config, 'num_lags', 1)
        for lag in range(1, num_lags + 1):
            for s in range(num_sensors):
                mapping[idx] = f"lag{lag}_sensor_{s}"
                idx += 1

    # Spatial features
    if getattr(config, 'use_spatial_features', False):
        spatial_dim = getattr(config, 'spatial_feature_dim', 336)
        for i in range(spatial_dim):
            mapping[idx] = f"spatial_{i}"
            idx += 1

    return mapping


# =============================================================================
# 1. plot_attention_weights
# =============================================================================

def plot_attention_weights(
    model: nn.Module,
    seq_length: int,
    results_dir: str,
    config: TrainingConfig,
    show_feature_groups: bool = True,
    include_pairwise: bool = True,
) -> None:
    """Enhanced attention weight visualization.

    Produces:
    * Row 1 -- Per-layer encoder attention heatmaps with optional feature
      group boundary lines.
    * Row 2 (if ``FeatureAttention`` present) -- Feature importance bar
      chart, pairwise cross-attention grid, and feature gate values.
    * Final row -- Model details text box.

    Parameters
    ----------
    model : nn.Module
        A trained ``TrafficTransformer`` whose encoder layers store
        ``attn_weights`` after a forward pass.
    seq_length : int
        Sequence length for labelling / boundary alignment.
    results_dir : str
        Directory to save the figure.
    config : TrainingConfig
        Experiment configuration.
    show_feature_groups : bool
        Overlay boundary lines for feature groups on heatmaps.
    include_pairwise : bool
        Include the pairwise cross-attention subplot.
    """
    try:
        model.eval()
        timestamp = get_maputo_timestamp()
        os.makedirs(results_dir, exist_ok=True)

        # ---- Collect encoder attention weights ----
        encoder_weights = _collect_encoder_attention(model)
        if len(encoder_weights) == 0:
            print("No encoder attention weights available for "
                  "plot_attention_weights.")
            return

        num_layers = len(encoder_weights)
        feature_groups = _get_feature_groups(config)

        # ---- Feature-attention module (optional) ----
        has_fa = (hasattr(model, 'feature_attention')
                  and model.feature_attention is not None)
        extra_row = 1 if has_fa else 0
        extra_cols = 3 if has_fa else 0
        total_cols = max(num_layers, extra_cols, 1)
        total_rows = 1 + extra_row + 1  # layers, [fa], details

        fig = plt.figure(figsize=(6 * total_cols, 5 * total_rows))
        gs = GridSpec(total_rows, total_cols, figure=fig,
                      hspace=0.35, wspace=0.3)

        # ---- Row 1: encoder attention heatmaps ----
        for li, attn_w in enumerate(encoder_weights):
            ax = fig.add_subplot(gs[0, li])
            attn_map = _safe_attn_map(attn_w)
            sns.heatmap(attn_map, ax=ax, cmap='viridis', cbar=True,
                        cbar_kws={'shrink': 0.6})
            ax.set_title(f'Layer {li + 1} Attention', fontsize=10)
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')

            if show_feature_groups:
                _draw_feature_group_boundaries(
                    ax, feature_groups, attn_map.shape[0])

        # Fill remaining columns in row 1 if num_layers < total_cols
        for extra in range(num_layers, total_cols):
            fig.add_subplot(gs[0, extra]).axis('off')

        # ---- Row 2 (optional): FeatureAttention diagnostics ----
        if has_fa:
            fa = model.feature_attention

            # 2a -- Feature importance bar chart
            ax_imp = fig.add_subplot(gs[1, 0])
            importances = getattr(fa, 'feature_importances', None)
            if importances is not None:
                imp_np = importances.cpu().numpy()
                imp_mean = imp_np.mean(axis=0) if imp_np.ndim == 2 else imp_np
                f_names = fa.feature_names
                n_bars = min(len(f_names), len(imp_mean))
                bar_colors = [BOUNDARY_COLORS[i % len(BOUNDARY_COLORS)]
                              for i in range(n_bars)]
                ax_imp.barh(range(n_bars), imp_mean[:n_bars],
                            color=bar_colors, alpha=0.8)
                ax_imp.set_yticks(range(n_bars))
                ax_imp.set_yticklabels(f_names[:n_bars], fontsize=8)
                ax_imp.set_xlabel('Importance Weight')
                ax_imp.set_title('Feature Group Importance', fontsize=10)
                ax_imp.invert_yaxis()
            else:
                ax_imp.text(0.5, 0.5, 'No importance data',
                            ha='center', va='center',
                            transform=ax_imp.transAxes)
                ax_imp.set_title('Feature Group Importance', fontsize=10)

            # 2b -- Pairwise attention grid
            if include_pairwise and total_cols > 1:
                ax_pair = fig.add_subplot(gs[1, 1])
                pw_dict = getattr(fa, 'pairwise_weights', {})
                if pw_dict:
                    f_names = fa.feature_names
                    nf = len(f_names)
                    pair_matrix = np.zeros((nf, nf))
                    for pk, pw in pw_dict.items():
                        if pw is None:
                            continue
                        parts = pk.split('_to_')
                        if len(parts) == 2:
                            si = (f_names.index(parts[0])
                                  if parts[0] in f_names else -1)
                            ti = (f_names.index(parts[1])
                                  if parts[1] in f_names else -1)
                            if si >= 0 and ti >= 0:
                                pair_matrix[si, ti] = (
                                    pw.cpu().numpy().mean())
                    sns.heatmap(pair_matrix, ax=ax_pair, cmap='YlOrRd',
                                xticklabels=f_names, yticklabels=f_names,
                                annot=True, fmt='.3f', cbar=True,
                                cbar_kws={'shrink': 0.6})
                    ax_pair.set_title('Pairwise Cross-Attention',
                                      fontsize=10)
                    ax_pair.tick_params(axis='both', labelsize=7)
                else:
                    ax_pair.text(0.5, 0.5, 'No pairwise data',
                                ha='center', va='center',
                                transform=ax_pair.transAxes)
                    ax_pair.set_title('Pairwise Cross-Attention',
                                      fontsize=10)

            # 2c -- Feature gate values
            if total_cols > 2:
                ax_gate = fig.add_subplot(gs[1, 2])
                gate_vals = getattr(fa, 'gate_values', {})
                if gate_vals:
                    g_names, g_means, g_stds = [], [], []
                    for gn, gv in gate_vals.items():
                        if gv is not None:
                            g_names.append(gn)
                            gv_np = gv.cpu().numpy()
                            g_means.append(gv_np.mean())
                            g_stds.append(gv_np.std())
                    if g_names:
                        bar_c = [BOUNDARY_COLORS[i % len(BOUNDARY_COLORS)]
                                 for i in range(len(g_names))]
                        ax_gate.barh(range(len(g_names)), g_means,
                                     xerr=g_stds, color=bar_c,
                                     alpha=0.8, capsize=3)
                        ax_gate.set_yticks(range(len(g_names)))
                        ax_gate.set_yticklabels(g_names, fontsize=8)
                        ax_gate.set_xlabel('Gate Value (mean +/- std)')
                        ax_gate.set_title('Feature Gate Values',
                                          fontsize=10)
                        ax_gate.invert_yaxis()
                    else:
                        ax_gate.text(0.5, 0.5, 'No gate data',
                                     ha='center', va='center',
                                     transform=ax_gate.transAxes)
                        ax_gate.set_title('Feature Gate Values',
                                          fontsize=10)
                else:
                    ax_gate.text(0.5, 0.5, 'No gate data',
                                ha='center', va='center',
                                transform=ax_gate.transAxes)
                    ax_gate.set_title('Feature Gate Values', fontsize=10)

            # Fill remaining cols in feature-attention row
            for extra in range(3, total_cols):
                fig.add_subplot(gs[1, extra]).axis('off')

        # ---- Model details row ----
        _add_model_details_subplot(fig, config, gs[-1, :])

        fig.suptitle('Attention Weight Analysis', fontsize=14,
                     fontweight='bold', y=0.98)

        save_path = os.path.join(
            results_dir, f'attention_weights_{timestamp}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Attention weights plot saved to: {save_path}")

    except Exception as e:
        print(f"Error in plot_attention_weights: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# 2. visualize_decoder_attention
# =============================================================================

def visualize_decoder_attention(
    model: nn.Module,
    test_loader: DataLoader,
    results_dir: str,
    config: TrainingConfig,
    max_samples: int = 3,
    adjacency_matrix: Optional[torch.Tensor] = None,
) -> None:
    """Visualize self-attention and cross-attention patterns in the
    transformer decoder for multiple samples.

    For each sample a figure is produced with:
    * Row 1 -- Self-attention heatmap per decoder layer (Blues cmap).
    * Row 2 -- Cross-attention heatmap per decoder layer (Reds cmap).
    * Row 3 -- Model details text box.

    Parameters
    ----------
    model : nn.Module
        Trained ``TrafficTransformer`` with ``decoder_type='transformer'``.
    test_loader : DataLoader
        Test data providing ``(input, target)`` batches.
    results_dir : str
        Directory to save figures.
    config : TrainingConfig
        Experiment configuration.
    max_samples : int
        Maximum number of samples to visualize.
    adjacency_matrix : torch.Tensor, optional
        Adjacency matrix for spatial models.
    """
    try:
        model.eval()
        timestamp = get_maputo_timestamp()
        os.makedirs(results_dir, exist_ok=True)

        decoder_type = getattr(model, 'decoder_type', 'linear')
        if decoder_type != 'transformer':
            print(f"Decoder type is '{decoder_type}'; skipping "
                  "decoder attention visualization.")
            return

        device = next(model.parameters()).device
        sample_count = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                if sample_count >= max_samples:
                    break

                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass to populate decoder attention weights
                kwargs: Dict[str, Any] = {}
                if adjacency_matrix is not None:
                    kwargs['adjacency_matrix'] = adjacency_matrix.to(device)
                _ = model(inputs, tgt=targets, **kwargs)

                decoder = getattr(model, 'decoder', None)
                if decoder is None:
                    print("No decoder attribute found on model.")
                    return

                sa_list = getattr(decoder, 'self_attn_weights_list', [])
                ca_list = getattr(decoder, 'cross_attn_weights_list', [])

                if not sa_list and not ca_list:
                    print("No decoder attention weights captured.")
                    return

                n_dec_layers = max(len(sa_list), len(ca_list))
                bs = inputs.size(0)

                for si in range(min(bs, max_samples - sample_count)):
                    fig = plt.figure(
                        figsize=(7 * n_dec_layers, 12))
                    gs = GridSpec(3, n_dec_layers, figure=fig,
                                 hspace=0.4, wspace=0.3)

                    # Row 1 -- self-attention
                    for li in range(n_dec_layers):
                        ax = fig.add_subplot(gs[0, li])
                        if li < len(sa_list) and sa_list[li] is not None:
                            sa = sa_list[li].cpu().numpy()
                            sa_map = _safe_attn_map(sa, si)
                            sns.heatmap(sa_map, ax=ax, cmap='Blues',
                                        cbar=True,
                                        cbar_kws={'shrink': 0.6})
                        else:
                            ax.text(0.5, 0.5, 'N/A', ha='center',
                                    va='center', transform=ax.transAxes)
                        ax.set_title(f'Dec L{li + 1} Self-Attn',
                                     fontsize=9)
                        ax.set_xlabel('Key')
                        ax.set_ylabel('Query')

                    # Row 2 -- cross-attention
                    for li in range(n_dec_layers):
                        ax = fig.add_subplot(gs[1, li])
                        if li < len(ca_list) and ca_list[li] is not None:
                            ca = ca_list[li].cpu().numpy()
                            ca_map = _safe_attn_map(ca, si)
                            sns.heatmap(ca_map, ax=ax, cmap='Reds',
                                        cbar=True,
                                        cbar_kws={'shrink': 0.6})
                        else:
                            ax.text(0.5, 0.5, 'N/A', ha='center',
                                    va='center', transform=ax.transAxes)
                        ax.set_title(f'Dec L{li + 1} Cross-Attn',
                                     fontsize=9)
                        ax.set_xlabel('Encoder Position')
                        ax.set_ylabel('Decoder Position')

                    # Row 3 -- model details
                    _add_model_details_subplot(fig, config, gs[2, :])

                    fig.suptitle(
                        f'Decoder Attention - Sample {sample_count + 1}',
                        fontsize=13, fontweight='bold', y=0.98)

                    save_path = os.path.join(
                        results_dir,
                        f'decoder_attention_sample'
                        f'{sample_count + 1}_{timestamp}.png')
                    fig.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"Decoder attention plot saved to: {save_path}")

                    sample_count += 1
                    if sample_count >= max_samples:
                        break

    except Exception as e:
        print(f"Error in visualize_decoder_attention: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# 3. plot_layer_progression
# =============================================================================

def plot_layer_progression(
    model: nn.Module,
    input_batch: torch.Tensor,
    results_dir: str,
    config: TrainingConfig,
    show_feature_groups: bool = True,
    adjacency_matrix: Optional[torch.Tensor] = None,
) -> None:
    """Show attention pattern evolution across encoder layers with
    difference maps between consecutive layers.

    Produces:
    * Row 1 -- Attention heatmap per encoder layer.
    * Row 2 -- Difference heatmap ``layer[i+1] - layer[i]``.
    * Final row -- Model details text box.

    Parameters
    ----------
    model : nn.Module
        Trained ``TrafficTransformer``.
    input_batch : torch.Tensor
        Input batch ``(batch, seq_len, input_dim)``.
    results_dir : str
        Directory to save the figure.
    config : TrainingConfig
        Experiment configuration.
    show_feature_groups : bool
        Overlay feature group boundary lines.
    adjacency_matrix : torch.Tensor, optional
        Adjacency matrix for spatial models.
    """
    try:
        model.eval()
        timestamp = get_maputo_timestamp()
        os.makedirs(results_dir, exist_ok=True)
        device = next(model.parameters()).device
        input_batch = input_batch.to(device)

        # Forward pass
        with torch.no_grad():
            kwargs: Dict[str, Any] = {}
            if adjacency_matrix is not None:
                kwargs['adjacency_matrix'] = adjacency_matrix.to(device)
            _ = model(input_batch, **kwargs)

        layer_weights = _collect_encoder_attention(model)
        if len(layer_weights) == 0:
            print("No attention weights captured for layer progression.")
            return

        num_layers = len(layer_weights)
        num_diff = max(num_layers - 1, 0)
        has_diff_row = num_diff > 0
        total_cols = max(num_layers, 1)
        total_rows = (1 + int(has_diff_row) + 1)  # maps, [diffs], details

        fig = plt.figure(figsize=(6 * total_cols, 5 * total_rows))
        gs = GridSpec(total_rows, total_cols, figure=fig,
                      hspace=0.4, wspace=0.3)
        feature_groups = _get_feature_groups(config)

        # Row 1 -- attention maps
        processed_maps: List[np.ndarray] = []
        for li, attn_w in enumerate(layer_weights):
            ax = fig.add_subplot(gs[0, li])
            attn_map = _safe_attn_map(attn_w)
            processed_maps.append(attn_map)
            sns.heatmap(attn_map, ax=ax, cmap='viridis', cbar=True,
                        cbar_kws={'shrink': 0.6})
            ax.set_title(f'Layer {li + 1}', fontsize=10)
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            if show_feature_groups:
                _draw_feature_group_boundaries(
                    ax, feature_groups, attn_map.shape[0])

        for extra in range(num_layers, total_cols):
            fig.add_subplot(gs[0, extra]).axis('off')

        # Row 2 -- difference maps
        if has_diff_row:
            for di in range(num_diff):
                ax = fig.add_subplot(gs[1, di])
                ma = processed_maps[di]
                mb = processed_maps[di + 1]
                mr = min(ma.shape[0], mb.shape[0])
                mc = min(ma.shape[1], mb.shape[1])
                diff = mb[:mr, :mc] - ma[:mr, :mc]
                sns.heatmap(diff, ax=ax, cmap='RdBu_r', center=0,
                            cbar=True, cbar_kws={'shrink': 0.6})
                ax.set_title(f'Diff: L{di + 2} - L{di + 1}', fontsize=10)
                ax.set_xlabel('Key Position')
                ax.set_ylabel('Query Position')
            for extra in range(num_diff, total_cols):
                fig.add_subplot(gs[1, extra]).axis('off')

        _add_model_details_subplot(fig, config, gs[-1, :])

        fig.suptitle('Attention Layer Progression', fontsize=14,
                     fontweight='bold', y=0.98)
        save_path = os.path.join(
            results_dir, f'layer_progression_{timestamp}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Layer progression plot saved to: {save_path}")

    except Exception as e:
        print(f"Error in plot_layer_progression: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# 4. visualize_feature_attribution
# =============================================================================

def visualize_feature_attribution(
    model: nn.Module,
    test_loader: DataLoader,
    config: TrainingConfig,
    results_dir: str,
    adjacency_matrix: Optional[torch.Tensor] = None,
    n_samples: int = 10,
    sensor_idx: int = 0,
) -> None:
    """Gradient-based feature attribution visualization by group and
    detailed.

    Computes input gradients w.r.t. the model output at *sensor_idx*,
    aggregates across samples, and produces:
    * Group-level importance bar chart.
    * Top-30 individual feature attribution bar chart.
    * Model details text box.

    Parameters
    ----------
    model : nn.Module
        Trained ``TrafficTransformer``.
    test_loader : DataLoader
        Test data loader.
    config : TrainingConfig
        Experiment configuration.
    results_dir : str
        Directory to save the figure.
    adjacency_matrix : torch.Tensor, optional
        Adjacency matrix for spatial models.
    n_samples : int
        Number of samples to aggregate gradients over.
    sensor_idx : int
        Which output sensor to attribute.
    """
    try:
        model.eval()
        timestamp = get_maputo_timestamp()
        os.makedirs(results_dir, exist_ok=True)
        device = next(model.parameters()).device
        feature_groups = _get_feature_groups(config)
        feature_mapping = create_feature_mapping(config)

        all_grads: List[np.ndarray] = []
        sample_count = 0

        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if sample_count >= n_samples:
                break
            inputs = inputs.to(device).requires_grad_(True)

            kwargs: Dict[str, Any] = {}
            if adjacency_matrix is not None:
                kwargs['adjacency_matrix'] = adjacency_matrix.to(device)
            output = model(inputs, **kwargs)

            bs = output.size(0)
            for si in range(min(bs, n_samples - sample_count)):
                model.zero_grad()
                if inputs.grad is not None:
                    inputs.grad.zero_()
                target_val = output[si, :, sensor_idx].mean()
                target_val.backward(retain_graph=True)

                if inputs.grad is not None:
                    g = inputs.grad[si].detach().cpu().numpy()
                    all_grads.append(g)
                    sample_count += 1
                if sample_count >= n_samples:
                    break

        if not all_grads:
            print("No gradients computed for feature attribution.")
            return

        avg_abs = np.mean([np.abs(g) for g in all_grads], axis=0)
        avg_per_feat = avg_abs.mean(axis=0)

        # Group-level
        g_names, g_imps = [], []
        for gname, (gs_start, gs_end) in feature_groups.items():
            ae = min(gs_end, len(avg_per_feat))
            if gs_start < len(avg_per_feat):
                g_names.append(gname)
                g_imps.append(avg_per_feat[gs_start:ae].mean())

        fig = plt.figure(figsize=(14, 16))
        gs_fig = GridSpec(3, 1, figure=fig, height_ratios=[1, 2, 0.3],
                          hspace=0.4)

        # Subplot 1 -- group importance
        ax1 = fig.add_subplot(gs_fig[0])
        if g_names:
            bc = [BOUNDARY_COLORS[i % len(BOUNDARY_COLORS)]
                  for i in range(len(g_names))]
            bars = ax1.barh(range(len(g_names)), g_imps, color=bc,
                            alpha=0.85)
            ax1.set_yticks(range(len(g_names)))
            ax1.set_yticklabels(g_names, fontsize=9)
            ax1.set_xlabel('Mean |Gradient|', fontsize=10)
            ax1.set_title(
                f'Feature Group Attribution (Sensor {sensor_idx})',
                fontsize=12)
            ax1.invert_yaxis()
            for bar, val in zip(bars, g_imps):
                ax1.text(bar.get_width() + 0.001,
                         bar.get_y() + bar.get_height() / 2,
                         f'{val:.4f}', va='center', fontsize=8)

        # Subplot 2 -- top-30 individual features
        ax2 = fig.add_subplot(gs_fig[1])
        top_k = min(30, len(avg_per_feat))
        top_idx = np.argsort(avg_per_feat)[-top_k:][::-1]
        top_vals = avg_per_feat[top_idx]
        labels, colors_d = [], []
        for fi in top_idx:
            labels.append(feature_mapping.get(fi, f'feat_{fi}'))
            c = 'gray'
            for gi, (gn, (gs2, ge2)) in enumerate(feature_groups.items()):
                if gs2 <= fi < ge2:
                    c = BOUNDARY_COLORS[gi % len(BOUNDARY_COLORS)]
                    break
            colors_d.append(c)
        ax2.barh(range(top_k), top_vals, color=colors_d, alpha=0.8)
        ax2.set_yticks(range(top_k))
        ax2.set_yticklabels(labels, fontsize=7)
        ax2.set_xlabel('Mean |Gradient|', fontsize=10)
        ax2.set_title(
            f'Top {top_k} Feature Attributions (Sensor {sensor_idx})',
            fontsize=12)
        ax2.invert_yaxis()

        _add_model_details_subplot(fig, config, gs_fig[2])
        fig.suptitle('Gradient-Based Feature Attribution', fontsize=14,
                     fontweight='bold', y=0.98)
        save_path = os.path.join(
            results_dir, f'feature_attribution_{timestamp}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Feature attribution plot saved to: {save_path}")

    except Exception as e:
        print(f"Error in visualize_feature_attribution: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# 5. visualize_prediction_explanation
# =============================================================================

def visualize_prediction_explanation(
    model: nn.Module,
    test_loader: DataLoader,
    data_scaler: Any,
    config: TrainingConfig,
    results_dir: str,
    adjacency_matrix: Optional[torch.Tensor] = None,
    sensor_idx: int = 0,
    pred_timestep: int = 0,
    batch_idx: int = 0,
    num_examples: int = 3,
) -> None:
    """Gradient-based explanation of specific predictions.

    For each example produces a multi-panel figure:
    * Input time series at the target sensor.
    * Predicted vs actual values with the explained timestep marked.
    * Gradient-attribution heatmap (time x top features).
    * Group contribution pie chart.
    * Temporal attribution profile bar chart.

    Parameters
    ----------
    model : nn.Module
        Trained ``TrafficTransformer``.
    test_loader : DataLoader
        Test data loader.
    data_scaler : object
        Scaler with ``inverse_transform`` for de-normalising.
    config : TrainingConfig
        Experiment configuration.
    results_dir : str
        Directory to save figures.
    adjacency_matrix : torch.Tensor, optional
        Adjacency matrix for spatial models.
    sensor_idx : int
        Sensor to explain.
    pred_timestep : int
        Which prediction timestep to explain.
    batch_idx : int
        Which batch to start from.
    num_examples : int
        How many example figures to produce.
    """
    try:
        model.eval()
        timestamp = get_maputo_timestamp()
        os.makedirs(results_dir, exist_ok=True)
        device = next(model.parameters()).device
        feature_groups = _get_feature_groups(config)
        feature_mapping = create_feature_mapping(config)

        example_count = 0

        for bi, (inputs, targets) in enumerate(test_loader):
            if bi < batch_idx:
                continue
            if example_count >= num_examples:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)
            bs = inputs.size(0)

            for si in range(min(bs, num_examples - example_count)):
                single_in = (inputs[si:si + 1].clone()
                             .detach().requires_grad_(True))
                single_tgt = targets[si:si + 1]

                kwargs: Dict[str, Any] = {}
                if adjacency_matrix is not None:
                    kwargs['adjacency_matrix'] = (
                        adjacency_matrix.to(device))
                output = model(single_in, **kwargs)

                pred_val = output[0, pred_timestep, sensor_idx]
                model.zero_grad()
                if single_in.grad is not None:
                    single_in.grad.zero_()
                pred_val.backward(retain_graph=True)

                grad = (single_in.grad[0].detach().cpu().numpy()
                        if single_in.grad is not None else None)
                input_np = single_in[0].detach().cpu().numpy()
                output_np = output[0].detach().cpu().numpy()
                target_np = single_tgt[0].detach().cpu().numpy()

                # Inverse-transform for display
                try:
                    pred_s = output_np[:, sensor_idx]
                    act_s = target_np[:, sensor_idx]
                    if (data_scaler is not None
                            and hasattr(data_scaler, 'inverse_transform')):
                        nf = output_np.shape[-1]
                        p_full = np.zeros((len(pred_s), nf))
                        p_full[:, sensor_idx] = pred_s
                        a_full = np.zeros((len(act_s), nf))
                        a_full[:, sensor_idx] = act_s
                        try:
                            pred_inv = data_scaler.inverse_transform(
                                p_full)[:, sensor_idx]
                            act_inv = data_scaler.inverse_transform(
                                a_full)[:, sensor_idx]
                        except Exception:
                            pred_inv, act_inv = pred_s, act_s
                    else:
                        pred_inv, act_inv = pred_s, act_s
                except Exception:
                    pred_inv = output_np[:, sensor_idx]
                    act_inv = target_np[:, sensor_idx]

                # ---- Figure ----
                fig = plt.figure(figsize=(16, 14))
                gs_fig = GridSpec(4, 2, figure=fig, hspace=0.45,
                                 wspace=0.3,
                                 height_ratios=[1, 1.5, 1, 0.3])

                # 1a -- input time series
                ax1 = fig.add_subplot(gs_fig[0, 0])
                in_s = (input_np[:, sensor_idx]
                        if sensor_idx < input_np.shape[1]
                        else input_np[:, 0])
                ax1.plot(in_s, 'b-o', markersize=3, linewidth=1.2,
                         label='Input')
                ax1.set_title(
                    f'Input Time Series (Sensor {sensor_idx})',
                    fontsize=10)
                ax1.set_xlabel('Time Step')
                ax1.set_ylabel('Value')
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3)

                # 1b -- predicted vs actual
                ax2 = fig.add_subplot(gs_fig[0, 1])
                steps = range(len(pred_inv))
                ax2.plot(steps, act_inv, 'g-o', markersize=4,
                         linewidth=1.2, label='Actual')
                ax2.plot(steps, pred_inv, 'r-s', markersize=4,
                         linewidth=1.2, label='Predicted')
                ax2.axvline(x=pred_timestep, color='black',
                            linestyle='--', alpha=0.5,
                            label=f'Explained t={pred_timestep}')
                ax2.set_title(
                    f'Prediction vs Actual (Sensor {sensor_idx})',
                    fontsize=10)
                ax2.set_xlabel('Prediction Step')
                ax2.set_ylabel('Value')
                ax2.legend(fontsize=8)
                ax2.grid(True, alpha=0.3)

                # 2 -- gradient attribution heatmap
                ax3 = fig.add_subplot(gs_fig[1, :])
                if grad is not None:
                    abs_g = np.abs(grad)
                    nf_show = min(50, abs_g.shape[1])
                    top_fi = np.argsort(
                        abs_g.mean(axis=0))[-nf_show:][::-1]
                    grad_sub = abs_g[:, top_fi]
                    fl = [feature_mapping.get(f, f'f{f}')
                          for f in top_fi]
                    sns.heatmap(
                        grad_sub.T, ax=ax3, cmap='hot', cbar=True,
                        cbar_kws={'shrink': 0.6, 'label': '|Gradient|'},
                        yticklabels=fl)
                    ax3.set_xlabel('Time Step')
                    ax3.set_ylabel('Feature')
                    ax3.set_title(
                        f'Gradient Attribution Heatmap '
                        f'(top {nf_show} features)', fontsize=10)
                    ax3.tick_params(axis='y', labelsize=6)
                else:
                    ax3.text(0.5, 0.5, 'No gradient data',
                             ha='center', va='center',
                             transform=ax3.transAxes)

                # 3a -- group contribution pie
                ax4 = fig.add_subplot(gs_fig[2, 0])
                if grad is not None:
                    abf = np.abs(grad).mean(axis=0)
                    gn_l, gv_l = [], []
                    for gn, (gs2, ge2) in feature_groups.items():
                        ae = min(ge2, len(abf))
                        if gs2 < len(abf):
                            gn_l.append(gn)
                            gv_l.append(abf[gs2:ae].sum())
                    if gn_l:
                        bc = [BOUNDARY_COLORS[i % len(BOUNDARY_COLORS)]
                              for i in range(len(gn_l))]
                        ax4.pie(gv_l, labels=gn_l, colors=bc,
                                autopct='%1.1f%%', startangle=90,
                                textprops={'fontsize': 8})
                        ax4.set_title('Group Contribution', fontsize=10)
                else:
                    ax4.text(0.5, 0.5, 'No gradient data',
                             ha='center', va='center',
                             transform=ax4.transAxes)

                # 3b -- temporal attribution profile
                ax5 = fig.add_subplot(gs_fig[2, 1])
                if grad is not None:
                    temp_imp = np.abs(grad).mean(axis=1)
                    ax5.bar(range(len(temp_imp)), temp_imp,
                            color='steelblue', alpha=0.8)
                    ax5.set_xlabel('Input Time Step')
                    ax5.set_ylabel('Mean |Gradient|')
                    ax5.set_title('Temporal Attribution Profile',
                                  fontsize=10)
                    ax5.grid(True, alpha=0.3)
                else:
                    ax5.text(0.5, 0.5, 'No gradient data',
                             ha='center', va='center',
                             transform=ax5.transAxes)

                _add_model_details_subplot(fig, config, gs_fig[3, :])

                fig.suptitle(
                    f'Prediction Explanation - Example '
                    f'{example_count + 1} '
                    f'(Sensor {sensor_idx}, t={pred_timestep})',
                    fontsize=13, fontweight='bold', y=0.99)

                save_path = os.path.join(
                    results_dir,
                    f'prediction_explanation_ex'
                    f'{example_count + 1}_{timestamp}.png')
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"Prediction explanation saved to: {save_path}")

                example_count += 1
                if example_count >= num_examples:
                    break

    except Exception as e:
        print(f"Error in visualize_prediction_explanation: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# 6. visualize_hidden_states
# =============================================================================

def visualize_hidden_states(
    model: nn.Module,
    input_batch: torch.Tensor,
    results_dir: str,
    config: TrainingConfig,
) -> None:
    """Visualize hidden state properties across encoder layers.

    Produces a 3x3 grid:
    * (0,0) PCA projection (2-D) of per-layer hidden states.
    * (0,1) Box plot of hidden state norms by layer.
    * (0,2) Mean variance per layer bar chart.
    * (1,0) Cosine similarity between consecutive layers.
    * (1,1) Attention-hidden-state correlation (entropy vs norm).
    * (1,2) Dimension-wise activation distribution sample.
    * (2,0:2) 3-D PCA trajectory of hidden states.
    * (2,2) Model details text box.

    Uses ``register_forward_hook`` to capture intermediate
    representations.

    Parameters
    ----------
    model : nn.Module
        Trained ``TrafficTransformer``.
    input_batch : torch.Tensor
        Input batch ``(batch, seq_len, input_dim)``.
    results_dir : str
        Directory to save the figure.
    config : TrainingConfig
        Experiment configuration.
    """
    try:
        model.eval()
        timestamp = get_maputo_timestamp()
        os.makedirs(results_dir, exist_ok=True)
        device = next(model.parameters()).device
        input_batch = input_batch.to(device)

        # ---- Hook-based capture ----
        hidden_states: List[torch.Tensor] = []
        hooks: List[Any] = []

        encoder = getattr(model, 'transformer',
                          getattr(model, 'encoder', None))
        if encoder is None:
            print("No encoder found for hidden state visualization.")
            return

        def _make_hook(layer_idx: int):
            def hook_fn(module: nn.Module,
                        inp: Any, out: Any) -> None:
                t = out[0] if isinstance(out, tuple) else out
                hidden_states.append(t.detach().cpu())
            return hook_fn

        for li, layer in enumerate(encoder):
            hooks.append(layer.register_forward_hook(_make_hook(li)))

        with torch.no_grad():
            _ = model(input_batch)

        for h in hooks:
            h.remove()

        if not hidden_states:
            print("No hidden states captured.")
            return

        num_layers = len(hidden_states)

        # Prepare per-layer 2-D arrays (seq_len, d_model) from first sample
        hs_list: List[np.ndarray] = []
        for hs in hidden_states:
            if hs.ndim == 3:
                hs_list.append(hs[:, 0, :].numpy())
            elif hs.ndim == 2:
                hs_list.append(hs.numpy())
            else:
                hs_list.append(
                    hs.reshape(-1, hs.shape[-1]).numpy())

        fig = plt.figure(figsize=(20, 18))
        gs_fig = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

        # ---- (0,0) PCA 2-D ----
        ax_pca = fig.add_subplot(gs_fig[0, 0])
        try:
            from sklearn.decomposition import PCA
            pca_colors = plt.cm.viridis(
                np.linspace(0, 1, num_layers))
            for li, hs in enumerate(hs_list):
                if hs.shape[0] >= 2 and hs.shape[1] >= 2:
                    pca = PCA(n_components=2)
                    proj = pca.fit_transform(hs)
                    ax_pca.scatter(proj[:, 0], proj[:, 1],
                                   c=[pca_colors[li]], s=15, alpha=0.7,
                                   label=f'Layer {li + 1}')
                    ax_pca.plot(proj[:, 0], proj[:, 1],
                                color=pca_colors[li], alpha=0.3,
                                linewidth=0.8)
            ax_pca.set_title('PCA of Hidden States', fontsize=10)
            ax_pca.set_xlabel('PC1')
            ax_pca.set_ylabel('PC2')
            ax_pca.legend(fontsize=7, loc='best')
            ax_pca.grid(True, alpha=0.3)
        except ImportError:
            ax_pca.text(0.5, 0.5, 'sklearn not available',
                        ha='center', va='center',
                        transform=ax_pca.transAxes)
            ax_pca.set_title('PCA of Hidden States', fontsize=10)

        # ---- (0,1) Hidden state norms ----
        ax_norms = fig.add_subplot(gs_fig[0, 1])
        norms = [np.linalg.norm(hs, axis=1) for hs in hs_list]
        bp = ax_norms.boxplot(
            norms,
            labels=[f'L{i + 1}' for i in range(num_layers)],
            patch_artist=True)
        box_colors = plt.cm.Set2(np.linspace(0, 1, num_layers))
        for patch, col in zip(bp['boxes'], box_colors):
            patch.set_facecolor(col)
        ax_norms.set_title('Hidden State Norms by Layer', fontsize=10)
        ax_norms.set_xlabel('Layer')
        ax_norms.set_ylabel('L2 Norm')
        ax_norms.grid(True, alpha=0.3)

        # ---- (0,2) Mean variance per layer ----
        ax_var = fig.add_subplot(gs_fig[0, 2])
        variances = [np.var(hs, axis=0).mean() for hs in hs_list]
        ax_var.bar(range(1, num_layers + 1), variances,
                   color=plt.cm.Set2(np.linspace(0, 1, num_layers)),
                   alpha=0.85)
        ax_var.set_title('Mean Variance by Layer', fontsize=10)
        ax_var.set_xlabel('Layer')
        ax_var.set_ylabel('Mean Variance')
        ax_var.grid(True, alpha=0.3)

        # ---- (1,0) Cosine similarity between consecutive layers ----
        ax_sim = fig.add_subplot(gs_fig[1, 0])
        if num_layers >= 2:
            sims, pairs = [], []
            for li in range(num_layers - 1):
                a, b = hs_list[li], hs_list[li + 1]
                ml = min(a.shape[0], b.shape[0])
                cs = []
                for t in range(ml):
                    na = np.linalg.norm(a[t])
                    nb = np.linalg.norm(b[t])
                    if na > 0 and nb > 0:
                        cs.append(np.dot(a[t], b[t]) / (na * nb))
                    else:
                        cs.append(0.0)
                sims.append(np.mean(cs))
                pairs.append(f'L{li + 1}-L{li + 2}')
            ax_sim.bar(range(len(sims)), sims, color='coral',
                       alpha=0.85)
            ax_sim.set_xticks(range(len(sims)))
            ax_sim.set_xticklabels(pairs, fontsize=8)
            ax_sim.set_ylabel('Mean Cosine Similarity')
            ax_sim.set_ylim(0, 1.05)
            ax_sim.grid(True, alpha=0.3)
        else:
            ax_sim.text(0.5, 0.5, 'Need >= 2 layers',
                        ha='center', va='center',
                        transform=ax_sim.transAxes)
        ax_sim.set_title('Cosine Similarity Between Layers',
                         fontsize=10)

        # ---- (1,1) Attention-hidden correlation ----
        ax_corr = fig.add_subplot(gs_fig[1, 1])
        attn_w_list = _collect_encoder_attention(model)
        if attn_w_list and len(attn_w_list) == num_layers:
            corrs = []
            for li in range(num_layers):
                hs = hs_list[li]
                aw = _safe_attn_map(attn_w_list[li])
                sl = min(hs.shape[0], aw.shape[0])
                hn = np.linalg.norm(hs[:sl], axis=1)
                ent = []
                for q in range(sl):
                    row = aw[q, :sl] + 1e-10
                    row = row / row.sum()
                    ent.append(-np.sum(row * np.log(row + 1e-10)))
                ent_arr = np.array(ent)
                if len(hn) > 1:
                    c = np.corrcoef(hn, ent_arr)[0, 1]
                    corrs.append(0.0 if np.isnan(c) else c)
                else:
                    corrs.append(0.0)
            ax_corr.bar(range(1, num_layers + 1), corrs,
                        color='mediumseagreen', alpha=0.85)
            ax_corr.axhline(y=0, color='gray', linestyle='--',
                            linewidth=0.5)
            ax_corr.set_xlabel('Layer')
            ax_corr.set_ylabel('Pearson Correlation')
            ax_corr.grid(True, alpha=0.3)
        else:
            ax_corr.text(0.5, 0.5, 'Attention data unavailable',
                         ha='center', va='center',
                         transform=ax_corr.transAxes)
        ax_corr.set_title('Attention-Hidden Correlation', fontsize=10)

        # ---- (1,2) Dimension-wise activation distribution ----
        ax_dist = fig.add_subplot(gs_fig[1, 2])
        d_model = hs_list[0].shape[1]
        n_dim_s = min(10, d_model)
        dim_idx = sorted(random.sample(range(d_model), n_dim_s))
        box_data, box_labels = [], []
        for li in range(num_layers):
            for di in dim_idx:
                box_data.append(hs_list[li][:, di])
                box_labels.append(f'L{li + 1}d{di}')
        n_show = min(20, len(box_data))
        if n_show > 0:
            bp2 = ax_dist.boxplot(
                box_data[:n_show],
                labels=box_labels[:n_show],
                patch_artist=True)
            for i, patch in enumerate(bp2['boxes']):
                patch.set_facecolor(plt.cm.tab20(i / max(n_show, 1)))
            ax_dist.tick_params(axis='x', rotation=90, labelsize=6)
            ax_dist.grid(True, alpha=0.3)
        ax_dist.set_title('Activation Distribution Sample', fontsize=10)

        # ---- (2,0:2) 3-D PCA trajectory ----
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            from sklearn.decomposition import PCA
            ax_3d = fig.add_subplot(gs_fig[2, 0:2], projection='3d')
            c3d = plt.cm.viridis(np.linspace(0, 1, num_layers))
            for li, hs in enumerate(hs_list):
                if hs.shape[0] >= 3 and hs.shape[1] >= 3:
                    pca3 = PCA(n_components=3)
                    p3 = pca3.fit_transform(hs)
                    ax_3d.plot(p3[:, 0], p3[:, 1], p3[:, 2],
                               color=c3d[li], alpha=0.7,
                               linewidth=1.2,
                               label=f'Layer {li + 1}')
                    ax_3d.scatter(p3[:, 0], p3[:, 1], p3[:, 2],
                                  c=[c3d[li]], s=10, alpha=0.5)
            ax_3d.set_title('3D Hidden State Trajectories', fontsize=10)
            ax_3d.set_xlabel('PC1', fontsize=8)
            ax_3d.set_ylabel('PC2', fontsize=8)
            ax_3d.set_zlabel('PC3', fontsize=8)
            ax_3d.legend(fontsize=7, loc='best')
        except Exception as exc:
            ax_fb = fig.add_subplot(gs_fig[2, 0:2])
            ax_fb.text(0.5, 0.5, f'3D plot unavailable: {exc}',
                       ha='center', va='center',
                       transform=ax_fb.transAxes)
            ax_fb.set_title('3D Hidden State Trajectories', fontsize=10)

        # ---- (2,2) Model details ----
        ax_det = fig.add_subplot(gs_fig[2, 2])
        ax_det.axis('off')
        ax_det.text(
            0.5, 0.5, _get_model_details_text(config),
            transform=ax_det.transAxes, ha='center', va='center',
            fontsize=7, family='monospace',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='lightyellow',
                      edgecolor='gray', alpha=0.8))

        fig.suptitle('Hidden State Analysis', fontsize=14,
                     fontweight='bold', y=0.99)
        save_path = os.path.join(
            results_dir, f'hidden_states_{timestamp}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Hidden states plot saved to: {save_path}")

    except Exception as e:
        print(f"Error in visualize_hidden_states: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# 7. plot_attention_heads
# =============================================================================

def plot_attention_heads(
    model: nn.Module,
    input_batch: torch.Tensor,
    results_dir: str,
    config: TrainingConfig,
    layer_idx: int = 0,
    feature_names: Optional[List[str]] = None,
    adjacency_matrix: Optional[torch.Tensor] = None,
) -> None:
    """Individual head visualization with specialization analysis.

    Produces:
    * Grid of per-head attention maps for *layer_idx*.
    * Entropy-based specialization bar chart (low entropy = specialized,
      high entropy = distributed).
    * Model details text box.

    Uses a temporary monkey-patch of the target layer's ``_sa_block``
    to retrieve per-head (non-averaged) attention weights.

    Parameters
    ----------
    model : nn.Module
        Trained ``TrafficTransformer``.
    input_batch : torch.Tensor
        Input batch ``(batch, seq_len, input_dim)``.
    results_dir : str
        Directory to save the figure.
    config : TrainingConfig
        Experiment configuration.
    layer_idx : int
        Which encoder layer to inspect (0-indexed).
    feature_names : list of str, optional
        Custom labels for the axes.
    adjacency_matrix : torch.Tensor, optional
        Adjacency matrix for spatial models.
    """
    try:
        model.eval()
        timestamp = get_maputo_timestamp()
        os.makedirs(results_dir, exist_ok=True)
        device = next(model.parameters()).device
        input_batch = input_batch.to(device)

        encoder = getattr(model, 'transformer',
                          getattr(model, 'encoder', None))
        if encoder is None:
            print("No encoder found for head attention analysis.")
            return

        enc_layers = list(encoder)
        n_total = len(enc_layers)
        if layer_idx >= n_total:
            print(f"layer_idx {layer_idx} >= num_layers {n_total}; "
                  "using last layer.")
            layer_idx = n_total - 1

        target_layer = enc_layers[layer_idx]

        # Monkey-patch _sa_block to get per-head weights
        per_head_weights: List[torch.Tensor] = []
        original_sa_block = target_layer._sa_block

        def _patched_sa_block(
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            is_causal: bool = False,
        ) -> torch.Tensor:
            x_out, attn_w = target_layer.self_attn(
                x, x, x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            per_head_weights.append(attn_w.detach().cpu())
            target_layer.attn_weights = attn_w.mean(dim=1).detach()
            return target_layer.dropout1(x_out)

        target_layer._sa_block = _patched_sa_block

        with torch.no_grad():
            kwargs: Dict[str, Any] = {}
            if adjacency_matrix is not None:
                kwargs['adjacency_matrix'] = (
                    adjacency_matrix.to(device))
            _ = model(input_batch, **kwargs)

        target_layer._sa_block = original_sa_block

        if not per_head_weights:
            print("No per-head attention weights captured.")
            return

        # (batch, num_heads, seq, seq) -> first sample
        hw = per_head_weights[0].numpy()
        num_heads = hw.shape[1]
        seq_len = hw.shape[2]
        head_maps = hw[0]  # (num_heads, seq, seq)

        # Head entropy
        head_entropies: List[float] = []
        for h in range(num_heads):
            ents = []
            for q in range(seq_len):
                row = head_maps[h, q] + 1e-10
                row = row / row.sum()
                ents.append(-np.sum(row * np.log(row + 1e-10)))
            head_entropies.append(float(np.mean(ents)))

        # Layout
        n_cols = min(4, num_heads)
        n_rows_h = math.ceil(num_heads / n_cols)
        total_rows = n_rows_h + 2  # heads, entropy, details

        fig = plt.figure(figsize=(5 * n_cols, 4 * total_rows))
        gs_fig = GridSpec(total_rows, n_cols, figure=fig,
                         hspace=0.4, wspace=0.3)

        # Head maps
        for h in range(num_heads):
            r, c = h // n_cols, h % n_cols
            ax = fig.add_subplot(gs_fig[r, c])

            tick_labels = feature_names if feature_names else True
            sns.heatmap(head_maps[h], ax=ax, cmap='viridis',
                        cbar=True, cbar_kws={'shrink': 0.6})
            ax.set_title(
                f'Head {h + 1} (H={head_entropies[h]:.2f})',
                fontsize=9)
            ax.set_xlabel('Key', fontsize=7)
            ax.set_ylabel('Query', fontsize=7)
            ax.tick_params(labelsize=6)

        # Empty slots
        for e in range(num_heads, n_rows_h * n_cols):
            fig.add_subplot(gs_fig[e // n_cols, e % n_cols]).axis('off')

        # Entropy bar chart
        ax_ent = fig.add_subplot(gs_fig[n_rows_h, :])
        ent_colors = plt.cm.RdYlGn_r(
            np.array(head_entropies)
            / max(max(head_entropies), 1e-8))
        ax_ent.bar(range(1, num_heads + 1), head_entropies,
                   color=ent_colors, alpha=0.85)
        ax_ent.set_xlabel('Head', fontsize=10)
        ax_ent.set_ylabel('Mean Entropy', fontsize=10)
        ax_ent.set_title(
            f'Head Specialization Analysis (Layer {layer_idx + 1}) '
            f'- Low entropy = specialized', fontsize=11)
        ax_ent.set_xticks(range(1, num_heads + 1))
        max_ent = np.log(seq_len)
        ax_ent.axhline(y=max_ent, color='red', linestyle='--',
                        alpha=0.5,
                        label=f'Max entropy (uniform) = {max_ent:.2f}')
        ax_ent.legend(fontsize=8)
        ax_ent.grid(True, alpha=0.3)

        _add_model_details_subplot(fig, config, gs_fig[-1, :])

        fig.suptitle(
            f'Attention Head Analysis - Layer {layer_idx + 1}',
            fontsize=14, fontweight='bold', y=0.99)
        save_path = os.path.join(
            results_dir,
            f'attention_heads_layer{layer_idx + 1}_{timestamp}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Attention heads plot saved to: {save_path}")

    except Exception as e:
        print(f"Error in plot_attention_heads: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# 8. visualize_feature_importance
# =============================================================================

def visualize_feature_importance(
    model: nn.Module,
    test_loader: DataLoader,
    results_dir: str,
    config: TrainingConfig,
) -> None:
    """Simple bar chart of feature group importance weights from the
    ``FeatureAttention`` module.

    Aggregates ``feature_importances`` across test batches and displays
    a horizontal bar chart with mean and standard deviation.

    Parameters
    ----------
    model : nn.Module
        Trained ``TrafficTransformer``.
    test_loader : DataLoader
        Test data loader.
    results_dir : str
        Directory to save the figure.
    config : TrainingConfig
        Experiment configuration.
    """
    try:
        model.eval()
        timestamp = get_maputo_timestamp()
        os.makedirs(results_dir, exist_ok=True)
        device = next(model.parameters()).device

        fa = getattr(model, 'feature_attention', None)
        if fa is None:
            print("No FeatureAttention module found. Skipping "
                  "feature importance visualization.")
            return

        all_imp: List[np.ndarray] = []

        with torch.no_grad():
            for bi, (inputs, targets) in enumerate(test_loader):
                # PATCH: dict batches (feature-attention mode) move per-key.
                if isinstance(inputs, dict):
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                else:
                    inputs = inputs.to(device)
                _ = model(inputs)
                imp = getattr(fa, 'feature_importances', None)
                if imp is not None:
                    all_imp.append(imp.cpu().numpy())
                if bi >= 20:
                    break

        if not all_imp:
            print("No feature importance data collected.")
            return

        arr = np.concatenate(all_imp, axis=0)
        mean_imp = arr.mean(axis=0)
        std_imp = arr.std(axis=0)

        f_names = fa.feature_names
        nf = min(len(f_names), len(mean_imp))

        fig = plt.figure(figsize=(10, max(6, nf * 0.8 + 3)))
        gs_fig = GridSpec(2, 1, figure=fig, height_ratios=[4, 1],
                          hspace=0.3)

        ax = fig.add_subplot(gs_fig[0])
        bc = [BOUNDARY_COLORS[i % len(BOUNDARY_COLORS)]
              for i in range(nf)]
        bars = ax.barh(range(nf), mean_imp[:nf], xerr=std_imp[:nf],
                        color=bc, alpha=0.85, capsize=4,
                        edgecolor='gray', linewidth=0.5)
        ax.set_yticks(range(nf))
        ax.set_yticklabels(f_names[:nf], fontsize=10)
        ax.set_xlabel('Importance Weight', fontsize=11)
        ax.set_title('Feature Group Importance Weights', fontsize=13)
        ax.invert_yaxis()
        ax.grid(True, axis='x', alpha=0.3)

        for bar, val, sd in zip(bars, mean_imp[:nf], std_imp[:nf]):
            ax.text(bar.get_width() + sd + 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=8)

        _add_model_details_subplot(fig, config, gs_fig[1])
        fig.suptitle('Feature Importance Analysis', fontsize=14,
                     fontweight='bold', y=0.98)
        save_path = os.path.join(
            results_dir, f'feature_importance_{timestamp}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Feature importance plot saved to: {save_path}")

    except Exception as e:
        print(f"Error in visualize_feature_importance: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# 10. plot_predictions_vs_actual
# =============================================================================

def plot_predictions_vs_actual(
    actuals: np.ndarray,
    predictions: np.ndarray,
    sensor_index: int,
    sensor_id: Union[int, str],
    fold: int,
    results_dir: str,
    pred_len: int,
    config: TrainingConfig,
) -> None:
    """Actual vs predicted with R2 score.

    Produces:
    * Time series overlay (first 200 points).
    * Scatter plot with perfect-prediction line.
    * Residual distribution histogram.
    * Model details text box.

    Parameters
    ----------
    actuals : np.ndarray
        Ground truth (any shape -- will be flattened).
    predictions : np.ndarray
        Predictions (same shape as *actuals*).
    sensor_index : int
        Positional index of the sensor.
    sensor_id : int or str
        Sensor identifier for titles.
    fold : int
        Cross-validation fold number.
    results_dir : str
        Directory to save the figure.
    pred_len : int
        Prediction horizon length (for annotation).
    config : TrainingConfig
        Experiment configuration.
    """
    try:
        timestamp = get_maputo_timestamp()
        os.makedirs(results_dir, exist_ok=True)

        af = np.asarray(actuals).flatten()
        pf = np.asarray(predictions).flatten()
        ml = min(len(af), len(pf))
        af, pf = af[:ml], pf[:ml]

        r2 = r2_score(af, pf)
        mae = mean_absolute_error(af, pf)
        rmse = math.sqrt(mean_squared_error(af, pf))
        nz = af != 0
        mape = (np.mean(np.abs((af[nz] - pf[nz]) / af[nz])) * 100
                if nz.sum() > 0 else float('nan'))

        fig = plt.figure(figsize=(18, 14))
        gs_fig = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3,
                          height_ratios=[2, 1.5, 0.5])

        # 1 -- time series overlay
        ax1 = fig.add_subplot(gs_fig[0, :])
        ns = min(200, ml)
        ax1.plot(range(ns), af[:ns], 'b-', linewidth=1.0, alpha=0.8,
                 label='Actual')
        ax1.plot(range(ns), pf[:ns], 'r-', linewidth=1.0, alpha=0.7,
                 label='Predicted')
        ax1.fill_between(range(ns), af[:ns], pf[:ns],
                         alpha=0.15, color='gray')
        ax1.set_title(
            f'Sensor {sensor_id} (idx={sensor_index}) - Fold {fold}\n'
            f'R2={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  '
            f'MAPE={mape:.2f}%', fontsize=11)
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Value')
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)

        # 2a -- scatter
        ax2 = fig.add_subplot(gs_fig[1, 0])
        ax2.scatter(af, pf, s=3, alpha=0.3, c='steelblue')
        lo = min(af.min(), pf.min())
        hi = max(af.max(), pf.max())
        ax2.plot([lo, hi], [lo, hi], 'r--', linewidth=1.0,
                 label='Perfect prediction')
        ax2.set_xlabel('Actual', fontsize=10)
        ax2.set_ylabel('Predicted', fontsize=10)
        ax2.set_title(f'Scatter (R2={r2:.4f})', fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal', adjustable='box')

        # 2b -- residual distribution
        ax3 = fig.add_subplot(gs_fig[1, 1])
        resid = pf - af
        ax3.hist(resid, bins=50, color='steelblue', alpha=0.7,
                 edgecolor='navy', density=True)
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=1.0)
        ax3.axvline(x=resid.mean(), color='orange', linestyle='-',
                    linewidth=1.0,
                    label=f'Mean={resid.mean():.4f}')
        ax3.set_xlabel('Residual (Predicted - Actual)', fontsize=10)
        ax3.set_ylabel('Density', fontsize=10)
        ax3.set_title('Residual Distribution', fontsize=10)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        _add_model_details_subplot(fig, config, gs_fig[2, :])

        save_path = os.path.join(
            results_dir,
            f'predictions_sensor{sensor_id}_fold{fold}_{timestamp}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Predictions vs actual plot saved to: {save_path}")

    except Exception as e:
        print(f"Error in plot_predictions_vs_actual: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# 11. plot_training_history
# =============================================================================

def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    results_dir: str,
    config: TrainingConfig,
) -> None:
    """Training and validation loss curves with improvement annotation.

    Produces:
    * Main loss curve with best-validation-loss marker.
    * Log-scale loss curve.
    * Per-epoch improvement bar chart.
    * Model details annotation.

    Parameters
    ----------
    train_losses : list of float
        Training loss per epoch.
    val_losses : list of float
        Validation loss per epoch.
    results_dir : str
        Directory to save the figure.
    config : TrainingConfig
        Experiment configuration.
    """
    try:
        timestamp = get_maputo_timestamp()
        os.makedirs(results_dir, exist_ok=True)

        if not train_losses:
            print("No training losses to plot.")
            return

        epochs = range(1, len(train_losses) + 1)

        fig = plt.figure(figsize=(14, 10))
        gs_fig = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3,
                         height_ratios=[3, 1])

        # ---- Main loss curves ----
        ax1 = fig.add_subplot(gs_fig[0, :])
        ax1.plot(epochs, train_losses, 'b-', linewidth=1.2, alpha=0.8,
                 label='Training Loss')
        if val_losses:
            ve = range(1, len(val_losses) + 1)
            ax1.plot(ve, val_losses, 'r-', linewidth=1.2, alpha=0.8,
                     label='Validation Loss')
            bvi = int(np.argmin(val_losses))
            bv = val_losses[bvi]
            ax1.scatter([bvi + 1], [bv], c='gold', s=100, zorder=5,
                        edgecolors='black', linewidth=1.5,
                        label=f'Best Val: {bv:.6f} (ep {bvi + 1})')

            if len(val_losses) > 1:
                iv = val_losses[0]
                imp_pct = ((iv - bv) / iv * 100) if iv > 0 else 0
                ax1.annotate(
                    f'Improvement: {imp_pct:.1f}%\n'
                    f'From {iv:.6f} to {bv:.6f}',
                    xy=(bvi + 1, bv),
                    xytext=(
                        bvi + 1 + len(train_losses) * 0.1,
                        bv + (max(val_losses) - min(val_losses)) * 0.2),
                    fontsize=9,
                    arrowprops=dict(arrowstyle='->',
                                   color='darkgreen', lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='lightgreen',
                              edgecolor='green', alpha=0.8))

        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('Training History', fontsize=13)
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)

        # ---- Log scale ----
        ax2 = fig.add_subplot(gs_fig[1, 0])
        ax2.semilogy(epochs, train_losses, 'b-', linewidth=1.0,
                     alpha=0.8, label='Train')
        if val_losses:
            ax2.semilogy(range(1, len(val_losses) + 1), val_losses,
                         'r-', linewidth=1.0, alpha=0.8, label='Val')
        ax2.set_xlabel('Epoch', fontsize=10)
        ax2.set_ylabel('Loss (log)', fontsize=10)
        ax2.set_title('Loss (Log Scale)', fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # ---- Per-epoch improvement ----
        ax3 = fig.add_subplot(gs_fig[1, 1])
        if val_losses and len(val_losses) > 1:
            va = np.array(val_losses)
            diff = np.diff(va)
            cols = ['green' if d < 0 else 'red' for d in diff]
            ax3.bar(range(2, len(val_losses) + 1), -diff,
                    color=cols, alpha=0.6, width=1.0)
            ax3.axhline(y=0, color='gray', linestyle='-',
                        linewidth=0.5)
            ax3.set_xlabel('Epoch', fontsize=10)
            ax3.set_ylabel('Val Loss Improvement', fontsize=10)
            ax3.set_title('Per-Epoch Improvement', fontsize=10)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.axis('off')
            txt = (f"Total Epochs: {len(train_losses)}\n"
                   f"Final Train Loss: {train_losses[-1]:.6f}\n"
                   f"Min Train Loss: {min(train_losses):.6f}")
            if val_losses:
                txt += (f"\nFinal Val Loss: {val_losses[-1]:.6f}\n"
                        f"Min Val Loss: {min(val_losses):.6f}")
            ax3.text(0.5, 0.5, txt, transform=ax3.transAxes,
                     ha='center', va='center', fontsize=10,
                     family='monospace',
                     bbox=dict(boxstyle='round,pad=0.5',
                               facecolor='lightyellow',
                               edgecolor='gray', alpha=0.8))
            ax3.set_title('Training Summary', fontsize=10)

        # Model details
        fig.text(0.5, 0.01, _get_model_details_text(config),
                 ha='center', fontsize=7, family='monospace',
                 bbox=dict(boxstyle='round,pad=0.3',
                           facecolor='lightyellow',
                           edgecolor='gray', alpha=0.8))

        fig.suptitle('Training & Validation Loss', fontsize=14,
                     fontweight='bold', y=0.98)
        save_path = os.path.join(
            results_dir, f'training_history_{timestamp}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Training history plot saved to: {save_path}")

    except Exception as e:
        print(f"Error in plot_training_history: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# 12. create_summary_comparison_plot
# =============================================================================

def create_summary_comparison_plot(
    transformer_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    results_dir: str,
    config: TrainingConfig,
) -> None:
    """Model performance metrics bar chart comparing transformer and
    baseline.

    For each common metric (MAE, RMSE, R2, MAPE) draws grouped bars with
    percentage-improvement annotations.

    Parameters
    ----------
    transformer_metrics : dict
        Metrics from the transformer, e.g.
        ``{'mae': 2.3, 'rmse': 3.1, 'r2': 0.95, 'mape': 4.5}``.
    baseline_metrics : dict
        Metrics from the baseline model (same key structure).
    results_dir : str
        Directory to save the figure.
    config : TrainingConfig
        Experiment configuration.
    """
    try:
        timestamp = get_maputo_timestamp()
        os.makedirs(results_dir, exist_ok=True)

        # Gather common metrics
        metric_names: List[str] = []
        t_vals: List[float] = []
        b_vals: List[float] = []

        for key in ['mae', 'rmse', 'r2', 'mape']:
            if key in transformer_metrics and key in baseline_metrics:
                metric_names.append(key.upper())
                t_vals.append(transformer_metrics[key])
                b_vals.append(baseline_metrics[key])

        # Fallback: try any common keys
        if not metric_names:
            for key in transformer_metrics:
                if key in baseline_metrics:
                    metric_names.append(key)
                    t_vals.append(transformer_metrics[key])
                    b_vals.append(baseline_metrics[key])

        if not metric_names:
            print("No common metrics found between transformer and "
                  "baseline.")
            return

        nm = len(metric_names)
        fig = plt.figure(figsize=(max(12, nm * 3), 10))
        gs_fig = GridSpec(2, 1, figure=fig, height_ratios=[4, 1],
                         hspace=0.3)

        ax = fig.add_subplot(gs_fig[0])
        x = np.arange(nm)
        w = 0.35

        bars_t = ax.bar(x - w / 2, t_vals, w, label='Transformer',
                        color='steelblue', alpha=0.85,
                        edgecolor='navy', linewidth=0.5)
        bars_b = ax.bar(x + w / 2, b_vals, w, label='Baseline',
                        color='coral', alpha=0.85,
                        edgecolor='darkred', linewidth=0.5)

        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, fontsize=11)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, axis='y', alpha=0.3)

        # Value labels
        for bar in bars_t:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f'{bar.get_height():.4f}',
                    ha='center', va='bottom', fontsize=8,
                    fontweight='bold')
        for bar in bars_b:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f'{bar.get_height():.4f}',
                    ha='center', va='bottom', fontsize=8)

        # Improvement annotations
        for i in range(nm):
            tv, bv = t_vals[i], b_vals[i]
            mn_lower = metric_names[i].lower()
            if bv != 0:
                if mn_lower == 'r2':
                    imp = ((tv - bv) / abs(bv)) * 100
                    better = tv > bv
                else:
                    imp = ((bv - tv) / abs(bv)) * 100
                    better = tv < bv
                sign = '+' if imp > 0 else ''
                clr = 'green' if better else 'red'
                ax.text(x[i], max(tv, bv) * 1.08,
                        f'{sign}{imp:.1f}%',
                        ha='center', va='bottom', fontsize=9,
                        fontweight='bold', color=clr)

        _add_model_details_subplot(fig, config, gs_fig[1])
        fig.suptitle('Transformer vs Baseline Comparison',
                     fontsize=14, fontweight='bold', y=0.98)
        save_path = os.path.join(
            results_dir, f'summary_comparison_{timestamp}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Summary comparison plot saved to: {save_path}")

    except Exception as e:
        print(f"Error in create_summary_comparison_plot: {e}")
        import traceback
        traceback.print_exc()
