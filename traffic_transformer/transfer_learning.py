"""
Transfer learning module for the traffic prediction transformer.

Provides the ``AdapterLayer`` nn.Module (FIX #11), the ``TransferLearningModule``
class for freezing / adapter-based fine-tuning, and the standalone
``train_transfer_model`` function used by the orchestration layer.
"""

import gc
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .config import TrainingConfig, TransferLearningConfig
from .model import TrafficTransformer
from .training import (
    _compute_loss,
    _forward_pass,
    _get_autocast_context,
    _inference_forward,
    _process_batch,
    create_criterion,
    evaluate_model,
    hybrid_loss,
    predict,
    quantile_loss,
    robust_mape,
)
from .data_module import TrafficDataset, prepare_data
from .utils import (
    AMP_AVAILABLE,
    DEVICE_TYPE_SUPPORTED,
    get_maputo_timestamp,
    make_grad_scaler,
)

# PATCH: the deprecated ``torch.cuda.amp`` import (and the
# ``GradScaler(device_type='cuda')`` call it fed — a keyword neither the old
# nor the new API accepts) has been replaced by utils.make_grad_scaler.


# =============================================================================
# AdapterLayer  (FIX #11)
# =============================================================================

class AdapterLayer(nn.Module):
    """Proper nn.Module adapter for transfer learning.

    FIX #11: This replaces the old monkey-patching approach with a proper
    nn.Module that is serializable via ``state_dict()`` and correctly
    participates in the module hierarchy.

    The adapter applies a bottleneck transformation (down-project, activate,
    up-project) and adds the result as a residual to the input.

    Parameters
    ----------
    hidden_dim : int
        Dimension of the input (and output) features.
    adapter_dim : int
        Bottleneck dimension of the adapter.
    """

    def __init__(self, hidden_dim: int, adapter_dim: int):
        super().__init__()
        self.down = nn.Linear(hidden_dim, adapter_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(adapter_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


# =============================================================================
# AdaptedEncoderLayer  (FIX #11 - proper wrapping)
# =============================================================================

class AdaptedEncoderLayer(nn.Module):
    """Wraps an existing encoder layer with an ``AdapterLayer`` as a proper
    nn.Module, replacing the old monkey-patching of forward methods with
    closures.

    FIX #11: Both the original layer and the adapter are registered as
    submodules so that ``state_dict()`` captures all parameters.

    Parameters
    ----------
    original_layer : nn.Module
        The transformer encoder layer to wrap.
    adapter : AdapterLayer
        The adapter module to apply after the original layer's forward pass.
    """

    def __init__(self, original_layer: nn.Module, adapter: AdapterLayer):
        super().__init__()
        self.original_layer = original_layer
        self.adapter = adapter

    def forward(self, src: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        output = self.original_layer(src, *args, **kwargs)
        return self.adapter(output)

    # PATCH: TrafficTransformer.encode() calls set_spatial_bias on every
    # encoder entry and the visualisation code reads attn_weights; the
    # wrapper previously exposed neither, so the first forward pass after
    # adding adapters raised AttributeError.
    def set_spatial_bias(self, bias: Optional[torch.Tensor]) -> None:
        if hasattr(self.original_layer, "set_spatial_bias"):
            self.original_layer.set_spatial_bias(bias)

    @property
    def attn_weights(self) -> Optional[torch.Tensor]:
        return getattr(self.original_layer, "attn_weights", None)


# =============================================================================
# TransferLearningModule
# =============================================================================

class TransferLearningModule:
    """Module for transfer learning with Traffic Transformer models.

    Enables fine-tuning models pre-trained on large traffic datasets (e.g.
    METR-LA, PEMS-BAY) for smaller or geographically different target datasets
    such as African traffic data.
    """

    def __init__(
        self,
        base_model: nn.Module,
        config: TrainingConfig,
        target_dataset_name: str = 'mozambique',
        freeze_encoder: bool = True,
        freeze_layers: int = 1,
        adapter_dim: int = 64,
    ):
        """Initialise transfer learning module.

        Parameters
        ----------
        base_model : nn.Module
            Pre-trained TrafficTransformer model.
        config : TrainingConfig
            Experiment configuration object.
        target_dataset_name : str
            Name of the target dataset (e.g. ``'mozambique'``).
        freeze_encoder : bool
            Whether to freeze embedding + first *freeze_layers* encoder layers.
        freeze_layers : int
            Number of transformer encoder layers to freeze (counted from the
            bottom).
        adapter_dim : int
            Bottleneck dimension of adapter layers.  Set to ``0`` to disable
            adapters entirely.
        """
        self.base_model = base_model
        self.config = config
        self.target_dataset_name = target_dataset_name
        self.freeze_encoder = freeze_encoder
        self.freeze_layers = freeze_layers
        self.adapter_dim = adapter_dim
        self.device = next(base_model.parameters()).device

        # Apply parameter freezing
        self._apply_parameter_freezing()

        # FIX #11: Add adapter layers as proper nn.Modules
        self._add_adapter_layers()

        # PATCH: report the trainable parameter count AFTER adapters exist
        # (the earlier print inside _apply_parameter_freezing excluded them).
        total_p = sum(p.numel() for p in self.base_model.parameters())
        trainable_p = sum(
            p.numel() for p in self.base_model.parameters() if p.requires_grad
        )
        print(
            f"Transfer setup complete — trainable (incl. adapters): "
            f"{trainable_p:,} / {total_p:,} parameters"
        )

        # Tracking history
        self.transfer_history: Dict[str, Any] = {
            'train_loss': [],
            'val_loss': [],
            'source_metrics': None,
            'target_metrics': None,
        }

    # ------------------------------------------------------------------
    # Parameter freezing
    # ------------------------------------------------------------------

    def _apply_parameter_freezing(self) -> None:
        """Freeze embedding, positional encoding, and first N encoder layers."""
        if not self.freeze_encoder:
            print("No parameter freezing applied - full fine-tuning")
            return

        # Freeze embedding layer
        for param in self.base_model.embedding.parameters():
            param.requires_grad = False

        # PATCH: the sinusoidal positional encoding is a registered buffer,
        # not a parameter — it never trains, so the old
        # ``pe.requires_grad = False`` line was a no-op and has been removed.

        # Freeze specified encoder layers
        for i, layer in enumerate(self.base_model.encoder):
            if i < self.freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        trainable = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.base_model.parameters())
        print(
            f"Froze embedding layer and {self.freeze_layers} transformer layers. "
            f"Trainable: {trainable:,} / {total:,} parameters"
        )

    # ------------------------------------------------------------------
    # Adapter layers  (FIX #11 - proper adapter registration)
    # ------------------------------------------------------------------

    def _add_adapter_layers(self) -> None:
        """Add ``AdapterLayer`` modules to unfrozen encoder layers.

        FIX #11: Instead of monkey-patching forward methods with closures,
        this method creates ``AdaptedEncoderLayer`` nn.Module instances that
        wrap each unfrozen encoder layer with an adapter.  The adapted layers
        are registered as proper submodules on the base model via
        ``nn.ModuleList`` so they are captured by ``state_dict()`` and
        correctly serialised / deserialised.
        """
        if self.adapter_dim <= 0:
            self.has_adapters = False
            return

        self.has_adapters = True

        # Resolve the model's hidden dimension.  TrafficTransformer uses
        # ``d_model``; some wrappers expose ``hidden_dim`` instead.
        hidden_dim = getattr(
            self.base_model, 'hidden_dim',
            getattr(self.base_model, 'd_model', None),
        )
        if hidden_dim is None:
            raise AttributeError(
                "Base model has neither 'hidden_dim' nor 'd_model' attribute. "
                "Cannot determine the hidden dimension for adapter layers."
            )

        # Build adapted layers for each unfrozen encoder layer
        adapters = nn.ModuleList()
        new_encoder = nn.ModuleList()

        for i, layer in enumerate(self.base_model.encoder):
            if i < self.freeze_layers:
                # Frozen layers pass through unchanged
                new_encoder.append(layer)
            else:
                # Create an adapter and wrap the unfrozen layer
                adapter = AdapterLayer(hidden_dim, self.adapter_dim).to(self.device)
                adapted = AdaptedEncoderLayer(layer, adapter).to(self.device)
                adapters.append(adapter)
                new_encoder.append(adapted)

        # Replace the encoder ModuleList so that adapted layers (and their
        # adapter parameters) are part of the model's module tree.
        self.base_model.encoder = new_encoder

        # PATCH: ``.transformer`` is now a read-only property on
        # TrafficTransformer that always mirrors ``.encoder``; assigning to
        # it would raise AttributeError, so the old re-assignment is gone.

        # Store reference for convenience
        self.adapters = adapters

        # Also register the adapters explicitly as a named module so they
        # appear clearly in ``state_dict`` keys.
        self.base_model.add_module('transfer_adapters', adapters)

        print(f"Added {len(self.adapters)} adapter layers with dimension {self.adapter_dim}")

    # ------------------------------------------------------------------
    # Input/output adaptation  (PATCH)
    # ------------------------------------------------------------------

    def adapt_input_output(
        self,
        new_input_dim: int,
        new_output_dim: int,
    ) -> None:
        """Replace the source-fitted input embedding and output head.

        PATCH: previously the source embedding (e.g. fitted for PEMS-BAY's
        325 sensors + auxiliaries) stayed frozen and the head kept the source
        output width, so cross-dataset transfer with a different sensor count
        (325 -> 12 for Maputo) crashed on the first batch. This implements
        the "embedding layer adaptation" the paper describes: fresh,
        trainable input/output layers sized for the target dataset, with the
        pretrained encoder in between.

        Parameters
        ----------
        new_input_dim : int
            Total input feature width of the target dataset (sensors +
            time/weather/holiday features as produced by TrafficDataset).
        new_output_dim : int
            Number of target-dataset sensors to predict.
        """
        m = self.base_model
        d_model = getattr(m, "d_model", None) or getattr(m, "hidden_dim")
        changed: List[str] = []

        if new_input_dim != getattr(m, "input_dim", new_input_dim):
            m.embedding = nn.Linear(new_input_dim, d_model).to(self.device)
            m.input_dim = new_input_dim
            changed.append(f"embedding in_dim -> {new_input_dim}")

        if new_output_dim != getattr(m, "output_dim", new_output_dim):
            decoder_type = getattr(m, "decoder_type", "linear")
            if decoder_type == "linear":
                m.decoder = nn.Linear(
                    d_model, m.pred_length * new_output_dim
                ).to(self.device)
            elif decoder_type == "mlp":
                last = m.decoder[-1]
                m.decoder[-1] = nn.Linear(
                    last.in_features, m.pred_length * new_output_dim
                ).to(self.device)
            else:  # transformer decoder
                m.decoder_proj = nn.Linear(d_model, new_output_dim).to(self.device)
                m.target_embedding = nn.Linear(new_output_dim, d_model).to(self.device)
            m.output_dim = new_output_dim
            m.num_sensors = new_output_dim
            changed.append(f"decoder out_dim -> {new_output_dim}")

        if changed:
            print("Adapted model I/O for target dataset: " + "; ".join(changed))

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def load_african_dataset(
        self,
        file_path: str,
        sequence_length: Optional[int] = None,
        prediction_length: Optional[int] = None,
        test_split: float = 0.2,
        val_split: float = 0.1,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, Any]:
        """Load and prepare an African traffic dataset for transfer learning.

        PATCH: three changes versus the original.
        (1) A chronological validation split is carved out of the training
        portion — early stopping previously used the *test* loader, leaking
        the test set into model selection.
        (2) The scaler is fitted on the training slice only (leakage fix).
        (3) The model's input embedding and output head are automatically
        re-sized to the target dataset via :meth:`adapt_input_output`.

        Parameters
        ----------
        file_path : str
            Path to the dataset CSV file.
        sequence_length : int, optional
            Input window length (defaults to ``config.seq_length``).
        prediction_length : int, optional
            Prediction horizon (defaults to ``config.pred_length``).
        test_split : float
            Fraction of data reserved for testing (chronological tail).
        val_split : float
            Fraction of data reserved for validation (between train & test).

        Returns
        -------
        train_loader : DataLoader
        val_loader : DataLoader
        test_loader : DataLoader
        data_scaler : sklearn scaler
        """
        seq_length = sequence_length or self.config.seq_length
        pred_length = prediction_length or self.config.pred_length

        try:
            # Load dataset
            df = pd.read_csv(file_path, parse_dates=True, index_col=0)
            print(f"Loaded dataset with shape: {df.shape}")

            # NOTE: Keep the df.replace(0.0, np.nan) here since this is
            # specifically for African datasets where zeros likely ARE missing
            # values.
            df.replace(0.0, np.nan, inplace=True)
            df.ffill(inplace=True)
            df.bfill(inplace=True)

            # Get timestamps and data
            timestamps = df.index
            sensor_data = df.values

            # Scale data
            if self.config.data_scaler_type == 'standard':
                data_scaler = StandardScaler()
            elif self.config.data_scaler_type == 'robust':
                data_scaler = RobustScaler()
            else:
                data_scaler = MinMaxScaler()

            # PATCH (leakage): chronological 3-way split with the scaler
            # fitted on the training slice only.
            total_samples = len(sensor_data)
            test_size = int(total_samples * test_split)
            val_size = int(total_samples * val_split)
            train_size = total_samples - test_size - val_size

            window = seq_length + pred_length
            if min(train_size, val_size, test_size) < window:
                raise ValueError(
                    f"Dataset too small for splits: train={train_size}, "
                    f"val={val_size}, test={test_size} timesteps, but each "
                    f"split needs at least seq+pred = {window}."
                )

            data_scaler.fit(sensor_data[:train_size])
            data_normalized = data_scaler.transform(sensor_data)

            train_data = data_normalized[:train_size]
            val_data = data_normalized[train_size:train_size + val_size]
            test_data = data_normalized[train_size + val_size:]
            train_times = timestamps[:train_size]
            val_times = timestamps[train_size:train_size + val_size]
            test_times = timestamps[train_size + val_size:]

            # Create datasets
            train_dataset = TrafficDataset(
                train_data, train_times, seq_length, pred_length, self.config,
            )
            val_dataset = TrafficDataset(
                val_data, val_times, seq_length, pred_length, self.config,
            )
            test_dataset = TrafficDataset(
                test_data, test_times, seq_length, pred_length, self.config,
            )

            # PATCH: re-size the input embedding / output head for the target
            # dataset's actual feature widths (sensors + auxiliaries).
            sample_x, sample_y = train_dataset[0]
            self.adapt_input_output(
                int(sample_x.shape[-1]), int(sample_y.shape[-1])
            )

            # Smaller batch size for typically smaller target datasets
            batch_size = min(32, self.config.batch_size)

            loader_kwargs = dict(
                batch_size=batch_size,
                num_workers=min(2, self.config.num_workers),
                pin_memory=self.config.pin_memory,
            )
            train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
            val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
            test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

            return train_loader, val_loader, test_loader, data_scaler

        except Exception as e:
            raise RuntimeError(f"Error loading African dataset: {e}") from e

    # ------------------------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------------------------

    def fine_tune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-4,
        num_epochs: int = 30,
        patience: int = 5,
        output_dir: Optional[str] = None,
        adjacency_matrix: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """Fine-tune the model on a new (target) dataset.

        Uses AdamW optimizer, ReduceLROnPlateau scheduler, gradient clipping,
        early stopping, and checkpoint saving.

        Parameters
        ----------
        train_loader : DataLoader
            Training data from the target dataset.
        val_loader : DataLoader
            Validation data from the target dataset.
        learning_rate : float
            Learning rate for fine-tuning.
        num_epochs : int
            Maximum number of training epochs.
        patience : int
            Early-stopping patience.
        output_dir : str, optional
            Directory for checkpoints and plots.

        Returns
        -------
        nn.Module
            The fine-tuned model.
        """
        model = self.base_model
        model.train()

        # Only optimise parameters that require gradients
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=patience // 2, factor=0.5,
        )
        # PATCH: honour config.loss_function (was hardcoded MSE, silently
        # overriding e.g. the paper's best MAE setting).
        criterion = create_criterion(self.config)

        # PATCH: pass the target-network adjacency once, if provided.
        if adjacency_matrix is not None:
            adjacency_matrix = adjacency_matrix.to(self.device)

        best_loss = float('inf')
        no_improve = 0
        train_losses: List[float] = []
        val_losses: List[float] = []

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                # PATCH: forward adjacency (when given) and supply the target
                # for teacher forcing with the transformer decoder.
                if getattr(model, 'decoder_type', '') == 'transformer':
                    output = model(data, target, adjacency_matrix)
                else:
                    output = model(data, adjacency_matrix=adjacency_matrix)                         if adjacency_matrix is not None else model(data)
                loss = criterion(output, target)
                loss.backward()

                # Gradient clipping
                if self.config.gradient_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)

                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data, adjacency_matrix=adjacency_matrix)                         if adjacency_matrix is not None else model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # Learning-rate schedule
            scheduler.step(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                no_improve = 0

                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    torch.save(
                        {
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': best_loss,
                            'target_dataset': self.target_dataset_name,
                        },
                        os.path.join(
                            output_dir,
                            f'fine_tuned_{self.target_dataset_name}_best.pth',
                        ),
                    )
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break

            print(
                f'Epoch {epoch + 1}/{num_epochs} | '
                f'Train Loss: {avg_train_loss:.6f} | '
                f'Val Loss: {avg_val_loss:.6f}'
            )

        # Store history
        self.transfer_history['train_loss'] = train_losses
        self.transfer_history['val_loss'] = val_losses

        # Plot training curve
        if output_dir:
            self._plot_transfer_learning_curve(train_losses, val_losses, output_dir)

        return model

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_transfer(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        source_scaler: Any,
        target_scaler: Any,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate transfer learning on both source and target datasets.

        Parameters
        ----------
        source_loader : DataLoader
        target_loader : DataLoader
        source_scaler, target_scaler : sklearn scaler
        output_dir : str, optional

        Returns
        -------
        dict
            ``{'source': {...}, 'target': {...}}`` with per-dataset metrics.
        """
        model = self.base_model
        model.eval()

        source_metrics = self._evaluate_on_dataset(model, source_loader, source_scaler, "METR-LA")
        target_metrics = self._evaluate_on_dataset(model, target_loader, target_scaler, self.target_dataset_name)

        self.transfer_history['source_metrics'] = source_metrics
        self.transfer_history['target_metrics'] = target_metrics

        if output_dir:
            self._plot_transfer_comparison(source_metrics, target_metrics, output_dir)

        return {
            'source': source_metrics,
            'target': target_metrics,
        }

    def _evaluate_on_dataset(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        scaler: Any,
        dataset_name: str,
    ) -> Dict[str, float]:
        """Evaluate model on a single dataset and return metrics.

        Parameters
        ----------
        model : nn.Module
            The model to evaluate.
        dataloader : DataLoader
            Data loader for the evaluation dataset.
        scaler : sklearn scaler
            Fitted scaler for inverse transformation.
        dataset_name : str
            Human-readable name for logging.

        Returns
        -------
        dict
            Dictionary with keys ``'mae'``, ``'rmse'``, ``'r2'``, ``'mape'``.
        """
        model.eval()
        all_preds: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)

                all_preds.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        predictions = np.concatenate(all_preds)
        actuals = np.concatenate(all_targets)

        # Reshape for inverse transformation
        num_samples, pred_window, num_features = predictions.shape
        predictions_2d = predictions.reshape(-1, num_features)
        actuals_2d = actuals.reshape(-1, num_features)

        # Inverse transform
        predictions_inv = scaler.inverse_transform(predictions_2d)
        actuals_inv = scaler.inverse_transform(actuals_2d)

        # Metrics
        mae = mean_absolute_error(actuals_inv.ravel(), predictions_inv.ravel())
        rmse = float(np.sqrt(mean_squared_error(actuals_inv.ravel(), predictions_inv.ravel())))
        r2 = float(r2_score(actuals_inv.ravel(), predictions_inv.ravel()))
        mape = float(
            np.mean(
                np.abs(
                    (actuals_inv.ravel() - predictions_inv.ravel())
                    / np.maximum(np.abs(actuals_inv.ravel()), 1e-10)
                )
            )
            * 100
        )

        print(
            f'Evaluation on {dataset_name}: '
            f'MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, MAPE={mape:.2f}%'
        )
        return {'mae': float(mae), 'rmse': rmse, 'r2': r2, 'mape': mape}

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def _plot_transfer_learning_curve(
        self,
        train_losses: List[float],
        val_losses: List[float],
        output_dir: str,
    ) -> None:
        """Plot and save the transfer-learning training / validation curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'Transfer Learning to {self.target_dataset_name.title()} Dataset')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.annotate(
            f"Frozen layers: {self.freeze_layers if self.freeze_encoder else 'None'}\n"
            f"Adapters: {'Yes' if self.has_adapters else 'No'}",
            xy=(0.02, 0.02),
            xycoords='figure fraction',
        )

        plt.tight_layout()
        save_path = os.path.join(
            output_dir,
            f'transfer_learning_curve_{self.target_dataset_name}.png',
        )
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved training curve to {save_path}")

    def _plot_transfer_comparison(
        self,
        source_metrics: Dict[str, float],
        target_metrics: Dict[str, float],
        output_dir: str,
    ) -> None:
        """Bar-chart comparison of source vs. target dataset performance."""
        metrics = ['mae', 'rmse', 'r2', 'mape']
        source_values = [source_metrics[m] for m in metrics]
        target_values = [target_metrics[m] for m in metrics]

        # Invert R2 so that "lower is better" applies uniformly
        r2_idx = metrics.index('r2')
        source_values[r2_idx] = 1 - source_values[r2_idx]
        target_values[r2_idx] = 1 - target_values[r2_idx]
        metrics[r2_idx] = 'r2 (inverted)'

        plt.figure(figsize=(12, 8))

        x = range(len(metrics))
        width = 0.35

        plt.bar([i - width / 2 for i in x], source_values, width, label='METR-LA (Source)')
        plt.bar(
            [i + width / 2 for i in x],
            target_values,
            width,
            label=f'{self.target_dataset_name.title()} (Target)',
        )

        plt.xlabel('Metrics')
        plt.ylabel('Value (Lower is Better)')
        plt.title('Transfer Learning Performance Comparison')
        plt.xticks(list(x), metrics)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Annotate improvement / degradation percentages
        for i, (source, target) in enumerate(zip(source_values, target_values)):
            if metrics[i] != 'r2 (inverted)':
                change_pct = (target - source) / (source + 1e-10) * 100
                color = 'green' if change_pct < 0 else 'red'
                plt.annotate(
                    f"{change_pct:.1f}%",
                    xy=(i, max(source, target) * 1.05),
                    ha='center',
                    color=color,
                )

        plt.tight_layout()
        save_path = os.path.join(
            output_dir,
            f'transfer_comparison_{self.target_dataset_name}.png',
        )
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved transfer comparison to {save_path}")

    # ------------------------------------------------------------------
    # Attention transfer visualisation (static method)
    # ------------------------------------------------------------------

    @staticmethod
    def visualize_attention_transfer(
        base_model: nn.Module,
        source_loader: DataLoader,
        target_loader: DataLoader,
        output_dir: str,
    ) -> None:
        """Compare attention patterns between source and target datasets.

        Produces a three-panel heatmap: source attention, target attention,
        and their element-wise difference.

        Parameters
        ----------
        base_model : nn.Module
            The (fine-tuned) model whose attention weights will be inspected.
        source_loader : DataLoader
        target_loader : DataLoader
        output_dir : str
        """
        device = next(base_model.parameters()).device
        base_model.eval()

        source_batch = next(iter(source_loader))
        target_batch = next(iter(target_loader))

        def get_attention_weights(
            batch: Tuple[torch.Tensor, torch.Tensor],
        ) -> Optional[np.ndarray]:
            data, _ = batch
            data = data.to(device)

            if hasattr(base_model, 'attention_weights'):
                base_model.attention_weights = None

            with torch.no_grad():
                _ = base_model(data)

            if (
                hasattr(base_model, 'attention_weights')
                and base_model.attention_weights is not None
            ):
                weights = base_model.attention_weights
                if isinstance(weights, torch.Tensor):
                    weights = weights.cpu().numpy()
                return weights
            return None

        source_attention = get_attention_weights(source_batch)
        target_attention = get_attention_weights(target_batch)

        if source_attention is not None and target_attention is not None:
            plt.figure(figsize=(15, 6))

            plt.subplot(1, 3, 1)
            sns.heatmap(source_attention, cmap='viridis')
            plt.title('Source Dataset (METR-LA)\nAttention Pattern')
            plt.xlabel('Key Position')
            plt.ylabel('Query Position')

            plt.subplot(1, 3, 2)
            sns.heatmap(target_attention, cmap='viridis')
            plt.title('Target Dataset\nAttention Pattern')
            plt.xlabel('Key Position')
            plt.ylabel('Query Position')

            plt.subplot(1, 3, 3)
            diff = target_attention - source_attention
            sns.heatmap(diff, cmap='coolwarm', center=0)
            plt.title('Attention Difference\n(Target - Source)')
            plt.xlabel('Key Position')
            plt.ylabel('Query Position')

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, 'attention_transfer_comparison.png'),
                dpi=300,
            )
            plt.close()
            print("Saved attention transfer comparison")
        else:
            print("Could not extract attention weights for visualization")


# =============================================================================
# train_transfer_model  (standalone function, from cell 25)
# =============================================================================

def train_transfer_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    source_loader: Optional[DataLoader] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    criterion: Optional[Union[Callable, nn.Module]] = None,
    config: Optional[TrainingConfig] = None,
    data_scaler: Optional[Union[MinMaxScaler, StandardScaler, RobustScaler]] = None,
    source_scaler: Optional[Union[MinMaxScaler, StandardScaler, RobustScaler]] = None,
    device: str = 'cpu',
    adjacency_matrix: Optional[torch.Tensor] = None,
) -> Tuple[nn.Module, Dict[str, List[float]], Dict[str, Optional[Dict[str, float]]]]:
    """Train a model using transfer learning from a pre-trained checkpoint.

    Parameters
    ----------
    model : nn.Module
        Pre-trained model to fine-tune.
    train_loader : DataLoader
        Training data from the target dataset.
    val_loader : DataLoader
        Validation data from the target dataset.
    source_loader : DataLoader, optional
        Test data from the source dataset (for comparison evaluation).
    optimizer : Optimizer, optional
        Will be created if ``None``.
    scheduler : _LRScheduler, optional
        Will be created if ``None``.
    criterion : callable, optional
        Loss function.  Will be created from ``config.loss_function`` if
        ``None``.
    config : TrainingConfig
        Experiment configuration (required).
    data_scaler : sklearn scaler, optional
        Scaler for the target dataset.
    source_scaler : sklearn scaler, optional
        Scaler for the source dataset (needed when *source_loader* is given).
    device : str
        ``'cpu'`` or ``'cuda'``.
    adjacency_matrix : Tensor, optional
        Adjacency matrix for GNN-based models.

    Returns
    -------
    model : nn.Module
        The fine-tuned model.
    history : dict
        ``{'train_loss': [...], 'val_loss': [...]}``.
    metrics : dict
        ``{'source': {...} | None, 'target': {...} | None}``.
    """
    if config is None:
        raise ValueError("Configuration object is required for transfer learning")

    # Setup directories
    output_dir = config.output_dir
    results_dir = config.results_dir
    model_dir = config.model_dir

    # Transfer-learning-specific hyper-parameters
    num_epochs = min(30, config.num_epochs)
    patience = min(5, config.patience)
    learning_rate = getattr(config, 'transfer_learning_rate', 5e-5)
    target_dataset_name = getattr(config, 'target_dataset_name', 'target')
    accumulation_steps = getattr(config, 'accumulation_steps', 1)

    print(f"\n=== Transfer Learning Setup ({target_dataset_name}) ===")
    print(f"Learning Rate: {learning_rate}")
    print(f"Max Epochs: {num_epochs}")
    print(f"Patience: {patience}")
    print(
        f"Trainable parameters: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # ---- Create optimizer if not provided ----
    if optimizer is None:
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=0.01,
        )

    # ---- Create scheduler if not provided ----
    if scheduler is None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=patience // 2, factor=0.5, verbose=True,
        )

    # ---- Create criterion if not provided ----
    if criterion is None:
        loss_fn_name = getattr(config, 'loss_function', 'mse')
        if loss_fn_name == 'mse':
            criterion = nn.MSELoss()
        elif loss_fn_name == 'mae':
            criterion = nn.L1Loss()
        elif loss_fn_name == 'huber':
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.MSELoss()

    # ---- Tracking variables ----
    best_loss = float('inf')
    no_improve = 0
    history: Dict[str, List[float]] = {
        'train_loss': [],
        'val_loss': [],
    }
    metrics: Dict[str, Optional[Dict[str, float]]] = {
        'source': None,
        'target': None,
    }

    # ---- Mixed precision setup ----
    use_amp = (
        getattr(config, 'use_mixed_precision', False)
        and AMP_AVAILABLE
        and device != 'cpu'
    )
    # PATCH: GradScaler(device_type='cuda') raised a TypeError on every
    # torch version; delegate to the utils helper instead.
    scaler = make_grad_scaler(device, enabled=True) if use_amp else None

    try:
        # ==================================================================
        # Fine-tuning loop
        # ==================================================================
        for epoch in range(num_epochs):
            # ---- Training phase ----
            model.train()
            train_loss = 0.0
            optimizer.zero_grad()

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = _process_batch(data, target, device)

                if use_amp:
                    ac_ctx = _get_autocast_context(True, device)
                    with ac_ctx:
                        output = _inference_forward(
                            model, data, config, adjacency_matrix, device,
                        )
                        loss = _compute_loss(
                            output, target, criterion, config,
                        ) / accumulation_steps

                    scaler.scale(loss).backward()

                    if (
                        (batch_idx + 1) % accumulation_steps == 0
                        or (batch_idx + 1) == len(train_loader)
                    ):
                        if config.gradient_clip is not None:
                            scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(
                                model.parameters(), config.gradient_clip,
                            )
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    output = _inference_forward(
                        model, data, config, adjacency_matrix, device,
                    )
                    loss = _compute_loss(
                        output, target, criterion, config,
                    ) / accumulation_steps

                    loss.backward()

                    if (
                        (batch_idx + 1) % accumulation_steps == 0
                        or (batch_idx + 1) == len(train_loader)
                    ):
                        if config.gradient_clip is not None:
                            nn.utils.clip_grad_norm_(
                                model.parameters(), config.gradient_clip,
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                train_loss += loss.item() * accumulation_steps

            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)

            # ---- Validation phase ----
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = _process_batch(data, target, device)

                    if use_amp:
                        ac_ctx = _get_autocast_context(True, device)
                        with ac_ctx:
                            output = _inference_forward(
                                model, data, config, adjacency_matrix, device,
                            )
                            loss = _compute_loss(
                                output, target, criterion, config,
                            )
                    else:
                        output = _inference_forward(
                            model, data, config, adjacency_matrix, device,
                        )
                        loss = _compute_loss(
                            output, target, criterion, config,
                        )

                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)

            # ---- Scheduler step ----
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()

            # ---- Logging ----
            print(
                f'Epoch {epoch + 1}/{num_epochs} | '
                f'Train Loss: {avg_train_loss:.6f} | '
                f'Val Loss: {avg_val_loss:.6f}'
            )

            # ---- Early stopping & checkpointing ----
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                no_improve = 0

                if model_dir:
                    try:
                        timestamp = get_maputo_timestamp()
                        model_path = os.path.join(
                            model_dir,
                            f'fine_tuned_{target_dataset_name}_{timestamp}.pth',
                        )
                        torch.save(
                            {
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': best_loss,
                                'target_dataset': target_dataset_name,
                                'config': {
                                    k: v
                                    for k, v in vars(config).items()
                                    if not k.startswith('_')
                                },
                            },
                            model_path,
                        )
                        print(f"Saved fine-tuned model: {model_path}")
                    except Exception as e:
                        print(f"Warning: Could not save model: {e}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break

        # ==================================================================
        # Post-training: plot history
        # ==================================================================
        if results_dir:
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(history['train_loss'], label='Training Loss')
                plt.plot(history['val_loss'], label='Validation Loss')
                plt.title(
                    f'Transfer Learning to {target_dataset_name.title()} Dataset'
                )
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)

                if hasattr(config, 'freeze_encoder') and hasattr(config, 'freeze_layers'):
                    plt.figtext(
                        0.02,
                        0.02,
                        f"Frozen layers: {config.freeze_layers if config.freeze_encoder else 'None'}\n"
                        f"Adapters: {'Yes' if hasattr(config, 'adapter_dim') and config.adapter_dim > 0 else 'No'}",
                        fontsize=10,
                    )

                plt.tight_layout()
                curve_path = os.path.join(
                    results_dir,
                    f'transfer_learning_curve_{target_dataset_name}.png',
                )
                plt.savefig(curve_path, dpi=300)
                plt.close()
                print(f"Saved training curve to {curve_path}")
            except Exception as e:
                print(f"Warning: Could not plot training history: {e}")

        # ==================================================================
        # Evaluate on target dataset
        # ==================================================================
        print(f"\n=== Evaluating Transfer Learning on {target_dataset_name} ===")
        target_eval = evaluate_model(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            data_scaler=data_scaler,
            device=device,
            config=config,
            adjacency_matrix=adjacency_matrix,
        )
        target_metrics_tuple = target_eval[1]
        metrics['target'] = {
            'mae': target_metrics_tuple[0],
            'rmse': target_metrics_tuple[1],
            'r2': target_metrics_tuple[2],
            'mape': target_metrics_tuple[3],
        }

        # ==================================================================
        # Evaluate on source dataset (if provided)
        # ==================================================================
        if source_loader is not None and source_scaler is not None:
            print("\n=== Evaluating Transfer Learning on Source Dataset (METR-LA) ===")
            source_eval = evaluate_model(
                model=model,
                data_loader=source_loader,
                criterion=criterion,
                data_scaler=source_scaler,
                device=device,
                config=config,
                adjacency_matrix=adjacency_matrix,
            )
            source_metrics_tuple = source_eval[1]
            metrics['source'] = {
                'mae': source_metrics_tuple[0],
                'rmse': source_metrics_tuple[1],
                'r2': source_metrics_tuple[2],
                'mape': source_metrics_tuple[3],
            }

            # ---- Comparative visualisation ----
            if metrics['source'] and metrics['target'] and results_dir:
                try:
                    plt.figure(figsize=(12, 8))
                    metric_names = ['mae', 'rmse', 'r2', 'mape']
                    source_values = [metrics['source'][m] for m in metric_names]
                    target_values = [metrics['target'][m] for m in metric_names]

                    # Invert R2 so lower-is-better applies uniformly
                    r2_idx = metric_names.index('r2')
                    source_values[r2_idx] = 1 - source_values[r2_idx]
                    target_values[r2_idx] = 1 - target_values[r2_idx]
                    metric_names[r2_idx] = 'r2 (inverted)'

                    x = range(len(metric_names))
                    width = 0.35

                    plt.bar(
                        [i - width / 2 for i in x],
                        source_values,
                        width,
                        label='METR-LA (Source)',
                    )
                    plt.bar(
                        [i + width / 2 for i in x],
                        target_values,
                        width,
                        label=f'{target_dataset_name.title()} (Target)',
                    )

                    plt.xlabel('Metrics')
                    plt.ylabel('Value (Lower is Better)')
                    plt.title('Transfer Learning Performance Comparison')
                    plt.xticks(list(x), metric_names)
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                    for i, (src_v, tgt_v) in enumerate(
                        zip(source_values, target_values)
                    ):
                        if metric_names[i] != 'r2 (inverted)':
                            change_pct = (tgt_v - src_v) / (src_v + 1e-10) * 100
                            color = 'green' if change_pct < 0 else 'red'
                            plt.annotate(
                                f"{change_pct:.1f}%",
                                xy=(i, max(src_v, tgt_v) * 1.05),
                                ha='center',
                                color=color,
                            )

                    plt.tight_layout()
                    cmp_path = os.path.join(
                        results_dir,
                        f'transfer_comparison_{target_dataset_name}.png',
                    )
                    plt.savefig(cmp_path, dpi=300)
                    plt.close()
                    print(f"Saved transfer comparison to {cmp_path}")
                except Exception as e:
                    print(f"Warning: Could not create comparison visualization: {e}")

        return model, history, metrics

    except Exception as e:
        print(f"\n=== Error during transfer learning: {e} ===")
        import traceback
        traceback.print_exc()

        # Return the model in its current state with partial history / metrics
        return model, history, metrics
