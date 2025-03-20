import os
import csv
import time
from datetime import datetime
import pytz
import torch
import pandas as pd
from typing import Dict, Any, Optional, List
from dataclasses import asdict
import uuid

class ExperimentLogger:
    """A class to handle logging of experiment parameters and metrics."""
    
    def __init__(self, output_dir: str):
        """Initialize the logger with the output directory."""
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, 'experiment_logs.csv')
        self.current_run_id = str(uuid.uuid4())[:8]  # Generate a unique run ID
        self.start_time = time.time()
        self.current_log = {}
        
        # Create the log file with headers if it doesn't exist
        self._initialize_log_file()
    
    def _initialize_log_file(self):
        """Initialize the CSV log file with headers if it doesn't exist."""
        headers = [
            # Run Information
            'timestamp', 'run_id', 'fold_number',
            
            # Model Architecture
            'hidden_dim', 'num_layers', 'num_heads', 'dropout',
            'ff_dim_multiplier', 'activation', 'decoder_type',
            'use_gnn_pre_transformer', 'gnn_type',
            
            # Input Features Configuration
            'seq_length', 'pred_length', 'use_time_features',
            'use_holiday_feature', 'use_weather_feature',
            'use_lagged_features', 'num_lags',
            'use_spatial_features', 'spatial_feature_dim',
            
            # Training Configuration
            'batch_size', 'num_epochs', 'learning_rate',
            'optimizer_type', 'loss_function', 'scaler_type',
            'gradient_clip', 'scheduler_type', 'warmup_epochs',
            'use_mixed_precision', 'accumulation_steps',
            
            # Performance Metrics
            'train_loss', 'val_loss', 'test_loss',
            'mae', 'rmse', 'r2', 'mape',
            'best_epoch', 'training_time',
            
            # Resource Usage
            'peak_gpu_memory', 'average_gpu_utilization',
            'total_parameters',
            
            # Baseline Comparisons
            'arima_mae', 'arima_rmse', 'arima_r2',
            'exp_smoothing_mae', 'exp_smoothing_rmse', 'exp_smoothing_r2',
            'lstm_mae', 'lstm_rmse', 'lstm_r2'
        ]
        
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    
    def _get_maputo_timestamp(self) -> str:
        """Get current timestamp in Maputo timezone."""
        maputo_tz = pytz.timezone('Africa/Maputo')
        return datetime.now(maputo_tz).strftime("%Y%m%d_%H%M%S")
    
    def _get_gpu_stats(self) -> Dict[str, float]:
        """Get GPU statistics if available."""
        if torch.cuda.is_available():
            return {
                'peak_gpu_memory': torch.cuda.max_memory_allocated() / 1e9,  # Convert to GB
                'average_gpu_utilization': torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else None
            }
        return {'peak_gpu_memory': None, 'average_gpu_utilization': None}
    
    def log_config(self, config: Any):
        """Log configuration parameters."""
        config_dict = asdict(config) if hasattr(config, '__dataclass_fields__') else vars(config)
        self.current_log.update(config_dict)
    
    def log_fold(self, fold_number: int):
        """Log the current fold number."""
        self.current_log['fold_number'] = fold_number
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log performance metrics."""
        self.current_log.update(metrics)
    
    def log_baseline_metrics(self, model_name: str, metrics: tuple):
        """Log baseline model metrics."""
        mae, rmse, r2 = metrics[:3]  # Extract first three metrics
        self.current_log.update({
            f'{model_name}_mae': mae,
            f'{model_name}_rmse': rmse,
            f'{model_name}_r2': r2
        })
    
    def log_model_stats(self, model: torch.nn.Module):
        """Log model statistics."""
        total_params = sum(p.numel() for p in model.parameters())
        self.current_log['total_parameters'] = total_params
    
    def save_log(self):
        """Save the current log to the CSV file."""
        # Add run information
        self.current_log.update({
            'timestamp': self._get_maputo_timestamp(),
            'run_id': self.current_run_id,
            'training_time': (time.time() - self.start_time) / 60  # Convert to minutes
        })
        
        # Add GPU stats
        self.current_log.update(self._get_gpu_stats())
        
        # Write to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._get_headers())
            writer.writerow(self.current_log)
        
        # Clear the current log
        self.current_log = {}
        
    def _get_headers(self) -> List[str]:
        """Get the headers from the CSV file."""
        with open(self.log_file, 'r') as f:
            reader = csv.reader(f)
            return next(reader)
    
    def get_logs_df(self) -> pd.DataFrame:
        """Return all logs as a pandas DataFrame."""
        return pd.read_csv(self.log_file)
    
    def get_best_run(self, metric: str = 'val_loss', minimize: bool = True) -> Dict[str, Any]:
        """Get the best run based on a specific metric."""
        df = self.get_logs_df()
        idx = df[metric].idxmin() if minimize else df[metric].idxmax()
        return df.iloc[idx].to_dict() 