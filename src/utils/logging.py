"""
Logging Utilities for Deep Hedging Experiments

Provides structured logging for:
1. Training progress
2. Experiment configuration
3. Results and metrics
4. Model checkpoints

Supports both console output and file logging.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np


class ExperimentLogger:
    """
    Comprehensive logger for deep hedging experiments.
    
    Handles:
    - Console and file logging
    - Metrics tracking
    - Configuration saving
    - Results export
    """
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = 'logs',
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        use_timestamp: bool = True
    ):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for log files
            console_level: Logging level for console
            file_level: Logging level for file
            use_timestamp: Whether to add timestamp to log directory
        """
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        
        # Create log directory
        if use_timestamp:
            timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
            self.log_dir = Path(log_dir) / f"{experiment_name}_{timestamp}"
        else:
            self.log_dir = Path(log_dir) / experiment_name
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Python logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = self.log_dir / 'experiment.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # Metrics storage
        self.metrics_history: List[Dict[str, Any]] = []
        self.best_metrics: Dict[str, float] = {}
        
        # Log start
        self.info(f"Experiment '{experiment_name}' started")
        self.info(f"Log directory: {self.log_dir}")
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)
    
    def log_config(self, config: Dict[str, Any]):
        """
        Log and save experiment configuration.
        
        Args:
            config: Configuration dictionary
        """
        # Save to file
        config_path = self.log_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        self.info(f"Configuration saved to {config_path}")
        
        # Log key parameters
        self.info("=" * 50)
        self.info("EXPERIMENT CONFIGURATION")
        self.info("=" * 50)
        self._log_dict(config, indent=0)
        self.info("=" * 50)
    
    def _log_dict(self, d: Dict, indent: int = 0):
        """Recursively log dictionary contents."""
        prefix = "  " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                self.info(f"{prefix}{key}:")
                self._log_dict(value, indent + 1)
            else:
                self.info(f"{prefix}{key}: {value}")
    
    def log_epoch(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metric values
            is_best: Whether this is the best epoch so far
        """
        # Add epoch to metrics
        metrics_with_epoch = {'epoch': epoch, **metrics}
        self.metrics_history.append(metrics_with_epoch)
        
        # Update best metrics
        if is_best:
            self.best_metrics = metrics.copy()
        
        # Format log message
        msg_parts = [f"Epoch {epoch:3d}"]
        for key, value in metrics.items():
            if isinstance(value, float):
                msg_parts.append(f"{key}: {value:.6f}")
            else:
                msg_parts.append(f"{key}: {value}")
        
        if is_best:
            msg_parts.append("*BEST*")
        
        self.info(" | ".join(msg_parts))
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """
        Log a set of metrics.
        
        Args:
            metrics: Dictionary of metrics
            prefix: Optional prefix for log message
        """
        if prefix:
            self.info(f"{prefix}:")
        
        for key, value in metrics.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.6f}")
            else:
                self.info(f"  {key}: {value}")
    
    def log_model_summary(self, model_summary: str):
        """Log model architecture summary."""
        self.info("MODEL ARCHITECTURE")
        self.info("-" * 40)
        for line in model_summary.split('\n'):
            self.info(line)
        self.info("-" * 40)
    
    def log_pruning_round(
        self,
        round_idx: int,
        sparsity: float,
        metrics: Dict[str, float]
    ):
        """
        Log pruning round results.
        
        Args:
            round_idx: Pruning round number
            sparsity: Current sparsity level
            metrics: Performance metrics
        """
        self.info(f"PRUNING ROUND {round_idx}")
        self.info(f"  Sparsity: {sparsity:.2%}")
        self.info(f"  Remaining weights: {(1-sparsity)*100:.1f}%")
        
        for key, value in metrics.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.6f}")
    
    def log_adversarial_results(
        self,
        clean_metrics: Dict[str, float],
        adversarial_metrics: Dict[str, float],
        attack_type: str = "PGD"
    ):
        """
        Log adversarial evaluation results.
        
        Args:
            clean_metrics: Metrics on clean data
            adversarial_metrics: Metrics on adversarial data
            attack_type: Type of attack used
        """
        self.info(f"ADVERSARIAL EVALUATION ({attack_type})")
        self.info("-" * 40)
        
        self.info("Clean Performance:")
        for key, value in clean_metrics.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.6f}")
        
        self.info("Adversarial Performance:")
        for key, value in adversarial_metrics.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.6f}")
        
        # Compute gaps
        for key in clean_metrics:
            if key in adversarial_metrics:
                clean_val = clean_metrics[key]
                adv_val = adversarial_metrics[key]
                if isinstance(clean_val, float) and isinstance(adv_val, float):
                    gap = clean_val - adv_val
                    self.info(f"  {key}_gap: {gap:.6f}")
        
        self.info("-" * 40)
    
    def save_metrics_history(self):
        """Save metrics history to JSON file."""
        metrics_path = self.log_dir / 'metrics_history.json'
        
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        self.info(f"Metrics history saved to {metrics_path}")
    
    def save_results(self, results: Dict[str, Any], filename: str = 'results.json'):
        """
        Save final results to JSON file.
        
        Args:
            results: Results dictionary
            filename: Output filename
        """
        results_path = self.log_dir / filename
        
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        results_converted = convert(results)
        
        with open(results_path, 'w') as f:
            json.dump(results_converted, f, indent=2)
        
        self.info(f"Results saved to {results_path}")
    
    def log_comparison_table(
        self,
        results: Dict[str, Dict[str, float]],
        metric_keys: Optional[List[str]] = None
    ):
        """
        Log a comparison table for multiple models/experiments.
        
        Args:
            results: Dict mapping name -> metrics
            metric_keys: Which metrics to include
        """
        if metric_keys is None:
            metric_keys = ['pnl_mean', 'pnl_std', 'cvar_05', 'sharpe_ratio']
        
        self.info("=" * 80)
        self.info("COMPARISON TABLE")
        self.info("=" * 80)
        
        # Header
        header = f"{'Model':<25}"
        for key in metric_keys:
            header += f" {key:>12}"
        self.info(header)
        self.info("-" * 80)
        
        # Rows
        for name, metrics in results.items():
            row = f"{name:<25}"
            for key in metric_keys:
                value = metrics.get(key, float('nan'))
                if isinstance(value, float):
                    row += f" {value:>12.4f}"
                else:
                    row += f" {str(value):>12}"
            self.info(row)
        
        self.info("=" * 80)
    
    def finalize(self):
        """Finalize logging and save all data."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        self.info("=" * 50)
        self.info("EXPERIMENT COMPLETED")
        self.info(f"Duration: {duration}")
        self.info(f"Log directory: {self.log_dir}")
        
        if self.best_metrics:
            self.info("Best metrics:")
            for key, value in self.best_metrics.items():
                if isinstance(value, float):
                    self.info(f"  {key}: {value:.6f}")
        
        self.info("=" * 50)
        
        # Save metrics history
        self.save_metrics_history()
        
        # Save summary
        summary = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'best_metrics': self.best_metrics,
            'total_epochs': len(self.metrics_history)
        }
        self.save_results(summary, 'summary.json')


def create_logger(
    experiment_name: str,
    config: Optional[Dict] = None,
    log_dir: str = 'logs'
) -> ExperimentLogger:
    """
    Factory function to create an experiment logger.
    
    Args:
        experiment_name: Name of experiment
        config: Optional configuration to log
        log_dir: Log directory
        
    Returns:
        ExperimentLogger instance
    """
    logger = ExperimentLogger(experiment_name, log_dir)
    
    if config is not None:
        logger.log_config(config)
    
    return logger


class ProgressBar:
    """
    Simple progress bar for training loops.
    """
    
    def __init__(
        self,
        total: int,
        prefix: str = '',
        length: int = 40,
        fill: str = '█'
    ):
        """
        Initialize progress bar.
        
        Args:
            total: Total iterations
            prefix: Prefix string
            length: Bar length in characters
            fill: Fill character
        """
        self.total = total
        self.prefix = prefix
        self.length = length
        self.fill = fill
        self.current = 0
    
    def update(self, current: Optional[int] = None, suffix: str = ''):
        """
        Update progress bar.
        
        Args:
            current: Current iteration (if None, increment by 1)
            suffix: Suffix string with additional info
        """
        if current is not None:
            self.current = current
        else:
            self.current += 1
        
        percent = self.current / self.total
        filled_length = int(self.length * percent)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        
        print(f'\r{self.prefix} |{bar}| {percent:.1%} {suffix}', end='', flush=True)
        
        if self.current >= self.total:
            print()
    
    def reset(self):
        """Reset progress bar."""
        self.current = 0


class MetricsTracker:
    """
    Track and compute running statistics for metrics.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics: Dict[str, List[float]] = {}
    
    def update(self, name: str, value: float):
        """Add a value to a metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def update_dict(self, metrics: Dict[str, float]):
        """Add multiple metrics at once."""
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.update(name, float(value))
    
    def get_mean(self, name: str) -> float:
        """Get mean of a metric."""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        return np.mean(self.metrics[name])
    
    def get_std(self, name: str) -> float:
        """Get standard deviation of a metric."""
        if name not in self.metrics or len(self.metrics[name]) < 2:
            return 0.0
        return np.std(self.metrics[name])
    
    def get_last(self, name: str) -> float:
        """Get last value of a metric."""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        return self.metrics[name][-1]
    
    def get_all_means(self) -> Dict[str, float]:
        """Get means of all metrics."""
        return {name: self.get_mean(name) for name in self.metrics}
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
    
    def reset_metric(self, name: str):
        """Reset a specific metric."""
        if name in self.metrics:
            self.metrics[name] = []