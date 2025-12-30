"""
Logging utilities for training and evaluation
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Logger for training metrics and events
    """
    
    def __init__(self, log_dir: str, experiment_name: str, use_tensorboard: bool = True):
        """
        Initialize logger
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
            use_tensorboard: Whether to use TensorBoard
        """
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard
        
        # TensorBoard writer
        self.writer = None
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Metrics storage
        self.metrics = {}
        
        # Log file
        self.log_file = self.log_dir / "training.log"
        
    def log(self, message: str, print_message: bool = True) -> None:
        """
        Log a message to file
        
        Args:
            message: Message to log
            print_message: Whether to print to console
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        if print_message:
            print(log_entry)
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        """
        Log metrics to TensorBoard and storage
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Current step (epoch)
            prefix: Prefix for metric names (e.g., 'train/', 'val/')
        """
        for name, value in metrics.items():
            full_name = f"{prefix}{name}" if prefix else name
            
            # Store metric
            if full_name not in self.metrics:
                self.metrics[full_name] = []
            self.metrics[full_name].append((step, value))
            
            # Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar(full_name, value, step)
    
    def log_histogram(self, name: str, values, step: int) -> None:
        """
        Log histogram to TensorBoard
        
        Args:
            name: Name of the histogram
            values: Values to plot
            step: Current step
        """
        if self.writer is not None:
            self.writer.add_histogram(name, values, step)
    
    def save_metrics(self, filename: str = "metrics.json") -> None:
        """
        Save metrics to JSON file
        
        Args:
            filename: Name of the file to save
        """
        save_path = self.log_dir / filename
        
        # Convert to serializable format
        metrics_dict = {}
        for name, values in self.metrics.items():
            metrics_dict[name] = {
                'steps': [v[0] for v in values],
                'values': [v[1] for v in values]
            }
        
        with open(save_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        self.log(f"Metrics saved to {save_path}")
    
    def close(self) -> None:
        """
        Close logger and TensorBoard writer
        """
        if self.writer is not None:
            self.writer.close()
        
        self.save_metrics()


def setup_logger(config: Dict[str, Any]) -> Logger:
    """
    Setup logger from config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Logger instance
    """
    log_dir = config['logging']['tensorboard_dir']
    experiment_name = config['experiment_name']
    use_tensorboard = config['logging']['use_tensorboard']
    
    logger = Logger(log_dir, experiment_name, use_tensorboard)
    logger.log(f"Logger initialized for experiment: {experiment_name}")
    
    return logger