"""
Metrics parser for nnUNet training logs
Extracts training metrics from nnUNet output
"""
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class TrainingMetrics:
    """Container for training metrics"""
    epochs: List[int] = field(default_factory=list)
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    dice_scores: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    
    def add_epoch(self, epoch: int, train_loss: Optional[float] = None, 
                  val_loss: Optional[float] = None, dice: Optional[float] = None,
                  lr: Optional[float] = None):
        """Add metrics for an epoch"""
        self.epochs.append(epoch)
        if train_loss is not None:
            self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if dice is not None:
            self.dice_scores.append(dice)
        if lr is not None:
            self.learning_rates.append(lr)
    
    def get_latest(self) -> Dict:
        """Get latest metrics"""
        if not self.epochs:
            return {}
        
        return {
            'epoch': self.epochs[-1] if self.epochs else None,
            'train_loss': self.train_losses[-1] if self.train_losses else None,
            'val_loss': self.val_losses[-1] if self.val_losses else None,
            'dice': self.dice_scores[-1] if self.dice_scores else None,
            'lr': self.learning_rates[-1] if self.learning_rates else None
        }


class MetricsParser:
    """Parse nnUNet training logs for metrics"""
    
    def __init__(self, max_epochs: int = 1000):
        self.metrics = TrainingMetrics()
        self.current_epoch = -1
        self.max_epochs = max_epochs
        self.best_dice = 0.0  # Track best dice score
        
        # Regex patterns for different log formats
        self.patterns = {
            'epoch': re.compile(r'epoch[:\s]+(\d+)', re.IGNORECASE),
            'train_loss': re.compile(r'train[_\s]?loss[:\s]+([-]?[\d\.]+)', re.IGNORECASE),
            'val_loss': re.compile(r'val[_\s]?loss[:\s]+([-]?[\d\.]+)', re.IGNORECASE),
            'dice': re.compile(r'dice[_\s]?score[:\s]+([\d\.]+)', re.IGNORECASE),
            'mean_dice': re.compile(r'mean[_\s]?dice[:\s]+([\d\.]+)', re.IGNORECASE),
            'lr': re.compile(r'lr[:\s]+([\d\.e\-]+)', re.IGNORECASE),
            
            # nnUNet specific patterns
            'nnunet_epoch': re.compile(r'Epoch\s+(\d+)'),
            'nnunet_train': re.compile(r'train_loss\s+([-]?[\d\.]+)'),
            'nnunet_val': re.compile(r'val_loss\s+([-]?[\d\.]+)'),
            'nnunet_dice': re.compile(r'Pseudo dice\s+([\d\.]+)'),
            'nnunet_dice_list': re.compile(r'Pseudo dice\s+\[(.*?)\]'),
            'best_dice': re.compile(r'New best EMA pseudo Dice: ([\d\.]+)')  # Pattern for best dice
        }

    
    def parse_line(self, line: str) -> Optional[Dict]:
        """
        Parse a single log line for metrics
        
        Args:
            line: Log line to parse
            
        Returns:
            Dictionary of extracted metrics or None
        """
        extracted = {}
        
        # Try to extract epoch
        for pattern_name in ['epoch', 'nnunet_epoch']:
            match = self.patterns[pattern_name].search(line)
            if match:
                extracted['epoch'] = int(match.group(1))
                break
        
        # Try to extract train loss
        for pattern_name in ['train_loss', 'nnunet_train']:
            match = self.patterns[pattern_name].search(line)
            if match:
                extracted['train_loss'] = float(match.group(1))
                break
        
        # Try to extract val loss
        for pattern_name in ['val_loss', 'nnunet_val']:
            match = self.patterns[pattern_name].search(line)
            if match:
                extracted['val_loss'] = float(match.group(1))
                break
        
        # Try to extract dice score
        for pattern_name in ['dice', 'mean_dice', 'nnunet_dice']:
            match = self.patterns[pattern_name].search(line)
            if match:
                extracted['dice'] = float(match.group(1))
                break
        
        # Try to extract dice score from list (nnUNet v2 format)
        if 'dice' not in extracted:
            match = self.patterns['nnunet_dice_list'].search(line)
            if match:
                try:
                    content = match.group(1)
                    # Clean up: remove np.float32( and )
                    content = content.replace('np.float32(', '').replace(')', '')
                    # Split and convert
                    values = [float(x.strip()) for x in content.split(',')]
                    if values:
                        extracted['dice'] = sum(values) / len(values)
                except Exception:
                    pass

        
        # Try to extract learning rate
        match = self.patterns['lr'].search(line)
        if match:
            extracted['lr'] = float(match.group(1))

        # Try to extract best dice
        match = self.patterns['best_dice'].search(line)
        if match:
            extracted['best_dice'] = float(match.group(1))
        
        return extracted if extracted else None
    
    def update_from_line(self, line: str):
        """
        Update metrics from a log line
        
        Args:
            line: Log line to parse
        """
        parsed = self.parse_line(line)
        if not parsed:
            return

        # Update current epoch if found
        if 'epoch' in parsed:
            self.current_epoch = parsed['epoch']
            
            # Initialize new epoch if needed
            if not self.metrics.epochs or self.current_epoch not in self.metrics.epochs:
                self.metrics.add_epoch(epoch=self.current_epoch)
        
        # Update metrics for current epoch
        if self.current_epoch >= 0:
            # Ensure epoch exists in metrics (handle case where we missed the Epoch line but got metrics)
            if not self.metrics.epochs or self.current_epoch not in self.metrics.epochs:
                self.metrics.add_epoch(epoch=self.current_epoch)
            
            idx = self.metrics.epochs.index(self.current_epoch)
            
            if 'train_loss' in parsed:
                # Update or append
                if idx < len(self.metrics.train_losses):
                    self.metrics.train_losses[idx] = parsed['train_loss']
                else:
                    self.metrics.train_losses.append(parsed['train_loss'])
                    
            if 'val_loss' in parsed:
                if idx < len(self.metrics.val_losses):
                    self.metrics.val_losses[idx] = parsed['val_loss']
                else:
                    self.metrics.val_losses.append(parsed['val_loss'])
                    
            if 'dice' in parsed:
                if idx < len(self.metrics.dice_scores):
                    self.metrics.dice_scores[idx] = parsed['dice']
                else:
                    self.metrics.dice_scores.append(parsed['dice'])
                    
            if 'lr' in parsed:
                if idx < len(self.metrics.learning_rates):
                    self.metrics.learning_rates[idx] = parsed['lr']
                else:
                    self.metrics.learning_rates.append(parsed['lr'])

        # Update best dice if found
        if 'best_dice' in parsed:
            self.best_dice = parsed['best_dice']

    def get_progress(self) -> tuple[int, int, float]:
        """
        Get training progress
        
        Returns:
            Tuple of (current_epoch, max_epochs, percentage)
        """
        current = self.current_epoch + 1 if self.current_epoch >= 0 else 0
        percentage = (current / self.max_epochs) * 100 if self.max_epochs > 0 else 0
        return current, self.max_epochs, min(percentage, 100.0)
    
    def get_metrics(self) -> TrainingMetrics:
        """Get all collected metrics"""
        return self.metrics
    
    def get_plot_data(self) -> Dict[str, List]:
        """
        Get data formatted for plotting
        
        Returns:
            Dictionary with lists for plotting
        """
        return {
            'epochs': self.metrics.epochs,
            'train_losses': self.metrics.train_losses,
            'val_losses': self.metrics.val_losses,
            'dice_scores': self.metrics.dice_scores,
            'learning_rates': self.metrics.learning_rates,
            'best_dice': self.best_dice  # Include best_dice in plot data
        }
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = TrainingMetrics()
        self.current_epoch = -1
        self.best_dice = 0.0  # Reset best_dice as well
