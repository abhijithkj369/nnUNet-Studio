"""
Plotter for nnUNetv2 Training Metrics
Creates visualizations for training progress
"""
import matplotlib
import matplotlib.style
matplotlib.use('Agg')  # Use non-interactive backend
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
import plotly.graph_objects as go
import os


class MetricsPlotter:
    """Creates plots for training metrics"""
    
    def __init__(self, output_dir: str = "./plots"):
        """
        Initialize plotter
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set style
        try:
            matplotlib.style.use('seaborn-v0_8-darkgrid')
        except Exception:
            matplotlib.style.use('default')
    
    def _save_figure(self, fig: Figure, filename: str) -> str:
        """Save figure to file and return path"""
        filepath = os.path.join(self.output_dir, filename)
        
        # Use FigureCanvasAgg to render the figure to a file
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(filepath)
        
        # Close the figure to free memory
        plt.close(fig)
        
        return filepath
    
    def plot_losses(self, epochs: List[int], train_losses: List[float], 
                   val_losses: List[float], save_path: Optional[str] = None) -> str:
        """
        Plot training and validation losses
        
        Args:
            epochs: List of epoch numbers
            train_losses: List of training losses
            val_losses: List of validation losses
            save_path: Path to save plot. If None, auto-generate
            
        Returns:
            Path to saved plot
        """
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        has_data = False
        
        if not epochs:
            ax.text(0.5, 0.5, 'No data available yet', 
                   ha='center', va='center', fontsize=14)
            ax.set_title('Training Progress')
        else:
            if train_losses and len(train_losses) == len(epochs):
                ax.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
                has_data = True
            
            if val_losses and len(val_losses) == len(epochs):
                ax.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=4)
                has_data = True
            
            ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
            if has_data:
                ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        
        if save_path is None:
            save_path = self.output_dir / "loss_plot.png"
        else:
            save_path = Path(save_path)
        
        canvas = FigureCanvasAgg(fig)
        fig.tight_layout()
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        
        return str(save_path)
    
    def plot_dice_scores(self, epochs: List[int], dice_scores: List[float],
                        save_path: Optional[str] = None) -> str:
        """
        Plot dice scores over epochs
        
        Args:
            epochs: List of epoch numbers
            dice_scores: List of dice scores
            save_path: Path to save plot. If None, auto-generate
            
        Returns:
            Path to saved plot
        """
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        has_data = False
        
        if not epochs or not dice_scores:
            ax.text(0.5, 0.5, 'No dice score data available yet', 
                   ha='center', va='center', fontsize=14)
            ax.set_title('Dice Score Progress')
        else:
            if len(dice_scores) == len(epochs):
                ax.plot(epochs, dice_scores, 'g-^', label='Dice Score', linewidth=2, markersize=4)
                has_data = True
                
                # Add horizontal line at 0.9 for reference (good dice score)
                if dice_scores:
                    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Target (0.9)')
            
            ax.set_title('Dice Score Progress', fontsize=14, fontweight='bold')
            ax.set_ylim([0, 1])
            if has_data:
                ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Dice Score', fontsize=12)
        
        if save_path is None:
            save_path = self.output_dir / "dice_plot.png"
        else:
            save_path = Path(save_path)
        
        canvas = FigureCanvasAgg(fig)
        fig.tight_layout()
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        
        return str(save_path)
    
    def plot_combined(self, epochs: List[int], train_losses: List[float],
                     val_losses: List[float], dice_scores: List[float],
                     save_path: Optional[str] = None) -> str:
        """
        Create combined plot with losses and dice scores
        
        Args:
            epochs: List of epoch numbers
            train_losses: List of training losses
            val_losses: List of validation losses
            dice_scores: List of dice scores
            save_path: Path to save plot. If None, auto-generate
            
        Returns:
            Path to saved plot
        """
        fig = Figure(figsize=(16, 6))
        
        if not epochs:
            # Create empty plot
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            for ax in [ax1, ax2]:
                ax.text(0.5, 0.5, 'No data available yet', 
                       ha='center', va='center', fontsize=14)
            ax1.set_title('Loss Progress')
            ax2.set_title('Dice Score Progress')
        else:
            ax1, ax2 = fig.subplots(1, 2)
            
            # Plot losses
            has_loss_data = False
            if train_losses and len(train_losses) == len(epochs):
                ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
                has_loss_data = True
            if val_losses and len(val_losses) == len(epochs):
                ax1.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=4)
                has_loss_data = True
            
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
            if has_loss_data:
                ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Plot dice scores
            has_dice_data = False
            if dice_scores and len(dice_scores) == len(epochs):
                ax2.plot(epochs, dice_scores, 'g-^', label='Dice Score', linewidth=2, markersize=4)
                ax2.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Target (0.9)')
                has_dice_data = True
            
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Dice Score', fontsize=12)
            ax2.set_title('Dice Score Progress', fontsize=14, fontweight='bold')
            ax2.set_ylim([0, 1])
            if has_dice_data:
                ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = self.output_dir / "combined_plot.png"
        else:
            save_path = Path(save_path)
        
        canvas = FigureCanvasAgg(fig)
        fig.tight_layout()
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        
        return str(save_path)
    
    def create_all_plots(self, plot_data: dict) -> Tuple[str, str, str]:
        """
        Create all plots at once
        
        Args:
            plot_data: Dictionary with 'epochs', 'train_losses', 'val_losses', 'dice_scores'
            
        Returns:
            Tuple of (loss_plot_path, dice_plot_path, combined_plot_path)
        """
        epochs = plot_data.get('epochs', [])
        train_losses = plot_data.get('train_losses', [])
        val_losses = plot_data.get('val_losses', [])
        dice_scores = plot_data.get('dice_scores', [])
        
        loss_plot = self.plot_losses(epochs, train_losses, val_losses)
        dice_plot = self.plot_dice_scores(epochs, dice_scores)
        combined_plot = self.plot_combined(epochs, train_losses, val_losses, dice_scores)
        
        return loss_plot, dice_plot, combined_plot

    def create_interactive_plots(self, data: Dict[str, List]) -> Tuple[Optional[go.Figure], Optional[go.Figure]]:
        """
        Create interactive plots using Plotly
        
        Args:
            data: Dictionary containing lists of metrics
            
        Returns:
            Tuple of (loss_figure, dice_figure)
        """
        epochs = data.get('epochs', [])
        train_losses = data.get('train_losses', [])
        val_losses = data.get('val_losses', [])
        dice_scores = data.get('dice_scores', [])
        
        if not epochs:
            return None, None
            
        # Create Loss Plot
        loss_fig = go.Figure()
        
        if train_losses and len(train_losses) == len(epochs):
            loss_fig.add_trace(go.Scatter(
                x=epochs, y=train_losses,
                mode='lines+markers',
                name='Train Loss',
                line=dict(color='blue')
            ))
            
        if val_losses and len(val_losses) == len(epochs):
            loss_fig.add_trace(go.Scatter(
                x=epochs, y=val_losses,
                mode='lines+markers',
                name='Validation Loss',
                line=dict(color='orange')
            ))
            
        loss_fig.update_layout(
            title='Training and Validation Loss',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            template='plotly_white',
            hovermode='x unified'
        )
        
        # Create Dice Plot
        dice_fig = go.Figure()
        
        if dice_scores and len(dice_scores) == len(epochs):
            dice_fig.add_trace(go.Scatter(
                x=epochs, y=dice_scores,
                mode='lines+markers',
                name='Dice Score',
                line=dict(color='green')
            ))
            
        dice_fig.update_layout(
            title='Dice Score',
            xaxis_title='Epoch',
            yaxis_title='Dice Score',
            template='plotly_white',
            hovermode='x unified',
            yaxis=dict(range=[0, 1])  # Dice is between 0 and 1
        )
        
        return loss_fig, dice_fig
