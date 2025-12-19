"""
nnUNetv2 Training Interface
A Gradio-based UI for training medical image segmentation models
"""
import gradio as gr
import os
import time
from pathlib import Path
from datetime import datetime

# Import backend modules
from backend.dataset_manager import DatasetManager
from backend.trainer import nnUNetTrainer
from backend.metrics_parser import MetricsParser
from backend.plotter import MetricsPlotter
from backend.system_monitor import SystemMonitor
import config

# Global variables
current_trainer = None
metrics_parser = None
plotter = None
system_monitor = SystemMonitor()
training_logs = []


def validate_dataset_folder(folder_path):
    """Validate dataset folder structure"""
    if not folder_path:
        return "⚠️ Please select a dataset folder", None
    
    try:
        manager = DatasetManager(folder_path)
        is_valid, message = manager.validate_structure()
        
        if is_valid:
            return f"✅ {message}", "valid"
        else:
            return f"❌ {message}", "invalid"
    except Exception as e:
        return f"❌ Error: {str(e)}", "invalid"


def generate_dataset_json(folder_path, dataset_name, description, modality_names):
    """Generate dataset.json file"""
    if not folder_path:
        return "⚠️ Please select a dataset folder first", None
    
    try:
        manager = DatasetManager(folder_path)
        
        # Parse modality names if provided
        modalities = None
        if modality_names:
            modalities = [m.strip() for m in modality_names.split(',')]
        
        success, message, dataset_dict = manager.generate_dataset_json(
            dataset_name=dataset_name,
            description=description,
            modality_names=modalities
        )
        
        if success:
            # Format the dataset.json for display
            import json
            json_str = json.dumps(dataset_dict, indent=2)
            return f"✅ {message}", json_str
        else:
            return f"❌ {message}", None
            
    except Exception as e:
        return f"❌ Error: {str(e)}", None


def setup_training(dataset_folder, nnunet_raw, nnunet_preprocessed, nnunet_results, dataset_id):
    """Setup training environment and prepare dataset"""
    global current_trainer
    
    try:
        # Create trainer instance
        current_trainer = nnUNetTrainer(
            nnunet_raw=nnunet_raw,
            nnunet_preprocessed=nnunet_preprocessed,
            nnunet_results=nnunet_results
        )
        
        # Setup environment
        env_vars = current_trainer.setup_environment()
        
        # Prepare dataset
        success, message = current_trainer.prepare_dataset(dataset_folder, dataset_id)
        
        if success:
            return f"✅ Training environment setup complete\n{message}\n\nEnvironment variables:\n" + \
                   "\n".join([f"  {k}: {v}" for k, v in env_vars.items()])
        else:
            return f"❌ {message}"
            
    except Exception as e:
        return f"❌ Error setting up training: {str(e)}"


def run_preprocessing(dataset_id, configurations, progress=gr.Progress()):
    """Run nnUNet preprocessing"""
    global current_trainer
    
    if current_trainer is None:
        return "❌ Please setup training environment first"
    
    try:
        logs = []
        
        def log_callback(line):
            logs.append(line)
            progress(0.5, desc="Preprocessing...")
        
        progress(0.1, desc="Starting preprocessing...")
        
        # Convert configurations to list if needed
        config_list = configurations if configurations else None
        
        success, message = current_trainer.run_preprocessing(dataset_id, config_list, log_callback)
        
        progress(1.0, desc="Preprocessing complete" if success else "Preprocessing failed")
        
        if success:
            return f"✅ {message}\n\n" + "".join(logs[-50:])  # Show last 50 lines
        else:
            return f"❌ {message}\n\n" + "".join(logs[-50:])
            
    except Exception as e:
        return f"❌ Error during preprocessing: {str(e)}"


def scan_datasets(nnunet_preprocessed):
    """Scan for available datasets in nnUNet_preprocessed"""
    try:
        path = Path(nnunet_preprocessed)
        if not path.exists():
            return []
        
        datasets = []
        for item in path.iterdir():
            if item.is_dir() and item.name.startswith("Dataset"):
                # Extract ID and Name
                # Format: DatasetXXX_Name
                datasets.append(item.name)
        
        return sorted(datasets)
    except Exception:
        return []


def start_training_process(dataset_selection, num_epochs, fold, configuration, 
                          nnunet_raw, nnunet_preprocessed, nnunet_results, num_workers, progress=gr.Progress()):
    """Start the training process"""
    global current_trainer, metrics_parser, plotter, training_logs
    
    # Parse dataset ID from selection (e.g., "Dataset035_KneeData" -> 35)
    try:
        if not dataset_selection:
            return "❌ Please select a dataset", None, None
            
        dataset_id = int(dataset_selection.split('_')[0].replace('Dataset', ''))
    except Exception:
        return "❌ Invalid dataset selection", None, None
    
    # Initialize trainer if needed (e.g. if skipped setup tab)
    if current_trainer is None:
        try:
            current_trainer = nnUNetTrainer(
                nnunet_raw=nnunet_raw,
                nnunet_preprocessed=nnunet_preprocessed,
                nnunet_results=nnunet_results
            )
            current_trainer.setup_environment()
        except Exception as e:
            return f"❌ Error initializing trainer: {str(e)}", None, None
    
    try:
        # Initialize metrics parser and plotter
        metrics_parser = MetricsParser(max_epochs=int(num_epochs))
        plotter = MetricsPlotter(output_dir="./plots")
        training_logs = []
        
        def log_callback(line):
            training_logs.append(line)
            # Update metrics
            metrics_parser.update_from_line(line)
        
        progress(0.1, desc="Starting training...")
        
        success, message = current_trainer.start_training(
            dataset_id=dataset_id,
            fold=fold,
            num_epochs=num_epochs,
            configuration=configuration,
            num_workers=int(num_workers),
            log_callback=log_callback
        )
        
        if success:
            return f"✅ {message}\n\nTraining is running. Check the logs and plots below.", None, None
        else:
            return f"❌ {message}", None, None
            
    except Exception as e:
        return f"❌ Error starting training: {str(e)}", None, None


def update_training_status():
    """Update training logs, plots, and system metrics"""
    global current_trainer, metrics_parser, plotter, training_logs, system_monitor
    
    # Get system metrics
    sys_metrics = system_monitor.get_metrics()
    
    # Format system metrics HTML
    sys_html = f"""
    <div style="display: flex; gap: 20px; margin-bottom: 10px;">
        <div style="flex: 1; background: #f0f0f0; padding: 10px; border-radius: 5px;">
            <strong>CPU Usage:</strong> {sys_metrics['cpu_percent']}%
            <div style="background: #ddd; height: 10px; border-radius: 5px; margin-top: 5px;">
                <div style="background: #4caf50; width: {sys_metrics['cpu_percent']}%; height: 100%; border-radius: 5px;"></div>
            </div>
            <small>RAM: {sys_metrics['ram_used_gb']}/{sys_metrics['ram_total_gb']} GB ({sys_metrics['ram_percent']}%)</small>
        </div>
        <div style="flex: 1; background: #f0f0f0; padding: 10px; border-radius: 5px;">
            <strong>GPU: {sys_metrics['gpu_name']}</strong>
            <div style="margin-top: 5px;">Util: {sys_metrics['gpu_util']}%</div>
            <div style="background: #ddd; height: 10px; border-radius: 5px; margin-top: 5px;">
                <div style="background: #2196f3; width: {sys_metrics['gpu_util']}%; height: 100%; border-radius: 5px;"></div>
            </div>
            <small>VRAM: {sys_metrics['gpu_mem_used_gb']}/{sys_metrics['gpu_mem_total_gb']} GB</small>
        </div>
    </div>
    """
    
    if current_trainer is None or not current_trainer.is_training_running():
        if training_logs:
            # Training finished, show final status
            log_text = "".join(training_logs[-100:])  # Last 100 lines
            
            # Generate final plots
            if metrics_parser:
                plot_data = metrics_parser.get_plot_data()
                loss_plot, dice_plot, combined_plot = plotter.create_all_plots(plot_data)
                # Get final progress
                current_epoch, max_epochs, percent = metrics_parser.get_progress()
                progress_html = f"""
                <div style="margin-top: 10px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <strong>Progress: Epoch {current_epoch}/{max_epochs}</strong>
                        <span>{percent:.1f}%</span>
                    </div>
                    <div style="background: #ddd; height: 20px; border-radius: 10px; overflow: hidden;">
                        <div style="background: #4caf50; width: {percent}%; height: 100%; transition: width 0.5s;"></div>
                    </div>
                </div>
                """
                return log_text, loss_plot, dice_plot, "Training completed!", sys_html, progress_html
        
        return "No training running", None, None, "Idle", sys_html, "<div>Ready to train</div>"
    
    # Get latest logs
    log_text = "".join(training_logs[-100:]) if training_logs else "Waiting for logs..."
    
    # Update plots
    loss_plot_path = None
    dice_plot_path = None
    
    if metrics_parser and plotter:
        try:
            plot_data = metrics_parser.get_plot_data()
            loss_plot_path, dice_plot_path, _ = plotter.create_all_plots(plot_data)
        except Exception as e:
            print(f"Error updating plots: {e}")
    
    
    # Get current metrics
    if metrics_parser:
        latest = metrics_parser.metrics.get_latest()
        epoch = latest.get('epoch', 'N/A')
        train_loss = latest.get('train_loss')
        val_loss = latest.get('val_loss')
        dice = latest.get('dice')
        
        # Format values
        train_loss_str = f"{train_loss:.4f}" if train_loss is not None else "N/A"
        val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
        dice_str = f"{dice:.4f}" if dice is not None else "N/A"
        
        # Get progress
        current_epoch, max_epochs, percent = metrics_parser.get_progress()
        
        # Create progress bar HTML
        progress_html = f"""
        <div style="margin-top: 10px; margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <strong>Progress: Epoch {current_epoch}/{max_epochs}</strong>
                <span>{percent:.1f}%</span>
            </div>
            <div style="background: #ddd; height: 20px; border-radius: 10px; overflow: hidden;">
                <div style="background: #4caf50; width: {percent}%; height: 100%; transition: width 0.5s;"></div>
            </div>
        </div>
        """
        
        status = f"Epoch: {epoch} | Train Loss: {train_loss_str} | Val Loss: {val_loss_str} | Dice: {dice_str}"
            
    else:
        status = "Training in progress..."
        progress_html = "<div>Waiting for metrics...</div>"
    
    return log_text, loss_plot_path, dice_plot_path, status, sys_html, progress_html


def stop_training():
    """Stop the training process"""
    global current_trainer
    
    if current_trainer:
        success, message = current_trainer.stop_training()
        if success:
            return f"✅ {message}"
        else:
            return f"⚠️ {message}"
    return "⚠️ No trainer instance found"


# Create Gradio Interface
with gr.Blocks(title=config.UI_TITLE) as app:
    gr.Markdown(f"# {config.UI_TITLE}")
    gr.Markdown(config.UI_DESCRIPTION)
    
    with gr.Tabs():
        # Tab 1: Dataset Preparation
        with gr.Tab("📁 Dataset Preparation"):
            gr.Markdown("### Step 1: Select and Validate Dataset")
            
            with gr.Row():
                with gr.Column():
                    dataset_folder_input = gr.Textbox(
                        label="Dataset Folder Path",
                        placeholder="Path to folder containing imagesTr and labelsTr",
                        info="Enter the path to your dataset folder"
                    )
                    validate_btn = gr.Button("🔍 Validate Dataset Structure", variant="primary")
                    validation_status = gr.Textbox(label="Validation Status", lines=3)
            
            gr.Markdown("### Step 2: Generate dataset.json")
            
            with gr.Row():
                with gr.Column():
                    dataset_name_input = gr.Textbox(
                        label="Dataset Name",
                        value="CustomDataset",
                        placeholder="e.g., BrainTumor, LiverSegmentation"
                    )
                    dataset_desc_input = gr.Textbox(
                        label="Dataset Description",
                        value="Medical Image Segmentation Dataset",
                        placeholder="Brief description of your dataset"
                    )
                    modality_input = gr.Textbox(
                        label="Modality Names (comma-separated)",
                        placeholder="e.g., CT, MRI, T1w, T2w",
                        info="Leave empty for auto-detection"
                    )
                    generate_json_btn = gr.Button("🔧 Generate dataset.json", variant="primary")
                
                with gr.Column():
                    json_status = gr.Textbox(label="Generation Status", lines=3)
                    json_preview = gr.Code(label="dataset.json Preview", language="json", lines=15)
            
            # Connect validation
            validate_btn.click(
                fn=validate_dataset_folder,
                inputs=[dataset_folder_input],
                outputs=[validation_status, gr.State()]
            )
            
            # Connect JSON generation
            generate_json_btn.click(
                fn=generate_dataset_json,
                inputs=[dataset_folder_input, dataset_name_input, dataset_desc_input, modality_input],
                outputs=[json_status, json_preview]
            )
        
        # Tab 2: Training Configuration
        with gr.Tab("⚙️ Training Configuration"):
            gr.Markdown("### Configure nnUNet Training")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**nnUNet Paths**")
                    nnunet_raw_input = gr.Textbox(
                        label="nnUNet_raw",
                        value=config.DEFAULT_NNUNET_RAW,
                        placeholder="Path for raw datasets"
                    )
                    nnunet_preprocessed_input = gr.Textbox(
                        label="nnUNet_preprocessed",
                        value=config.DEFAULT_NNUNET_PREPROCESSED,
                        placeholder="Path for preprocessed data"
                    )
                    nnunet_results_input = gr.Textbox(
                        label="nnUNet_results",
                        value=config.DEFAULT_NNUNET_RESULTS,
                        placeholder="Path for model results/checkpoints"
                    )
                
                with gr.Column():
                    gr.Markdown("**Training Parameters**")
                    dataset_id_input = gr.Number(
                        label="Dataset ID",
                        value=config.DEFAULT_DATASET_ID,
                        precision=0,
                        info="Integer ID for your dataset (e.g., 1 for Dataset001)"
                    )
                    
                    gr.Markdown("**Preprocessing Configuration**")
                    preprocess_configs = gr.CheckboxGroup(
                        label="Configurations to Preprocess",
                        choices=["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"],
                        value=["3d_fullres"],
                        info="Select which configurations to generate plans for"
                    )
            
            with gr.Row():
                setup_training_btn = gr.Button("🚀 Setup Training Environment", variant="primary")
                
            setup_status = gr.Textbox(label="Setup Status", lines=8)
            
            setup_training_btn.click(
                fn=setup_training,
                inputs=[dataset_folder_input, nnunet_raw_input, nnunet_preprocessed_input, 
                       nnunet_results_input, dataset_id_input],
                outputs=[setup_status]
            )
            
            with gr.Row():
                preprocess_btn = gr.Button("⚡ Run Preprocessing", variant="secondary")
            
            gr.Markdown("ℹ️ **Note**: Dataset verification is skipped by default to save RAM. Your data will still be preprocessed correctly.")
                
            preprocess_status = gr.Textbox(label="Preprocessing Status", lines=10)
            
            preprocess_btn.click(
                fn=run_preprocessing,
                inputs=[dataset_id_input, preprocess_configs],
                outputs=[preprocess_status]
            )
        
        # Tab 3: Training & Monitoring
        with gr.Tab("🎯 Training & Monitoring"):
            gr.Markdown("### Start Training and Monitor Progress")
            
            # System Monitor
            system_metrics_html = gr.HTML(label="System Metrics")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Experiment Selection**")
                    with gr.Row():
                        dataset_dropdown = gr.Dropdown(
                            label="Select Dataset",
                            choices=[],
                            interactive=True,
                            info="Datasets found in nnUNet_preprocessed"
                        )
                        refresh_datasets_btn = gr.Button("🔄", size="sm")
                    
                    config_dropdown = gr.Dropdown(
                        label="Configuration",
                        choices=["3d_fullres", "2d", "3d_lowres", "3d_cascade_fullres"],
                        value="3d_fullres",
                        info="Model configuration"
                    )
                    
                    with gr.Row():
                        fold_input = gr.Number(label="Fold", value=0, precision=0)
                        num_epochs_input = gr.Number(label="Epochs", value=config.DEFAULT_EPOCHS, precision=0)
                        num_workers_input = gr.Slider(
                            label="Number of Workers", 
                            value=config.DEFAULT_NUM_WORKERS, 
                            minimum=0, 
                            maximum=16, 
                            step=1,
                            info="Reduce this if you encounter memory errors (e.g., 2 for Windows)"
                        )
                
                with gr.Column():
                    gr.Markdown("**Control**")
                    start_train_btn = gr.Button("▶️ Start Training", variant="primary", size="lg")
                    stop_train_btn = gr.Button("⏹️ Stop Training", variant="stop", size="lg")
                    train_status = gr.Textbox(label="Status", lines=2)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Training Logs**")
                    log_output = gr.Textbox(label="Live Logs", lines=20, max_lines=30, autoscroll=True)
                    
                with gr.Column():
                    gr.Markdown("**Metrics Visualization**")
                    progress_output = gr.HTML(label="Progress")
                    current_status = gr.Textbox(label="Current Metrics", lines=2)
                    loss_plot = gr.Image(label="Loss Plot", type="filepath")
                    dice_plot = gr.Image(label="Dice Score Plot", type="filepath")
            
            # Auto-refresh logs and plots
            refresh_timer = gr.Timer(value=2, active=True)  # Refresh every 2 seconds
            
            # Event handlers
            def refresh_datasets(preprocessed_path):
                datasets = scan_datasets(preprocessed_path)
                return gr.Dropdown(choices=datasets)
            
            refresh_datasets_btn.click(
                fn=refresh_datasets,
                inputs=[nnunet_preprocessed_input],
                outputs=[dataset_dropdown]
            )
            
            # Initial load of datasets
            app.load(
                fn=refresh_datasets,
                inputs=[nnunet_preprocessed_input],
                outputs=[dataset_dropdown]
            )
            
            start_train_btn.click(
                fn=start_training_process,
                inputs=[dataset_dropdown, num_epochs_input, fold_input, config_dropdown,
                       nnunet_raw_input, nnunet_preprocessed_input, nnunet_results_input, num_workers_input],
                outputs=[train_status, loss_plot, dice_plot]
            )
            
            stop_train_btn.click(
                fn=stop_training,
                outputs=[train_status]
            )
            
            refresh_timer.tick(
                fn=update_training_status,
                outputs=[log_output, loss_plot, dice_plot, current_status, system_metrics_html, progress_output]
            )
    
    gr.Markdown("---")
    gr.Markdown("💡 **Tips**: Make sure nnUNetv2 is installed in your environment. Follow the tabs from left to right for the complete workflow.")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("./plots", exist_ok=True)
    
    print("🚀 Starting nnUNetv2 Training Interface...")
    print("📍 Access the interface at: http://localhost:7860")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
