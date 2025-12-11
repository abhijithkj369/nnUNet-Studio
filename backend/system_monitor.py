"""
System Monitor
Collects CPU and GPU usage metrics
"""
import psutil
import shutil
import subprocess
import time
from typing import Dict, Optional

class SystemMonitor:
    """Monitors system resources (CPU, RAM, GPU)"""
    
    def __init__(self):
        self.has_gpu = False
        self.gpu_name = "Unknown"
        self._check_gpu()
        
    def _check_gpu(self):
        """Check if NVIDIA GPU is available"""
        try:
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
            if result.returncode == 0:
                self.has_gpu = True
                # Parse first GPU name
                # Output format: GPU 0: NVIDIA RTX 6000 Ada Generation (UUID: ...)
                line = result.stdout.strip().split('\n')[0]
                if ':' in line:
                    self.gpu_name = line.split(':', 1)[1].split('(')[0].strip()
        except Exception:
            self.has_gpu = False
            
    def get_metrics(self) -> Dict:
        """Get current system metrics"""
        metrics = {
            'cpu_percent': psutil.cpu_percent(interval=None),
            'ram_percent': psutil.virtual_memory().percent,
            'ram_used_gb': round(psutil.virtual_memory().used / (1024**3), 1),
            'ram_total_gb': round(psutil.virtual_memory().total / (1024**3), 1),
            'gpu_util': 0,
            'gpu_mem_used_gb': 0,
            'gpu_mem_total_gb': 0,
            'gpu_name': self.gpu_name
        }
        
        if self.has_gpu:
            try:
                # Get GPU metrics via nvidia-smi
                # query: utilization.gpu, memory.used, memory.total
                cmd = ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                       '--format=csv,noheader,nounits']
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    line = result.stdout.strip().split('\n')[0]
                    parts = [x.strip() for x in line.split(',')]
                    if len(parts) >= 3:
                        metrics['gpu_util'] = float(parts[0])
                        metrics['gpu_mem_used_gb'] = round(float(parts[1]) / 1024, 1)
                        metrics['gpu_mem_total_gb'] = round(float(parts[2]) / 1024, 1)
            except Exception:
                pass
                
        return metrics
