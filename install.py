#!/usr/bin/env python3
"""
Enhanced Chicken Counter - Automatic Installation Script
Handles dependency installation, environment setup, and system validation

Features:
- Automatic dependency installation
- Environment validation
- GPU detection and setup
- Model downloading
- Configuration file creation
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time

# ANSI color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_colored(text: str, color: str = Colors.WHITE, bold: bool = False):
    """Print colored text to terminal"""
    style = Colors.BOLD if bold else ""
    print(f"{style}{color}{text}{Colors.END}")

def print_header(text: str):
    """Print section header"""
    print_colored(f"\n{'='*60}", Colors.CYAN, bold=True)
    print_colored(f" {text}", Colors.CYAN, bold=True)
    print_colored(f"{'='*60}", Colors.CYAN, bold=True)

def print_step(step: str, status: str = ""):
    """Print installation step"""
    if status == "OK":
        print_colored(f"✅ {step}", Colors.GREEN)
    elif status == "SKIP":
        print_colored(f"⏭️  {step}", Colors.YELLOW)
    elif status == "ERROR":
        print_colored(f"❌ {step}", Colors.RED)
    else:
        print_colored(f"🔧 {step}", Colors.BLUE)

class SystemChecker:
    """System requirements checker"""
    
    @staticmethod
    def check_python_version() -> bool:
        """Check if Python version is compatible"""
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            print_step(f"Python {version.major}.{version.minor}.{version.micro}", "OK")
            return True
        else:
            print_step(f"Python {version.major}.{version.minor}.{version.micro} (Required: 3.8+)", "ERROR")
            return False
    
    @staticmethod
    def check_pip() -> bool:
        """Check if pip is available"""
        try:
            import pip
            print_step("pip package manager", "OK")
            return True
        except ImportError:
            print_step("pip package manager", "ERROR")
            return False
    
    @staticmethod
    def check_gpu() -> Dict[str, Any]:
        """Check GPU availability"""
        gpu_info = {
            'cuda_available': False,
            'cuda_version': None,
            'gpu_count': 0,
            'gpu_names': []
        }
        
        try:
            import torch
            gpu_info['cuda_available'] = torch.cuda.is_available()
            
            if gpu_info['cuda_available']:
                gpu_info['cuda_version'] = torch.version.cuda
                gpu_info['gpu_count'] = torch.cuda.device_count()
                gpu_info['gpu_names'] = [torch.cuda.get_device_name(i) 
                                       for i in range(gpu_info['gpu_count'])]
                print_step(f"CUDA {gpu_info['cuda_version']} with {gpu_info['gpu_count']} GPU(s)", "OK")
                for i, name in enumerate(gpu_info['gpu_names']):
                    print_colored(f"   GPU {i}: {name}", Colors.GREEN)
            else:
                print_step("CUDA support", "SKIP")
                print_colored("   Will use CPU inference", Colors.YELLOW)
                
        except ImportError:
            print_step("PyTorch not installed yet", "SKIP")
        
        return gpu_info
    
    @staticmethod
    def check_system_resources() -> Dict[str, Any]:
        """Check system resources"""
        resources = {}
        
        try:
            import psutil
            
            # Memory
            memory = psutil.virtual_memory()
            resources['total_memory_gb'] = round(memory.total / (1024**3), 1)
            resources['available_memory_gb'] = round(memory.available / (1024**3), 1)
            
            # CPU
            resources['cpu_count'] = psutil.cpu_count()
            resources['cpu_percent'] = psutil.cpu_percent(interval=1)
            
            # Disk space
            disk = psutil.disk_usage('.')
            resources['free_space_gb'] = round(disk.free / (1024**3), 1)
            
            print_step(f"Memory: {resources['available_memory_gb']:.1f}GB available", "OK")
            print_step(f"CPU: {resources['cpu_count']} cores", "OK")
            print_step(f"Disk: {resources['free_space_gb']:.1f}GB free", "OK")
            
        except ImportError:
            print_step("System resource check", "SKIP")
        
        return resources

class DependencyInstaller:
    """Handles dependency installation"""
    
    def __init__(self):
        self.requirements_file = Path("requirements.txt")
        self.python_executable = sys.executable
    
    def install_package(self, package: str, upgrade: bool = False) -> bool:
        """Install a single package"""
        try:
            cmd = [self.python_executable, "-m", "pip", "install"]
            if upgrade:
                cmd.append("--upgrade")
            cmd.append(package)
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            print_step(f"Package {package} installation timed out", "ERROR")
            return False
        except Exception as e:
            print_step(f"Failed to install {package}: {e}", "ERROR")
            return False
    
    def install_requirements(self) -> bool:
        """Install all requirements from requirements.txt"""
        if not self.requirements_file.exists():
            print_step("requirements.txt not found", "ERROR")
            return False
        
        print_step("Installing dependencies from requirements.txt...")
        
        try:
            cmd = [
                self.python_executable, "-m", "pip", "install", 
                "-r", str(self.requirements_file),
                "--upgrade"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print_step("All dependencies installed", "OK")
                return True
            else:
                print_step(f"Dependency installation failed: {result.stderr}", "ERROR")
                return False
                
        except subprocess.TimeoutExpired:
            print_step("Dependency installation timed out", "ERROR")
            return False
        except Exception as e:
            print_step(f"Dependency installation error: {e}", "ERROR")
            return False
    
    def install_pytorch_gpu(self) -> bool:
        """Install PyTorch with GPU support"""
        print_step("Installing PyTorch with CUDA support...")
        
        # Detect CUDA version
        cuda_version = self._detect_cuda_version()
        
        if cuda_version:
            if cuda_version.startswith('11'):
                torch_cmd = "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            elif cuda_version.startswith('12'):
                torch_cmd = "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
            else:
                torch_cmd = "torch torchvision torchaudio"
        else:
            torch_cmd = "torch torchvision torchaudio"
        
        return self.install_package(torch_cmd)
    
    def _detect_cuda_version(self) -> Optional[str]:
        """Detect CUDA version on system"""
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                # Parse CUDA version from nvidia-smi output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'CUDA Version:' in line:
                        version = line.split('CUDA Version:')[1].strip().split()[0]
                        return version
        except FileNotFoundError:
            pass
        
        return None

class ConfigurationManager:
    """Manages configuration file creation and setup"""
    
    def __init__(self):
        self.config_dir = Path("config")
        self.env_file = Path(".env")
        self.env_example = Path(".env.example")
    
    def create_directories(self) -> bool:
        """Create necessary directories"""
        directories = ["config", "models", "logs", "logs/data", "logs/performance", 
                      "logs/analysis", "logs/exports", "src"]
        
        try:
            for directory in directories:
                Path(directory).mkdir(exist_ok=True)
            
            print_step("Project directories created", "OK")
            return True
            
        except Exception as e:
            print_step(f"Directory creation failed: {e}", "ERROR")
            return False
    
    def create_env_file(self) -> bool:
        """Create .env file from example if it doesn't exist"""
        try:
            if not self.env_file.exists() and self.env_example.exists():
                self.env_file.write_text(self.env_example.read_text())
                print_step(".env file created from template", "OK")
                print_colored("   Please edit .env file with your API keys", Colors.YELLOW)
                return True
            elif self.env_file.exists():
                print_step(".env file already exists", "SKIP")
                return True
            else:
                print_step(".env.example not found", "ERROR")
                return False
                
        except Exception as e:
            print_step(f"Environment file creation failed: {e}", "ERROR")
            return False
    
    def create_config_files(self) -> bool:
        """Create configuration files"""
        try:
            # Create settings.json
            settings = {
                "system": {
                    "name": "Enhanced Chicken Counter",
                    "version": "1.0.0",
                    "debug": False,
                    "auto_save": True
                },
                "detection": {
                    "confidence_threshold": 0.25,
                    "nms_threshold": 0.45,
                    "max_detections": 100,
                    "input_size": 640
                },
                "tracking": {
                    "track_buffer": 30,
                    "track_high_thresh": 0.25,
                    "track_low_thresh": 0.1,
                    "match_thresh": 0.8,
                    "max_tracked_objects": 100
                },
                "display": {
                    "show_video": True,
                    "show_fps": True,
                    "show_confidence": True,
                    "window_width": 1280,
                    "window_height": 720
                },
                "logging": {
                    "log_level": "INFO",
                    "performance_logging": True,
                    "save_images": False,
                    "cleanup_days": 7
                }
            }
            
            settings_file = self.config_dir / "settings.json"
            with open(settings_file, 'w') as f:
                import json
                json.dump(settings, f, indent=2)
            
            # Create model_config.json
            model_config = {
                "available_models": {
                    "yolov8n": {
                        "name": "YOLOv8 Nano",
                        "size": "6MB",
                        "speed": "Fast",
                        "accuracy": "Good"
                    },
                    "yolov8s": {
                        "name": "YOLOv8 Small", 
                        "size": "22MB",
                        "speed": "Medium",
                        "accuracy": "Better"
                    },
                    "yolov8m": {
                        "name": "YOLOv8 Medium",
                        "size": "50MB", 
                        "speed": "Slower",
                        "accuracy": "Best"
                    }
                },
                "default_model": "yolov8n.pt",
                "auto_download": True,
                "cache_models": True
            }
            
            model_config_file = self.config_dir / "model_config.json"
            with open(model_config_file, 'w') as f:
                json.dump(model_config, f, indent=2)
            
            print_step("Configuration files created", "OK")
            return True
            
        except Exception as e:
            print_step(f"Configuration creation failed: {e}", "ERROR")
            return False

class ModelDownloader:
    """Handles model downloading and validation"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
    
    def download_default_models(self) -> bool:
        """Download default YOLO models"""
        print_step("Downloading default models...")
        
        try:
            # Import ultralytics to trigger model download
            from ultralytics import YOLO
            
            # Download nano model (smallest, fastest)
            model = YOLO('yolov8n.pt')
            print_step("YOLOv8 Nano model ready", "OK")
            
            return True
            
        except Exception as e:
            print_step(f"Model download failed: {e}", "ERROR")
            return False
    
    def validate_models(self) -> bool:
        """Validate downloaded models"""
        try:
            from ultralytics import YOLO
            import torch
            
            # Test model loading
            model = YOLO('yolov8n.pt')
            
            # Test inference with dummy data
            import numpy as np
            dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            results = model(dummy_img, verbose=False)
            
            print_step("Model validation", "OK")
            return True
            
        except Exception as e:
            print_step(f"Model validation failed: {e}", "ERROR")
            return False

def run_installation():
    """Main installation routine"""
    print_colored("""
╔══════════════════════════════════════════════════════════════╗
║            Enhanced Chicken Counter Installation            ║
║                      Setup & Validation                      ║
╚══════════════════════════════════════════════════════════════╝
    """, Colors.CYAN, bold=True)
    
    success_count = 0
    total_steps = 0
    
    # Step 1: System Requirements Check
    print_header("SYSTEM REQUIREMENTS CHECK")
    total_steps += 3
    
    checker = SystemChecker()
    
    if checker.check_python_version():
        success_count += 1
    
    if checker.check_pip():
        success_count += 1
    
    # Resource check (not critical)
    checker.check_system_resources()
    success_count += 1
    
    # Step 2: Directory Setup
    print_header("PROJECT SETUP")
    total_steps += 2
    
    config_manager = ConfigurationManager()
    
    if config_manager.create_directories():
        success_count += 1
    
    if config_manager.create_env_file():
        success_count += 1
    
    if config_manager.create_config_files():
        success_count += 1
        total_steps += 1
    
    # Step 3: Dependency Installation
    print_header("DEPENDENCY INSTALLATION")
    total_steps += 1
    
    installer = DependencyInstaller()
    
    if installer.install_requirements():
        success_count += 1
    
    # Step 4: GPU Setup (optional)
    print_header("GPU DETECTION")
    gpu_info = checker.check_gpu()
    
    # Step 5: Model Download
    print_header("MODEL SETUP")
    total_steps += 2
    
    downloader = ModelDownloader()
    
    if downloader.download_default_models():
        success_count += 1
    
    if downloader.validate_models():
        success_count += 1
    
    # Installation Summary
    print_header("INSTALLATION SUMMARY")
    
    if success_count == total_steps:
        print_colored("🎉 Installation completed successfully!", Colors.GREEN, bold=True)
        print_colored("\nNext steps:", Colors.CYAN)
        print_colored("1. Edit .env file with your OpenRouter API key", Colors.WHITE)
        print_colored("2. Run: python launcher.py", Colors.WHITE)
        print_colored("3. Enjoy chicken counting! 🐔", Colors.WHITE)
        
        return True
    else:
        print_colored(f"⚠️  Installation completed with issues: {success_count}/{total_steps} steps successful", 
                     Colors.YELLOW, bold=True)
        print_colored("\nPlease check the errors above and run the installation again.", Colors.RED)
        return False

def main():
    """Main entry point"""
    try:
        # Check if running in correct directory
        if not Path("requirements.txt").exists():
            print_colored("❌ Please run this script from the project root directory", Colors.RED)
            sys.exit(1)
        
        # Run installation
        success = run_installation()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print_colored("\n🛑 Installation interrupted by user", Colors.YELLOW)
        sys.exit(1)
    except Exception as e:
        print_colored(f"\n❌ Installation failed: {e}", Colors.RED)
        sys.exit(1)

if __name__ == "__main__":
    main()