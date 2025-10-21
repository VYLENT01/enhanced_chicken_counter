"""
Enhanced Chicken Counter - Setup Utilities
System setup, configuration, and initialization helpers

Features:
- Environment validation and setup
- Configuration file management
- Model download and validation
- System optimization
- Hardware detection and configuration
"""

import os
import json
import shutil
import platform
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import subprocess
import time

from loguru import logger

class SystemSetup:
    """System setup and configuration utilities"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.config_dir = self.project_root / "config"
        self.models_dir = self.project_root / "models"
        self.logs_dir = self.project_root / "logs"
        self.src_dir = self.project_root / "src"
        
        # System information
        self.system_info = self._gather_system_info()
        
        logger.info("🔧 SystemSetup initialized")
    
    def _gather_system_info(self) -> Dict[str, Any]:
        """Gather comprehensive system information"""
        info = {
            'platform': platform.platform(),
            'system': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'architecture': platform.architecture(),
        }
        
        # Memory information
        try:
            import psutil
            memory = psutil.virtual_memory()
            info['total_memory_gb'] = round(memory.total / (1024**3), 2)
            info['available_memory_gb'] = round(memory.available / (1024**3), 2)
            info['cpu_count'] = psutil.cpu_count()
        except ImportError:
            logger.warning("⚠️  psutil not available for system monitoring")
        
        # GPU information
        try:
            import torch
            info['cuda_available'] = torch.cuda.is_available()
            if info['cuda_available']:
                info['cuda_version'] = torch.version.cuda
                info['gpu_count'] = torch.cuda.device_count()
                info['gpu_names'] = [torch.cuda.get_device_name(i) 
                                   for i in range(info['gpu_count'])]
        except ImportError:
            info['cuda_available'] = False
        
        return info
    
    def create_project_structure(self) -> bool:
        """Create complete project directory structure"""
        try:
            directories = [
                "config",
                "models", 
                "logs",
                "logs/data",
                "logs/performance",
                "logs/analysis", 
                "logs/exports",
                "src",
                "utils",
                "tests",
                "docs",
                "examples"
            ]
            
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(exist_ok=True)
                logger.debug(f"📁 Created directory: {directory}")
            
            logger.info("✅ Project structure created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create project structure: {e}")
            return False
    
    def setup_environment_file(self) -> bool:
        """Setup .env file with system-specific defaults"""
        try:
            env_file = self.project_root / ".env"
            env_example = self.project_root / ".env.example"
            
            if env_file.exists():
                logger.info("⏭️  .env file already exists")
                return True
            
            if not env_example.exists():
                logger.error("❌ .env.example not found")
                return False
            
            # Read example file
            env_content = env_example.read_text()
            
            # Apply system-specific defaults
            replacements = self._get_system_defaults()
            
            for key, value in replacements.items():
                env_content = env_content.replace(f"{key}=your_key_here", f"{key}={value}")
                env_content = env_content.replace(f"{key}=0", f"{key}={value}")
            
            # Write customized .env file
            env_file.write_text(env_content)
            
            logger.info("✅ .env file created with system defaults")
            return True
            
        except Exception as e:
            logger.error(f"❌ Environment file setup failed: {e}")
            return False
    
    def _get_system_defaults(self) -> Dict[str, str]:
        """Get system-specific default values"""
        defaults = {}
        
        # GPU settings
        if self.system_info.get('cuda_available'):
            defaults['USE_GPU'] = 'true'
            defaults['DEFAULT_MODEL'] = 'yolov8s.pt'  # Use better model with GPU
        else:
            defaults['USE_GPU'] = 'false'
            defaults['DEFAULT_MODEL'] = 'yolov8n.pt'  # Use lighter model for CPU
        
        # Memory-based settings
        memory_gb = self.system_info.get('available_memory_gb', 4)
        if memory_gb >= 8:
            defaults['MAX_TRACKED_OBJECTS'] = '200'
            defaults['WINDOW_WIDTH'] = '1280'
            defaults['WINDOW_HEIGHT'] = '720'
        else:
            defaults['MAX_TRACKED_OBJECTS'] = '50'
            defaults['WINDOW_WIDTH'] = '640'
            defaults['WINDOW_HEIGHT'] = '480'
        
        # Platform-specific settings
        if self.system_info.get('system') == 'Windows':
            defaults['INPUT_SOURCE'] = '0'
        else:
            defaults['INPUT_SOURCE'] = '0'
        
        return defaults
    
    def optimize_configuration(self) -> bool:
        """Optimize configuration files based on system capabilities"""
        try:
            # Load current settings
            settings_file = self.config_dir / "settings.json"
            if not settings_file.exists():
                logger.warning("⚠️  settings.json not found, skipping optimization")
                return True
            
            with open(settings_file, 'r') as f:
                settings = json.load(f)
            
            # Apply system-specific optimizations
            optimizations = self._get_performance_optimizations()
            
            # Update settings
            for section, updates in optimizations.items():
                if section in settings:
                    settings[section].update(updates)
            
            # Save optimized settings
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            
            logger.info("✅ Configuration optimized for system")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration optimization failed: {e}")
            return False
    
    def _get_performance_optimizations(self) -> Dict[str, Dict[str, Any]]:
        """Get performance optimizations based on system specs"""
        optimizations = {}
        
        # GPU optimizations
        if self.system_info.get('cuda_available'):
            optimizations['detection'] = {
                'device': 'auto',
                'half_precision': True,
                'max_detections': 200
            }
            optimizations['performance'] = {
                'gpu_memory_limit_mb': 2048,
                'auto_optimization': True
            }
        else:
            optimizations['detection'] = {
                'device': 'cpu',
                'half_precision': False,
                'max_detections': 50,
                'input_size': 416
            }
        
        # Memory optimizations
        memory_gb = self.system_info.get('available_memory_gb', 4)
        if memory_gb < 4:
            optimizations['performance'] = {
                'memory_limit_mb': 1024,
                'fps_target': 15
            }
            optimizations['display'] = {
                'window_width': 640,
                'window_height': 480
            }
        elif memory_gb >= 8:
            optimizations['performance'] = {
                'memory_limit_mb': 4096,
                'fps_target': 30
            }
            optimizations['display'] = {
                'window_width': 1280,
                'window_height': 720
            }
        
        # CPU optimizations
        cpu_count = self.system_info.get('cpu_count', 4)
        optimizations['advanced'] = {
            'multi_threading': cpu_count > 2,
            'async_processing': cpu_count > 4
        }
        
        return optimizations
    
    def validate_dependencies(self) -> Tuple[bool, List[str]]:
        """Validate all required dependencies"""
        required_packages = [
            'torch', 'torchvision', 'ultralytics', 'opencv-python',
            'numpy', 'pandas', 'pillow', 'loguru', 'rich',
            'httpx', 'python-dotenv', 'pydantic', 'filterpy',
            'scipy', 'scikit-learn'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                logger.debug(f"✅ {package} available")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"❌ {package} missing")
        
        if missing_packages:
            logger.error(f"❌ Missing packages: {', '.join(missing_packages)}")
            return False, missing_packages
        else:
            logger.info("✅ All dependencies validated")
            return True, []
    
    def download_default_models(self) -> bool:
        """Download and validate default models"""
        try:
            from ultralytics import YOLO
            
            models_to_download = ['yolov8n.pt', 'yolov8s.pt']
            
            for model_name in models_to_download:
                logger.info(f"📥 Downloading {model_name}...")
                
                try:
                    model = YOLO(model_name)
                    
                    # Test the model with dummy data
                    import numpy as np
                    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                    results = model(dummy_img, verbose=False)
                    
                    logger.info(f"✅ {model_name} downloaded and validated")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to download {model_name}: {e}")
                    return False
            
            logger.info("✅ All default models ready")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model download failed: {e}")
            return False
    
    def setup_logging_system(self) -> bool:
        """Setup comprehensive logging system"""
        try:
            # Create log directories
            log_dirs = ['data', 'performance', 'analysis', 'exports']
            for log_dir in log_dirs:
                (self.logs_dir / log_dir).mkdir(exist_ok=True)
            
            # Setup loguru configuration
            from loguru import logger
            
            # Remove default handler
            logger.remove()
            
            # Add console handler
            logger.add(
                lambda msg: print(msg, end=''),
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                level="INFO"
            )
            
            # Add file handlers
            logger.add(
                self.logs_dir / "system.log",
                rotation="10 MB",
                retention="7 days",
                level="DEBUG",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
            )
            
            logger.add(
                self.logs_dir / "errors.log",
                rotation="5 MB",
                retention="30 days",
                level="ERROR",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
            )
            
            logger.info("✅ Logging system configured")
            return True
            
        except Exception as e:
            print(f"❌ Logging setup failed: {e}")
            return False
    
    def create_test_configuration(self) -> bool:
        """Create test configuration for validation"""
        try:
            test_config = {
                "test_mode": True,
                "input_source": "test_video.mp4",
                "output_dir": "./tests/output",
                "duration_seconds": 30,
                "expected_chickens": 10,
                "validation_thresholds": {
                    "min_fps": 15,
                    "min_accuracy": 0.85,
                    "max_memory_mb": 2048
                }
            }
            
            test_config_file = self.config_dir / "test_config.json"
            with open(test_config_file, 'w') as f:
                json.dump(test_config, f, indent=2)
            
            logger.info("✅ Test configuration created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Test configuration creation failed: {e}")
            return False
    
    def run_system_diagnostic(self) -> Dict[str, Any]:
        """Run comprehensive system diagnostic"""
        diagnostic_results = {
            'timestamp': time.time(),
            'system_info': self.system_info,
            'checks': {}
        }
        
        # Check project structure
        diagnostic_results['checks']['project_structure'] = self._check_project_structure()
        
        # Check dependencies
        deps_ok, missing = self.validate_dependencies()
        diagnostic_results['checks']['dependencies'] = {
            'status': deps_ok,
            'missing_packages': missing
        }
        
        # Check configuration files
        diagnostic_results['checks']['configuration'] = self._check_configuration_files()
        
        # Check models
        diagnostic_results['checks']['models'] = self._check_models()
        
        # Check permissions
        diagnostic_results['checks']['permissions'] = self._check_permissions()
        
        # Overall health score
        passed_checks = sum(1 for check in diagnostic_results['checks'].values() 
                          if check.get('status', False))
        total_checks = len(diagnostic_results['checks'])
        diagnostic_results['health_score'] = round(passed_checks / total_checks * 100, 1)
        
        return diagnostic_results
    
    def _check_project_structure(self) -> Dict[str, Any]:
        """Check if project structure is complete"""
        required_dirs = ['config', 'models', 'logs', 'src', 'utils']
        required_files = ['.env', 'requirements.txt', 'launcher.py']
        
        missing_dirs = [d for d in required_dirs if not (self.project_root / d).exists()]
        missing_files = [f for f in required_files if not (self.project_root / f).exists()]
        
        return {
            'status': len(missing_dirs) == 0 and len(missing_files) == 0,
            'missing_directories': missing_dirs,
            'missing_files': missing_files
        }
    
    def _check_configuration_files(self) -> Dict[str, Any]:
        """Check configuration files"""
        config_files = ['settings.json', 'model_config.json']
        missing_configs = []
        invalid_configs = []
        
        for config_file in config_files:
            config_path = self.config_dir / config_file
            if not config_path.exists():
                missing_configs.append(config_file)
            else:
                try:
                    with open(config_path, 'r') as f:
                        json.load(f)
                except json.JSONDecodeError:
                    invalid_configs.append(config_file)
        
        return {
            'status': len(missing_configs) == 0 and len(invalid_configs) == 0,
            'missing_files': missing_configs,
            'invalid_files': invalid_configs
        }
    
    def _check_models(self) -> Dict[str, Any]:
        """Check model availability"""
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')
            return {'status': True, 'message': 'Models accessible'}
        except Exception as e:
            return {'status': False, 'error': str(e)}
    
    def _check_permissions(self) -> Dict[str, Any]:
        """Check file and directory permissions"""
        permissions_ok = True
        issues = []
        
        # Check write permissions
        test_dirs = [self.logs_dir, self.models_dir, self.config_dir]
        
        for test_dir in test_dirs:
            try:
                test_file = test_dir / "test_write.tmp"
                test_file.write_text("test")
                test_file.unlink()
            except Exception as e:
                permissions_ok = False
                issues.append(f"Cannot write to {test_dir}: {e}")
        
        return {
            'status': permissions_ok,
            'issues': issues
        }
    
    def export_diagnostic_report(self, diagnostic_results: Dict[str, Any]) -> Path:
        """Export diagnostic results to file"""
        try:
            report_file = self.logs_dir / f"diagnostic_report_{int(time.time())}.json"
            
            with open(report_file, 'w') as f:
                json.dump(diagnostic_results, f, indent=2, default=str)
            
            logger.info(f"📋 Diagnostic report saved: {report_file}")
            return report_file
            
        except Exception as e:
            logger.error(f"❌ Failed to export diagnostic report: {e}")
            raise

def run_complete_setup() -> bool:
    """Run complete system setup"""
    setup = SystemSetup()
    
    print("🔧 Starting Enhanced Chicken Counter Setup...")
    
    success_count = 0
    total_steps = 8
    
    # Step 1: Create project structure
    if setup.create_project_structure():
        success_count += 1
        print("✅ Project structure created")
    else:
        print("❌ Failed to create project structure")
    
    # Step 2: Setup environment file
    if setup.setup_environment_file():
        success_count += 1
        print("✅ Environment file configured")
    else:
        print("❌ Failed to setup environment file")
    
    # Step 3: Setup logging
    if setup.setup_logging_system():
        success_count += 1
        print("✅ Logging system configured")
    else:
        print("❌ Failed to setup logging system")
    
    # Step 4: Validate dependencies
    deps_ok, missing = setup.validate_dependencies()
    if deps_ok:
        success_count += 1
        print("✅ All dependencies validated")
    else:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
    
    # Step 5: Download models
    if setup.download_default_models():
        success_count += 1
        print("✅ Default models downloaded")
    else:
        print("❌ Failed to download models")
    
    # Step 6: Optimize configuration
    if setup.optimize_configuration():
        success_count += 1
        print("✅ Configuration optimized")
    else:
        print("❌ Failed to optimize configuration")
    
    # Step 7: Create test configuration
    if setup.create_test_configuration():
        success_count += 1
        print("✅ Test configuration created")
    else:
        print("❌ Failed to create test configuration")
    
    # Step 8: Run diagnostic
    diagnostic_results = setup.run_system_diagnostic()
    if diagnostic_results['health_score'] >= 80:
        success_count += 1
        print(f"✅ System diagnostic passed ({diagnostic_results['health_score']}%)")
    else:
        print(f"❌ System diagnostic failed ({diagnostic_results['health_score']}%)")
    
    # Export diagnostic report
    try:
        report_file = setup.export_diagnostic_report(diagnostic_results)
        print(f"📋 Diagnostic report: {report_file}")
    except Exception as e:
        print(f"⚠️  Could not export diagnostic report: {e}")
    
    # Final summary
    print(f"\n📊 Setup Summary: {success_count}/{total_steps} steps completed")
    
    if success_count == total_steps:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file with your API keys")
        print("2. Run: python launcher.py")
        print("3. Start counting chickens! 🐔")
        return True
    else:
        print("⚠️  Setup completed with issues")
        print("Please check the diagnostic report for details")
        return False

if __name__ == "__main__":
    run_complete_setup()