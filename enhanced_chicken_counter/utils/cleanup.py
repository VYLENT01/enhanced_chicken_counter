"""
Enhanced Chicken Counter - Cleanup Utilities
System cleanup, maintenance, and optimization tools

Features:
- Log file management and rotation
- Model cache cleanup
- Temporary file removal
- Performance optimization
- Storage management
- System reset capabilities
"""

import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

from loguru import logger

class SystemCleaner:
    """Comprehensive system cleanup and maintenance"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.logs_dir = self.project_root / "logs"
        self.models_dir = self.project_root / "models"
        self.config_dir = self.project_root / "config"
        
        # Cleanup statistics
        self.cleanup_stats = {
            'files_removed': 0,
            'space_freed_mb': 0,
            'directories_cleaned': 0,
            'errors': []
        }
        
        logger.info("🧹 SystemCleaner initialized")
    
    def cleanup_logs(self, days_to_keep: int = 7, max_size_mb: int = 100) -> Dict[str, Any]:
        """Clean up old log files and large logs"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            space_freed = 0
            files_removed = 0
            
            if not self.logs_dir.exists():
                return {'status': 'skipped', 'reason': 'logs directory not found'}
            
            # Clean old log files
            for log_file in self.logs_dir.rglob("*.log"):
                try:
                    file_date = datetime.fromtimestamp(log_file.stat().st_mtime)
                    file_size = log_file.stat().st_size
                    
                    # Remove old files
                    if file_date < cutoff_date:
                        space_freed += file_size
                        log_file.unlink()
                        files_removed += 1
                        logger.debug(f"🗑️  Removed old log: {log_file.name}")
                    
                    # Remove large files (with backup)
                    elif file_size > max_size_mb * 1024 * 1024:
                        # Create backup with timestamp
                        backup_name = f"{log_file.stem}_{int(time.time())}.bak"
                        backup_path = log_file.parent / backup_name
                        shutil.move(log_file, backup_path)
                        
                        logger.info(f"📦 Large log backed up: {log_file.name} -> {backup_name}")
                        
                except Exception as e:
                    self.cleanup_stats['errors'].append(f"Log cleanup error: {e}")
                    logger.warning(f"⚠️  Failed to process {log_file}: {e}")
            
            # Clean old JSON data files
            for data_dir in ['data', 'performance', 'analysis']:
                data_path = self.logs_dir / data_dir
                if data_path.exists():
                    files_removed += self._cleanup_json_files(data_path, cutoff_date)
            
            # Clean old exports
            exports_dir = self.logs_dir / "exports"
            if exports_dir.exists():
                files_removed += self._cleanup_old_exports(exports_dir, days_to_keep)
            
            self.cleanup_stats['files_removed'] += files_removed
            self.cleanup_stats['space_freed_mb'] += round(space_freed / (1024 * 1024), 2)
            
            logger.info(f"✅ Log cleanup completed: {files_removed} files, {space_freed/(1024*1024):.1f}MB freed")
            
            return {
                'status': 'success',
                'files_removed': files_removed,
                'space_freed_mb': round(space_freed / (1024 * 1024), 2)
            }
            
        except Exception as e:
            error_msg = f"Log cleanup failed: {e}"
            self.cleanup_stats['errors'].append(error_msg)
            logger.error(f"❌ {error_msg}")
            return {'status': 'error', 'error': str(e)}
    
    def _cleanup_json_files(self, directory: Path, cutoff_date: datetime) -> int:
        """Clean up old JSON data files"""
        files_removed = 0
        
        for json_file in directory.glob("*.json"):
            try:
                file_date = datetime.fromtimestamp(json_file.stat().st_mtime)
                if file_date < cutoff_date:
                    json_file.unlink()
                    files_removed += 1
                    logger.debug(f"🗑️  Removed old data file: {json_file.name}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to remove {json_file}: {e}")
        
        return files_removed
    
    def _cleanup_old_exports(self, exports_dir: Path, days_to_keep: int) -> int:
        """Clean up old export files"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep * 2)  # Keep exports longer
        files_removed = 0
        
        for export_file in exports_dir.glob("*"):
            try:
                if export_file.is_file():
                    file_date = datetime.fromtimestamp(export_file.stat().st_mtime)
                    if file_date < cutoff_date:
                        export_file.unlink()
                        files_removed += 1
                        logger.debug(f"🗑️  Removed old export: {export_file.name}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to remove {export_file}: {e}")
        
        return files_removed
    
    def cleanup_models(self, keep_essential: bool = True) -> Dict[str, Any]:
        """Clean up model cache and unused models"""
        try:
            if not self.models_dir.exists():
                return {'status': 'skipped', 'reason': 'models directory not found'}
            
            essential_models = ['yolov8n.pt', 'yolov8s.pt'] if keep_essential else []
            space_freed = 0
            files_removed = 0
            
            # Clean ultralytics cache
            try:
                from ultralytics.utils import SETTINGS
                cache_dir = Path(SETTINGS['datasets_dir']).parent / 'models'
                if cache_dir.exists():
                    for cache_file in cache_dir.glob("*.pt"):
                        if cache_file.name not in essential_models:
                            file_size = cache_file.stat().st_size
                            cache_file.unlink()
                            space_freed += file_size
                            files_removed += 1
                            logger.debug(f"🗑️  Removed cached model: {cache_file.name}")
            except Exception as e:
                logger.warning(f"⚠️  Cache cleanup warning: {e}")
            
            # Clean local models directory
            for model_file in self.models_dir.glob("*.pt"):
                if keep_essential and model_file.name in essential_models:
                    continue
                
                # Check if model is very old or large unused models
                file_age = time.time() - model_file.stat().st_mtime
                file_size = model_file.stat().st_size
                
                # Remove if older than 30 days and larger than 100MB
                if file_age > 30 * 24 * 3600 and file_size > 100 * 1024 * 1024:
                    space_freed += file_size
                    model_file.unlink()
                    files_removed += 1
                    logger.info(f"🗑️  Removed old large model: {model_file.name}")
            
            self.cleanup_stats['files_removed'] += files_removed
            self.cleanup_stats['space_freed_mb'] += round(space_freed / (1024 * 1024), 2)
            
            logger.info(f"✅ Model cleanup completed: {files_removed} files, {space_freed/(1024*1024):.1f}MB freed")
            
            return {
                'status': 'success',
                'files_removed': files_removed,
                'space_freed_mb': round(space_freed / (1024 * 1024), 2)
            }
            
        except Exception as e:
            error_msg = f"Model cleanup failed: {e}"
            self.cleanup_stats['errors'].append(error_msg)
            logger.error(f"❌ {error_msg}")
            return {'status': 'error', 'error': str(e)}
    
    def cleanup_temp_files(self) -> Dict[str, Any]:
        """Clean up temporary files and caches"""
        try:
            space_freed = 0
            files_removed = 0
            
            # Python cache files
            for pycache_dir in self.project_root.rglob("__pycache__"):
                if pycache_dir.is_dir():
                    for cache_file in pycache_dir.rglob("*.pyc"):
                        space_freed += cache_file.stat().st_size
                        cache_file.unlink()
                        files_removed += 1
                    
                    # Remove empty __pycache__ directories
                    try:
                        pycache_dir.rmdir()
                        logger.debug(f"🗑️  Removed empty cache directory: {pycache_dir}")
                    except OSError:
                        pass  # Directory not empty
            
            # Temporary files
            temp_patterns = ['*.tmp', '*.temp', '*.lock', '*.pid']
            for pattern in temp_patterns:
                for temp_file in self.project_root.rglob(pattern):
                    if temp_file.is_file():
                        # Only remove files older than 1 hour
                        file_age = time.time() - temp_file.stat().st_mtime
                        if file_age > 3600:
                            space_freed += temp_file.stat().st_size
                            temp_file.unlink()
                            files_removed += 1
                            logger.debug(f"🗑️  Removed temp file: {temp_file.name}")
            
            # System-specific cleanup
            if os.name == 'nt':  # Windows
                self._cleanup_windows_temp()
            else:  # Unix-like
                self._cleanup_unix_temp()
            
            self.cleanup_stats['files_removed'] += files_removed
            self.cleanup_stats['space_freed_mb'] += round(space_freed / (1024 * 1024), 2)
            
            logger.info(f"✅ Temp cleanup completed: {files_removed} files, {space_freed/(1024*1024):.1f}MB freed")
            
            return {
                'status': 'success',
                'files_removed': files_removed,
                'space_freed_mb': round(space_freed / (1024 * 1024), 2)
            }
            
        except Exception as e:
            error_msg = f"Temp cleanup failed: {e}"
            self.cleanup_stats['errors'].append(error_msg)
            logger.error(f"❌ {error_msg}")
            return {'status': 'error', 'error': str(e)}
    
    def _cleanup_windows_temp(self):
        """Windows-specific temporary file cleanup"""
        try:
            # Windows temp directory
            import tempfile
            temp_dir = Path(tempfile.gettempdir())
            
            # Look for our app's temp files
            for temp_file in temp_dir.glob("chicken_counter_*"):
                if temp_file.is_file():
                    file_age = time.time() - temp_file.stat().st_mtime
                    if file_age > 3600:  # Older than 1 hour
                        temp_file.unlink()
                        logger.debug(f"🗑️  Removed Windows temp: {temp_file.name}")
        except Exception as e:
            logger.warning(f"⚠️  Windows temp cleanup warning: {e}")
    
    def _cleanup_unix_temp(self):
        """Unix-specific temporary file cleanup"""
        try:
            # Unix temp directories
            temp_dirs = [Path('/tmp'), Path('/var/tmp')]
            
            for temp_dir in temp_dirs:
                if temp_dir.exists():
                    for temp_file in temp_dir.glob("chicken_counter_*"):
                        if temp_file.is_file():
                            file_age = time.time() - temp_file.stat().st_mtime
                            if file_age > 3600:  # Older than 1 hour
                                temp_file.unlink()
                                logger.debug(f"🗑️  Removed Unix temp: {temp_file.name}")
        except Exception as e:
            logger.warning(f"⚠️  Unix temp cleanup warning: {e}")
    
    def optimize_storage(self) -> Dict[str, Any]:
        """Optimize storage usage and organization"""
        try:
            optimizations = []
            
            # Compress old log files
            compressed_count = self._compress_old_logs()
            if compressed_count > 0:
                optimizations.append(f"Compressed {compressed_count} log files")
            
            # Organize export files
            organized_count = self._organize_exports()
            if organized_count > 0:
                optimizations.append(f"Organized {organized_count} export files")
            
            # Deduplicate data files
            dedupe_count = self._deduplicate_data_files()
            if dedupe_count > 0:
                optimizations.append(f"Removed {dedupe_count} duplicate files")
            
            logger.info(f"✅ Storage optimization completed: {len(optimizations)} optimizations")
            
            return {
                'status': 'success',
                'optimizations': optimizations
            }
            
        except Exception as e:
            error_msg = f"Storage optimization failed: {e}"
            self.cleanup_stats['errors'].append(error_msg)
            logger.error(f"❌ {error_msg}")
            return {'status': 'error', 'error': str(e)}
    
    def _compress_old_logs(self) -> int:
        """Compress old log files to save space"""
        compressed_count = 0
        
        try:
            import gzip
            
            # Find log files older than 3 days
            cutoff_date = datetime.now() - timedelta(days=3)
            
            for log_file in self.logs_dir.rglob("*.log"):
                file_date = datetime.fromtimestamp(log_file.stat().st_mtime)
                
                if file_date < cutoff_date and not log_file.name.endswith('.gz'):
                    # Compress the file
                    compressed_path = log_file.with_suffix(log_file.suffix + '.gz')
                    
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(compressed_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Remove original
                    log_file.unlink()
                    compressed_count += 1
                    logger.debug(f"📦 Compressed log: {log_file.name}")
        
        except ImportError:
            logger.warning("⚠️  gzip not available for compression")
        except Exception as e:
            logger.warning(f"⚠️  Log compression warning: {e}")
        
        return compressed_count
    
    def _organize_exports(self) -> int:
        """Organize export files by date"""
        organized_count = 0
        
        try:
            exports_dir = self.logs_dir / "exports"
            if not exports_dir.exists():
                return 0
            
            for export_file in exports_dir.glob("*"):
                if export_file.is_file():
                    file_date = datetime.fromtimestamp(export_file.stat().st_mtime)
                    year_month = file_date.strftime("%Y-%m")
                    
                    # Create year-month directory
                    month_dir = exports_dir / year_month
                    month_dir.mkdir(exist_ok=True)
                    
                    # Move file if not already organized
                    if export_file.parent == exports_dir:
                        new_path = month_dir / export_file.name
                        export_file.rename(new_path)
                        organized_count += 1
                        logger.debug(f"📁 Organized export: {export_file.name} -> {year_month}/")
        
        except Exception as e:
            logger.warning(f"⚠️  Export organization warning: {e}")
        
        return organized_count
    
    def _deduplicate_data_files(self) -> int:
        """Remove duplicate data files"""
        duplicates_removed = 0
        
        try:
            import hashlib
            
            # Check for duplicate files in data directories
            data_dirs = [self.logs_dir / 'data', self.logs_dir / 'analysis']
            
            for data_dir in data_dirs:
                if not data_dir.exists():
                    continue
                
                file_hashes = {}
                
                for data_file in data_dir.glob("*.json"):
                    # Calculate file hash
                    with open(data_file, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    
                    if file_hash in file_hashes:
                        # Duplicate found - remove the newer file
                        original_file = file_hashes[file_hash]
                        duplicate_file = data_file
                        
                        if duplicate_file.stat().st_mtime > original_file.stat().st_mtime:
                            duplicate_file.unlink()
                            duplicates_removed += 1
                            logger.debug(f"🗑️  Removed duplicate: {duplicate_file.name}")
                        else:
                            original_file.unlink()
                            file_hashes[file_hash] = duplicate_file
                            duplicates_removed += 1
                            logger.debug(f"🗑️  Removed duplicate: {original_file.name}")
                    else:
                        file_hashes[file_hash] = data_file
        
        except Exception as e:
            logger.warning(f"⚠️  Deduplication warning: {e}")
        
        return duplicates_removed
    
    def reset_system(self, keep_config: bool = True, keep_models: bool = True) -> Dict[str, Any]:
        """Reset system to clean state"""
        try:
            reset_actions = []
            
            # Clear logs
            if self.logs_dir.exists():
                for log_item in self.logs_dir.iterdir():
                    if log_item.is_file():
                        log_item.unlink()
                    elif log_item.is_dir():
                        shutil.rmtree(log_item)
                
                # Recreate log structure
                for subdir in ['data', 'performance', 'analysis', 'exports']:
                    (self.logs_dir / subdir).mkdir(exist_ok=True)
                
                reset_actions.append("Cleared all logs")
            
            # Clear models (optional)
            if not keep_models and self.models_dir.exists():
                for model_file in self.models_dir.glob("*.pt"):
                    model_file.unlink()
                reset_actions.append("Removed all models")
            
            # Reset configuration (optional)
            if not keep_config:
                env_file = self.project_root / ".env"
                if env_file.exists():
                    env_file.unlink()
                reset_actions.append("Reset environment configuration")
            
            # Clear Python cache
            for pycache_dir in self.project_root.rglob("__pycache__"):
                if pycache_dir.is_dir():
                    shutil.rmtree(pycache_dir)
            reset_actions.append("Cleared Python cache")
            
            logger.info(f"✅ System reset completed: {len(reset_actions)} actions")
            
            return {
                'status': 'success',
                'actions': reset_actions
            }
            
        except Exception as e:
            error_msg = f"System reset failed: {e}"
            self.cleanup_stats['errors'].append(error_msg)
            logger.error(f"❌ {error_msg}")
            return {'status': 'error', 'error': str(e)}
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get detailed storage usage information"""
        try:
            storage_info = {}
            
            # Check each directory
            directories = {
                'logs': self.logs_dir,
                'models': self.models_dir,
                'config': self.config_dir,
                'project_root': self.project_root
            }
            
            for name, directory in directories.items():
                if directory.exists():
                    total_size = 0
                    file_count = 0
                    
                    for item in directory.rglob("*"):
                        if item.is_file():
                            total_size += item.stat().st_size
                            file_count += 1
                    
                    storage_info[name] = {
                        'size_mb': round(total_size / (1024 * 1024), 2),
                        'file_count': file_count
                    }
                else:
                    storage_info[name] = {'size_mb': 0, 'file_count': 0}
            
            # Calculate total
            total_size_mb = sum(info['size_mb'] for info in storage_info.values())
            total_files = sum(info['file_count'] for info in storage_info.values())
            
            storage_info['total'] = {
                'size_mb': round(total_size_mb, 2),
                'file_count': total_files
            }
            
            return storage_info
            
        except Exception as e:
            logger.error(f"❌ Failed to get storage info: {e}")
            return {}
    
    def run_full_cleanup(self, aggressive: bool = False) -> Dict[str, Any]:
        """Run complete system cleanup"""
        logger.info("🧹 Starting full system cleanup...")
        
        results = {
            'timestamp': time.time(),
            'operations': {},
            'summary': self.cleanup_stats.copy()
        }
        
        # Log cleanup
        results['operations']['logs'] = self.cleanup_logs(
            days_to_keep=3 if aggressive else 7
        )
        
        # Model cleanup
        results['operations']['models'] = self.cleanup_models(
            keep_essential=not aggressive
        )
        
        # Temp file cleanup
        results['operations']['temp_files'] = self.cleanup_temp_files()
        
        # Storage optimization
        results['operations']['storage'] = self.optimize_storage()
        
        # Final summary
        results['summary'] = self.cleanup_stats.copy()
        results['storage_info'] = self.get_storage_info()
        
        logger.info(f"✅ Full cleanup completed: {self.cleanup_stats['files_removed']} files, "
                   f"{self.cleanup_stats['space_freed_mb']:.1f}MB freed")
        
        return results

def run_cleanup(aggressive: bool = False) -> Dict[str, Any]:
    """Main cleanup function"""
    cleaner = SystemCleaner()
    return cleaner.run_full_cleanup(aggressive=aggressive)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Chicken Counter - System Cleanup")
    parser.add_argument("--aggressive", action="store_true", 
                       help="Aggressive cleanup (removes more files)")
    parser.add_argument("--logs-only", action="store_true",
                       help="Clean only log files")
    parser.add_argument("--models-only", action="store_true", 
                       help="Clean only model cache")
    parser.add_argument("--reset", action="store_true",
                       help="Reset system to clean state")
    
    args = parser.parse_args()
    
    cleaner = SystemCleaner()
    
    if args.reset:
        result = cleaner.reset_system()
        print(f"System reset: {result}")
    elif args.logs_only:
        result = cleaner.cleanup_logs()
        print(f"Log cleanup: {result}")
    elif args.models_only:
        result = cleaner.cleanup_models()
        print(f"Model cleanup: {result}")
    else:
        result = cleaner.run_full_cleanup(aggressive=args.aggressive)
        print(f"Full cleanup completed: {result['summary']}")