"""
Enhanced Chicken Counter - Data Logging and Storage System
Handles all data logging, storage, and retrieval operations

Features:
- JSON-based logging for structured data
- Performance metrics tracking
- Automatic file rotation and cleanup
- Real-time data export capabilities
- Statistical analysis and reporting
"""

import os
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import asdict

import pandas as pd
from loguru import logger

from detector_classes import (
    CountingEvent, PerformanceMetrics, SystemStatus, 
    AIAnalysisResult, ConfigSettings
)

class DataLogger:
    """Smart data logging system for chicken counting"""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir or os.getenv('OUTPUT_DIR', './logs'))
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.data_dir = self.output_dir / "data"
        self.performance_dir = self.output_dir / "performance" 
        self.analysis_dir = self.output_dir / "analysis"
        self.export_dir = self.output_dir / "exports"
        
        for directory in [self.data_dir, self.performance_dir, self.analysis_dir, self.export_dir]:
            directory.mkdir(exist_ok=True)
        
        # Initialize log files
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.counting_log_file = self.data_dir / f"counting_data_{self.current_date}.json"
        self.performance_log_file = self.performance_dir / f"performance_{self.current_date}.json"
        self.system_log_file = self.data_dir / f"system_status_{self.current_date}.json"
        self.ai_analysis_file = self.analysis_dir / f"ai_analysis_{self.current_date}.json"
        
        # In-memory buffers for real-time data
        self.counting_buffer = deque(maxlen=1000)  # Last 1000 counting events
        self.performance_buffer = deque(maxlen=500)  # Last 500 performance samples
        self.analysis_buffer = deque(maxlen=100)   # Last 100 AI analyses
        
        # Statistics tracking
        self.session_stats = {
            'start_time': time.time(),
            'total_counts': 0,
            'total_frames': 0,
            'peak_fps': 0.0,
            'avg_processing_time': 0.0,
            'error_count': 0
        }
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Background writer thread
        self.writer_thread = None
        self.is_writing = False
        self.write_queue = deque()
        
        self._start_background_writer()
        
        logger.info(f"📊 DataLogger initialized - Output: {self.output_dir}")
    
    def _start_background_writer(self):
        """Start background thread for writing data"""
        self.is_writing = True
        self.writer_thread = threading.Thread(target=self._background_writer, daemon=True)
        self.writer_thread.start()
    
    def _background_writer(self):
        """Background thread that writes data to files"""
        while self.is_writing:
            try:
                if self.write_queue:
                    with self.lock:
                        if self.write_queue:
                            write_task = self.write_queue.popleft()
                            self._execute_write_task(write_task)
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                
            except Exception as e:
                logger.error(f"❌ Background writer error: {e}")
    
    def _execute_write_task(self, task: Dict[str, Any]):
        """Execute a write task"""
        try:
            task_type = task.get('type')
            data = task.get('data')
            file_path = task.get('file_path')
            
            if task_type == 'append_json':
                self._append_to_json_file(file_path, data)
            elif task_type == 'write_json':
                self._write_json_file(file_path, data)
            elif task_type == 'export_csv':
                self._export_to_csv(file_path, data)
                
        except Exception as e:
            logger.error(f"❌ Write task execution failed: {e}")
    
    def _append_to_json_file(self, file_path: Path, data: Dict[str, Any]):
        """Append data to JSON file"""
        try:
            # Read existing data
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []
            
            # Append new data
            existing_data.append(data)
            
            # Write back
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"❌ JSON append failed: {e}")
    
    def _write_json_file(self, file_path: Path, data: Any):
        """Write data to JSON file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"❌ JSON write failed: {e}")
    
    def log_counting_data(self, data: Dict[str, Any]):
        """Log chicken counting data"""
        try:
            # Add timestamp and session info
            counting_entry = {
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                'session_id': self._get_session_id(),
                **data
            }
            
            # Add to buffer
            with self.lock:
                self.counting_buffer.append(counting_entry)
            
            # Queue for background writing
            self.write_queue.append({
                'type': 'append_json',
                'file_path': self.counting_log_file,
                'data': counting_entry
            })
            
            # Update session stats
            if 'total_count' in data:
                self.session_stats['total_counts'] = data['total_count']
            if 'frame_count' in data:
                self.session_stats['total_frames'] = data['frame_count']
            
            logger.debug(f"📊 Counting data logged: {data.get('total_count', 0)} chickens")
            
        except Exception as e:
            logger.error(f"❌ Counting data logging failed: {e}")
    
    def log_performance_data(self, metrics: PerformanceMetrics):
        """Log performance metrics"""
        try:
            performance_entry = {
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                'session_id': self._get_session_id(),
                **asdict(metrics)
            }
            
            # Add to buffer
            with self.lock:
                self.performance_buffer.append(performance_entry)
            
            # Queue for background writing
            self.write_queue.append({
                'type': 'append_json',
                'file_path': self.performance_log_file,
                'data': performance_entry
            })
            
            # Update session stats
            if metrics.fps > self.session_stats['peak_fps']:
                self.session_stats['peak_fps'] = metrics.fps
            
            logger.debug(f"📈 Performance logged: {metrics.fps:.1f} FPS")
            
        except Exception as e:
            logger.error(f"❌ Performance logging failed: {e}")
    
    def log_system_status(self, status: SystemStatus):
        """Log system status"""
        try:
            status_entry = {
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                'session_id': self._get_session_id(),
                **asdict(status)
            }
            
            # Queue for background writing
            self.write_queue.append({
                'type': 'append_json',
                'file_path': self.system_log_file,
                'data': status_entry
            })
            
            logger.debug("🔧 System status logged")
            
        except Exception as e:
            logger.error(f"❌ System status logging failed: {e}")
    
    def log_ai_analysis(self, analysis: AIAnalysisResult):
        """Log AI analysis results"""
        try:
            analysis_entry = {
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                'session_id': self._get_session_id(),
                **asdict(analysis)
            }
            
            # Add to buffer
            with self.lock:
                self.analysis_buffer.append(analysis_entry)
            
            # Queue for background writing
            self.write_queue.append({
                'type': 'append_json',
                'file_path': self.ai_analysis_file,
                'data': analysis_entry
            })
            
            logger.debug("🧠 AI analysis logged")
            
        except Exception as e:
            logger.error(f"❌ AI analysis logging failed: {e}")
    
    def log_counting_event(self, event: CountingEvent):
        """Log individual counting event"""
        try:
            event_data = {
                'type': 'counting_event',
                'event': asdict(event),
                'session_id': self._get_session_id()
            }
            
            self.log_counting_data(event_data)
            
        except Exception as e:
            logger.error(f"❌ Counting event logging failed: {e}")
    
    def get_recent_counting_data(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent counting data from buffer"""
        with self.lock:
            return list(self.counting_buffer)[-limit:]
    
    def get_recent_performance_data(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent performance data from buffer"""
        with self.lock:
            return list(self.performance_buffer)[-limit:]
    
    def get_recent_analysis_data(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent AI analysis data from buffer"""
        with self.lock:
            return list(self.analysis_buffer)[-limit:]
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get current session statistics"""
        current_time = time.time()
        session_duration = current_time - self.session_stats['start_time']
        
        # Calculate averages from buffers
        avg_fps = 0.0
        avg_processing_time = 0.0
        
        if self.performance_buffer:
            recent_perf = list(self.performance_buffer)[-30:]  # Last 30 samples
            avg_fps = sum(p.get('fps', 0) for p in recent_perf) / len(recent_perf)
            avg_processing_time = sum(p.get('total_processing_time', 0) for p in recent_perf) / len(recent_perf)
        
        return {
            'session_duration': session_duration,
            'start_time': self.session_stats['start_time'],
            'total_counts': self.session_stats['total_counts'],
            'total_frames': self.session_stats['total_frames'],
            'peak_fps': self.session_stats['peak_fps'],
            'average_fps': avg_fps,
            'average_processing_time': avg_processing_time,
            'error_count': self.session_stats['error_count'],
            'data_points_logged': len(self.counting_buffer),
            'performance_samples': len(self.performance_buffer),
            'ai_analyses': len(self.analysis_buffer)
        }
    
    def export_session_data(self, format: str = 'json') -> Optional[Path]:
        """Export current session data"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format.lower() == 'json':
                export_file = self.export_dir / f"session_export_{timestamp}.json"
                export_data = {
                    'session_info': self.get_session_statistics(),
                    'counting_data': self.get_recent_counting_data(),
                    'performance_data': self.get_recent_performance_data(),
                    'ai_analysis_data': self.get_recent_analysis_data()
                }
                
                with open(export_file, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str)
                    
            elif format.lower() == 'csv':
                export_file = self.export_dir / f"session_export_{timestamp}.csv"
                
                # Convert counting data to DataFrame and export
                counting_df = pd.DataFrame(self.get_recent_counting_data())
                counting_df.to_csv(export_file, index=False)
                
            else:
                logger.error(f"❌ Unsupported export format: {format}")
                return None
            
            logger.info(f"📤 Session data exported: {export_file}")
            return export_file
            
        except Exception as e:
            logger.error(f"❌ Export failed: {e}")
            return None
    
    def generate_daily_report(self) -> Dict[str, Any]:
        """Generate daily statistics report"""
        try:
            # Load today's data
            counting_data = self._load_json_file(self.counting_log_file)
            performance_data = self._load_json_file(self.performance_log_file)
            
            if not counting_data:
                return {'error': 'No data available for today'}
            
            # Calculate statistics
            total_chickens = max((entry.get('total_count', 0) for entry in counting_data), default=0)
            total_frames = max((entry.get('frame_count', 0) for entry in counting_data), default=0)
            
            if performance_data:
                avg_fps = sum(entry.get('fps', 0) for entry in performance_data) / len(performance_data)
                peak_fps = max(entry.get('fps', 0) for entry in performance_data)
                avg_processing_time = sum(entry.get('total_processing_time', 0) for entry in performance_data) / len(performance_data)
            else:
                avg_fps = peak_fps = avg_processing_time = 0.0
            
            # Generate hourly breakdown
            hourly_counts = defaultdict(int)
            for entry in counting_data:
                if 'datetime' in entry:
                    hour = datetime.fromisoformat(entry['datetime']).hour
                    count = entry.get('total_count', 0)
                    if count > hourly_counts[hour]:
                        hourly_counts[hour] = count
            
            report = {
                'date': self.current_date,
                'generated_at': datetime.now().isoformat(),
                'summary': {
                    'total_chickens_counted': total_chickens,
                    'total_frames_processed': total_frames,
                    'average_fps': round(avg_fps, 2),
                    'peak_fps': round(peak_fps, 2),
                    'average_processing_time': round(avg_processing_time * 1000, 2),  # Convert to ms
                    'data_points': len(counting_data),
                    'performance_samples': len(performance_data)
                },
                'hourly_breakdown': dict(hourly_counts),
                'performance_trends': self._calculate_performance_trends(performance_data)
            }
            
            # Save report
            report_file = self.export_dir / f"daily_report_{self.current_date}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📋 Daily report generated: {report_file}")
            return report
            
        except Exception as e:
            logger.error(f"❌ Daily report generation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_performance_trends(self, performance_data: List[Dict]) -> Dict[str, Any]:
        """Calculate performance trends from data"""
        if not performance_data or len(performance_data) < 2:
            return {}
        
        try:
            # Extract FPS values over time
            fps_values = [entry.get('fps', 0) for entry in performance_data]
            processing_times = [entry.get('total_processing_time', 0) for entry in performance_data]
            
            # Calculate simple trends (first half vs second half)
            mid_point = len(fps_values) // 2
            first_half_fps = sum(fps_values[:mid_point]) / mid_point if mid_point > 0 else 0
            second_half_fps = sum(fps_values[mid_point:]) / (len(fps_values) - mid_point)
            
            fps_trend = "improving" if second_half_fps > first_half_fps else "declining"
            
            return {
                'fps_trend': fps_trend,
                'fps_change': round(second_half_fps - first_half_fps, 2),
                'peak_fps': max(fps_values),
                'lowest_fps': min(fps_values),
                'fps_variance': round(np.var(fps_values), 2) if len(fps_values) > 1 else 0,
                'average_processing_time': round(sum(processing_times) / len(processing_times) * 1000, 2)  # ms
            }
            
        except Exception as e:
            logger.warning(f"⚠️  Performance trends calculation failed: {e}")
            return {}
    
    def _load_json_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load data from JSON file"""
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.warning(f"⚠️  Failed to load {file_path}: {e}")
            return []
    
    def _get_session_id(self) -> str:
        """Get unique session identifier"""
        return f"session_{int(self.session_stats['start_time'])}"
    
    def cleanup_old_files(self, days_to_keep: int = 7):
        """Clean up old log files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            for directory in [self.data_dir, self.performance_dir, self.analysis_dir]:
                for file_path in directory.glob("*.json"):
                    file_date = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_date < cutoff_date:
                        file_path.unlink()
                        logger.info(f"🗑️  Cleaned up old file: {file_path.name}")
            
        except Exception as e:
            logger.warning(f"⚠️  Cleanup failed: {e}")
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage usage information"""
        try:
            total_size = 0
            file_count = 0
            
            for directory in [self.data_dir, self.performance_dir, self.analysis_dir, self.export_dir]:
                for file_path in directory.rglob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                        file_count += 1
            
            return {
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'file_count': file_count,
                'directories': {
                    'data': len(list(self.data_dir.glob("*"))),
                    'performance': len(list(self.performance_dir.glob("*"))),
                    'analysis': len(list(self.analysis_dir.glob("*"))),
                    'exports': len(list(self.export_dir.glob("*")))
                }
            }
            
        except Exception as e:
            logger.warning(f"⚠️  Storage info calculation failed: {e}")
            return {}
    
    def close(self):
        """Close logger and cleanup resources"""
        try:
            # Stop background writer
            self.is_writing = False
            if self.writer_thread and self.writer_thread.is_alive():
                self.writer_thread.join(timeout=2)
            
            # Process remaining write tasks
            while self.write_queue:
                try:
                    task = self.write_queue.popleft()
                    self._execute_write_task(task)
                except:
                    break
            
            # Generate final session summary
            session_summary = self.get_session_statistics()
            summary_file = self.export_dir / f"session_summary_{self._get_session_id()}.json"
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(session_summary, f, indent=2, default=str)
            
            logger.info(f"📊 DataLogger closed - Session summary: {summary_file}")
            
        except Exception as e:
            logger.warning(f"⚠️  Logger close warning: {e}")