"""
Enhanced Chicken Counter - Main Detection and Counting System
Integrates YOLOv8 detection with ByteTrack tracking for accurate chicken counting

Features:
- Real-time chicken detection using YOLOv8
- Advanced tracking with ByteTrack algorithm
- Smart counting with anti-duplication
- Performance monitoring and logging
"""

import os
import cv2
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque

import numpy as np
from loguru import logger

try:
    from ultralytics import YOLO
    import torch
except ImportError as e:
    logger.error(f"Missing ML dependencies: {e}")
    raise

from model_manager import ModelManager
from tracking_system import ByTrackSystem
from data_logger import DataLogger
from ai_analyzer import AIAnalyzer
from detector_classes import DetectionResult, TrackingResult

@dataclass
class CountingStats:
    """Statistics for chicken counting performance"""
    total_count: int = 0
    active_tracks: int = 0
    fps: float = 0.0
    processing_time: float = 0.0
    detection_confidence: float = 0.0
    frame_count: int = 0
    errors: int = 0
    start_time: float = 0.0

class ChickenCounterSystem:
    """Main chicken counting system with integrated detection and tracking"""
    
    def __init__(self, data_logger: DataLogger, ai_analyzer: AIAnalyzer):
        self.data_logger = data_logger
        self.ai_analyzer = ai_analyzer
        
        # Core components
        self.model_manager = ModelManager()
        self.tracking_system = ByTrackSystem()
        
        # System state
        self.is_running = False
        self.is_initialized = False
        self.stats = CountingStats()
        
        # Configuration from environment
        self.input_source = os.getenv('INPUT_SOURCE', '0')
        self.show_video = os.getenv('SHOW_VIDEO', 'true').lower() == 'true'
        self.save_images = os.getenv('SAVE_IMAGES', 'false').lower() == 'true'
        self.window_width = int(os.getenv('WINDOW_WIDTH', '1280'))
        self.window_height = int(os.getenv('WINDOW_HEIGHT', '720'))
        self.show_fps = os.getenv('SHOW_FPS', 'true').lower() == 'true'
        self.show_confidence = os.getenv('SHOW_CONFIDENCE', 'true').lower() == 'true'
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)  # Track last 30 frames for FPS
        self.frame_times = deque(maxlen=100)  # Track processing times
        
        # Counting logic
        self.counting_lines = self._setup_counting_lines()
        self.counted_ids = set()  # Track which objects have been counted
        self.id_positions = defaultdict(list)  # Track object position history
        
        # Video capture
        self.cap = None
        self.output_writer = None
        
        logger.info("🎯 ChickenCounterSystem initialized")
    
    def _setup_counting_lines(self) -> List[Dict]:
        """Setup virtual counting lines for chicken counting"""
        # Define counting zones - can be customized based on setup
        lines = [
            {
                'name': 'entry_line',
                'p1': (self.window_width // 4, self.window_height // 2),
                'p2': (3 * self.window_width // 4, self.window_height // 2),
                'direction': 'down',  # Count when crossing downward
                'color': (0, 255, 0)  # Green line
            }
        ]
        return lines
    
    def initialize(self) -> bool:
        """Initialize all system components"""
        try:
            logger.info("🔧 Initializing chicken counter system...")
            
            # Initialize model manager
            if not self.model_manager.initialize():
                logger.error("❌ Failed to initialize model manager")
                return False
            
            # Initialize tracking system
            if not self.tracking_system.initialize():
                logger.error("❌ Failed to initialize tracking system")
                return False
            
            # Setup video capture
            if not self._setup_video_capture():
                logger.error("❌ Failed to setup video capture")
                return False
            
            # Initialize stats
            self.stats.start_time = time.time()
            self.is_initialized = True
            
            logger.info("✅ Chicken counter system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
    
    def _setup_video_capture(self) -> bool:
        """Setup video capture from source"""
        try:
            # Determine source type
            if self.input_source.isdigit():
                source = int(self.input_source)
                logger.info(f"📹 Using camera source: {source}")
            else:
                source = self.input_source
                if not Path(source).exists():
                    logger.error(f"❌ Video file not found: {source}")
                    return False
                logger.info(f"📁 Using video file: {source}")
            
            # Initialize capture
            self.cap = cv2.VideoCapture(source)
            
            if not self.cap.isOpened():
                logger.error(f"❌ Failed to open video source: {source}")
                return False
            
            # Set video properties for optimal performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.window_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.window_height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Get actual properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"📺 Video resolution: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
            
            # Update window dimensions if needed
            if actual_width != self.window_width or actual_height != self.window_height:
                self.window_width = actual_width
                self.window_height = actual_height
                self.counting_lines = self._setup_counting_lines()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Video capture setup failed: {e}")
            return False
    
    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """Process single frame for detection and tracking"""
        start_time = time.time()
        
        try:
            # Run detection
            detection_result = self.model_manager.detect(frame)
            
            if detection_result is None:
                return frame, 0
            
            # Run tracking
            tracking_result = self.tracking_system.update(detection_result, frame)
            
            # Update counting based on tracking
            current_count = self._update_counting(tracking_result)
            
            # Draw visualizations
            vis_frame = self._draw_visualizations(frame, tracking_result)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.frame_times.append(processing_time)
            self.stats.processing_time = processing_time
            self.stats.active_tracks = len(tracking_result.tracks) if tracking_result else 0
            
            return vis_frame, current_count
            
        except Exception as e:
            logger.error(f"❌ Frame processing error: {e}")
            self.stats.errors += 1
            return frame, 0
    
    def _update_counting(self, tracking_result: TrackingResult) -> int:
        """Update chicken count based on tracking results"""
        if not tracking_result or not tracking_result.tracks:
            return self.stats.total_count
        
        new_counts = 0
        
        for track in tracking_result.tracks:
            track_id = track.track_id
            center = track.get_center()
            
            # Store position history
            self.id_positions[track_id].append(center)
            
            # Keep only recent positions (for performance)
            if len(self.id_positions[track_id]) > 10:
                self.id_positions[track_id] = self.id_positions[track_id][-10:]
            
            # Check if this ID has already been counted
            if track_id in self.counted_ids:
                continue
            
            # Check crossing of counting lines
            if self._check_line_crossing(track_id, center):
                self.counted_ids.add(track_id)
                new_counts += 1
                logger.info(f"🐔 New chicken counted! ID: {track_id}, Total: {self.stats.total_count + new_counts}")
        
        self.stats.total_count += new_counts
        return self.stats.total_count
    
    def _check_line_crossing(self, track_id: int, current_pos: Tuple[int, int]) -> bool:
        """Check if track has crossed any counting line"""
        if track_id not in self.id_positions or len(self.id_positions[track_id]) < 2:
            return False
        
        prev_pos = self.id_positions[track_id][-2]
        
        for line in self.counting_lines:
            if self._line_intersect(prev_pos, current_pos, line['p1'], line['p2']):
                # Check direction if specified
                if 'direction' in line:
                    if line['direction'] == 'down' and current_pos[1] > prev_pos[1]:
                        return True
                    elif line['direction'] == 'up' and current_pos[1] < prev_pos[1]:
                        return True
                    elif line['direction'] == 'right' and current_pos[0] > prev_pos[0]:
                        return True
                    elif line['direction'] == 'left' and current_pos[0] < prev_pos[0]:
                        return True
                else:
                    return True  # Any crossing
        
        return False
    
    def _line_intersect(self, p1: Tuple[int, int], p2: Tuple[int, int], 
                       p3: Tuple[int, int], p4: Tuple[int, int]) -> bool:
        """Check if two line segments intersect"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    def _draw_visualizations(self, frame: np.ndarray, tracking_result: TrackingResult) -> np.ndarray:
        """Draw detection and tracking visualizations on frame"""
        vis_frame = frame.copy()
        
        try:
            # Draw counting lines
            for line in self.counting_lines:
                cv2.line(vis_frame, line['p1'], line['p2'], line['color'], 3)
                cv2.putText(vis_frame, line['name'], 
                          (line['p1'][0], line['p1'][1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, line['color'], 2)
            
            # Draw tracking results
            if tracking_result and tracking_result.tracks:
                for track in tracking_result.tracks:
                    bbox = track.bbox
                    track_id = track.track_id
                    confidence = track.confidence
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw track ID and confidence
                    label = f"ID:{track_id}"
                    if self.show_confidence:
                        label += f" {confidence:.2f}"
                    
                    cv2.putText(vis_frame, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Draw center point
                    center = track.get_center()
                    cv2.circle(vis_frame, center, 4, (255, 0, 0), -1)
            
            # Draw statistics overlay
            self._draw_stats_overlay(vis_frame)
            
        except Exception as e:
            logger.warning(f"⚠️  Visualization error: {e}")
        
        return vis_frame
    
    def _draw_stats_overlay(self, frame: np.ndarray):
        """Draw statistics overlay on frame"""
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Statistics text
        stats_text = [
            f"Total Chickens: {self.stats.total_count}",
            f"Active Tracks: {self.stats.active_tracks}",
            f"Frame: {self.stats.frame_count}",
        ]
        
        if self.show_fps:
            stats_text.append(f"FPS: {self.stats.fps:.1f}")
        
        stats_text.append(f"Processing: {self.stats.processing_time*1000:.1f}ms")
        
        for i, text in enumerate(stats_text):
            y_pos = 30 + i * 25
            cv2.putText(frame, text, (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _calculate_fps(self):
        """Calculate and update FPS"""
        current_time = time.time()
        self.fps_counter.append(current_time)
        
        if len(self.fps_counter) > 1:
            time_diff = self.fps_counter[-1] - self.fps_counter[0]
            if time_diff > 0:
                self.stats.fps = (len(self.fps_counter) - 1) / time_diff
    
    def run(self):
        """Main execution loop"""
        if not self.is_initialized:
            logger.error("❌ System not initialized")
            return
        
        logger.info("🚀 Starting chicken counting system...")
        self.is_running = True
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning("⚠️  Failed to read frame, retrying...")
                    time.sleep(0.1)
                    continue
                
                # Process frame
                vis_frame, current_count = self._process_frame(frame)
                
                # Update frame counter and FPS
                self.stats.frame_count += 1
                self._calculate_fps()
                
                # Log data periodically
                if self.stats.frame_count % 30 == 0:  # Every 30 frames
                    self.data_logger.log_counting_data({
                        'timestamp': time.time(),
                        'frame_count': self.stats.frame_count,
                        'total_count': self.stats.total_count,
                        'active_tracks': self.stats.active_tracks,
                        'fps': self.stats.fps,
                        'processing_time': self.stats.processing_time
                    })
                
                # AI analysis periodically
                if (self.stats.frame_count % 900 == 0 and  # Every 30 seconds at 30fps
                    os.getenv('ENABLE_AI_ANALYSIS', 'true').lower() == 'true'):
                    threading.Thread(
                        target=self.ai_analyzer.analyze_frame_async,
                        args=(vis_frame, self.stats.total_count),
                        daemon=True
                    ).start()
                
                # Display video if enabled
                if self.show_video:
                    cv2.imshow('Enhanced Chicken Counter', vis_frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        logger.info("🛑 User requested shutdown")
                        break
                    elif key == ord('r'):  # Reset count
                        self.stats.total_count = 0
                        self.counted_ids.clear()
                        logger.info("🔄 Count reset by user")
                    elif key == ord('s'):  # Save screenshot
                        timestamp = int(time.time())
                        filename = f"logs/screenshot_{timestamp}.jpg"
                        cv2.imwrite(filename, vis_frame)
                        logger.info(f"📸 Screenshot saved: {filename}")
                
                # Save images if enabled
                if self.save_images and self.stats.frame_count % 30 == 0:
                    timestamp = int(time.time())
                    filename = f"logs/frame_{timestamp}_{self.stats.frame_count}.jpg"
                    cv2.imwrite(filename, vis_frame)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            logger.info("🛑 Interrupted by user")
        except Exception as e:
            logger.error(f"❌ Runtime error: {e}")
            self.stats.errors += 1
        finally:
            self.cleanup()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        return {
            'total_count': self.stats.total_count,
            'active_tracks': self.stats.active_tracks,
            'fps': self.stats.fps,
            'processing_time': self.stats.processing_time,
            'frame_count': self.stats.frame_count,
            'errors': self.stats.errors,
            'uptime': time.time() - self.stats.start_time if self.stats.start_time > 0 else 0,
            'is_running': self.is_running
        }
    
    def shutdown(self):
        """Shutdown the system gracefully"""
        logger.info("🛑 Shutting down chicken counter system...")
        self.is_running = False
        
        # Final data log
        self.data_logger.log_counting_data({
            'timestamp': time.time(),
            'final_count': self.stats.total_count,
            'total_frames': self.stats.frame_count,
            'session_duration': time.time() - self.stats.start_time,
            'average_fps': self.stats.fps,
            'total_errors': self.stats.errors
        })
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.cap:
                self.cap.release()
            
            if self.output_writer:
                self.output_writer.release()
            
            cv2.destroyAllWindows()
            
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.warning(f"⚠️  Cleanup warning: {e}")