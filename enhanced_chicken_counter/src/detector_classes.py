"""
Enhanced Chicken Counter - Data Classes and Type Definitions
Defines all data structures used throughout the system

Features:
- Detection and tracking result classes
- Performance metrics containers
- Configuration data structures
- Type hints and validation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
import numpy as np
import time

class DetectionType(Enum):
    """Types of objects that can be detected"""
    CHICKEN = "chicken"
    BIRD = "bird"
    PERSON = "person"
    UNKNOWN = "unknown"

class TrackState(Enum):
    """Possible states for tracking objects"""
    ACTIVE = "active"
    LOST = "lost"
    REMOVED = "removed"
    TENTATIVE = "tentative"

@dataclass
class Detection:
    """Single object detection result"""
    bbox: np.ndarray           # Bounding box [x1, y1, x2, y2]
    confidence: float          # Detection confidence (0.0-1.0)
    class_id: int             # Class ID from model
    class_name: str           # Human-readable class name
    detection_time: float = field(default_factory=time.time)
    
    def get_center(self) -> Tuple[int, int]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def get_size(self) -> Tuple[float, float]:
        """Get size (width, height) of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1, y2 - y1)
    
    def get_area(self) -> float:
        """Get area of bounding box"""
        w, h = self.get_size()
        return w * h
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            'bbox': self.bbox.tolist(),
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'center': self.get_center(),
            'size': self.get_size(),
            'area': self.get_area(),
            'detection_time': self.detection_time
        }

@dataclass
class DetectionResult:
    """Result of detection on a single frame"""
    detections: List[Detection]                    # List of detected objects
    inference_time: float                          # Time taken for inference (seconds)
    frame_shape: Tuple[int, int, int]             # Original frame shape (H, W, C)
    model_name: str                               # Name of model used
    timestamp: float = field(default_factory=time.time)
    
    def get_detection_count(self) -> int:
        """Get total number of detections"""
        return len(self.detections)
    
    def get_high_confidence_detections(self, threshold: float = 0.5) -> List[Detection]:
        """Get detections above confidence threshold"""
        return [det for det in self.detections if det.confidence >= threshold]
    
    def get_detections_by_class(self, class_name: str) -> List[Detection]:
        """Get detections of specific class"""
        return [det for det in self.detections if det.class_name == class_name]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            'detection_count': self.get_detection_count(),
            'detections': [det.to_dict() for det in self.detections],
            'inference_time': self.inference_time,
            'frame_shape': self.frame_shape,
            'model_name': self.model_name,
            'timestamp': self.timestamp
        }

@dataclass
class TrackingResult:
    """Result of tracking update"""
    tracks: List['Track']                         # Active tracks (imported from tracking_system)
    frame_count: int                              # Current frame number
    stats: Dict[str, Any]                        # Tracking statistics
    timestamp: float = field(default_factory=time.time)
    
    def get_track_count(self) -> int:
        """Get number of active tracks"""
        return len(self.tracks)
    
    def get_tracks_by_confidence(self, threshold: float = 0.5) -> List['Track']:
        """Get tracks above confidence threshold"""
        return [track for track in self.tracks if track.confidence >= threshold]
    
    def get_valid_tracks(self) -> List['Track']:
        """Get tracks that are valid for counting"""
        return [track for track in self.tracks if track.is_valid()]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            'track_count': self.get_track_count(),
            'valid_tracks': len(self.get_valid_tracks()),
            'frame_count': self.frame_count,
            'stats': self.stats,
            'timestamp': self.timestamp,
            'track_ids': [track.track_id for track in self.tracks]
        }

@dataclass
class CountingEvent:
    """Record of a counting event"""
    track_id: int                                 # ID of tracked object
    event_type: str                              # 'entry', 'exit', 'cross'
    location: Tuple[int, int]                    # Location where counting occurred
    confidence: float                            # Confidence of the count
    timestamp: float = field(default_factory=time.time)
    frame_number: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            'track_id': self.track_id,
            'event_type': self.event_type,
            'location': self.location,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'frame_number': self.frame_number
        }

@dataclass
class PerformanceMetrics:
    """Performance metrics for the system"""
    fps: float = 0.0                             # Frames per second
    inference_time: float = 0.0                  # Average inference time
    tracking_time: float = 0.0                   # Average tracking time
    total_processing_time: float = 0.0           # Total processing time per frame
    memory_usage: float = 0.0                    # Memory usage percentage
    gpu_usage: float = 0.0                       # GPU usage percentage
    cpu_usage: float = 0.0                       # CPU usage percentage
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            'fps': self.fps,
            'inference_time': self.inference_time,
            'tracking_time': self.tracking_time,
            'total_processing_time': self.total_processing_time,
            'memory_usage': self.memory_usage,
            'gpu_usage': self.gpu_usage,
            'cpu_usage': self.cpu_usage
        }

@dataclass
class SystemStatus:
    """Overall system status"""
    is_running: bool = False
    is_initialized: bool = False
    current_source: str = ""
    total_chickens_counted: int = 0
    active_tracks: int = 0
    uptime: float = 0.0
    last_update: float = field(default_factory=time.time)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, error: str):
        """Add error to error list"""
        self.errors.append(f"{time.time()}: {error}")
        # Keep only last 10 errors
        if len(self.errors) > 10:
            self.errors = self.errors[-10:]
    
    def add_warning(self, warning: str):
        """Add warning to warning list"""
        self.warnings.append(f"{time.time()}: {warning}")
        # Keep only last 10 warnings
        if len(self.warnings) > 10:
            self.warnings = self.warnings[-10:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            'is_running': self.is_running,
            'is_initialized': self.is_initialized,
            'current_source': self.current_source,
            'total_chickens_counted': self.total_chickens_counted,
            'active_tracks': self.active_tracks,
            'uptime': self.uptime,
            'last_update': self.last_update,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'latest_errors': self.errors[-3:] if self.errors else [],
            'latest_warnings': self.warnings[-3:] if self.warnings else []
        }

@dataclass
class ConfigSettings:
    """Configuration settings for the system"""
    # Model settings
    model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.25
    nms_threshold: float = 0.45
    device: str = "auto"
    
    # Tracking settings
    track_buffer: int = 30
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    match_thresh: float = 0.8
    max_tracked_objects: int = 100
    
    # Input/Output settings
    input_source: str = "0"
    output_dir: str = "./logs"
    save_images: bool = False
    show_video: bool = True
    
    # Display settings
    window_width: int = 1280
    window_height: int = 720
    show_fps: bool = True
    show_confidence: bool = True
    
    # AI Analysis settings
    enable_ai_analysis: bool = False
    ai_model: str = "anthropic/claude-3-haiku"
    ai_analysis_interval: int = 30
    
    # Logging settings
    log_level: str = "INFO"
    performance_logging: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'model_name': self.model_name,
            'confidence_threshold': self.confidence_threshold,
            'nms_threshold': self.nms_threshold,
            'device': self.device,
            'track_buffer': self.track_buffer,
            'track_high_thresh': self.track_high_thresh,
            'track_low_thresh': self.track_low_thresh,
            'match_thresh': self.match_thresh,
            'max_tracked_objects': self.max_tracked_objects,
            'input_source': self.input_source,
            'output_dir': self.output_dir,
            'save_images': self.save_images,
            'show_video': self.show_video,
            'window_width': self.window_width,
            'window_height': self.window_height,
            'show_fps': self.show_fps,
            'show_confidence': self.show_confidence,
            'enable_ai_analysis': self.enable_ai_analysis,
            'ai_model': self.ai_model,
            'ai_analysis_interval': self.ai_analysis_interval,
            'log_level': self.log_level,
            'performance_logging': self.performance_logging
        }
    
    @classmethod
    def from_env(cls) -> 'ConfigSettings':
        """Create configuration from environment variables"""
        import os
        
        return cls(
            model_name=os.getenv('DEFAULT_MODEL', 'yolov8n.pt'),
            confidence_threshold=float(os.getenv('MODEL_CONFIDENCE', '0.25')),
            nms_threshold=float(os.getenv('NMS_THRESHOLD', '0.45')),
            device=os.getenv('USE_GPU', 'auto'),
            track_buffer=int(os.getenv('TRACK_BUFFER', '30')),
            track_high_thresh=float(os.getenv('TRACK_HIGH_THRESH', '0.25')),
            track_low_thresh=float(os.getenv('TRACK_LOW_THRESH', '0.1')),
            match_thresh=float(os.getenv('MATCH_THRESH', '0.8')),
            max_tracked_objects=int(os.getenv('MAX_TRACKED_OBJECTS', '100')),
            input_source=os.getenv('INPUT_SOURCE', '0'),
            output_dir=os.getenv('OUTPUT_DIR', './logs'),
            save_images=os.getenv('SAVE_IMAGES', 'false').lower() == 'true',
            show_video=os.getenv('SHOW_VIDEO', 'true').lower() == 'true',
            window_width=int(os.getenv('WINDOW_WIDTH', '1280')),
            window_height=int(os.getenv('WINDOW_HEIGHT', '720')),
            show_fps=os.getenv('SHOW_FPS', 'true').lower() == 'true',
            show_confidence=os.getenv('SHOW_CONFIDENCE', 'true').lower() == 'true',
            enable_ai_analysis=os.getenv('ENABLE_AI_ANALYSIS', 'false').lower() == 'true',
            ai_model=os.getenv('AI_MODEL', 'anthropic/claude-3-haiku'),
            ai_analysis_interval=int(os.getenv('AI_ANALYSIS_INTERVAL', '30')),
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            performance_logging=os.getenv('PERFORMANCE_LOGGING', 'true').lower() == 'true'
        )

@dataclass
class AIAnalysisResult:
    """Result from AI analysis of behavior"""
    analysis_text: str                           # AI-generated analysis
    confidence: float                            # Confidence in analysis
    behavior_tags: List[str] = field(default_factory=list)  # Detected behaviors
    anomalies: List[str] = field(default_factory=list)      # Detected anomalies
    recommendations: List[str] = field(default_factory=list) # AI recommendations
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            'analysis_text': self.analysis_text,
            'confidence': self.confidence,
            'behavior_tags': self.behavior_tags,
            'anomalies': self.anomalies,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp
        }

# Type aliases for cleaner code
BoundingBox = Tuple[float, float, float, float]  # x1, y1, x2, y2
Point = Tuple[int, int]                          # x, y
Size = Tuple[float, float]                       # width, height
Color = Tuple[int, int, int]                     # R, G, B
FrameData = np.ndarray                           # Image frame data

# Validation functions
def validate_bbox(bbox: BoundingBox) -> bool:
    """Validate bounding box coordinates"""
    x1, y1, x2, y2 = bbox
    return x1 < x2 and y1 < y2 and all(coord >= 0 for coord in bbox)

def validate_confidence(confidence: float) -> bool:
    """Validate confidence score"""
    return 0.0 <= confidence <= 1.0

def validate_frame_shape(shape: Tuple[int, int, int]) -> bool:
    """Validate frame shape"""
    return len(shape) == 3 and all(dim > 0 for dim in shape)

# Utility functions
def iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """Calculate Intersection over Union of two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def bbox_center(bbox: BoundingBox) -> Point:
    """Get center point of bounding box"""
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def bbox_area(bbox: BoundingBox) -> float:
    """Calculate area of bounding box"""
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def distance(point1: Point, point2: Point) -> float:
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)