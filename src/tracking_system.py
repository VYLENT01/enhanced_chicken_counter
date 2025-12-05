"""
Enhanced Chicken Counter - ByteTrack Tracking System
Implementation of ByteTrack algorithm for robust multi-object tracking

Features:
- ByteTrack algorithm with dual-threshold association
- Kalman filter for state prediction
- ID management and recovery
- Occlusion handling
- Track lifecycle management
"""

import os
import time
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque

from loguru import logger

try:
    from filterpy.kalman import KalmanFilter
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
except ImportError as e:
    logger.error(f"Missing tracking dependencies: {e}")
    raise

from detector_classes import DetectionResult, Detection, TrackingResult

@dataclass
class Track:
    """Individual track representation"""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    state: str = "active"  # active, lost, removed
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    kalman_filter: Optional[KalmanFilter] = field(default=None, init=False)
    history: deque = field(default_factory=lambda: deque(maxlen=30), init=False)
    
    def __post_init__(self):
        """Initialize Kalman filter after creation"""
        self._init_kalman_filter()
        self.history.append(self.bbox.copy())
    
    def _init_kalman_filter(self):
        """Initialize Kalman filter for state estimation"""
        # State: [x, y, vx, vy, w, h, vw, vh] (center_x, center_y, width, height + velocities)
        self.kalman_filter = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix (constant velocity model)
        self.kalman_filter.F = np.array([
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix (we observe [x, y, w, h])
        self.kalman_filter.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0]
        ])
        
        # Process noise
        self.kalman_filter.Q *= 0.1
        
        # Measurement noise
        self.kalman_filter.R *= 10
        
        # Initial state covariance
        self.kalman_filter.P *= 100
        
        # Initialize state
        bbox_center = self.get_center()
        bbox_size = self.get_size()
        self.kalman_filter.x = np.array([
            bbox_center[0], bbox_center[1], 0, 0,  # x, y, vx, vy
            bbox_size[0], bbox_size[1], 0, 0       # w, h, vw, vh
        ])
    
    def update(self, detection: Detection):
        """Update track with new detection"""
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.hits += 1
        self.time_since_update = 0
        self.state = "active"
        
        # Update Kalman filter
        bbox_center = self.get_center()
        bbox_size = self.get_size()
        measurement = np.array([bbox_center[0], bbox_center[1], bbox_size[0], bbox_size[1]])
        self.kalman_filter.update(measurement)
        
        # Add to history
        self.history.append(self.bbox.copy())
    
    def predict(self):
        """Predict next state using Kalman filter"""
        self.kalman_filter.predict()
        self.age += 1
        self.time_since_update += 1
        
        # Update bbox from predicted state
        predicted_state = self.kalman_filter.x
        center_x, center_y = predicted_state[0], predicted_state[1]
        width, height = predicted_state[4], predicted_state[5]
        
        # Convert back to bbox format [x1, y1, x2, y2]
        self.bbox = np.array([
            center_x - width/2,
            center_y - height/2,
            center_x + width/2,
            center_y + height/2
        ])
        
        # Update state based on time since last update
        if self.time_since_update > 1:
            self.state = "lost"
    
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
    
    def is_valid(self) -> bool:
        """Check if track is valid for counting"""
        return (self.hits >= 3 and 
                self.time_since_update < 5 and
                self.state == "active")

class ByTrackSystem:
    """ByteTrack implementation for multi-object tracking"""
    
    def __init__(self):
        # Configuration from environment
        self.track_buffer = int(os.getenv('TRACK_BUFFER', '30'))
        self.track_high_thresh = float(os.getenv('TRACK_HIGH_THRESH', '0.25'))
        self.track_low_thresh = float(os.getenv('TRACK_LOW_THRESH', '0.1'))
        self.match_thresh = float(os.getenv('MATCH_THRESH', '0.8'))
        self.max_tracked_objects = int(os.getenv('MAX_TRACKED_OBJECTS', '100'))
        
        # Tracking state
        self.tracks: List[Track] = []
        self.next_id = 1
        self.frame_count = 0
        
        # Performance tracking
        self.track_stats = {
            'total_tracks': 0,
            'active_tracks': 0,
            'lost_tracks': 0,
            'id_switches': 0,
            'association_time': 0.0
        }
        
        logger.info(f"🔄 ByteTrack initialized - Buffer: {self.track_buffer}, "
                   f"High thresh: {self.track_high_thresh}, Low thresh: {self.track_low_thresh}")
    
    def initialize(self) -> bool:
        """Initialize tracking system"""
        try:
            logger.info("🔄 Initializing ByteTrack tracking system...")
            return True
        except Exception as e:
            logger.error(f"❌ Tracking initialization failed: {e}")
            return False
    
    def update(self, detection_result: DetectionResult, frame: np.ndarray) -> Optional[TrackingResult]:
        """Update tracks with new detections (main ByteTrack algorithm)"""
        if not detection_result or not detection_result.detections:
            # Predict existing tracks even without detections
            self._predict_tracks()
            self._remove_stale_tracks()
            return self._create_result()
        
        start_time = time.time()
        self.frame_count += 1
        
        try:
            # Step 1: Predict existing tracks
            self._predict_tracks()
            
            # Step 2: Separate high and low confidence detections
            high_conf_dets, low_conf_dets = self._separate_detections(detection_result.detections)
            
            # Step 3: First association (high confidence detections with tracks)
            unmatched_tracks, unmatched_high_dets = self._first_association(high_conf_dets)
            
            # Step 4: Second association (low confidence detections with unmatched tracks)
            unmatched_tracks, unmatched_low_dets = self._second_association(low_conf_dets, unmatched_tracks)
            
            # Step 5: Create new tracks from unmatched high confidence detections
            self._create_new_tracks(unmatched_high_dets)
            
            # Step 6: Remove stale tracks
            self._remove_stale_tracks()
            
            # Update statistics
            self.track_stats['association_time'] = time.time() - start_time
            self._update_stats()
            
            return self._create_result()
            
        except Exception as e:
            logger.error(f"❌ Tracking update failed: {e}")
            return None
    
    def _predict_tracks(self):
        """Predict all existing tracks"""
        for track in self.tracks:
            track.predict()
    
    def _separate_detections(self, detections: List[Detection]) -> Tuple[List[Detection], List[Detection]]:
        """Separate detections into high and low confidence"""
        high_conf = [det for det in detections if det.confidence >= self.track_high_thresh]
        low_conf = [det for det in detections if self.track_low_thresh <= det.confidence < self.track_high_thresh]
        
        return high_conf, low_conf
    
    def _first_association(self, high_conf_dets: List[Detection]) -> Tuple[List[Track], List[Detection]]:
        """First association: high confidence detections with active tracks"""
        active_tracks = [t for t in self.tracks if t.state == "active"]
        
        if not active_tracks or not high_conf_dets:
            return active_tracks, high_conf_dets
        
        # Calculate IoU matrix
        iou_matrix = self._calculate_iou_matrix(active_tracks, high_conf_dets)
        
        # Perform assignment
        matched_tracks, unmatched_tracks, unmatched_dets = self._associate(
            active_tracks, high_conf_dets, iou_matrix, self.match_thresh
        )
        
        # Update matched tracks
        for track_idx, det_idx in matched_tracks:
            active_tracks[track_idx].update(high_conf_dets[det_idx])
        
        # Get unmatched items
        unmatched_track_list = [active_tracks[i] for i in unmatched_tracks]
        unmatched_det_list = [high_conf_dets[i] for i in unmatched_dets]
        
        return unmatched_track_list, unmatched_det_list
    
    def _second_association(self, low_conf_dets: List[Detection], 
                          unmatched_tracks: List[Track]) -> Tuple[List[Track], List[Detection]]:
        """Second association: low confidence detections with unmatched tracks"""
        if not unmatched_tracks or not low_conf_dets:
            return unmatched_tracks, low_conf_dets
        
        # Calculate IoU matrix
        iou_matrix = self._calculate_iou_matrix(unmatched_tracks, low_conf_dets)
        
        # Perform assignment with lower threshold
        matched_tracks, unmatched_tracks_idx, unmatched_dets = self._associate(
            unmatched_tracks, low_conf_dets, iou_matrix, 0.5  # Lower threshold for recovery
        )
        
        # Update matched tracks
        for track_idx, det_idx in matched_tracks:
            unmatched_tracks[track_idx].update(low_conf_dets[det_idx])
        
        # Get remaining unmatched tracks
        remaining_unmatched = [unmatched_tracks[i] for i in unmatched_tracks_idx]
        unmatched_det_list = [low_conf_dets[i] for i in unmatched_dets]
        
        return remaining_unmatched, unmatched_det_list
    
    def _calculate_iou_matrix(self, tracks: List[Track], detections: List[Detection]) -> np.ndarray:
        """Calculate IoU matrix between tracks and detections"""
        if not tracks or not detections:
            return np.zeros((len(tracks), len(detections)))
        
        track_boxes = np.array([track.bbox for track in tracks])
        det_boxes = np.array([det.bbox for det in detections])
        
        return self._batch_iou(track_boxes, det_boxes)
    
    def _batch_iou(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Calculate IoU between two sets of boxes"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # Left-top corner
        rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # Right-bottom corner
        
        wh = np.maximum(0, rb - lt)
        intersection = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2 - intersection
        iou = intersection / np.maximum(union, 1e-6)
        
        return iou
    
    def _associate(self, tracks: List[Track], detections: List[Detection], 
                  iou_matrix: np.ndarray, thresh: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate tracks and detections using Hungarian algorithm"""
        if iou_matrix.size == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Convert IoU to cost (1 - IoU)
        cost_matrix = 1 - iou_matrix
        
        # Set high cost for pairs below threshold
        cost_matrix[iou_matrix < thresh] = 1e6
        
        # Hungarian assignment
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        
        # Filter out assignments with cost too high
        matches = []
        for i, (t_idx, d_idx) in enumerate(zip(track_indices, det_indices)):
            if cost_matrix[t_idx, d_idx] < 1e6:
                matches.append((t_idx, d_idx))
        
        # Get unmatched tracks and detections
        matched_track_indices = [m[0] for m in matches]
        matched_det_indices = [m[1] for m in matches]
        
        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_track_indices]
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_det_indices]
        
        return matches, unmatched_tracks, unmatched_dets
    
    def _create_new_tracks(self, detections: List[Detection]):
        """Create new tracks from unmatched detections"""
        for detection in detections:
            # Limit total number of tracks
            if len(self.tracks) >= self.max_tracked_objects:
                logger.warning(f"⚠️  Maximum tracks reached: {self.max_tracked_objects}")
                break
            
            # Create new track
            new_track = Track(
                track_id=self.next_id,
                bbox=detection.bbox,
                confidence=detection.confidence,
                class_id=detection.class_id,
                class_name=detection.class_name
            )
            
            self.tracks.append(new_track)
            self.next_id += 1
            self.track_stats['total_tracks'] += 1
            
            logger.debug(f"🆕 New track created: ID {new_track.track_id}")
    
    def _remove_stale_tracks(self):
        """Remove tracks that haven't been updated for too long"""
        tracks_to_remove = []
        
        for track in self.tracks:
            # Remove tracks that are lost for too long
            if track.time_since_update > self.track_buffer:
                tracks_to_remove.append(track)
            # Remove tracks with very low hits
            elif track.hits < 3 and track.age > 10:
                tracks_to_remove.append(track)
        
        for track in tracks_to_remove:
            self.tracks.remove(track)
            logger.debug(f"🗑️  Removed stale track: ID {track.track_id}")
    
    def _update_stats(self):
        """Update tracking statistics"""
        self.track_stats['active_tracks'] = len([t for t in self.tracks if t.state == "active"])
        self.track_stats['lost_tracks'] = len([t for t in self.tracks if t.state == "lost"])
    
    def _create_result(self) -> TrackingResult:
        """Create tracking result"""
        return TrackingResult(
            tracks=[t for t in self.tracks if t.state == "active"],
            frame_count=self.frame_count,
            stats=self.track_stats.copy()
        )
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Get track by ID"""
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None
    
    def get_active_tracks(self) -> List[Track]:
        """Get all active tracks"""
        return [t for t in self.tracks if t.state == "active"]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracking statistics"""
        return {
            'total_tracks': len(self.tracks),
            'active_tracks': len([t for t in self.tracks if t.state == "active"]),
            'lost_tracks': len([t for t in self.tracks if t.state == "lost"]),
            'frame_count': self.frame_count,
            'next_id': self.next_id,
            'performance': self.track_stats.copy()
        }
    
    def reset(self):
        """Reset tracking system"""
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0
        self.track_stats = {
            'total_tracks': 0,
            'active_tracks': 0,
            'lost_tracks': 0,
            'id_switches': 0,
            'association_time': 0.0
        }
        logger.info("🔄 Tracking system reset")
    
    def shutdown(self):
        """Cleanup tracking resources"""
        self.tracks.clear()
        logger.info("✅ Tracking system shutdown completed")