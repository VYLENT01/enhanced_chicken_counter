"""
Enhanced Chicken Counter - Model Management System
Handles YOLOv8 model loading, optimization, and inference

Features:
- Automatic model downloading and caching
- GPU/CPU optimization
- Confidence and NMS threshold management
- Performance monitoring
- Model switching capabilities
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass

import numpy as np
import cv2
from loguru import logger

try:
    from ultralytics import YOLO
    import torch
    import torchvision
except ImportError as e:
    logger.error(f"Missing ML dependencies: {e}")
    raise

from detector_classes import DetectionResult, Detection

@dataclass
class ModelConfig:
    """Configuration for YOLO models"""
    name: str
    path: str
    input_size: int = 640
    confidence: float = 0.25
    nms_threshold: float = 0.45
    max_detections: int = 100
    device: str = "auto"
    half_precision: bool = False

class ModelManager:
    """Manages YOLOv8 models for chicken detection"""
    
    # Available models with their configurations
    AVAILABLE_MODELS = {
        'yolov8n.pt': {
            'name': 'YOLOv8 Nano',
            'size': '6MB',
            'speed': 'Fast',
            'accuracy': 'Good'
        },
        'yolov8s.pt': {
            'name': 'YOLOv8 Small',
            'size': '22MB', 
            'speed': 'Medium',
            'accuracy': 'Better'
        },
        'yolov8m.pt': {
            'name': 'YOLOv8 Medium',
            'size': '50MB',
            'speed': 'Slower', 
            'accuracy': 'Best'
        }
    }
    
    # COCO class index for birds/chickens
    BIRD_CLASS_ID = 14  # 'bird' in COCO dataset
    PERSON_CLASS_ID = 0  # Sometimes useful for scale reference
    
    def __init__(self):
        self.model: Optional[YOLO] = None
        self.current_config: Optional[ModelConfig] = None
        self.model_dir = Path("models")
        self.is_initialized = False
        
        # Performance tracking
        self.inference_times = []
        self.total_detections = 0
        
        # Create models directory
        self.model_dir.mkdir(exist_ok=True)
        
        # Load configuration from environment FIRST
        self.default_model = os.getenv('DEFAULT_MODEL', 'yolov8n.pt')
        self.confidence_threshold = float(os.getenv('MODEL_CONFIDENCE', '0.25'))
        self.nms_threshold = float(os.getenv('NMS_THRESHOLD', '0.45'))
        self.use_gpu = os.getenv('USE_GPU', 'true').lower() == 'true'
        
        # Now detect device (after use_gpu is defined)
        self.device = self._detect_device()
        
        logger.info(f"🎯 ModelManager initialized - Device: {self.device}")
    
    def _detect_device(self) -> str:
        """Detect best available device for inference"""
        try:
            if torch.cuda.is_available() and self.use_gpu:
                device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"🚀 GPU detected: {gpu_name}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon
                logger.info("🍎 Apple MPS detected")
            else:
                device = "cpu"
                logger.info("💻 Using CPU inference")
            
            return device
        except Exception as e:
            logger.warning(f"⚠️  Device detection error: {e}, using CPU")
            return "cpu"
    
    def initialize(self) -> bool:
        """Initialize the model manager with default model"""
        try:
            logger.info(f"🔧 Loading model: {self.default_model}")
            return self.load_model(self.default_model)
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            return False
    
    def load_model(self, model_path: str) -> bool:
        """Load a model from path"""
        try:
            start_time = time.time()
            
            # Get model info
            model_info = self.AVAILABLE_MODELS.get(model_path, {'name': f'Custom ({model_path})'})
            
            logger.info(f"📥 Loading model: {model_info['name']}")
            
            # Create config
            config = ModelConfig(
                name=model_info['name'],
                path=model_path,
                confidence=self.confidence_threshold,
                nms_threshold=self.nms_threshold,
                device=self.device
            )
            
            return self._load_model_from_config(config)
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            return False
    
    def _load_model_from_config(self, config: ModelConfig) -> bool:
        """Load model from configuration"""
        try:
            start_time = time.time()
            
            # Load YOLO model - let Ultralytics handle path resolution and downloading
            logger.info(f"📥 Initializing YOLO model: {config.path}")
            self.model = YOLO(config.path)
            
            # Configure device
            if self.device != "cpu":
                try:
                    self.model.to(self.device)
                    logger.info(f"✅ Model moved to {self.device}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to move to {self.device}, using CPU: {e}")
                    self.device = "cpu"
            
            # Set model to evaluation mode
            if hasattr(self.model, 'model'):
                self.model.model.eval()
            
            # Store configuration
            self.current_config = config
            self.is_initialized = True
            
            load_time = time.time() - start_time
            logger.info(f"✅ Model loaded successfully in {load_time:.2f}s")
            
            # Test model with dummy input
            if self._test_model():
                logger.info("🧪 Model test passed")
                return True
            else:
                logger.error("❌ Model test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            return False
    
    def _test_model(self) -> bool:
        """Test model with dummy input"""
        try:
            # Create dummy RGB image
            dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Run inference
            results = self.model(dummy_img, verbose=False)
            
            logger.info(f"🧪 Model test: inference successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model test error: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> Optional[DetectionResult]:
        """Run detection on frame"""
        if not self.is_initialized or self.model is None:
            logger.warning("⚠️  Model not initialized")
            return None
        
        start_time = time.time()
        
        try:
            # Ensure frame is in correct format
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Convert BGR to RGB for YOLO
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame
            
            # Run inference
            results = self.model(
                rgb_frame,
                conf=self.current_config.confidence,
                iou=self.current_config.nms_threshold,
                max_det=self.current_config.max_detections,
                verbose=False
            )
            
            # Process results
            detections = self._process_results(results, frame.shape)
            
            # Update performance metrics
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-100:]  # Keep last 100
            
            self.total_detections += len(detections)
            
            return DetectionResult(
                detections=detections,
                inference_time=inference_time,
                frame_shape=frame.shape,
                model_name=self.current_config.name
            )
            
        except Exception as e:
            logger.error(f"❌ Detection error: {e}")
            return None
    
    def _process_results(self, results, frame_shape: Tuple[int, int, int]) -> List[Detection]:
        """Process YOLO results into Detection objects"""
        detections = []
        
        try:
            if not results or len(results) == 0:
                return detections
            
            result = results[0]  # YOLO returns list, we use first result
            
            if result.boxes is None or len(result.boxes) == 0:
                return detections
            
            # Extract boxes, scores, and classes
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for i in range(len(boxes)):
                class_id = int(classes[i])
                confidence = float(scores[i])
                bbox = boxes[i]
                
                # Filter for birds/chickens or all objects if chicken-specific model
                if self._is_relevant_class(class_id):
                    detection = Detection(
                        bbox=bbox,
                        confidence=confidence,
                        class_id=class_id,
                        class_name=self._get_class_name(class_id)
                    )
                    detections.append(detection)
        
        except Exception as e:
            logger.warning(f"⚠️  Result processing error: {e}")
        
        return detections
    
    def _is_relevant_class(self, class_id: int) -> bool:
        """Check if class is relevant for chicken counting"""
        # For standard COCO models, filter for birds
        if class_id == self.BIRD_CLASS_ID:
            return True
        
        # For custom chicken models, accept all detections
        # (assuming custom model is trained specifically for chickens)
        if self.current_config and 'custom' in self.current_config.name.lower():
            return True
        
        # Add more class IDs if needed
        return False
    
    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID"""
        try:
            if hasattr(self.model, 'names') and self.model.names:
                return self.model.names.get(class_id, f"class_{class_id}")
            else:
                # Default COCO class names
                coco_names = {
                    0: "person", 14: "bird", 15: "cat", 16: "dog"
                }
                return coco_names.get(class_id, f"class_{class_id}")
        except:
            return f"class_{class_id}"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get model performance statistics"""
        if not self.inference_times:
            return {}
        
        avg_inference_time = np.mean(self.inference_times)
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        return {
            'average_inference_time': avg_inference_time,
            'fps': fps,
            'total_detections': self.total_detections,
            'model_name': self.current_config.name if self.current_config else "None",
            'device': self.device,
            'confidence_threshold': self.current_config.confidence if self.current_config else 0,
            'nms_threshold': self.current_config.nms_threshold if self.current_config else 0
        }
    
    def update_thresholds(self, confidence: float, nms: float):
        """Update detection thresholds"""
        if self.current_config:
            self.current_config.confidence = confidence
            self.current_config.nms_threshold = nms
            logger.info(f"🔧 Updated thresholds - Conf: {confidence}, NMS: {nms}")
    
    def get_available_models(self) -> Dict[str, str]:
        """Get list of available models"""
        return {key: config['name'] for key, config in self.AVAILABLE_MODELS.items()}
    
    def optimize_for_speed(self):
        """Optimize model for speed over accuracy"""
        if self.current_config:
            self.current_config.confidence = max(0.3, self.current_config.confidence)
            self.current_config.nms_threshold = min(0.6, self.current_config.nms_threshold)
            self.current_config.max_detections = 50
            logger.info("🚀 Model optimized for speed")
    
    def optimize_for_accuracy(self):
        """Optimize model for accuracy over speed"""
        if self.current_config:
            self.current_config.confidence = 0.1
            self.current_config.nms_threshold = 0.3
            self.current_config.max_detections = 200
            logger.info("🎯 Model optimized for accuracy")
    
    def shutdown(self):
        """Cleanup resources"""
        try:
            if self.model:
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            self.is_initialized = False
            logger.info("✅ Model manager shutdown completed")
            
        except Exception as e:
            logger.warning(f"⚠️  Shutdown warning: {e}")