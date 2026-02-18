#!/usr/bin/env python3
"""
Enhanced Chicken Counter - Performance Benchmarking Tool
Comprehensive performance testing and benchmarking system

Features:
- Model performance comparison
- Hardware optimization testing
- Throughput benchmarking
- Memory usage analysis
- Real-world scenario testing
- Automated report generation
"""

import os
import sys
import time
import threading
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import json
import argparse

import numpy as np
import cv2
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests"""
    duration: int = 60
    models: List[str] = field(default_factory=lambda: ['yolov8n.pt', 'yolov8s.pt'])
    input_sizes: List[int] = field(default_factory=lambda: [416, 640])
    confidence_thresholds: List[float] = field(default_factory=lambda: [0.25, 0.5])
    batch_sizes: List[int] = field(default_factory=lambda: [1])
    warmup_time: int = 10
    test_video: Optional[str] = None
    output_file: Optional[str] = None

@dataclass
class BenchmarkResult:
    """Results of a single benchmark test"""
    test_id: str
    model_name: str
    input_size: int
    confidence: float
    batch_size: int
    fps: float
    avg_inference_time: float
    min_inference_time: float
    max_inference_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: float
    gpu_memory_mb: float
    total_frames: int
    dropped_frames: int
    accuracy_score: float
    timestamp: float = field(default_factory=time.time)

class PerformanceBenchmark:
    """Comprehensive performance benchmarking system"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.test_video_path: Optional[Path] = None
        
        # System monitoring
        self.system_monitor = SystemMonitor()
        
        # Create test video if not provided
        if not config.test_video:
            self.create_test_video()
        else:
            self.test_video_path = Path(config.test_video)
        
        logger.info("🚀 PerformanceBenchmark initialized")
    
    def create_test_video(self, duration: int = 30) -> bool:
        """Create standardized test video for benchmarking"""
        try:
            self.test_video_path = Path("benchmark_test_video.mp4")
            
            if self.test_video_path.exists():
                logger.info("📹 Using existing test video")
                return True
            
            logger.info("📹 Creating test video for benchmarking...")
            
            # Video properties
            fps = 30
            width, height = 1280, 720
            total_frames = duration * fps
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(self.test_video_path), fourcc, fps, (width, height))
            
            for frame_num in range(total_frames):
                # Create realistic farm scene
                frame = self._generate_farm_scene(frame_num, width, height)
                writer.write(frame)
                
                if frame_num % 100 == 0:
                    logger.debug(f"Generated {frame_num}/{total_frames} frames")
            
            writer.release()
            logger.info(f"✅ Test video created: {self.test_video_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create test video: {e}")
            return False
    
    def _generate_farm_scene(self, frame_num: int, width: int, height: int) -> np.ndarray:
        """Generate realistic farm scene for testing"""
        # Create background
        frame = np.random.randint(80, 120, (height, width, 3), dtype=np.uint8)
        
        # Add ground texture
        ground_color = (101, 67, 33)  # Brown ground
        cv2.rectangle(frame, (0, height//2), (width, height), ground_color, -1)
        
        # Add some noise for realism
        noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add moving "chickens"
        num_chickens = 8 + int(4 * np.sin(frame_num * 0.02))  # Varying count
        
        for i in range(num_chickens):
            # Calculate chicken position with realistic movement
            base_x = (width // (num_chickens + 1)) * (i + 1)
            base_y = height // 2 + 100
            
            # Add movement
            x_offset = int(30 * np.sin(frame_num * 0.05 + i))
            y_offset = int(15 * np.cos(frame_num * 0.03 + i * 1.5))
            
            chicken_x = base_x + x_offset
            chicken_y = base_y + y_offset
            
            # Draw chicken-like object
            self._draw_chicken(frame, chicken_x, chicken_y, i)
        
        # Add some static objects (feeders, etc.)
        self._add_farm_objects(frame, width, height)
        
        return frame
    
    def _draw_chicken(self, frame: np.ndarray, x: int, y: int, chicken_id: int):
        """Draw a chicken-like object"""
        # Chicken colors
        colors = [
            (200, 200, 200),  # White
            (139, 69, 19),    # Brown
            (255, 215, 0),    # Yellow
            (169, 169, 169)   # Gray
        ]
        
        color = colors[chicken_id % len(colors)]
        
        # Body (ellipse)
        cv2.ellipse(frame, (x, y), (25, 15), 0, 0, 360, color, -1)
        
        # Head (circle)
        cv2.circle(frame, (x + 20, y - 10), 8, color, -1)
        
        # Beak
        cv2.circle(frame, (x + 28, y - 10), 3, (255, 140, 0), -1)
        
        # Legs
        cv2.line(frame, (x - 10, y + 15), (x - 10, y + 25), (255, 140, 0), 2)
        cv2.line(frame, (x + 5, y + 15), (x + 5, y + 25), (255, 140, 0), 2)
    
    def _add_farm_objects(self, frame: np.ndarray, width: int, height: int):
        """Add static farm objects"""
        # Feeder
        cv2.rectangle(frame, (50, height//2 + 50), (100, height//2 + 80), (101, 67, 33), -1)
        
        # Water container
        cv2.circle(frame, (width - 100, height//2 + 70), 20, (64, 164, 223), -1)
        
        # Fence posts
        for x in range(0, width, 200):
            cv2.rectangle(frame, (x, height//2), (x + 10, height//2 + 100), (101, 67, 33), -1)
    
    def benchmark_model_performance(self, model_name: str) -> List[BenchmarkResult]:
        """Benchmark performance for a specific model"""
        logger.info(f"🔬 Benchmarking model: {model_name}")
        
        try:
            from ultralytics import YOLO
            
            model_results = []
            
            for input_size in self.config.input_sizes:
                for confidence in self.config.confidence_thresholds:
                    for batch_size in self.config.batch_sizes:
                        
                        test_id = f"{model_name}_{input_size}_{confidence}_{batch_size}"
                        logger.info(f"🧪 Running test: {test_id}")
                        
                        # Load model
                        model = YOLO(model_name)
                        
                        # Configure model
                        model.conf = confidence
                        model.iou = 0.45
                        
                        # Warmup
                        logger.debug("🔥 Warming up model...")
                        dummy_img = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)
                        for _ in range(10):
                            model(dummy_img, verbose=False)
                        
                        # Start monitoring
                        self.system_monitor.start_monitoring()
                        
                        # Run benchmark
                        result = self._run_single_benchmark(
                            model, test_id, model_name, input_size, confidence, batch_size
                        )
                        
                        # Stop monitoring
                        monitor_stats = self.system_monitor.stop_monitoring()
                        
                        # Update result with monitoring data
                        result.memory_usage_mb = monitor_stats['avg_memory_mb']
                        result.cpu_usage_percent = monitor_stats['avg_cpu_percent']
                        result.gpu_usage_percent = monitor_stats['avg_gpu_percent']
                        result.gpu_memory_mb = monitor_stats['max_gpu_memory_mb']
                        
                        model_results.append(result)
                        
                        # Cleanup
                        del model
                        
                        # Small break between tests
                        time.sleep(2)
            
            return model_results
            
        except Exception as e:
            logger.error(f"❌ Model benchmark failed: {e}")
            return []
    
    def _run_single_benchmark(self, model, test_id: str, model_name: str, 
                            input_size: int, confidence: float, batch_size: int) -> BenchmarkResult:
        """Run a single benchmark test"""
        
        # Initialize video capture
        cap = cv2.VideoCapture(str(self.test_video_path))
        
        # Performance tracking
        inference_times = []
        frame_count = 0
        dropped_frames = 0
        start_time = time.time()
        
        logger.debug(f"🎬 Starting benchmark for {self.config.duration}s...")
        
        while time.time() - start_time < self.config.duration:
            ret, frame = cap.read()
            
            if not ret:
                # Reset video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Resize frame if needed
            if frame.shape[:2] != (input_size, input_size):
                frame = cv2.resize(frame, (input_size, input_size))
            
            # Run inference
            inference_start = time.time()
            
            try:
                results = model(frame, verbose=False)
                inference_time = time.time() - inference_start
                inference_times.append(inference_time)
                frame_count += 1
            except Exception as e:
                dropped_frames += 1
                logger.warning(f"⚠️  Frame dropped: {e}")
        
        cap.release()
        
        # Calculate metrics
        total_duration = time.time() - start_time
        avg_fps = frame_count / total_duration
        avg_inference_time = np.mean(inference_times) if inference_times else 0
        min_inference_time = np.min(inference_times) if inference_times else 0
        max_inference_time = np.max(inference_times) if inference_times else 0
        
        # Calculate accuracy score (simplified - would need ground truth for real accuracy)
        accuracy_score = max(0.0, 1.0 - (dropped_frames / max(frame_count + dropped_frames, 1)))
        
        return BenchmarkResult(
            test_id=test_id,
            model_name=model_name,
            input_size=input_size,
            confidence=confidence,
            batch_size=batch_size,
            fps=avg_fps,
            avg_inference_time=avg_inference_time,
            min_inference_time=min_inference_time,
            max_inference_time=max_inference_time,
            memory_usage_mb=0,  # Will be filled by monitoring
            cpu_usage_percent=0,  # Will be filled by monitoring
            gpu_usage_percent=0,  # Will be filled by monitoring
            gpu_memory_mb=0,  # Will be filled by monitoring
            total_frames=frame_count,
            dropped_frames=dropped_frames,
            accuracy_score=accuracy_score
        )
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark across all configurations"""
        logger.info("🚀 Starting comprehensive performance benchmark")
        
        start_time = time.time()
        
        for model_name in self.config.models:
            try:
                model_results = self.benchmark_model_performance(model_name)
                self.results.extend(model_results)
            except Exception as e:
                logger.error(f"❌ Failed to benchmark {model_name}: {e}")
        
        total_duration = time.time() - start_time
        
        # Generate summary
        summary = self._generate_benchmark_summary(total_duration)
        
        # Save results
        if self.config.output_file:
            self._save_results(summary)
        
        logger.info(f"✅ Benchmark completed in {total_duration:.1f}s")
        return summary
    
    def _generate_benchmark_summary(self, total_duration: float) -> Dict[str, Any]:
        """Generate comprehensive benchmark summary"""
        if not self.results:
            return {'error': 'No benchmark results available'}
        
        # Best performance by metric
        best_fps = max(self.results, key=lambda x: x.fps)
        best_inference = min(self.results, key=lambda x: x.avg_inference_time)
        most_efficient = min(self.results, key=lambda x: x.memory_usage_mb)
        
        # Model comparison
        model_comparison = {}
        for model in self.config.models:
            model_results = [r for r in self.results if r.model_name == model]
            if model_results:
                avg_fps = np.mean([r.fps for r in model_results])
                avg_inference = np.mean([r.avg_inference_time for r in model_results])
                avg_memory = np.mean([r.memory_usage_mb for r in model_results])
                
                model_comparison[model] = {
                    'avg_fps': round(avg_fps, 2),
                    'avg_inference_time': round(avg_inference * 1000, 2),  # ms
                    'avg_memory_mb': round(avg_memory, 2),
                    'test_count': len(model_results)
                }
        
        # Performance recommendations
        recommendations = self._generate_recommendations()
        
        return {
            'benchmark_info': {
                'total_duration': total_duration,
                'total_tests': len(self.results),
                'models_tested': len(self.config.models),
                'timestamp': time.time()
            },
            'best_performance': {
                'highest_fps': {
                    'test_id': best_fps.test_id,
                    'fps': round(best_fps.fps, 2),
                    'model': best_fps.model_name,
                    'input_size': best_fps.input_size
                },
                'fastest_inference': {
                    'test_id': best_inference.test_id,
                    'inference_time_ms': round(best_inference.avg_inference_time * 1000, 2),
                    'model': best_inference.model_name,
                    'input_size': best_inference.input_size
                },
                'most_memory_efficient': {
                    'test_id': most_efficient.test_id,
                    'memory_mb': round(most_efficient.memory_usage_mb, 2),
                    'model': most_efficient.model_name
                }
            },
            'model_comparison': model_comparison,
            'detailed_results': [
                {
                    'test_id': r.test_id,
                    'model': r.model_name,
                    'input_size': r.input_size,
                    'confidence': r.confidence,
                    'fps': round(r.fps, 2),
                    'inference_time_ms': round(r.avg_inference_time * 1000, 2),
                    'memory_mb': round(r.memory_usage_mb, 2),
                    'cpu_percent': round(r.cpu_usage_percent, 2),
                    'accuracy_score': round(r.accuracy_score, 3),
                    'total_frames': r.total_frames,
                    'dropped_frames': r.dropped_frames
                }
                for r in self.results
            ],
            'recommendations': recommendations,
            'system_info': self._get_system_info()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations based on results"""
        recommendations = []
        
        if not self.results:
            return recommendations
        
        # Find best configurations
        best_realtime = max([r for r in self.results if r.fps >= 25], 
                          key=lambda x: x.fps, default=None)
        best_accuracy = max(self.results, key=lambda x: x.accuracy_score)
        best_efficiency = min(self.results, key=lambda x: x.memory_usage_mb)
        
        if best_realtime:
            recommendations.append(
                f"For real-time applications: Use {best_realtime.model_name} "
                f"with input size {best_realtime.input_size} "
                f"(achieves {best_realtime.fps:.1f} FPS)"
            )
        
        if best_accuracy and best_accuracy.accuracy_score > 0.95:
            recommendations.append(
                f"For maximum accuracy: Use {best_accuracy.model_name} "
                f"with confidence {best_accuracy.confidence} "
                f"(accuracy score: {best_accuracy.accuracy_score:.3f})"
            )
        
        if best_efficiency:
            recommendations.append(
                f"For memory-constrained environments: Use {best_efficiency.model_name} "
                f"(uses only {best_efficiency.memory_usage_mb:.1f}MB)"
            )
        
        # General recommendations
        avg_memory = np.mean([r.memory_usage_mb for r in self.results])
        if avg_memory > 2000:
            recommendations.append(
                "Consider using smaller models or reducing input size for better memory efficiency"
            )
        
        avg_fps = np.mean([r.fps for r in self.results])
        if avg_fps < 15:
            recommendations.append(
                "Consider GPU acceleration or model optimization for better performance"
            )
        
        return recommendations
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for the report"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 1),
            'platform': sys.platform
        }
        
        try:
            import torch
            info['torch_version'] = torch.__version__
            info['cuda_available'] = torch.cuda.is_available()
            if info['cuda_available']:
                info['cuda_version'] = torch.version.cuda
                info['gpu_count'] = torch.cuda.device_count()
                info['gpu_names'] = [torch.cuda.get_device_name(i) 
                                   for i in range(info['gpu_count'])]
        except ImportError:
            info['torch_available'] = False
        
        return info
    
    def _save_results(self, summary: Dict[str, Any]):
        """Save benchmark results to file"""
        try:
            output_path = Path(self.config.output_file)
            
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"📊 Results saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")

class SystemMonitor:
    """Real-time system resource monitoring"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.samples = {
            'cpu': [],
            'memory': [],
            'gpu_percent': [],
            'gpu_memory': []
        }
    
    def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        self.samples = {k: [] for k in self.samples}
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return statistics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        return {
            'avg_cpu_percent': np.mean(self.samples['cpu']) if self.samples['cpu'] else 0,
            'avg_memory_mb': np.mean(self.samples['memory']) if self.samples['memory'] else 0,
            'avg_gpu_percent': np.mean(self.samples['gpu_percent']) if self.samples['gpu_percent'] else 0,
            'max_gpu_memory_mb': max(self.samples['gpu_memory']) if self.samples['gpu_memory'] else 0
        }
    
    def _monitor_loop(self):
        """Monitoring loop"""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # CPU and memory
                self.samples['cpu'].append(process.cpu_percent())
                self.samples['memory'].append(process.memory_info().rss / 1024 / 1024)  # MB
                
                # GPU monitoring
                try:
                    import torch
                    if torch.cuda.is_available():
                        # GPU utilization (approximation)
                        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                        self.samples['gpu_memory'].append(gpu_memory)
                        
                        # Simple GPU usage estimation
                        gpu_percent = min(100, gpu_memory / 1024 * 10)  # Rough estimation
                        self.samples['gpu_percent'].append(gpu_percent)
                except:
                    pass
                
                time.sleep(0.5)  # Sample every 500ms
                
            except Exception as e:
                logger.warning(f"⚠️  Monitoring error: {e}")
                break

def main():
    """Main benchmark function"""
    parser = argparse.ArgumentParser(description="Enhanced Chicken Counter - Performance Benchmark")
    parser.add_argument("--duration", type=int, default=60, help="Benchmark duration per test (seconds)")
    parser.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolov8s.pt"], help="Models to benchmark")
    parser.add_argument("--input-sizes", nargs="+", type=int, default=[416, 640], help="Input sizes to test")
    parser.add_argument("--confidence", nargs="+", type=float, default=[0.25, 0.5], help="Confidence thresholds")
    parser.add_argument("--test-video", type=str, help="Custom test video path")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark (30s duration, limited configs)")
    
    args = parser.parse_args()
    
    # Quick mode adjustments
    if args.quick:
        args.duration = 30
        args.models = ["yolov8n.pt"]
        args.input_sizes = [640]
        args.confidence = [0.25]
    
    # Create configuration
    config = BenchmarkConfig(
        duration=args.duration,
        models=args.models,
        input_sizes=args.input_sizes,
        confidence_thresholds=args.confidence,
        test_video=args.test_video,
        output_file=args.output
    )
    
    # Run benchmark
    benchmark = PerformanceBenchmark(config)
    results = benchmark.run_comprehensive_benchmark()
    
    # Print summary
    print("\n" + "="*60)
    print("🚀 BENCHMARK RESULTS SUMMARY")
    print("="*60)
    
    if 'best_performance' in results:
        best = results['best_performance']
        print(f"🏆 Highest FPS: {best['highest_fps']['fps']} FPS ({best['highest_fps']['model']})")
        print(f"⚡ Fastest Inference: {best['fastest_inference']['inference_time_ms']} ms ({best['fastest_inference']['model']})")
        print(f"💾 Most Memory Efficient: {best['most_memory_efficient']['memory_mb']} MB ({best['most_memory_efficient']['model']})")
    
    if 'recommendations' in results:
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"   {i}. {rec}")
    
    print(f"\n📊 Full results saved to: {args.output}")

if __name__ == "__main__":
    main()