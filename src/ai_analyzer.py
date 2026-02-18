"""
Enhanced Chicken Counter - AI Analysis System
OpenRouter API integration for intelligent behavior analysis

Features:
- Real-time image analysis using state-of-the-art AI models
- Behavior pattern recognition
- Health and welfare monitoring
- Anomaly detection
- Smart recommendations
"""

import os
import time
import base64
import asyncio
from io import BytesIO
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import threading
from queue import Queue, Empty

import cv2
import numpy as np
from PIL import Image
import httpx
from loguru import logger

from detector_classes import AIAnalysisResult

@dataclass
class AnalysisRequest:
    """Request for AI analysis"""
    frame: np.ndarray
    chicken_count: int
    timestamp: float
    frame_number: int
    metadata: Dict[str, Any]

class OpenRouterClient:
    """Client for OpenRouter API communication"""
    
    def __init__(self, api_key: str, model: str = "anthropic/claude-3-haiku"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/enhanced-chicken-counter",
            "X-Title": "Enhanced Chicken Counter",
            "Content-Type": "application/json"
        }
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum 1 second between requests
        
        # Performance tracking
        self.request_count = 0
        self.total_response_time = 0.0
        self.error_count = 0
        
        logger.info(f"🤖 OpenRouter client initialized - Model: {model}")
    
    async def analyze_image(self, image_data: str, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """Send image analysis request to OpenRouter"""
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - time_since_last)
            
            start_time = time.time()
            
            # Prepare request payload
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.1,  # Low temperature for consistent analysis
                "top_p": 0.9
            }
            
            # Make API request
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Extract response text
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    
                    # Update performance metrics
                    response_time = time.time() - start_time
                    self.total_response_time += response_time
                    self.request_count += 1
                    self.last_request_time = time.time()
                    
                    logger.debug(f"🤖 AI analysis completed in {response_time:.2f}s")
                    return content.strip()
                else:
                    logger.warning("⚠️  Unexpected API response format")
                    return None
                    
        except httpx.HTTPStatusError as e:
            self.error_count += 1
            if e.response.status_code == 429:
                logger.warning("⚠️  Rate limit hit, backing off...")
                self.min_request_interval = min(self.min_request_interval * 1.5, 10.0)
            else:
                logger.error(f"❌ API request failed: {e.response.status_code} - {e.response.text}")
            return None
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ AI analysis request failed: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client performance statistics"""
        avg_response_time = (self.total_response_time / self.request_count 
                           if self.request_count > 0 else 0.0)
        
        return {
            'request_count': self.request_count,
            'error_count': self.error_count,
            'success_rate': ((self.request_count - self.error_count) / self.request_count * 100 
                           if self.request_count > 0 else 0.0),
            'average_response_time': avg_response_time,
            'current_interval': self.min_request_interval
        }

class AIAnalyzer:
    """AI-powered analysis system for chicken behavior and health monitoring"""
    
    # Analysis prompts for different scenarios
    ANALYSIS_PROMPTS = {
        'general': """
        Analyze this poultry farm image and provide insights about:
        1. Overall chicken behavior and activity levels
        2. Distribution and density of chickens in the space
        3. Any signs of stress, illness, or abnormal behavior
        4. Environmental conditions (lighting, space utilization)
        5. Welfare indicators and recommendations for improvement
        
        Current count: {count} chickens detected
        
        Provide a concise analysis focusing on animal welfare and farm management insights.
        """,
        
        'health': """
        Focus on health and welfare assessment of chickens in this image:
        1. Posture and movement patterns
        2. Signs of lameness or mobility issues
        3. Feather condition and cleanliness
        4. Social behaviors and grouping patterns
        5. Environmental stress indicators
        
        Current count: {count} chickens
        
        Identify any health concerns and provide actionable recommendations.
        """,
        
        'behavior': """
        Analyze chicken behavior patterns in this farm environment:
        1. Feeding and drinking behaviors
        2. Social interactions and hierarchy
        3. Territorial behaviors and spacing
        4. Activity levels and movement patterns
        5. Response to environmental stimuli
        
        Current count: {count} chickens observed
        
        Provide behavioral insights for optimizing farm management.
        """,
        
        'anomaly': """
        Examine this image for any unusual or concerning patterns:
        1. Unexpected chicken distributions or clustering
        2. Signs of distress or panic behavior
        3. Environmental hazards or issues
        4. Unusual activity levels (too high/low)
        5. Equipment malfunctions or environmental problems
        
        Current count: {count} chickens
        
        Flag any anomalies requiring immediate attention.
        """
    }
    
    def __init__(self):
        # Configuration from environment
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        self.model = os.getenv('AI_MODEL', 'anthropic/claude-3-haiku')
        self.analysis_interval = int(os.getenv('AI_ANALYSIS_INTERVAL', '30'))
        self.enabled = os.getenv('ENABLE_AI_ANALYSIS', 'true').lower() == 'true'
        
        # Initialize client if API key available
        self.client = None
        if self.api_key and self.enabled:
            self.client = OpenRouterClient(self.api_key, self.model)
        else:
            logger.warning("⚠️  AI analysis disabled - No API key or disabled in config")
        
        # Analysis queue and processing
        self.analysis_queue = Queue(maxsize=10)
        self.processing_thread = None
        self.is_processing = False
        
        # Analysis history and patterns
        self.analysis_history = []
        self.behavior_patterns = {
            'feeding_times': [],
            'activity_peaks': [],
            'stress_indicators': [],
            'health_alerts': []
        }
        
        # Performance tracking
        self.analysis_count = 0
        self.last_analysis_time = 0
        
        if self.client:
            self._start_processing_thread()
            logger.info("🧠 AI Analyzer initialized and ready")
        else:
            logger.info("🧠 AI Analyzer initialized (disabled)")
    
    def _start_processing_thread(self):
        """Start background thread for processing analysis requests"""
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._process_analysis_queue, daemon=True)
        self.processing_thread.start()
    
    def _process_analysis_queue(self):
        """Background thread that processes analysis requests"""
        while self.is_processing:
            try:
                # Get request from queue with timeout
                request = self.analysis_queue.get(timeout=1.0)
                
                # Process the analysis request
                asyncio.run(self._process_single_analysis(request))
                
                # Mark task as done
                self.analysis_queue.task_done()
                
            except Empty:
                continue  # No request available, continue waiting
            except Exception as e:
                logger.error(f"❌ Analysis processing error: {e}")
    
    async def _process_single_analysis(self, request: AnalysisRequest):
        """Process a single analysis request"""
        try:
            if not self.client:
                return
            
            # Convert frame to base64
            image_data = self._frame_to_base64(request.frame)
            if not image_data:
                return
            
            # Select analysis type based on patterns and time
            analysis_type = self._select_analysis_type(request)
            
            # Get appropriate prompt
            prompt = self.ANALYSIS_PROMPTS[analysis_type].format(count=request.chicken_count)
            
            # Perform AI analysis
            analysis_text = await self.client.analyze_image(image_data, prompt)
            
            if analysis_text:
                # Parse and structure the analysis
                analysis_result = self._parse_analysis_result(
                    analysis_text, analysis_type, request
                )
                
                # Store analysis
                self.analysis_history.append(analysis_result)
                if len(self.analysis_history) > 100:  # Keep last 100 analyses
                    self.analysis_history = self.analysis_history[-100:]
                
                # Update behavior patterns
                self._update_behavior_patterns(analysis_result)
                
                self.analysis_count += 1
                self.last_analysis_time = time.time()
                
                logger.info(f"🧠 AI analysis completed: {analysis_type}")
                
                # Log the analysis (this would be handled by data logger)
                # self.data_logger.log_ai_analysis(analysis_result)
                
            else:
                logger.warning("⚠️  AI analysis returned empty result")
                
        except Exception as e:
            logger.error(f"❌ Single analysis processing failed: {e}")
    
    def _frame_to_base64(self, frame: np.ndarray, quality: int = 70) -> Optional[str]:
        """Convert frame to base64 encoded JPEG"""
        try:
            # Resize frame if too large (to save bandwidth)
            height, width = frame.shape[:2]
            if width > 1024:
                scale = 1024 / width
                new_width = 1024
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)
            
            # Compress to JPEG
            buffer = BytesIO()
            pil_image.save(buffer, format='JPEG', quality=quality, optimize=True)
            
            # Encode to base64
            image_bytes = buffer.getvalue()
            base64_string = base64.b64encode(image_bytes).decode('utf-8')
            
            return base64_string
            
        except Exception as e:
            logger.error(f"❌ Frame to base64 conversion failed: {e}")
            return None
    
    def _select_analysis_type(self, request: AnalysisRequest) -> str:
        """Select appropriate analysis type based on context"""
        current_time = time.time()
        
        # Check if this is a scheduled general analysis
        if current_time - self.last_analysis_time > self.analysis_interval:
            return 'general'
        
        # Check for anomaly conditions
        if self._detect_anomaly_conditions(request):
            return 'anomaly'
        
        # Rotate between health and behavior analysis
        if self.analysis_count % 3 == 1:
            return 'health'
        elif self.analysis_count % 3 == 2:
            return 'behavior'
        else:
            return 'general'
    
    def _detect_anomaly_conditions(self, request: AnalysisRequest) -> bool:
        """Detect if current conditions warrant anomaly analysis"""
        # Check for sudden count changes
        if len(self.analysis_history) > 0:
            last_count = self.analysis_history[-1].confidence  # Using confidence as proxy for count
            count_change_ratio = abs(request.chicken_count - last_count) / max(last_count, 1)
            if count_change_ratio > 0.3:  # 30% change
                return True
        
        # Check for unusual timing
        hour = time.localtime().tm_hour
        if hour < 6 or hour > 22:  # Outside normal hours
            return True
        
        return False
    
    def _parse_analysis_result(self, analysis_text: str, analysis_type: str, 
                             request: AnalysisRequest) -> AIAnalysisResult:
        """Parse AI analysis text into structured result"""
        
        # Extract behavior tags from text
        behavior_tags = self._extract_behavior_tags(analysis_text)
        
        # Extract anomalies
        anomalies = self._extract_anomalies(analysis_text)
        
        # Extract recommendations
        recommendations = self._extract_recommendations(analysis_text)
        
        # Calculate confidence based on text quality and length
        confidence = self._calculate_analysis_confidence(analysis_text)
        
        return AIAnalysisResult(
            analysis_text=analysis_text,
            confidence=confidence,
            behavior_tags=behavior_tags,
            anomalies=anomalies,
            recommendations=recommendations,
            timestamp=request.timestamp
        )
    
    def _extract_behavior_tags(self, text: str) -> List[str]:
        """Extract behavior tags from analysis text"""
        tags = []
        
        # Define behavior keywords to look for
        behavior_keywords = {
            'active': ['active', 'moving', 'mobile', 'energetic'],
            'feeding': ['feeding', 'eating', 'food', 'pecking'],
            'resting': ['resting', 'sleeping', 'inactive', 'calm'],
            'social': ['grouping', 'social', 'together', 'flocking'],
            'stressed': ['stress', 'agitated', 'anxious', 'disturbed'],
            'healthy': ['healthy', 'normal', 'good', 'well'],
            'crowded': ['crowded', 'dense', 'packed', 'congested'],
            'dispersed': ['dispersed', 'spread', 'scattered', 'distributed']
        }
        
        text_lower = text.lower()
        for tag, keywords in behavior_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.append(tag)
        
        return tags
    
    def _extract_anomalies(self, text: str) -> List[str]:
        """Extract anomalies from analysis text"""
        anomalies = []
        
        # Look for anomaly indicators
        anomaly_keywords = [
            'unusual', 'abnormal', 'concerning', 'alert', 'warning',
            'problem', 'issue', 'distress', 'emergency', 'immediate'
        ]
        
        text_lower = text.lower()
        sentences = text.split('.')
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in anomaly_keywords):
                anomalies.append(sentence.strip())
        
        return anomalies
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract recommendations from analysis text"""
        recommendations = []
        
        # Look for recommendation indicators
        rec_keywords = [
            'recommend', 'suggest', 'should', 'consider', 'improve',
            'optimize', 'adjust', 'modify', 'implement', 'monitor'
        ]
        
        sentences = text.split('.')
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in rec_keywords):
                recommendations.append(sentence.strip())
        
        return recommendations
    
    def _calculate_analysis_confidence(self, text: str) -> float:
        """Calculate confidence score for analysis"""
        if not text:
            return 0.0
        
        # Base confidence on text length and content quality
        length_score = min(len(text) / 500.0, 1.0)  # Normalize to 500 chars
        
        # Check for specific indicators
        quality_indicators = [
            'observe', 'appear', 'indicate', 'suggest', 'show',
            'evidence', 'pattern', 'behavior', 'condition'
        ]
        
        quality_score = sum(1 for indicator in quality_indicators 
                          if indicator in text.lower()) / len(quality_indicators)
        
        # Combine scores
        confidence = (length_score * 0.6 + quality_score * 0.4)
        return min(confidence, 1.0)
    
    def _update_behavior_patterns(self, analysis: AIAnalysisResult):
        """Update behavior pattern tracking"""
        current_time = time.time()
        
        # Track feeding patterns
        if 'feeding' in analysis.behavior_tags:
            self.behavior_patterns['feeding_times'].append(current_time)
        
        # Track activity patterns
        if 'active' in analysis.behavior_tags:
            self.behavior_patterns['activity_peaks'].append(current_time)
        
        # Track stress indicators
        if 'stressed' in analysis.behavior_tags or analysis.anomalies:
            self.behavior_patterns['stress_indicators'].append({
                'timestamp': current_time,
                'tags': analysis.behavior_tags,
                'anomalies': analysis.anomalies
            })
        
        # Track health alerts
        if any('health' in tag or 'sick' in tag for tag in analysis.behavior_tags):
            self.behavior_patterns['health_alerts'].append({
                'timestamp': current_time,
                'analysis': analysis.analysis_text[:200]  # First 200 chars
            })
        
        # Cleanup old patterns (keep last 24 hours)
        cutoff_time = current_time - 86400  # 24 hours
        for pattern_list in self.behavior_patterns.values():
            if isinstance(pattern_list, list) and pattern_list:
                # Remove old entries
                if isinstance(pattern_list[0], dict):
                    self.behavior_patterns = [p for p in pattern_list 
                                            if p.get('timestamp', 0) > cutoff_time]
                else:
                    pattern_list[:] = [t for t in pattern_list if t > cutoff_time]
    
    def analyze_frame_async(self, frame: np.ndarray, chicken_count: int, 
                          metadata: Dict[str, Any] = None):
        """Queue frame for asynchronous AI analysis"""
        if not self.client or not self.enabled:
            return
        
        current_time = time.time()
        
        # Rate limiting - only analyze if enough time has passed
        if current_time - self.last_analysis_time < self.analysis_interval:
            return
        
        try:
            request = AnalysisRequest(
                frame=frame.copy(),
                chicken_count=chicken_count,
                timestamp=current_time,
                frame_number=metadata.get('frame_number', 0) if metadata else 0,
                metadata=metadata or {}
            )
            
            # Add to queue (non-blocking)
            if not self.analysis_queue.full():
                self.analysis_queue.put_nowait(request)
                logger.debug("🧠 Frame queued for AI analysis")
            else:
                logger.warning("⚠️  Analysis queue full, skipping frame")
                
        except Exception as e:
            logger.error(f"❌ Failed to queue analysis: {e}")
    
    def get_recent_analyses(self, limit: int = 10) -> List[AIAnalysisResult]:
        """Get recent analysis results"""
        return self.analysis_history[-limit:] if self.analysis_history else []
    
    def get_behavior_summary(self) -> Dict[str, Any]:
        """Get summary of behavior patterns"""
        current_time = time.time()
        
        # Count recent activities (last hour)
        recent_cutoff = current_time - 3600  # 1 hour
        
        recent_feeding = sum(1 for t in self.behavior_patterns['feeding_times'] 
                           if t > recent_cutoff)
        recent_activity = sum(1 for t in self.behavior_patterns['activity_peaks'] 
                            if t > recent_cutoff)
        recent_stress = sum(1 for s in self.behavior_patterns['stress_indicators'] 
                          if s.get('timestamp', 0) > recent_cutoff)
        
        return {
            'analysis_count': self.analysis_count,
            'last_analysis': self.last_analysis_time,
            'recent_feeding_events': recent_feeding,
            'recent_activity_peaks': recent_activity,
            'recent_stress_indicators': recent_stress,
            'total_health_alerts': len(self.behavior_patterns['health_alerts']),
            'api_stats': self.client.get_stats() if self.client else {}
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        return {
            'enabled': self.enabled,
            'analysis_count': self.analysis_count,
            'queue_size': self.analysis_queue.qsize(),
            'last_analysis_time': self.last_analysis_time,
            'behavior_summary': self.get_behavior_summary(),
            'client_stats': self.client.get_stats() if self.client else {}
        }
    
    def shutdown(self):
        """Shutdown AI analyzer"""
        try:
            self.is_processing = False
            
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2)
            
            # Clear queue
            while not self.analysis_queue.empty():
                try:
                    self.analysis_queue.get_nowait()
                except Empty:
                    break
            
            logger.info("🧠 AI Analyzer shutdown completed")
            
        except Exception as e:
            logger.warning(f"⚠️  AI Analyzer shutdown warning: {e}")