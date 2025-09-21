import mediapipe as mp
import cv2
import numpy as np

class MediaPipeGazeDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def detect_gaze(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks
            gaze_vector = self.calculate_gaze_vector(face_landmarks)
            return self.is_looking_at_screen(gaze_vector)




"""
Core gaze detection module implementing advanced gaze tracking algorithms.
This module provides the main GazeDetector class for real-time gaze detection.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional, Dict, List
import time
import math
from dataclasses import dataclass

@dataclass
class GazeResult:
    """Data class for gaze detection results"""
    looking_at_screen: bool
    gaze_direction: Tuple[float, float]
    confidence: float
    processing_time: float
    eye_center: Tuple[int, int]
    iris_center: Tuple[int, int]

class GazeDetector:
    """
    Advanced gaze detection system using MediaPipe and custom algorithms.
    Provides real-time gaze tracking with high accuracy and performance.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the gaze detector.
        
        Args:
            config: Configuration dictionary with detection parameters
        """
        self.config = config or self._default_config()
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.config['min_detection_confidence'],
            min_tracking_confidence=self.config['min_tracking_confidence']
        )
        
        # Eye and iris landmark indices
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        # State variables
        self.screen_center = self.config['screen_center']
        self.attention_threshold = self.config['attention_threshold']
        self.last_valid_result = None
        
        # Performance tracking
        self.frame_count = 0
        self.total_processing_time = 0
        
    def _default_config(self) -> Dict:
        """Get default configuration parameters"""
        return {
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5,
            'screen_center': (320, 240),
            'attention_threshold': 0.3,
            'smoothing_factor': 0.7,
            'iris_weight': 0.8,
            'eye_weight': 0.2
        }
    
    def detect_gaze(self, frame: np.ndarray) -> GazeResult:
        """
        Detect gaze direction and screen attention from a video frame.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            GazeResult object with detection results
        """
        start_time = time.time()
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return self._create_empty_result(time.time() - start_time)
        
        # Extract face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Get eye and iris coordinates
        eye_data = self._extract_eye_data(face_landmarks, w, h)
        
        # Calculate gaze direction
        gaze_direction = self._calculate_gaze_direction(eye_data)
        
        # Apply smoothing if previous result exists
        if self.last_valid_result:
            gaze_direction = self._smooth_gaze(gaze_direction)
        
        # Determine screen attention
        looking_at_screen = self._is_looking_at_screen(gaze_direction)
        
        # Calculate confidence
        confidence = self._calculate_confidence(gaze_direction, eye_data)
        
        processing_time = time.time() - start_time
        
        # Create result
        result = GazeResult(
            looking_at_screen=looking_at_screen,
            gaze_direction=gaze_direction,
            confidence=confidence,
            processing_time=processing_time,
            eye_center=eye_data.get('eye_center', (0, 0)),
            iris_center=eye_data.get('iris_center', (0, 0))
        )
        
        self.last_valid_result = result
        self._update_performance_stats(processing_time)
        
        return result
    
    def _extract_eye_data(self, face_landmarks, width: int, height: int) -> Dict:
        """Extract eye and iris coordinates from face landmarks"""
        landmarks = face_landmarks.landmark
        
        # Extract eye coordinates
        left_eye_coords = [(int(landmarks[i].x * width), int(landmarks[i].y * height)) 
                          for i in self.LEFT_EYE]
        right_eye_coords = [(int(landmarks[i].x * width), int(landmarks[i].y * height)) 
                           for i in self.RIGHT_EYE]
        
        # Extract iris coordinates (if refined landmarks available)
        left_iris_coords = []
        right_iris_coords = []
        
        if len(landmarks) > 468:  # Refined landmarks available
            left_iris_coords = [(int(landmarks[i].x * width), int(landmarks[i].y * height)) 
                               for i in self.LEFT_IRIS]
            right_iris_coords = [(int(landmarks[i].x * width), int(landmarks[i].y * height)) 
                                for i in self.RIGHT_IRIS]
        
        # Calculate centers
        left_eye_center = self._calculate_center(left_eye_coords)
        right_eye_center = self._calculate_center(right_eye_coords)
        eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                     (left_eye_center[1] + right_eye_center[1]) // 2)
        
        iris_center = (0, 0)
        if left_iris_coords and right_iris_coords:
            left_iris_center = self._calculate_center(left_iris_coords)
            right_iris_center = self._calculate_center(right_iris_coords)
            iris_center = ((left_iris_center[0] + right_iris_center[0]) // 2,
                          (left_iris_center[1] + right_iris_center[1]) // 2)
        
        return {
            'left_eye': left_eye_coords,
            'right_eye': right_eye_coords,
            'left_iris': left_iris_coords,
            'right_iris': right_iris_coords,
            'left_eye_center': left_eye_center,
            'right_eye_center': right_eye_center,
            'eye_center': eye_center,
            'iris_center': iris_center
        }
    
    def _calculate_center(self, coordinates: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Calculate center point of a list of coordinates"""
        if not coordinates:
            return (0, 0)
        
        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]
        
        return (int(np.mean(x_coords)), int(np.mean(y_coords)))
    
    def _calculate_gaze_direction(self, eye_data: Dict) -> Tuple[float, float]:
        """Calculate gaze direction from eye and iris data"""
        
        # Use iris data if available for more accurate gaze estimation
        if eye_data['left_iris'] and eye_data['right_iris']:
            return self._calculate_iris_gaze(eye_data)
        else:
            return self._calculate_eye_gaze(eye_data)
    
    def _calculate_iris_gaze(self, eye_data: Dict) -> Tuple[float, float]:
        """Calculate gaze direction using iris positions"""
        left_eye_center = eye_data['left_eye_center']
        right_eye_center = eye_data['right_eye_center']
        left_iris_center = self._calculate_center(eye_data['left_iris'])
        right_iris_center = self._calculate_center(eye_data['right_iris'])
        
        # Calculate iris displacement from eye center
        left_displacement = (
            (left_iris_center[0] - left_eye_center[0]) / 30.0,
            (left_iris_center[1] - left_eye_center[1]) / 30.0
        )
        
        right_displacement = (
            (right_iris_center[0] - right_eye_center[0]) / 30.0,
            (right_iris_center[1] - right_eye_center[1]) / 30.0
        )
        
        # Average both eyes
        gaze_x = (left_displacement[0] + right_displacement[0]) / 2.0
        gaze_y = (left_displacement[1] + right_displacement[1]) / 2.0
        
        return (gaze_x, gaze_y)
    
    def _calculate_eye_gaze(self, eye_data: Dict) -> Tuple[float, float]:
        """Calculate gaze direction using eye positions (fallback method)"""
        eye_center = eye_data['eye_center']
        
        # Calculate relative position to screen center
        gaze_x = (eye_center[0] - self.screen_center[0]) / self.screen_center[0]
        gaze_y = (eye_center[1] - self.screen_center[1]) / self.screen_center[1]
        
        return (gaze_x, gaze_y)
    
    def _smooth_gaze(self, current_gaze: Tuple[float, float]) -> Tuple[float, float]:
        """Apply temporal smoothing to gaze direction"""
        if not self.last_valid_result:
            return current_gaze
        
        prev_gaze = self.last_valid_result.gaze_direction
        smoothing = self.config['smoothing_factor']
        
        smoothed_x = smoothing * prev_gaze[0] + (1 - smoothing) * current_gaze[0]
        smoothed_y = smoothing * prev_gaze[1] + (1 - smoothing) * current_gaze[1]
        
        return (smoothed_x, smoothed_y)
    
    def _is_looking_at_screen(self, gaze_direction: Tuple[float, float]) -> bool:
        """Determine if the person is looking at the screen"""
        gaze_x, gaze_y = gaze_direction
        gaze_magnitude = math.sqrt(gaze_x**2 + gaze_y**2)
        
        return gaze_magnitude < self.attention_threshold
    
    def _calculate_confidence(self, gaze_direction: Tuple[float, float], eye_data: Dict) -> float:
        """Calculate confidence score for the gaze detection"""
        gaze_x, gaze_y = gaze_direction
        gaze_magnitude = math.sqrt(gaze_x**2 + gaze_y**2)
        
        # Base confidence from gaze magnitude
        base_confidence = max(0.0, 1.0 - gaze_magnitude)
        
        # Bonus confidence if iris data is available
        iris_bonus = 0.1 if eye_data['left_iris'] and eye_data['right_iris'] else 0.0
        
        # Eye detection quality bonus
        eye_quality = min(len(eye_data['left_eye']), len(eye_data['right_eye'])) / 16.0
        
        final_confidence = min(1.0, base_confidence + iris_bonus + eye_quality * 0.1)
        
        return final_confidence
    
    def _create_empty_result(self, processing_time: float) -> GazeResult:
        """Create empty result when no face is detected"""
        return GazeResult(
            looking_at_screen=False,
            gaze_direction=(0.0, 0.0),
            confidence=0.0,
            processing_time=processing_time,
            eye_center=(0, 0),
            iris_center=(0, 0)
        )
    
    def _update_performance_stats(self, processing_time: float):
        """Update performance statistics"""
        self.frame_count += 1
        self.total_processing_time += processing_time
    
    def get_average_fps(self) -> float:
        """Get average FPS performance"""
        if self.total_processing_time == 0:
            return 0.0
        return self.frame_count / self.total_processing_time
    
    def update_config(self, new_config: Dict):
        """Update configuration parameters"""
        self.config.update(new_config)
        self.attention_threshold = self.config['attention_threshold']
        self.screen_center = self.config['screen_center']
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.frame_count = 0
        self.total_processing_time = 0
        self.last_valid_result = None