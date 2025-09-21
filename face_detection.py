"""
Face detection module providing multiple detection backends and optimizations.
Supports MediaPipe, OpenCV Haar Cascades, and Dlib-based face detection.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class FaceDetectionResult:
    """Data class for face detection results"""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    landmarks: List[Tuple[int, int]]
    confidence: float
    face_id: int = 0

class FaceDetectorBase(ABC):
    """Abstract base class for face detectors"""
    
    @abstractmethod
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetectionResult]:
        """Detect faces in the given frame"""
        pass
    
    @abstractmethod
    def get_face_landmarks(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
        """Get facial landmarks for a detected face"""
        pass

class MediaPipeFaceDetector(FaceDetectorBase):
    """
    MediaPipe-based face detector with high accuracy and performance.
    Provides 468 facial landmarks including eye and iris landmarks.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize MediaPipe face detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=self.config['model_selection'],
            min_detection_confidence=self.config['min_detection_confidence']
        )
        
        # Initialize MediaPipe Face Mesh for landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.config['min_detection_confidence'],
            min_tracking_confidence=self.config['min_tracking_confidence']
        )
        
    def _default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'model_selection': 0,  # 0 for short-range (2m), 1 for full-range (5m)
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5
        }
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetectionResult]:
        """Detect faces using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            h, w = frame.shape[:2]
            
            for i, detection in enumerate(results.detections):
                bbox_rel = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to absolute
                x = int(bbox_rel.xmin * w)
                y = int(bbox_rel.ymin * h)
                width = int(bbox_rel.width * w)
                height = int(bbox_rel.height * h)
                
                # Get landmarks
                landmarks = self.get_face_landmarks(frame, (x, y, width, height))
                
                face_result = FaceDetectionResult(
                    bbox=(x, y, width, height),
                    landmarks=landmarks,
                    confidence=detection.score[0],
                    face_id=i
                )
                
                faces.append(face_result)
        
        return faces
    
    def get_face_landmarks(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
        """Get facial landmarks using MediaPipe Face Mesh"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        landmarks = []
        if results.multi_face_landmarks:
            h, w = frame.shape[:2]
            face_landmarks = results.multi_face_landmarks[0]
            
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append((x, y))
        
        return landmarks

class OpenCVFaceDetector(FaceDetectorBase):
    """
    OpenCV Haar Cascade-based face detector.
    Fast and lightweight but less accurate than MediaPipe.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize OpenCV face detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        
        # Load Haar cascade classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Try to load dlib predictor if available
        self.dlib_predictor = None
        try:
            import dlib
            self.dlib_detector = dlib.get_frontal_face_detector()
            # You need to download shape_predictor_68_face_landmarks.dat
            self.dlib_predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
        except (ImportError, RuntimeError):
            print("Dlib not available or landmark file not found. Using basic OpenCV detection.")
    
    def _default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'scale_factor': 1.1,
            'min_neighbors': 5,
            'min_size': (30, 30),
            'max_size': ()
        }
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetectionResult]:
        """Detect faces using OpenCV Haar Cascades"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.config['scale_factor'],
            minNeighbors=self.config['min_neighbors'],
            minSize=self.config['min_size']
        )
        
        results = []
        for i, (x, y, w, h) in enumerate(faces):
            landmarks = self.get_face_landmarks(frame, (x, y, w, h))
            
            face_result = FaceDetectionResult(
                bbox=(x, y, w, h),
                landmarks=landmarks,
                confidence=0.8,  # Haar cascades don't provide confidence scores
                face_id=i
            )
            
            results.append(face_result)
        
        return results
    
    def get_face_landmarks(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
        """Get facial landmarks using dlib or basic eye detection"""
        landmarks = []
        
        if self.dlib_predictor:
            landmarks = self._get_dlib_landmarks(frame, bbox)
        else:
            landmarks = self._get_basic_landmarks(frame, bbox)
        
        return landmarks
    
    def _get_dlib_landmarks(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
        """Get 68 facial landmarks using dlib"""
        import dlib
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = bbox
        
        # Convert to dlib rectangle
        rect = dlib.rectangle(x, y, x + w, y + h)
        
        # Get landmarks
        landmarks = self.dlib_predictor(gray, rect)
        
        landmark_points = []
        for i in range(landmarks.num_parts):
            point = landmarks.part(i)
            landmark_points.append((point.x, point.y))
        
        return landmark_points
    
    def _get_basic_landmarks(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
        """Get basic eye landmarks using OpenCV eye detection"""
        x, y, w, h = bbox
        face_roi = frame[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        eyes = self.eye_cascade.detectMultiScale(gray_roi)
        
        landmarks = []
        for (ex, ey, ew, eh) in eyes:
            # Convert eye coordinates to full frame coordinates
            eye_center_x = x + ex + ew // 2
            eye_center_y = y + ey + eh // 2
            landmarks.append((eye_center_x, eye_center_y))
        
        # Add face center and corners as landmarks if no eyes detected
        if not landmarks:
            landmarks = [
                (x + w // 2, y + h // 3),  # Approximate left eye
                (x + 2 * w // 3, y + h // 3),  # Approximate right eye
                (x + w // 2, y + 2 * h // 3),  # Approximate nose
                (x + w // 2, y + 4 * h // 5)   # Approximate mouth
            ]
        
        return landmarks

class AdaptiveFaceDetector:
    """
    Adaptive face detector that switches between different detection methods
    based on performance and accuracy requirements.
    """
    
    def __init__(self, primary_detector: str = 'mediapipe'):
        """
        Initialize adaptive face detector.
        
        Args:
            primary_detector: Primary detection method ('mediapipe' or 'opencv')
        """
        self.primary_detector = primary_detector
        self.current_detector = None
        
        # Initialize detectors
        self.mediapipe_detector = MediaPipeFaceDetector()
        self.opencv_detector = OpenCVFaceDetector()
        
        # Performance tracking
        self.performance_history = []
        self.detection_failures = 0
        self.max_failures = 5
        
        # Set initial detector
        self._switch_detector(primary_detector)
    
    def _switch_detector(self, detector_type: str):
        """Switch to a different detector"""
        if detector_type == 'mediapipe':
            self.current_detector = self.mediapipe_detector
        elif detector_type == 'opencv':
            self.current_detector = self.opencv_detector
        
        print(f"Switched to {detector_type} face detector")
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetectionResult]:
        """Detect faces using adaptive method selection"""
        try:
            results = self.current_detector.detect_faces(frame)
            
            if results:
                self.detection_failures = 0
                return results
            else:
                self.detection_failures += 1
                
                # Switch detector if too many failures
                if self.detection_failures >= self.max_failures:
                    self._fallback_detector()
                    self.detection_failures = 0
                
                return []
                
        except Exception as e:
            print(f"Detection error: {e}")
            self._fallback_detector()
            return []
    
    def _fallback_detector(self):
        """Switch to fallback detector"""
        if isinstance(self.current_detector, MediaPipeFaceDetector):
            self._switch_detector('opencv')
        else:
            self._switch_detector('mediapipe')
    
    def get_face_landmarks(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
        """Get facial landmarks using current detector"""
        return self.current_detector.get_face_landmarks(frame, bbox)
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            'current_detector': type(self.current_detector).__name__,
            'detection_failures': self.detection_failures,
            'performance_history': self.performance_history[-10:]  # Last 10 entries
        }

class FaceROIExtractor:
    """
    Utility class for extracting regions of interest from detected faces.
    Provides methods for extracting eyes, nose, mouth regions.
    """
    
    @staticmethod
    def extract_eye_regions(frame: np.ndarray, landmarks: List[Tuple[int, int]]) -> Dict[str, np.ndarray]:
        """
        Extract left and right eye regions from face landmarks.
        
        Args:
            frame: Input frame
            landmarks: Facial landmarks
            
        Returns:
            Dictionary with 'left_eye' and 'right_eye' image regions
        """
        if len(landmarks) < 68:  # Basic landmarks
            return FaceROIExtractor._extract_basic_eye_regions(frame, landmarks)
        else:  # Full 68-point landmarks
            return FaceROIExtractor._extract_detailed_eye_regions(frame, landmarks)
    
    @staticmethod
    def _extract_basic_eye_regions(frame: np.ndarray, landmarks: List[Tuple[int, int]]) -> Dict[str, np.ndarray]:
        """Extract eye regions from basic landmarks"""
        eye_regions = {'left_eye': None, 'right_eye': None}
        
        if len(landmarks) >= 2:
            # Assume first two landmarks are eyes
            for i, eye_key in enumerate(['left_eye', 'right_eye']):
                if i < len(landmarks):
                    x, y = landmarks[i]
                    # Extract 60x30 pixel region around eye
                    x1, y1 = max(0, x - 30), max(0, y - 15)
                    x2, y2 = min(frame.shape[1], x + 30), min(frame.shape[0], y + 15)
                    eye_regions[eye_key] = frame[y1:y2, x1:x2]
        
        return eye_regions
    
    @staticmethod
    def _extract_detailed_eye_regions(frame: np.ndarray, landmarks: List[Tuple[int, int]]) -> Dict[str, np.ndarray]:
        """Extract eye regions from detailed 68-point landmarks"""
        # Left eye landmarks: 36-41, Right eye landmarks: 42-47
        left_eye_points = landmarks[36:42]
        right_eye_points = landmarks[42:48]
        
        eye_regions = {}
        
        for eye_points, eye_key in [(left_eye_points, 'left_eye'), (right_eye_points, 'right_eye')]:
            if eye_points:
                # Find bounding box of eye region
                x_coords = [p[0] for p in eye_points]
                y_coords = [p[1] for p in eye_points]
                
                x1, x2 = min(x_coords) - 10, max(x_coords) + 10
                y1, y2 = min(y_coords) - 10, max(y_coords) + 10
                
                # Ensure bounds are within frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                eye_regions[eye_key] = frame[y1:y2, x1:x2]
            else:
                eye_regions[eye_key] = None
        
        return eye_regions
    
    @staticmethod
    def extract_face_region(frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                           padding: float = 0.2) -> np.ndarray:
        """
        Extract face region with optional padding.
        
        Args:
            frame: Input frame
            bbox: Face bounding box (x, y, width, height)
            padding: Padding ratio (0.2 = 20% padding)
            
        Returns:
            Extracted face region
        """
        x, y, w, h = bbox
        
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(frame.shape[1], x + w + pad_w)
        y2 = min(frame.shape[0], y + h + pad_h)
        
        return frame[y1:y2, x1:x2]