"""
Calibration module for personalizing gaze detection to individual users.
Provides calibration data collection, processing, and adaptation algorithms.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import json
import time
import math
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class CalibrationPoint:
    """Data class for calibration point information"""
    screen_position: Tuple[float, float]  # Normalized screen coordinates (0-1)
    gaze_direction: Tuple[float, float]   # Detected gaze direction
    eye_position: Tuple[int, int]         # Eye center position
    iris_position: Tuple[int, int]        # Iris center position
    timestamp: float                      # Collection timestamp
    confidence: float                     # Detection confidence

@dataclass
class CalibrationResult:
    """Data class for calibration results"""
    offset_x: float
    offset_y: float
    scale_x: float
    scale_y: float
    accuracy: float
    num_points: int
    calibration_date: str

class CalibrationSystem:
    """
    Advanced calibration system for personalizing gaze detection.
    Collects user-specific calibration data and computes correction parameters.
    """
    
    def __init__(self, screen_resolution: Tuple[int, int] = (1920, 1080)):
        """
        Initialize calibration system.
        
        Args:
            screen_resolution: Target screen resolution (width, height)
        """
        self.screen_resolution = screen_resolution
        self.calibration_points = []
        self.calibration_result = None
        
        # Calibration parameters
        self.min_points_required = 9
        self.max_points = 25
        self.target_accuracy = 0.15  # 15% screen accuracy
        
        # Calibration grid positions (normalized 0-1)
        self.grid_positions = self._generate_grid_positions()
        self.current_grid_index = 0
        
        # Quality thresholds
        self.min_confidence = 0.6
        self.max_distance_error = 0.3
        
    def _generate_grid_positions(self) -> List[Tuple[float, float]]:
        """Generate calibration grid positions"""
        positions = []
        
        # 3x3 grid
        for y in [0.2, 0.5, 0.8]:
            for x in [0.2, 0.5, 0.8]:
                positions.append((x, y))
        
        # Additional points for better accuracy
        extra_points = [
            (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9),  # Corners
            (0.35, 0.35), (0.65, 0.35), (0.35, 0.65), (0.65, 0.65),  # Inner grid
            (0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5)   # Edge midpoints
        ]
        
        positions.extend(extra_points)
        return positions
    
    def start_calibration(self) -> bool:
        """
        Start a new calibration session.
        
        Returns:
            True if calibration started successfully
        """
        self.calibration_points.clear()
        self.current_grid_index = 0
        self.calibration_result = None
        
        print("Starting calibration session...")
        print(f"Please look at {len(self.grid_positions)} calibration points")
        
        return True
    
    def get_next_calibration_point(self) -> Optional[Tuple[float, float]]:
        """
        Get the next calibration point position.
        
        Returns:
            Next calibration point position (normalized coordinates) or None if complete
        """
        if self.current_grid_index >= len(self.grid_positions):
            return None
        
        position = self.grid_positions[self.current_grid_index]
        return position
    
    def add_calibration_point(self, screen_position: Tuple[float, float], 
                            gaze_direction: Tuple[float, float],
                            eye_position: Tuple[int, int],
                            iris_position: Tuple[int, int],
                            confidence: float) -> bool:
        """
        Add a calibration point to the dataset.
        
        Args:
            screen_position: Target screen position (normalized 0-1)
            gaze_direction: Detected gaze direction
            eye_position: Eye center position in pixels
            iris_position: Iris center position in pixels
            confidence: Detection confidence (0-1)
            
        Returns:
            True if point was accepted
        """
        
        # Quality check
        if confidence < self.min_confidence:
            print(f"Point rejected: low confidence ({confidence:.2f})")
            return False
        
        # Create calibration point
        cal_point = CalibrationPoint(
            screen_position=screen_position,
            gaze_direction=gaze_direction,
            eye_position=eye_position,
            iris_position=iris_position,
            timestamp=time.time(),
            confidence=confidence
        )
        
        self.calibration_points.append(cal_point)
        self.current_grid_index += 1
        
        print(f"Calibration point {len(self.calibration_points)}/{len(self.grid_positions)} collected")
        
        return True
    
    def is_calibration_complete(self) -> bool:
        """Check if calibration has enough points"""
        return len(self.calibration_points) >= self.min_points_required
    
    def compute_calibration(self) -> CalibrationResult:
        """
        Compute calibration parameters from collected points.
        
        Returns:
            CalibrationResult with computed parameters
        """
        if not self.is_calibration_complete():
            raise ValueError(f"Need at least {self.min_points_required} calibration points")
        
        # Extract data for computation
        screen_positions = np.array([cp.screen_position for cp in self.calibration_points])
        gaze_directions = np.array([cp.gaze_direction for cp in self.calibration_points])
        
        # Compute offset and scale corrections
        offset_x, offset_y, scale_x, scale_y = self._compute_transformation_parameters(
            screen_positions, gaze_directions
        )
        
        # Evaluate calibration accuracy
        accuracy = self._evaluate_calibration_accuracy(
            screen_positions, gaze_directions, offset_x, offset_y, scale_x, scale_y
        )
        
        # Create result
        self.calibration_result = CalibrationResult(
            offset_x=offset_x,
            offset_y=offset_y,
            scale_x=scale_x,
            scale_y=scale_y,
            accuracy=accuracy,
            num_points=len(self.calibration_points),
            calibration_date=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        print(f"Calibration complete! Accuracy: {accuracy:.1%}")
        
        return self.calibration_result
    
    def _compute_transformation_parameters(self, screen_pos: np.ndarray, 
                                         gaze_dir: np.ndarray) -> Tuple[float, float, float, float]:
        """Compute linear transformation parameters using least squares"""
        
        # Convert screen positions to centered coordinates (-0.5 to 0.5)
        screen_centered = screen_pos - 0.5
        
        # Use least squares to find best fit transformation
        # gaze = scale * screen + offset
        
        # X direction
        A_x = np.column_stack([screen_centered[:, 0], np.ones(len(screen_centered))])
        params_x = np.linalg.lstsq(A_x, gaze_dir[:, 0], rcond=None)[0]
        scale_x, offset_x = params_x
        
        # Y direction
        A_y = np.column_stack([screen_centered[:, 1], np.ones(len(screen_centered))])
        params_y = np.linalg.lstsq(A_y, gaze_dir[:, 1], rcond=None)[0]
        scale_y, offset_y = params_y
        
        return offset_x, offset_y, scale_x, scale_y
    
    def _evaluate_calibration_accuracy(self, screen_pos: np.ndarray, gaze_dir: np.ndarray,
                                     offset_x: float, offset_y: float,
                                     scale_x: float, scale_y: float) -> float:
        """Evaluate calibration accuracy"""
        
        # Apply calibration to gaze directions
        corrected_gaze = self._apply_calibration(gaze_dir, offset_x, offset_y, scale_x, scale_y)
        
        # Convert to screen coordinates
        corrected_screen = corrected_gaze + 0.5
        
        # Calculate RMS error
        errors = np.sqrt(np.sum((corrected_screen - screen_pos) ** 2, axis=1))
        rms_error = np.sqrt(np.mean(errors ** 2))
        
        # Convert to accuracy percentage (1.0 - normalized_error)
        max_error = np.sqrt(2)  # Maximum possible error (corner to corner)
        accuracy = max(0.0, 1.0 - (rms_error / max_error))
        
        return accuracy
    
    def _apply_calibration(self, gaze_direction: np.ndarray, 
                          offset_x: float, offset_y: float,
                          scale_x: float, scale_y: float) -> np.ndarray:
        """Apply calibration transformation to gaze directions"""
        
        corrected = gaze_direction.copy()
        corrected[:, 0] = (gaze_direction[:, 0] - offset_x) / scale_x
        corrected[:, 1] = (gaze_direction[:, 1] - offset_y) / scale_y
        
        return corrected
    
    def apply_calibration_to_gaze(self, gaze_direction: Tuple[float, float]) -> Tuple[float, float]:
        """
        Apply calibration to a single gaze direction measurement.
        
        Args:
            gaze_direction: Raw gaze direction
            
        Returns:
            Calibrated gaze direction
        """
        if not self.calibration_result:
            return gaze_direction
        
        result = self.calibration_result
        
        # Apply inverse transformation
        corrected_x = (gaze_direction[0] - result.offset_x) / result.scale_x
        corrected_y = (gaze_direction[1] - result.offset_y) / result.scale_y
        
        return (corrected_x, corrected_y)
    
    def save_calibration(self, filepath: str) -> bool:
        """
        Save calibration data to file.
        
        Args:
            filepath: Path to save calibration data
            
        Returns:
            True if saved successfully
        """
        if not self.calibration_result:
            print("No calibration result to save")
            return False
        
        try:
            # Prepare data for saving
            save_data = {
                'calibration_result': asdict(self.calibration_result),
                'screen_resolution': self.screen_resolution,
                'calibration_points': [asdict(cp) for cp in self.calibration_points]
            }
            
            # Save to JSON file
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"Calibration saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving calibration: {e}")
            return False
    
    def load_calibration(self, filepath: str) -> bool:
        """
        Load calibration data from file.
        
        Args:
            filepath: Path to calibration file
            
        Returns:
            True if loaded successfully
        """
        try:
            if not Path(filepath).exists():
                print(f"Calibration file not found: {filepath}")
                return False
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load calibration result
            result_data = data['calibration_result']
            self.calibration_result = CalibrationResult(**result_data)
            
            # Load screen resolution
            self.screen_resolution = tuple(data['screen_resolution'])
            
            # Load calibration points
            self.calibration_points = []
            for cp_data in data['calibration_points']:
                cp = CalibrationPoint(**cp_data)
                self.calibration_points.append(cp)
            
            print(f"Calibration loaded from {filepath}")
            print(f"Accuracy: {self.calibration_result.accuracy:.1%}, Points: {self.calibration_result.num_points}")
            
            return True
            
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False
    
    def get_calibration_quality(self) -> Dict:
        """Get calibration quality metrics"""
        if not self.calibration_result:
            return {'status': 'No calibration available'}
        
        result = self.calibration_result
        
        quality_level = "Poor"
        if result.accuracy > 0.8:
            quality_level = "Excellent"
        elif result.accuracy > 0.7:
            quality_level = "Good"
        elif result.accuracy > 0.6:
            quality_level = "Fair"
        
        return {
            'status': 'Calibrated',
            'accuracy': result.accuracy,
            'quality_level': quality_level,
            'num_points': result.num_points,
            'date': result.calibration_date,
            'offset_magnitude': math.sqrt(result.offset_x**2 + result.offset_y**2),
            'scale_consistency': abs(result.scale_x - result.scale_y)
        }
    
    def needs_recalibration(self) -> bool:
        """Check if recalibration is recommended"""
        if not self.calibration_result:
            return True
        
        # Check accuracy
        if self.calibration_result.accuracy < 0.6:
            return True
        
        # Check age (recommend recalibration after 7 days)
        calibration_age = time.time() - time.mktime(
            time.strptime(self.calibration_result.calibration_date, "%Y-%m-%d %H:%M:%S")
        )
        if calibration_age > 7 * 24 * 3600:  # 7 days in seconds
            return True
        
        return False
    
    def reset_calibration(self):
        """Reset calibration data"""
        self.calibration_points.clear()
        self.calibration_result = None
        self.current_grid_index = 0
        print("Calibration data reset")

class CalibrationUI:
    """
    UI helper class for calibration visualization and interaction.
    """
    
    def __init__(self, screen_size: Tuple[int, int] = (800, 600)):
        """Initialize calibration UI"""
        self.screen_size = screen_size
        self.point_radius = 15
        self.animation_frame = 0
        
    def draw_calibration_point(self, frame: np.ndarray, 
                             position: Tuple[float, float],
                             is_active: bool = True) -> np.ndarray:
        """
        Draw calibration point on frame.
        
        Args:
            frame: Input frame
            position: Normalized position (0-1)
            is_active: Whether point is currently active
            
        Returns:
            Frame with calibration point drawn
        """
        h, w = frame.shape[:2]
        
        # Convert normalized position to pixel coordinates
        x = int(position[0] * w)
        y = int(position[1] * h)
        
        # Draw calibration point
        if is_active:
            # Animated pulsing circle
            self.animation_frame += 1
            pulse = int(10 * (1 + 0.3 * math.sin(self.animation_frame * 0.1)))
            radius = self.point_radius + pulse
            
            # Outer circle (red)
            cv2.circle(frame, (x, y), radius, (0, 0, 255), 3)
            # Inner circle (white)
            cv2.circle(frame, (x, y), 8, (255, 255, 255), -1)
            # Center dot (red)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
            
        else:
            # Static completed point (green)
            cv2.circle(frame, (x, y), self.point_radius, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        return frame
    
    def draw_calibration_progress(self, frame: np.ndarray, 
                                current_point: int, total_points: int) -> np.ndarray:
        """Draw calibration progress indicator"""
        
        # Progress bar
        bar_width = 300
        bar_height = 20
        bar_x = (frame.shape[1] - bar_width) // 2
        bar_y = 50
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Progress fill
        progress = current_point / total_points
        fill_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                     (0, 255, 0), -1)
        
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (255, 255, 255), 2)
        
        # Progress text
        progress_text = f"Calibration Progress: {current_point}/{total_points}"
        cv2.putText(frame, progress_text, (bar_x, bar_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def draw_instructions(self, frame: np.ndarray, instruction: str) -> np.ndarray:
        """Draw calibration instructions"""
        
        # Instruction text
        text_lines = instruction.split('\n')
        y_start = frame.shape[0] - 100
        
        for i, line in enumerate(text_lines):
            y = y_start + i * 30
            
            # Text shadow
            cv2.putText(frame, line, (22, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 0), 2)
            # Main text
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
        
        return frame