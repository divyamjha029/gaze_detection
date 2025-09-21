import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional, Dict
import time
import math
import os
import pickle
import json
from datetime import datetime
import threading
import tkinter as tk
from tkinter import messagebox
import pygame

class FaceRecognitionSystem:
    """
    Face recognition system to identify authorized users vs intruders.
    Uses face embeddings for secure identification.
    """
    
    def __init__(self):
        self.authorized_faces = []
        self.face_database_file = "authorized_faces.pkl"
        self.recognition_threshold = 0.6
        self.face_embeddings = []
        
        # Load existing authorized faces
        self.load_authorized_faces()
        
        # Initialize face recognition
        try:
            import face_recognition
            self.face_recognition_available = True
            print("✅ Face recognition library available")
        except ImportError:
            self.face_recognition_available = False
            print("⚠️ Face recognition library not available. Install with: pip install face-recognition")
    
    def register_authorized_face(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> bool:
        """
        Register a new authorized face from the current frame.
        
        Args:
            frame: Current video frame
            face_bbox: Face bounding box (x, y, width, height)
            
        Returns:
            True if face was successfully registered
        """
        if not self.face_recognition_available:
            print("Face recognition not available for registration")
            return False
        
        try:
            import face_recognition
            
            # Extract face region
            x, y, w, h = face_bbox
            face_region = frame[y:y+h, x:x+w]
            
            # Convert BGR to RGB
            rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            
            # Get face encoding
            face_encodings = face_recognition.face_encodings(rgb_face)
            
            if face_encodings:
                # Store the first face encoding
                face_encoding = face_encodings[0]
                self.face_embeddings.append(face_encoding)
                
                # Save to file
                self.save_authorized_faces()
                
                print(f"✅ Authorized face registered! Total authorized faces: {len(self.face_embeddings)}")
                return True
            else:
                print("❌ Could not extract face encoding for registration")
                return False
                
        except Exception as e:
            print(f"Error registering face: {e}")
            return False
    
    def is_authorized_face(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> bool:
        """
        Check if the detected face belongs to an authorized user.
        
        Args:
            frame: Current video frame
            face_bbox: Face bounding box
            
        Returns:
            True if face is authorized, False if intruder
        """
        if not self.face_recognition_available or not self.face_embeddings:
            return True  # If no face recognition or no registered faces, assume authorized
        
        try:
            import face_recognition
            
            # Extract face region
            x, y, w, h = face_bbox
            face_region = frame[y:y+h, x:x+w]
            
            # Convert BGR to RGB
            rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            
            # Get face encoding
            face_encodings = face_recognition.face_encodings(rgb_face)
            
            if not face_encodings:
                return False  # Could not detect face properly
            
            current_encoding = face_encodings[0]
            
            # Compare with authorized faces
            matches = face_recognition.compare_faces(
                self.face_embeddings, 
                current_encoding, 
                tolerance=self.recognition_threshold
            )
            
            return any(matches)
            
        except Exception as e:
            print(f"Error checking face authorization: {e}")
            return True  # Default to authorized on error

    def save_authorized_faces(self):
        """Save authorized face embeddings to file"""
        try:
            with open(self.face_database_file, 'wb') as f:
                pickle.dump(self.face_embeddings, f)
            print(f"Authorized faces saved to {self.face_database_file}")
        except Exception as e:
            print(f"Error saving authorized faces: {e}")
    
    def load_authorized_faces(self):
        """Load authorized face embeddings from file"""
        try:
            if os.path.exists(self.face_database_file):
                with open(self.face_database_file, 'rb') as f:
                    self.face_embeddings = pickle.load(f)
                print(f"Loaded {len(self.face_embeddings)} authorized faces")
            else:
                print("No authorized faces database found. Will create on first registration.")
        except Exception as e:
            print(f"Error loading authorized faces: {e}")
            self.face_embeddings = []
    
    def reset_authorized_faces(self):
        """Reset all authorized faces"""
        self.face_embeddings = []
        if os.path.exists(self.face_database_file):
            os.remove(self.face_database_file)
        print("All authorized faces have been reset")

class IntruderNotificationSystem:
    """
    Notification system for intruder detection alerts.
    Supports multiple notification methods: popup, sound, system tray.
    """
    
    def __init__(self):
        self.notification_enabled = True
        self.sound_enabled = True
        self.popup_enabled = True
        self.log_file = "intruder_log.json"
        
        # Initialize sound system
        self.init_sound_system()
        
        # Intruder detection history
        self.intruder_detections = []
        self.load_intruder_log()
    
    def init_sound_system(self):
        """Initialize pygame for sound alerts"""
        try:
            pygame.mixer.init()
            self.sound_available = True
            print("✅ Sound system initialized")
        except Exception as e:
            self.sound_available = False
            print(f"⚠️ Sound system not available: {e}")
    
    def show_intruder_alert(self, confidence: float = 0.0):
        """
        Show intruder detection alert with multiple notification methods.
        
        Args:
            confidence: Detection confidence level
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log the detection
        detection_record = {
            "timestamp": timestamp,
            "confidence": confidence,
            "alert_type": "intruder_detected"
        }
        
        self.intruder_detections.append(detection_record)
        self.save_intruder_log()
        
        print(f"🚨 INTRUDER DETECTED at {timestamp} (Confidence: {confidence:.2f})")
        
        # Show popup notification
        if self.popup_enabled:
            self.show_popup_alert(timestamp, confidence)
        
        # Play sound alert
        if self.sound_enabled and self.sound_available:
            self.play_alert_sound()
        
        # System tray notification (if available)
        self.show_system_notification(timestamp, confidence)
    
    def show_popup_alert(self, timestamp: str, confidence: float):
        """Show popup alert window"""
        def show_alert():
            root = tk.Tk()
            root.withdraw()  # Hide main window
            
            alert_message = f"""
🚨 SECURITY ALERT 🚨

Someone else is looking at your screen!

Time: {timestamp}
Confidence: {confidence:.1%}

This is not the registered user.
            """
            
            messagebox.showwarning(
                "Intruder Detection Alert", 
                alert_message
            )
            
            root.destroy()
        
        # Run in separate thread to avoid blocking
        alert_thread = threading.Thread(target=show_alert, daemon=True)
        alert_thread.start()
    
    def play_alert_sound(self):
        """Play sound alert"""
        try:
            # Create a simple beep sound
            duration = 500  # milliseconds
            freq = 800  # Hz
            
            # Generate sine wave
            sample_rate = 22050
            frames = int(duration * sample_rate / 1000)
            arr = np.zeros((frames, 2))
            
            for i in range(frames):
                wave = np.sin(2 * np.pi * freq * i / sample_rate)
                arr[i] = [wave, wave]
            
            # Convert to pygame sound
            arr = (arr * 32767).astype(np.int16)
            sound = pygame.sndarray.make_sound(arr)
            sound.play()
            
        except Exception as e:
            print(f"Error playing alert sound: {e}")
    
    def show_system_notification(self, timestamp: str, confidence: float):
        """Show system tray notification"""
        try:
            # Try to use plyer for cross-platform notifications
            try:
                from plyer import notification
                notification.notify(
                    title="🚨 Intruder Detected",
                    message=f"Someone is looking at your screen!\nTime: {timestamp}",
                    timeout=10
                )
            except ImportError:
                # Fallback to OS-specific notifications
                import platform
                system = platform.system()
                
                if system == "Windows":
                    import subprocess
                    subprocess.run([
                        'powershell', '-Command',
                        f'Add-Type -AssemblyName System.Windows.Forms; '
                        f'[System.Windows.Forms.MessageBox]::Show("Intruder detected at {timestamp}", "Security Alert")'
                    ])
                elif system == "Darwin":  # macOS
                    os.system(f'osascript -e \'display notification "Intruder detected at {timestamp}" with title "Security Alert"\'')
                elif system == "Linux":
                    os.system(f'notify-send "Security Alert" "Intruder detected at {timestamp}"')
                    
        except Exception as e:
            print(f"System notification error: {e}")
    
    def save_intruder_log(self):
        """Save intruder detection log"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.intruder_detections, f, indent=2)
        except Exception as e:
            print(f"Error saving intruder log: {e}")
    
    def load_intruder_log(self):
        """Load intruder detection log"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    self.intruder_detections = json.load(f)
                print(f"Loaded {len(self.intruder_detections)} intruder detection records")
        except Exception as e:
            print(f"Error loading intruder log: {e}")
            self.intruder_detections = []
    
    def get_intruder_statistics(self) -> Dict:
        """Get intruder detection statistics"""
        if not self.intruder_detections:
            return {"total_detections": 0}
        
        total_detections = len(self.intruder_detections)
        latest_detection = self.intruder_detections[-1]["timestamp"]
        
        # Count detections in last 24 hours
        from datetime import datetime, timedelta
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        
        recent_detections = 0
        for detection in self.intruder_detections:
            detection_time = datetime.strptime(detection["timestamp"], "%Y-%m-%d %H:%M:%S")
            if detection_time > yesterday:
                recent_detections += 1
        
        return {
            "total_detections": total_detections,
            "recent_detections_24h": recent_detections,
            "latest_detection": latest_detection
        }

class EnhancedGazeDetectionSystem:
    """
    Enhanced gaze detection system with face recognition and intruder detection.
    Combines gaze tracking with security features.
    """
    
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        
        # Calibration parameters
        self.screen_center = (320, 240)
        self.attention_threshold = 0.3
        self.calibration_data = []
        
        # Eye landmarks indices
        self.LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        # Security components
        self.face_recognition = FaceRecognitionSystem()
        self.notification_system = IntruderNotificationSystem()
        
        # System states
        self.registration_mode = len(self.face_recognition.face_embeddings) == 0
        self.intruder_detected = False
        self.last_intruder_alert = 0
        self.alert_cooldown = 5  # seconds between alerts
    
    def detect_face_and_eyes(self, frame: np.ndarray) -> Optional[dict]:
        """Detect face and extract eye regions with security checking"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
            
        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Calculate face bounding box for recognition
        x_coords = [int(landmark.x * w) for landmark in face_landmarks.landmark]
        y_coords = [int(landmark.y * h) for landmark in face_landmarks.landmark]
        
        face_bbox = (
            min(x_coords), min(y_coords),
            max(x_coords) - min(x_coords),
            max(y_coords) - min(y_coords)
        )
        
        # Security check - is this an authorized user?
        if self.registration_mode:
            # In registration mode, register the current face
            if self.face_recognition.register_authorized_face(frame, face_bbox):
                self.registration_mode = False
                print("✅ Face registration complete! Now monitoring for intruders...")
        else:
            # Check if this is an authorized user
            is_authorized = self.face_recognition.is_authorized_face(frame, face_bbox)
            
            if not is_authorized:
                self.intruder_detected = True
                current_time = time.time()
                
                # Send alert if cooldown period has passed
                if current_time - self.last_intruder_alert > self.alert_cooldown:
                    self.notification_system.show_intruder_alert(confidence=0.85)
                    self.last_intruder_alert = current_time
            else:
                self.intruder_detected = False
        
        # Extract eye coordinates
        left_eye_coords = [(int(landmark.x * w), int(landmark.y * h)) 
                          for idx, landmark in enumerate(face_landmarks.landmark) 
                          if idx in self.LEFT_EYE_LANDMARKS]
        
        right_eye_coords = [(int(landmark.x * w), int(landmark.y * h)) 
                           for idx, landmark in enumerate(face_landmarks.landmark) 
                           if idx in self.RIGHT_EYE_LANDMARKS]
        
        left_iris_coords = []
        right_iris_coords = []
        
        if len(face_landmarks.landmark) > 468:
            left_iris_coords = [(int(landmark.x * w), int(landmark.y * h)) 
                               for idx, landmark in enumerate(face_landmarks.landmark) 
                               if idx in self.LEFT_IRIS]
            
            right_iris_coords = [(int(landmark.x * w), int(landmark.y * h)) 
                                for idx, landmark in enumerate(face_landmarks.landmark) 
                                if idx in self.RIGHT_IRIS]
        
        return {
            'left_eye': left_eye_coords,
            'right_eye': right_eye_coords,
            'left_iris': left_iris_coords,
            'right_iris': right_iris_coords,
            'face_landmarks': face_landmarks,
            'face_bbox': face_bbox,
            'is_authorized': not self.intruder_detected
        }
    
    def calculate_eye_center(self, eye_coords: list) -> Tuple[int, int]:
        """Calculate the center of an eye region"""
        if not eye_coords:
            return (0, 0)
        
        x_coords = [coord[0] for coord in eye_coords]
        y_coords = [coord[1] for coord in eye_coords]
        
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        
        return (center_x, center_y)
    
    def estimate_gaze_direction(self, eye_data: dict) -> Tuple[float, float]:
        """Estimate gaze direction"""
        left_eye_center = self.calculate_eye_center(eye_data['left_eye'])
        right_eye_center = self.calculate_eye_center(eye_data['right_eye'])
        
        if eye_data['left_iris'] and eye_data['right_iris']:
            left_iris_center = self.calculate_eye_center(eye_data['left_iris'])
            right_iris_center = self.calculate_eye_center(eye_data['right_iris'])
            
            left_gaze_x = (left_iris_center[0] - left_eye_center[0]) / 50.0
            left_gaze_y = (left_iris_center[1] - left_eye_center[1]) / 50.0
            
            right_gaze_x = (right_iris_center[0] - right_eye_center[0]) / 50.0
            right_gaze_y = (right_iris_center[1] - right_eye_center[1]) / 50.0
            
            gaze_x = (left_gaze_x + right_gaze_x) / 2.0
            gaze_y = (left_gaze_y + right_gaze_y) / 2.0
        else:
            avg_eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                             (left_eye_center[1] + right_eye_center[1]) // 2)
            
            gaze_x = (avg_eye_center[0] - self.screen_center[0]) / self.screen_center[0]
            gaze_y = (avg_eye_center[1] - self.screen_center[1]) / self.screen_center[1]
        
        return (gaze_x, gaze_y)
    
    def is_looking_at_screen(self, gaze_direction: Tuple[float, float]) -> bool:
        """Determine if looking at screen"""
        gaze_x, gaze_y = gaze_direction
        gaze_magnitude = math.sqrt(gaze_x**2 + gaze_y**2)
        return gaze_magnitude < self.attention_threshold
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """Process frame with security features"""
        start_time = time.time()
        
        eye_data = self.detect_face_and_eyes(frame)
        
        if not eye_data:
            return {
                'looking_at_screen': False,
                'gaze_direction': (0, 0),
                'confidence': 0.0,
                'processing_time': time.time() - start_time,
                'eye_data': None,
                'is_authorized': True,
                'registration_mode': self.registration_mode,
                'intruder_detected': False
            }
        
        gaze_direction = self.estimate_gaze_direction(eye_data)
        looking_at_screen = self.is_looking_at_screen(gaze_direction)
        
        gaze_magnitude = math.sqrt(gaze_direction[0]**2 + gaze_direction[1]**2)
        confidence = max(0.0, 1.0 - gaze_magnitude)
        
        processing_time = time.time() - start_time
        
        return {
            'looking_at_screen': looking_at_screen,
            'gaze_direction': gaze_direction,
            'confidence': confidence,
            'processing_time': processing_time,
            'eye_data': eye_data,
            'is_authorized': eye_data['is_authorized'],
            'registration_mode': self.registration_mode,  
            'intruder_detected': self.intruder_detected
        }
    
    def draw_annotations(self, frame: np.ndarray, results: dict) -> np.ndarray:
        """Draw enhanced annotations with security status"""
        annotated_frame = frame.copy()
        
        if not results['eye_data']:
            cv2.putText(annotated_frame, "No face detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return annotated_frame
        
        eye_data = results['eye_data']
        
        # Draw face bounding box with color based on authorization
        x, y, w, h = eye_data['face_bbox']
        if results['intruder_detected']:
            bbox_color = (0, 0, 255)  # Red for intruder
            status_text = "🚨 INTRUDER DETECTED!"
            status_color = (0, 0, 255)
        elif results['registration_mode']:
            bbox_color = (0, 255, 255)  # Yellow for registration
            status_text = "📝 REGISTERING YOUR FACE..."
            status_color = (0, 255, 255)
        else:
            bbox_color = (0, 255, 0)  # Green for authorized
            status_text = "✅ AUTHORIZED USER"
            status_color = (0, 255, 0)
        
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), bbox_color, 2)
        
        # Draw security status
        cv2.putText(annotated_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Draw eye regions
        if eye_data['left_eye'] and not results['intruder_detected']:
            for point in eye_data['left_eye']:
                cv2.circle(annotated_frame, point, 2, (255, 0, 0), -1)
        
        if eye_data['right_eye'] and not results['intruder_detected']:
            for point in eye_data['right_eye']:
                cv2.circle(annotated_frame, point, 2, (255, 0, 0), -1)
        
        # Draw iris points
        if eye_data['left_iris'] and not results['intruder_detected']:
            for point in eye_data['left_iris']:
                cv2.circle(annotated_frame, point, 3, (0, 255, 0), -1)
        
        if eye_data['right_iris'] and not results['intruder_detected']:
            for point in eye_data['right_iris']:
                cv2.circle(annotated_frame, point, 3, (0, 255, 0), -1)
        
        # Only show gaze info for authorized users
        if results['is_authorized'] and not results['registration_mode']:
            # Draw gaze direction
            gaze_x, gaze_y = results['gaze_direction']
            center_x, center_y = annotated_frame.shape[1] // 2, annotated_frame.shape[0] // 2
            
            end_x = int(center_x + gaze_x * 100)
            end_y = int(center_y + gaze_y * 100)
            
            cv2.arrowedLine(annotated_frame, (center_x, center_y), (end_x, end_y), 
                           (0, 255, 255), 3, tipLength=0.3)
            
            # Draw gaze status
            gaze_status = "LOOKING AT SCREEN" if results['looking_at_screen'] else "NOT LOOKING AT SCREEN"
            gaze_color = (0, 255, 0) if results['looking_at_screen'] else (0, 0, 255)
            
            cv2.putText(annotated_frame, gaze_status, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, gaze_color, 2)
            
            cv2.putText(annotated_frame, f"Confidence: {results['confidence']:.2f}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(annotated_frame, f"Processing: {results['processing_time']*1000:.1f}ms", (10, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_frame


def main():
    """Enhanced main function with face recognition and intruder detection"""
    
    # Initialize the enhanced gaze detection system
    gaze_detector = EnhancedGazeDetectionSystem()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("🔐 Enhanced Gaze Detection System with Intruder Detection")
    print("=" * 60)
    
    if gaze_detector.registration_mode:
        print("👤 FIRST TIME SETUP")
        print("   Look directly at the camera to register your face...")
        print("   Your face will be saved securely for future recognition.")
    else:
        print("✅ Face recognition enabled")
        print(f"   {len(gaze_detector.face_recognition.face_embeddings)} authorized face(s) loaded")
        
        # Show intruder statistics
        stats = gaze_detector.notification_system.get_intruder_statistics()
        if stats["total_detections"] > 0:
            print(f"📊 Intruder Statistics:")
            print(f"   Total detections: {stats['total_detections']}")
            print(f"   Recent (24h): {stats['recent_detections_24h']}")
            print(f"   Latest: {stats['latest_detection']}")
    
    print("\n🎮 Controls:")
    print("   'q' - Quit")
    print("   'r' - Reset authorized faces")
    print("   's' - Save screenshot")
    print("   'n' - Toggle notifications")
    print("   'c' - Show statistics")
    
    # Performance tracking
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Process frame
        results = gaze_detector.process_frame(frame)
        
        # Draw annotations
        annotated_frame = gaze_detector.draw_annotations(frame, results)
        
        # Calculate FPS
        fps_counter += 1
        if fps_counter % 30 == 0:
            fps_end_time = time.time()
            current_fps = 30 / (fps_end_time - fps_start_time)
            fps_start_time = fps_end_time
        
        # Display FPS
        cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (10, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Show registration instructions
        if results['registration_mode']:
            instruction = "Look at camera to register your face"
            cv2.putText(annotated_frame, instruction, 
                       (10, annotated_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display the frame
        cv2.imshow('Enhanced Gaze Detection - Security Mode', annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset authorized faces
            gaze_detector.face_recognition.reset_authorized_faces()
            gaze_detector.registration_mode = True
            print("\n🔄 Authorized faces reset. Please register your face again.")
        elif key == ord('s'):
            # Save screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"security_screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"📸 Screenshot saved: {filename}")
        elif key == ord('n'):
            # Toggle notifications
            gaze_detector.notification_system.notification_enabled = not gaze_detector.notification_system.notification_enabled
            status = "enabled" if gaze_detector.notification_system.notification_enabled else "disabled"
            print(f"🔔 Notifications {status}")
        elif key == ord('c'):
            # Show statistics
            stats = gaze_detector.notification_system.get_intruder_statistics()
            print("\n📊 Security Statistics:")
            print(f"   Authorized faces: {len(gaze_detector.face_recognition.face_embeddings)}")
            print(f"   Total intruder detections: {stats.get('total_detections', 0)}")
            print(f"   Recent detections (24h): {stats.get('recent_detections_24h', 0)}")
            if stats.get('latest_detection'):
                print(f"   Latest detection: {stats['latest_detection']}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n🔒 Enhanced Gaze Detection System stopped")
    print("Your security data has been saved.")


if __name__ == "__main__":
    main()