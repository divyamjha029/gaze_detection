"""
Settings module for configuring gaze detection parameters and user preferences.
Provides a comprehensive settings interface with validation and persistence.
"""

import tkinter as tk
from tkinter import ttk, messagebox, colorchooser
import json
import os
from typing import Dict, Any, Callable
from dataclasses import dataclass, asdict, field

@dataclass
class GazeSettings:
    """Data class for gaze detection settings"""
    
    # Detection parameters
    attention_threshold: float = 0.3
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    smoothing_factor: float = 0.7
    
    # Performance settings
    target_fps: int = 30
    frame_skip: int = 1
    processing_resolution: tuple = (640, 480)
    enable_gpu_acceleration: bool = True
    
    # Privacy settings
    privacy_mode: bool = True
    auto_pause_on_away: bool = True
    data_retention_days: int = 0  # 0 = no retention
    anonymize_logs: bool = True
    
    # UI settings
    show_gaze_vector: bool = True
    show_eye_regions: bool = True
    show_confidence_meter: bool = True
    annotation_color: tuple = (0, 255, 0)
    
    # Calibration settings
    calibration_points: int = 9
    calibration_timeout: int = 30
    auto_recalibrate_days: int = 7
    
    # Notification settings
    enable_notifications: bool = True
    sound_alerts: bool = False
    visual_alerts: bool = True
    
    # Advanced settings
    detector_type: str = "mediapipe"  # mediapipe, opencv, adaptive
    use_iris_tracking: bool = True
    enable_face_mesh: bool = True
    multiface_detection: bool = False

class SettingsDialog:
    """
    Comprehensive settings dialog for configuring all application parameters.
    """
    
    def __init__(self, parent, current_settings: GazeSettings, 
                 on_settings_changed: Callable[[GazeSettings], None]):
        """
        Initialize settings dialog.
        
        Args:
            parent: Parent window
            current_settings: Current settings object
            on_settings_changed: Callback when settings change
        """
        self.parent = parent
        self.settings = current_settings
        self.on_settings_changed = on_settings_changed
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Settings")
        self.dialog.geometry("800x600")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center dialog on parent
        self.center_dialog()
        
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(self.dialog)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_detection_tab()
        self.create_performance_tab()
        self.create_privacy_tab()
        self.create_ui_tab()
        self.create_calibration_tab()
        self.create_notifications_tab()
        self.create_advanced_tab()
        
        # Create buttons
        self.create_buttons()
        
        # Load current settings into UI
        self.load_settings_to_ui()
    
    def center_dialog(self):
        """Center dialog on parent window"""
        
        self.dialog.update_idletasks()
        
        # Get parent window position and size
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        # Calculate center position
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        self.dialog.geometry(f"+{x}+{y}")
    
    def create_detection_tab(self):
        """Create detection parameters tab"""
        
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Detection")
        
        # Attention threshold
        ttk.Label(frame, text="Attention Threshold:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.attention_threshold_var = tk.DoubleVar()
        threshold_scale = ttk.Scale(frame, from_=0.1, to=0.8, 
                                   variable=self.attention_threshold_var, 
                                   orient="horizontal")
        threshold_scale.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.threshold_label = ttk.Label(frame, text="0.3")
        self.threshold_label.grid(row=0, column=2, padx=5, pady=5)
        
        # Update label when scale changes
        def update_threshold_label(*args):
            value = self.attention_threshold_var.get()
            self.threshold_label.config(text=f"{value:.2f}")
        
        self.attention_threshold_var.trace("w", update_threshold_label)
        
        # Detection confidence
        ttk.Label(frame, text="Min Detection Confidence:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.detection_confidence_var = tk.DoubleVar()
        detection_scale = ttk.Scale(frame, from_=0.3, to=0.9, 
                                   variable=self.detection_confidence_var, 
                                   orient="horizontal")
        detection_scale.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        self.detection_label = ttk.Label(frame, text="0.5")
        self.detection_label.grid(row=1, column=2, padx=5, pady=5)
        
        def update_detection_label(*args):
            value = self.detection_confidence_var.get()
            self.detection_label.config(text=f"{value:.2f}")
        
        self.detection_confidence_var.trace("w", update_detection_label)
        
        # Tracking confidence
        ttk.Label(frame, text="Min Tracking Confidence:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.tracking_confidence_var = tk.DoubleVar()
        tracking_scale = ttk.Scale(frame, from_=0.3, to=0.9, 
                                  variable=self.tracking_confidence_var, 
                                  orient="horizontal")
        tracking_scale.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        self.tracking_label = ttk.Label(frame, text="0.5")
        self.tracking_label.grid(row=2, column=2, padx=5, pady=5)
        
        def update_tracking_label(*args):
            value = self.tracking_confidence_var.get()
            self.tracking_label.config(text=f"{value:.2f}")
        
        self.tracking_confidence_var.trace("w", update_tracking_label)
        
        # Smoothing factor
        ttk.Label(frame, text="Smoothing Factor:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.smoothing_var = tk.DoubleVar()
        smoothing_scale = ttk.Scale(frame, from_=0.0, to=0.9, 
                                   variable=self.smoothing_var, 
                                   orient="horizontal")
        smoothing_scale.grid(row=3, column=1, sticky="ew", padx=5, pady=5)
        self.smoothing_label = ttk.Label(frame, text="0.7")
        self.smoothing_label.grid(row=3, column=2, padx=5, pady=5)
        
        def update_smoothing_label(*args):
            value = self.smoothing_var.get()
            self.smoothing_label.config(text=f"{value:.2f}")
        
        self.smoothing_var.trace("w", update_smoothing_label)
        
        frame.grid_columnconfigure(1, weight=1)
    
    def create_performance_tab(self):
        """Create performance settings tab"""
        
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Performance")
        
        # Target FPS
        ttk.Label(frame, text="Target FPS:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.fps_var = tk.IntVar()
        fps_spinbox = ttk.Spinbox(frame, from_=10, to=60, textvariable=self.fps_var, width=10)
        fps_spinbox.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Frame skip
        ttk.Label(frame, text="Frame Skip:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.frame_skip_var = tk.IntVar()
        skip_spinbox = ttk.Spinbox(frame, from_=1, to=5, textvariable=self.frame_skip_var, width=10)
        skip_spinbox.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # Processing resolution
        ttk.Label(frame, text="Processing Resolution:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.resolution_var = tk.StringVar()
        resolution_combo = ttk.Combobox(frame, textvariable=self.resolution_var, 
                                       values=["320x240", "640x480", "800x600", "1024x768"])
        resolution_combo.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        # GPU acceleration
        self.gpu_var = tk.BooleanVar()
        gpu_check = ttk.Checkbutton(frame, text="Enable GPU Acceleration", 
                                   variable=self.gpu_var)
        gpu_check.grid(row=3, column=0, columnspan=2, sticky="w", padx=5, pady=5)
    
    def create_privacy_tab(self):
        """Create privacy settings tab"""
        
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Privacy")
        
        # Privacy mode
        self.privacy_mode_var = tk.BooleanVar()
        privacy_check = ttk.Checkbutton(frame, text="Enable Privacy Mode", 
                                       variable=self.privacy_mode_var)
        privacy_check.grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        # Auto pause
        self.auto_pause_var = tk.BooleanVar()
        pause_check = ttk.Checkbutton(frame, text="Auto-pause when looking away", 
                                     variable=self.auto_pause_var)
        pause_check.grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        # Data retention
        ttk.Label(frame, text="Data Retention (days):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.retention_var = tk.IntVar()
        retention_spinbox = ttk.Spinbox(frame, from_=0, to=90, textvariable=self.retention_var, width=10)
        retention_spinbox.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        # Anonymize logs
        self.anonymize_var = tk.BooleanVar()
        anonymize_check = ttk.Checkbutton(frame, text="Anonymize logs", 
                                         variable=self.anonymize_var)
        anonymize_check.grid(row=3, column=0, columnspan=2, sticky="w", padx=5, pady=5)
    
    def create_ui_tab(self):
        """Create UI settings tab"""
        
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Interface")
        
        # Show gaze vector
        self.show_gaze_var = tk.BooleanVar()
        gaze_check = ttk.Checkbutton(frame, text="Show gaze direction vector", 
                                    variable=self.show_gaze_var)
        gaze_check.grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        # Show eye regions
        self.show_eyes_var = tk.BooleanVar()
        eyes_check = ttk.Checkbutton(frame, text="Show eye regions", 
                                    variable=self.show_eyes_var)
        eyes_check.grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        # Show confidence meter
        self.show_confidence_var = tk.BooleanVar()
        confidence_check = ttk.Checkbutton(frame, text="Show confidence meter", 
                                          variable=self.show_confidence_var)
        confidence_check.grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        # Annotation color
        ttk.Label(frame, text="Annotation Color:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.color_button = tk.Button(frame, text="Select Color", 
                                     command=self.choose_color, width=15)
        self.color_button.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        
        self.annotation_color = (0, 255, 0)  # Default green
    
    def create_calibration_tab(self):
        """Create calibration settings tab"""
        
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Calibration")
        
        # Number of calibration points
        ttk.Label(frame, text="Calibration Points:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.cal_points_var = tk.IntVar()
        points_combo = ttk.Combobox(frame, textvariable=self.cal_points_var, 
                                   values=[9, 16, 25], width=10)
        points_combo.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Calibration timeout
        ttk.Label(frame, text="Timeout (seconds):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.cal_timeout_var = tk.IntVar()
        timeout_spinbox = ttk.Spinbox(frame, from_=10, to=60, textvariable=self.cal_timeout_var, width=10)
        timeout_spinbox.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # Auto-recalibration
        ttk.Label(frame, text="Auto-recalibrate (days):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.auto_recal_var = tk.IntVar()
        recal_spinbox = ttk.Spinbox(frame, from_=1, to=30, textvariable=self.auto_recal_var, width=10)
        recal_spinbox.grid(row=2, column=1, sticky="w", padx=5, pady=5)
    
    def create_notifications_tab(self):
        """Create notifications settings tab"""
        
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Notifications")
        
        # Enable notifications
        self.notifications_var = tk.BooleanVar()
        notifications_check = ttk.Checkbutton(frame, text="Enable notifications", 
                                             variable=self.notifications_var)
        notifications_check.grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        # Sound alerts
        self.sound_var = tk.BooleanVar()
        sound_check = ttk.Checkbutton(frame, text="Sound alerts", 
                                     variable=self.sound_var)
        sound_check.grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        # Visual alerts
        self.visual_var = tk.BooleanVar()
        visual_check = ttk.Checkbutton(frame, text="Visual alerts", 
                                      variable=self.visual_var)
        visual_check.grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=5)
    
    def create_advanced_tab(self):
        """Create advanced settings tab"""
        
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Advanced")
        
        # Detector type
        ttk.Label(frame, text="Detector Type:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.detector_var = tk.StringVar()
        detector_combo = ttk.Combobox(frame, textvariable=self.detector_var, 
                                     values=["mediapipe", "opencv", "adaptive"])
        detector_combo.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Use iris tracking
        self.iris_var = tk.BooleanVar()
        iris_check = ttk.Checkbutton(frame, text="Use iris tracking", 
                                    variable=self.iris_var)
        iris_check.grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        # Enable face mesh
        self.face_mesh_var = tk.BooleanVar()
        mesh_check = ttk.Checkbutton(frame, text="Enable face mesh", 
                                    variable=self.face_mesh_var)
        mesh_check.grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        # Multi-face detection
        self.multiface_var = tk.BooleanVar()
        multiface_check = ttk.Checkbutton(frame, text="Multi-face detection", 
                                         variable=self.multiface_var)
        multiface_check.grid(row=3, column=0, columnspan=2, sticky="w", padx=5, pady=5)
    
    def create_buttons(self):
        """Create dialog buttons"""
        
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        # OK button
        ok_button = ttk.Button(button_frame, text="OK", command=self.apply_settings)
        ok_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Cancel button
        cancel_button = ttk.Button(button_frame, text="Cancel", command=self.cancel)
        cancel_button.pack(side=tk.RIGHT)
        
        # Apply button
        apply_button = ttk.Button(button_frame, text="Apply", command=self.apply_settings_no_close)
        apply_button.pack(side=tk.RIGHT, padx=(0, 5))
        
        # Reset to defaults button
        reset_button = ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_defaults)
        reset_button.pack(side=tk.LEFT)
    
    def load_settings_to_ui(self):
        """Load current settings into UI controls"""
        
        # Detection tab
        self.attention_threshold_var.set(self.settings.attention_threshold)
        self.detection_confidence_var.set(self.settings.min_detection_confidence)
        self.tracking_confidence_var.set(self.settings.min_tracking_confidence)
        self.smoothing_var.set(self.settings.smoothing_factor)
        
        # Performance tab
        self.fps_var.set(self.settings.target_fps)
        self.frame_skip_var.set(self.settings.frame_skip)
        resolution_str = f"{self.settings.processing_resolution[0]}x{self.settings.processing_resolution[1]}"
        self.resolution_var.set(resolution_str)
        self.gpu_var.set(self.settings.enable_gpu_acceleration)
        
        # Privacy tab
        self.privacy_mode_var.set(self.settings.privacy_mode)
        self.auto_pause_var.set(self.settings.auto_pause_on_away)
        self.retention_var.set(self.settings.data_retention_days)
        self.anonymize_var.set(self.settings.anonymize_logs)
        
        # UI tab
        self.show_gaze_var.set(self.settings.show_gaze_vector)
        self.show_eyes_var.set(self.settings.show_eye_regions)
        self.show_confidence_var.set(self.settings.show_confidence_meter)
        self.annotation_color = self.settings.annotation_color
        self.update_color_button()
        
        # Calibration tab
        self.cal_points_var.set(self.settings.calibration_points)
        self.cal_timeout_var.set(self.settings.calibration_timeout)
        self.auto_recal_var.set(self.settings.auto_recalibrate_days)
        
        # Notifications tab
        self.notifications_var.set(self.settings.enable_notifications)
        self.sound_var.set(self.settings.sound_alerts)
        self.visual_var.set(self.settings.visual_alerts)
        
        # Advanced tab
        self.detector_var.set(self.settings.detector_type)
        self.iris_var.set(self.settings.use_iris_tracking)
        self.face_mesh_var.set(self.settings.enable_face_mesh)
        self.multiface_var.set(self.settings.multiface_detection)
    
    def get_settings_from_ui(self) -> GazeSettings:
        """Extract settings from UI controls"""
        
        # Parse resolution string
        resolution_str = self.resolution_var.get()
        width, height = map(int, resolution_str.split('x'))
        
        return GazeSettings(
            # Detection
            attention_threshold=self.attention_threshold_var.get(),
            min_detection_confidence=self.detection_confidence_var.get(),
            min_tracking_confidence=self.tracking_confidence_var.get(),
            smoothing_factor=self.smoothing_var.get(),
            
            # Performance
            target_fps=self.fps_var.get(),
            frame_skip=self.frame_skip_var.get(),
            processing_resolution=(width, height),
            enable_gpu_acceleration=self.gpu_var.get(),
            
            # Privacy
            privacy_mode=self.privacy_mode_var.get(),
            auto_pause_on_away=self.auto_pause_var.get(),
            data_retention_days=self.retention_var.get(),
            anonymize_logs=self.anonymize_var.get(),
            
            # UI
            show_gaze_vector=self.show_gaze_var.get(),
            show_eye_regions=self.show_eyes_var.get(),
            show_confidence_meter=self.show_confidence_var.get(),
            annotation_color=self.annotation_color,
            
            # Calibration
            calibration_points=self.cal_points_var.get(),
            calibration_timeout=self.cal_timeout_var.get(),
            auto_recalibrate_days=self.auto_recal_var.get(),
            
            # Notifications
            enable_notifications=self.notifications_var.get(),
            sound_alerts=self.sound_var.get(),
            visual_alerts=self.visual_var.get(),
            
            # Advanced
            detector_type=self.detector_var.get(),
            use_iris_tracking=self.iris_var.get(),
            enable_face_mesh=self.face_mesh_var.get(),
            multiface_detection=self.multiface_var.get()
        )
    
    def choose_color(self):
        """Open color chooser dialog"""
        
        color = colorchooser.askcolor(title="Choose annotation color")
        if color[0]:  # User didn't cancel
            # Convert RGB to BGR for OpenCV
            r, g, b = map(int, color[0])
            self.annotation_color = (b, g, r)
            self.update_color_button()
    
    def update_color_button(self):
        """Update color button appearance"""
        
        # Convert BGR to hex for tkinter
        b, g, r = self.annotation_color
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        self.color_button.config(bg=hex_color)
    
    def validate_settings(self, settings: GazeSettings) -> bool:
        """Validate settings values"""
        
        if not (0.1 <= settings.attention_threshold <= 0.8):
            messagebox.showerror("Error", "Attention threshold must be between 0.1 and 0.8")
            return False
        
        if not (0.3 <= settings.min_detection_confidence <= 0.9):
            messagebox.showerror("Error", "Detection confidence must be between 0.3 and 0.9")
            return False
        
        if not (10 <= settings.target_fps <= 60):
            messagebox.showerror("Error", "Target FPS must be between 10 and 60")
            return False
        
        return True
    
    def apply_settings(self):
        """Apply settings and close dialog"""
        
        new_settings = self.get_settings_from_ui()
        
        if self.validate_settings(new_settings):
            self.settings = new_settings
            self.on_settings_changed(self.settings)
            self.dialog.destroy()
    
    def apply_settings_no_close(self):
        """Apply settings without closing dialog"""
        
        new_settings = self.get_settings_from_ui()
        
        if self.validate_settings(new_settings):
            self.settings = new_settings
            self.on_settings_changed(self.settings)
    
    def cancel(self):
        """Cancel dialog without applying changes"""
        
        self.dialog.destroy()
    
    def reset_defaults(self):
        """Reset all settings to defaults"""
        
        result = messagebox.askyesno("Confirm", "Reset all settings to defaults?")
        if result:
            default_settings = GazeSettings()
            self.settings = default_settings
            self.load_settings_to_ui()

class SettingsManager:
    """
    Settings manager for loading, saving, and managing application settings.
    """
    
    def __init__(self, settings_file: str = "settings.json"):
        """
        Initialize settings manager.
        
        Args:
            settings_file: Path to settings file
        """
        self.settings_file = settings_file
        self.current_settings = GazeSettings()
        
        # Load settings if file exists
        self.load_settings()
    
    def load_settings(self) -> bool:
        """
        Load settings from file.
        
        Returns:
            True if loaded successfully
        """
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    data = json.load(f)
                
                # Create settings object from loaded data
                self.current_settings = GazeSettings(**data)
                return True
            
        except Exception as e:
            print(f"Error loading settings: {e}")
        
        return False
    
    def save_settings(self) -> bool:
        """
        Save current settings to file.
        
        Returns:
            True if saved successfully
        """
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(asdict(self.current_settings), f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False
    
    def get_settings(self) -> GazeSettings:
        """Get current settings"""
        return self.current_settings
    
    def update_settings(self, new_settings: GazeSettings):
        """Update and save settings"""
        self.current_settings = new_settings
        self.save_settings()
    
    def reset_to_defaults(self):
        """Reset settings to defaults"""
        self.current_settings = GazeSettings()
        self.save_settings()
    
    def export_settings(self, filepath: str) -> bool:
        """Export settings to specified file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(asdict(self.current_settings), f, indent=2)
            return True
        except Exception as e:
            print(f"Error exporting settings: {e}")
            return False
    
    def import_settings(self, filepath: str) -> bool:
        """Import settings from specified file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.current_settings = GazeSettings(**data)
            self.save_settings()
            return True
            
        except Exception as e:
            print(f"Error importing settings: {e}")
            return False