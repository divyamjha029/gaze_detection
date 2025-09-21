"""
Performance monitoring module for tracking system performance and optimization.
Provides real-time performance metrics, profiling, and resource usage monitoring.
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import json
import numpy as np

@dataclass
class PerformanceMetrics:
    """Data class for performance metrics"""
    fps: float = 0.0
    avg_processing_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    detection_accuracy: float = 0.0
    frame_drops: int = 0
    timestamp: float = field(default_factory=time.time)

@dataclass
class SystemSpecs:
    """System specifications for performance analysis"""
    cpu_count: int
    cpu_freq: float
    total_memory: float
    gpu_available: bool
    gpu_memory: float
    opencv_version: str
    mediapipe_version: str

class PerformanceProfiler:
    """
    Performance profiler for measuring execution times of different components.
    """
    
    def __init__(self):
        """Initialize performance profiler"""
        self.profiles = {}
        self.active_profiles = {}
    
    def start_profile(self, name: str):
        """Start profiling a component"""
        self.active_profiles[name] = time.perf_counter()
    
    def end_profile(self, name: str) -> float:
        """
        End profiling and return execution time.
        
        Args:
            name: Profile name
            
        Returns:
            Execution time in seconds
        """
        if name not in self.active_profiles:
            return 0.0
        
        start_time = self.active_profiles.pop(name)
        execution_time = time.perf_counter() - start_time
        
        # Store in profile history
        if name not in self.profiles:
            self.profiles[name] = deque(maxlen=1000)  # Keep last 1000 measurements
        
        self.profiles[name].append(execution_time)
        
        return execution_time
    
    def get_profile_stats(self, name: str) -> Dict[str, float]:
        """Get statistical analysis of profile data"""
        if name not in self.profiles or not self.profiles[name]:
            return {}
        
        times = list(self.profiles[name])
        
        return {
            'mean': np.mean(times),
            'median': np.median(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99),
            'count': len(times)
        }
    
    def get_all_profiles(self) -> Dict[str, Dict[str, float]]:
        """Get stats for all profiles"""
        return {name: self.get_profile_stats(name) for name in self.profiles.keys()}

class ResourceMonitor:
    """
    Monitor system resources including CPU, memory, and GPU usage.
    """
    
    def __init__(self, update_interval: float = 1.0):
        """
        Initialize resource monitor.
        
        Args:
            update_interval: Update interval in seconds
        """
        self.update_interval = update_interval
        self.running = False
        self.monitor_thread = None
        
        # Resource usage history
        self.cpu_history = deque(maxlen=60)  # Last 60 measurements
        self.memory_history = deque(maxlen=60)
        self.gpu_history = deque(maxlen=60)
        
        # Current values
        self.current_cpu = 0.0
        self.current_memory = 0.0
        self.current_gpu = 0.0
        
        # Callbacks for alerts
        self.alert_callbacks = []
        
        # Alert thresholds
        self.cpu_threshold = 80.0  # %
        self.memory_threshold = 80.0  # %
        self.gpu_threshold = 80.0  # %
    
    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Get CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                self.current_cpu = cpu_percent
                self.cpu_history.append(cpu_percent)
                
                # Get memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                self.current_memory = memory_percent
                self.memory_history.append(memory_percent)
                
                # Get GPU usage (if available)
                gpu_percent = self._get_gpu_usage()
                self.current_gpu = gpu_percent
                self.gpu_history.append(gpu_percent)
                
                # Check for alerts
                self._check_alerts()
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Error in resource monitoring: {e}")
                time.sleep(self.update_interval)
    
    def _get_gpu_usage(self) -> float:
        """Get GPU usage percentage"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(utilization.gpu)
        except:
            return 0.0  # GPU monitoring not available
    
    def _check_alerts(self):
        """Check if resource usage exceeds thresholds"""
        alerts = []
        
        if self.current_cpu > self.cpu_threshold:
            alerts.append(f"High CPU usage: {self.current_cpu:.1f}%")
        
        if self.current_memory > self.memory_threshold:
            alerts.append(f"High memory usage: {self.current_memory:.1f}%")
        
        if self.current_gpu > self.gpu_threshold:
            alerts.append(f"High GPU usage: {self.current_gpu:.1f}%")
        
        # Trigger callbacks for alerts
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except:
                    pass  # Don't let callback errors stop monitoring
    
    def add_alert_callback(self, callback: Callable[[str], None]):
        """Add callback for resource alerts"""
        self.alert_callbacks.append(callback)
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        return {
            'cpu': self.current_cpu,
            'memory': self.current_memory,
            'gpu': self.current_gpu
        }
    
    def get_usage_history(self) -> Dict[str, List[float]]:
        """Get resource usage history"""
        return {
            'cpu': list(self.cpu_history),
            'memory': list(self.memory_history),
            'gpu': list(self.gpu_history)
        }
    
    def get_usage_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistical analysis of resource usage"""
        stats = {}
        
        for name, history in [('cpu', self.cpu_history), 
                             ('memory', self.memory_history), 
                             ('gpu', self.gpu_history)]:
            if history:
                values = list(history)
                stats[name] = {
                    'current': values[-1],
                    'mean': np.mean(values),
                    'max': np.max(values),
                    'min': np.min(values),
                    'std': np.std(values)
                }
            else:
                stats[name] = {'current': 0.0, 'mean': 0.0, 'max': 0.0, 'min': 0.0, 'std': 0.0}
        
        return stats

class PerformanceOptimizer:
    """
    Automatic performance optimizer that adjusts settings based on system performance.
    """
    
    def __init__(self, resource_monitor: ResourceMonitor):
        """
        Initialize performance optimizer.
        
        Args:
            resource_monitor: Resource monitor instance
        """
        self.resource_monitor = resource_monitor
        self.optimization_callbacks = []
        
        # Optimization parameters
        self.target_fps = 30
        self.min_fps = 15
        self.cpu_limit = 70.0
        self.memory_limit = 70.0
        
        # Current optimization level (0 = no optimization, 3 = maximum)
        self.optimization_level = 0
        
        # Optimization history
        self.optimization_history = []
    
    def add_optimization_callback(self, callback: Callable[[Dict[str, any]], None]):
        """Add callback for optimization suggestions"""
        self.optimization_callbacks.append(callback)
    
    def analyze_performance(self, current_fps: float, processing_time: float) -> Dict[str, any]:
        """
        Analyze current performance and suggest optimizations.
        
        Args:
            current_fps: Current frames per second
            processing_time: Current processing time per frame
            
        Returns:
            Optimization suggestions
        """
        suggestions = {
            'level': self.optimization_level,
            'actions': [],
            'reasoning': [],
            'estimated_improvement': 0.0
        }
        
        # Get current resource usage
        usage = self.resource_monitor.get_current_usage()
        
        # Analyze FPS performance
        if current_fps < self.min_fps:
            suggestions['actions'].append('reduce_resolution')
            suggestions['reasoning'].append(f'FPS too low: {current_fps:.1f} < {self.min_fps}')
            suggestions['estimated_improvement'] += 5.0
        
        # Analyze CPU usage
        if usage['cpu'] > self.cpu_limit:
            suggestions['actions'].append('increase_frame_skip')
            suggestions['reasoning'].append(f'High CPU usage: {usage["cpu"]:.1f}%')
            suggestions['estimated_improvement'] += 3.0
        
        # Analyze memory usage
        if usage['memory'] > self.memory_limit:
            suggestions['actions'].append('reduce_buffer_size')
            suggestions['reasoning'].append(f'High memory usage: {usage["memory"]:.1f}%')
            suggestions['estimated_improvement'] += 2.0
        
        # Analyze processing time
        target_processing_time = 1.0 / self.target_fps
        if processing_time > target_processing_time * 1.5:
            suggestions['actions'].append('optimize_algorithms')
            suggestions['reasoning'].append(f'Slow processing: {processing_time*1000:.1f}ms')
            suggestions['estimated_improvement'] += 4.0
        
        # Determine optimization level
        if suggestions['actions']:
            self.optimization_level = min(3, self.optimization_level + 1)
        else:
            self.optimization_level = max(0, self.optimization_level - 1)
        
        suggestions['level'] = self.optimization_level
        
        # Store optimization history
        self.optimization_history.append({
            'timestamp': time.time(),
            'fps': current_fps,
            'cpu_usage': usage['cpu'],
            'memory_usage': usage['memory'],
            'optimization_level': self.optimization_level,
            'suggestions': suggestions['actions']
        })
        
        # Trigger callbacks
        for callback in self.optimization_callbacks:
            try:
                callback(suggestions)
            except:
                pass
        
        return suggestions
    
    def get_optimization_config(self) -> Dict[str, any]:
        """Get current optimization configuration"""
        
        base_config = {
            'frame_skip': 1,
            'resolution_scale': 1.0,
            'buffer_size': 10,
            'use_gpu': True,
            'quality_preset': 'high'
        }
        
        # Apply optimizations based on level
        if self.optimization_level >= 1:
            base_config['frame_skip'] = 2
            base_config['quality_preset'] = 'medium'
        
        if self.optimization_level >= 2:
            base_config['resolution_scale'] = 0.75
            base_config['buffer_size'] = 5
        
        if self.optimization_level >= 3:
            base_config['frame_skip'] = 3
            base_config['resolution_scale'] = 0.5
            base_config['buffer_size'] = 3
            base_config['quality_preset'] = 'low'
        
        return base_config

class PerformanceMonitor:
    """
    Main performance monitoring system coordinating all performance components.
    """
    
    def __init__(self):
        """Initialize performance monitor"""
        
        # Initialize components
        self.profiler = PerformanceProfiler()
        self.resource_monitor = ResourceMonitor()
        self.optimizer = PerformanceOptimizer(self.resource_monitor)
        
        # Performance metrics history
        self.metrics_history = deque(maxlen=300)  # 5 minutes at 1 FPS
        
        # Frame timing
        self.frame_times = deque(maxlen=100)
        self.last_frame_time = time.time()
        
        # Detection accuracy tracking
        self.accuracy_samples = deque(maxlen=100)
        
        # System specifications
        self.system_specs = self._get_system_specs()
        
        # Start monitoring
        self.resource_monitor.start_monitoring()
    
    def _get_system_specs(self) -> SystemSpecs:
        """Get system specifications"""
        
        try:
            import cv2
            opencv_version = cv2.__version__
        except:
            opencv_version = "unknown"
        
        try:
            import mediapipe
            mediapipe_version = mediapipe.__version__
        except:
            mediapipe_version = "unknown"
        
        # Check GPU availability
        gpu_available = False
        gpu_memory = 0.0
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_available = True
            gpu_memory = memory_info.total / (1024**3)  # Convert to GB
        except:
            pass
        
        return SystemSpecs(
            cpu_count=psutil.cpu_count(),
            cpu_freq=psutil.cpu_freq().max if psutil.cpu_freq() else 0.0,
            total_memory=psutil.virtual_memory().total / (1024**3),  # GB
            gpu_available=gpu_available,
            gpu_memory=gpu_memory,
            opencv_version=opencv_version,
            mediapipe_version=mediapipe_version
        )
    
    def update_frame_time(self, processing_time: float):
        """Update frame processing time"""
        
        current_time = time.time()
        frame_interval = current_time - self.last_frame_time
        self.frame_times.append(frame_interval)
        self.last_frame_time = current_time
        
        # Store processing time in profiler
        self.profiler.profiles.setdefault('frame_processing', deque(maxlen=1000))
        self.profiler.profiles['frame_processing'].append(processing_time)
    
    def update_accuracy(self, accuracy: float):
        """Update detection accuracy"""
        self.accuracy_samples.append(accuracy)
    
    def get_current_fps(self) -> float:
        """Calculate current FPS"""
        if len(self.frame_times) < 2:
            return 0.0
        
        recent_times = list(self.frame_times)[-10:]  # Last 10 frames
        if not recent_times:
            return 0.0
        
        avg_interval = np.mean(recent_times)
        return 1.0 / avg_interval if avg_interval > 0 else 0.0
    
    def get_stats(self) -> PerformanceMetrics:
        """Get current performance statistics"""
        
        # Calculate FPS
        current_fps = self.get_current_fps()
        
        # Get average processing time
        processing_times = self.profiler.profiles.get('frame_processing', [])
        avg_processing_time = np.mean(processing_times) if processing_times else 0.0
        
        # Get resource usage
        usage = self.resource_monitor.get_current_usage()
        
        # Get detection accuracy
        accuracy = np.mean(self.accuracy_samples) if self.accuracy_samples else 0.0
        
        # Count frame drops (FPS significantly below target)
        target_fps = 30
        frame_drops = max(0, int((target_fps - current_fps) * len(self.frame_times) / 100))
        
        metrics = PerformanceMetrics(
            fps=current_fps,
            avg_processing_time=avg_processing_time,
            cpu_usage=usage['cpu'],
            memory_usage=usage['memory'],
            gpu_usage=usage['gpu'],
            detection_accuracy=accuracy,
            frame_drops=frame_drops
        )
        
        # Store in history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def get_performance_report(self) -> Dict[str, any]:
        """Generate comprehensive performance report"""
        
        current_stats = self.get_stats()
        
        report = {
            'timestamp': time.time(),
            'current_metrics': {
                'fps': current_stats.fps,
                'processing_time_ms': current_stats.avg_processing_time * 1000,
                'cpu_usage': current_stats.cpu_usage,
                'memory_usage': current_stats.memory_usage,
                'gpu_usage': current_stats.gpu_usage,
                'accuracy': current_stats.detection_accuracy
            },
            'system_specs': {
                'cpu_count': self.system_specs.cpu_count,
                'cpu_freq_ghz': self.system_specs.cpu_freq / 1000,
                'total_memory_gb': self.system_specs.total_memory,
                'gpu_available': self.system_specs.gpu_available,
                'gpu_memory_gb': self.system_specs.gpu_memory,
                'opencv_version': self.system_specs.opencv_version,
                'mediapipe_version': self.system_specs.mediapipe_version
            },
            'performance_history': self._get_history_stats(),
            'profiling_data': self.profiler.get_all_profiles(),
            'resource_stats': self.resource_monitor.get_usage_stats(),
            'optimization_suggestions': self.optimizer.analyze_performance(
                current_stats.fps, current_stats.avg_processing_time
            )
        }
        
        return report
    
    def _get_history_stats(self) -> Dict[str, any]:
        """Get statistical analysis of performance history"""
        
        if not self.metrics_history:
            return {}
        
        fps_values = [m.fps for m in self.metrics_history]
        cpu_values = [m.cpu_usage for m in self.metrics_history]
        memory_values = [m.memory_usage for m in self.metrics_history]
        
        return {
            'fps': {
                'mean': np.mean(fps_values),
                'min': np.min(fps_values),
                'max': np.max(fps_values),
                'std': np.std(fps_values)
            },
            'cpu': {
                'mean': np.mean(cpu_values),
                'min': np.min(cpu_values),
                'max': np.max(cpu_values),
                'std': np.std(cpu_values)
            },
            'memory': {
                'mean': np.mean(memory_values),
                'min': np.min(memory_values),
                'max': np.max(memory_values),
                'std': np.std(memory_values)
            }
        }
    
    def save_performance_log(self, filepath: str) -> bool:
        """Save performance data to file"""
        
        try:
            report = self.get_performance_report()
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            print(f"Error saving performance log: {e}")
            return False
    
    def cleanup(self):
        """Cleanup performance monitor"""
        self.resource_monitor.stop_monitoring()