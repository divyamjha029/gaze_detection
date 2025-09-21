#!/usr/bin/env python3
"""
Gaze Detection System Setup Script
Handles installation, configuration, and deployment of the gaze detection software.
"""

from setuptools import setup, find_packages
import os
import sys
from pathlib import Path

# Read version from file
def get_version():
    version_file = Path(__file__).parent / "src" / "__version__.py"
    if version_file.exists():
        with open(version_file, 'r') as f:
            exec(f.read())
            return locals()['__version__']
    return "1.0.0"

# Read long description from README
def get_long_description():
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return "Advanced gaze detection system using computer vision and machine learning"

# Read requirements from requirements.txt
def get_requirements():
    requirements_file = Path(__file__).parent / "requirements.txt"
    requirements = []
    
    if requirements_file.exists():
        with open(requirements_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    # Remove inline comments
                    if '#' in line:
                        line = line.split('#')[0].strip()
                    if line:
                        requirements.append(line)
    
    return requirements

# Platform-specific requirements
def get_platform_requirements():
    platform_requirements = []
    
    # Windows-specific
    if sys.platform.startswith('win'):
        platform_requirements.extend([
            'pywin32>=227',  # For Windows-specific features
        ])
    
    # macOS-specific
    elif sys.platform.startswith('darwin'):
        platform_requirements.extend([
            'pyobjc-framework-AVFoundation>=8.0',  # For camera access
        ])
    
    # Linux-specific
    elif sys.platform.startswith('linux'):
        platform_requirements.extend([
            'python3-dev',  # Development headers
        ])
    
    return platform_requirements

# Optional dependencies for different features
extras_require = {
    'gpu': [
        'tensorflow-gpu>=2.10.0',
        'torch>=1.12.0+cu116',
        'pynvml>=11.0.0',
    ],
    'mobile': [
        'kivy>=2.1.0',
        'buildozer>=1.4.0',
        'python-for-android>=2022.09.04',
    ],
    'advanced': [
        'dlib>=19.24.0',
        'scikit-learn>=1.1.0',
        'xgboost>=1.6.0',
    ],
    'dev': [
        'pytest>=7.0.0',
        'pytest-cov>=4.0.0',
        'black>=22.0.0',
        'flake8>=5.0.0',
        'mypy>=0.991',
        'pre-commit>=2.20.0',
    ],
    'docs': [
        'sphinx>=5.0.0',
        'sphinx-rtd-theme>=1.0.0',
        'myst-parser>=0.18.0',
    ]
}

# Combine all optional dependencies
extras_require['all'] = list(set(sum(extras_require.values(), [])))

setup(
    name="gaze-detection-system",
    version=get_version(),
    author="Gaze Detection Team",
    author_email="contact@gazedetection.com",
    description="Advanced real-time gaze detection system",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gaze-detection-system",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/gaze-detection-system/issues",
        "Source": "https://github.com/yourusername/gaze-detection-system",
        "Documentation": "https://gaze-detection-system.readthedocs.io/",
    },
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "gaze_detection": [
            "data/*.json",
            "data/*.dat",
            "models/*.tflite",
            "models/*.onnx",
            "ui/assets/*",
        ]
    },
    include_package_data=True,
    
    # Requirements
    python_requires=">=3.7",
    install_requires=get_requirements() + get_platform_requirements(),
    extras_require=extras_require,
    
    # Entry points
    entry_points={
        "console_scripts": [
            "gaze-detection=src.ui.main_app:main",
            "gaze-calibrate=src.core.calibration:main",
            "gaze-test=src.utils.performance:main",
        ],
        "gui_scripts": [
            "gaze-detection-gui=src.ui.main_app:main",
        ]
    },
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Multimedia :: Video :: Capture",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: Desktop Environment",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    
    # Keywords
    keywords=[
        "gaze detection", "eye tracking", "computer vision", "machine learning",
        "mediapipe", "opencv", "facial landmarks", "human-computer interaction",
        "accessibility", "privacy", "real-time", "calibration"
    ],
    
    # Additional metadata
    zip_safe=False,
    platforms=["any"],
    
    # Test configuration
    test_suite="tests",
    tests_require=extras_require['dev'],
    
    # Options
    options={
        "build_exe": {
            "packages": ["opencv", "mediapipe", "numpy", "tkinter"],
            "include_files": [
                ("models/", "models/"),
                ("src/ui/assets/", "assets/"),
                ("README.md", "README.md"),
                ("LICENSE", "LICENSE"),
            ],
            "excludes": ["test", "tests"],
        },
        "bdist_mac": {
            "bundle_name": "Gaze Detection System",
            "iconfile": "src/ui/assets/icon.icns",
        },
        "bdist_dmg": {
            "applications_shortcut": True,
            "volume_label": "Gaze Detection System",
        }
    }
)

# Post-installation setup
def post_install():
    """Run post-installation setup tasks"""
    
    print("Setting up Gaze Detection System...")
    
    # Create necessary directories
    directories = [
        "data/calibration",
        "data/logs",
        "data/temp",
        "models",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Download required model files if not present
    model_files = [
        {
            "name": "shape_predictor_68_face_landmarks.dat",
            "url": "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
            "path": "models/",
            "description": "68-point facial landmark predictor for dlib"
        }
    ]
    
    print("\nChecking model files...")
    for model in model_files:
        model_path = Path(model["path"]) / model["name"]
        if not model_path.exists():
            print(f"Model file missing: {model['name']}")
            print(f"Description: {model['description']}")
            print(f"Download from: {model['url']}")
            print("Please download manually and extract to the models/ directory")
        else:
            print(f"✓ Model file found: {model['name']}")
    
    # Create default configuration
    config_file = Path("config/default_settings.json")
    if not config_file.exists():
        default_config = {
            "detection": {
                "attention_threshold": 0.3,
                "min_detection_confidence": 0.5,
                "min_tracking_confidence": 0.5,
                "smoothing_factor": 0.7
            },
            "performance": {
                "target_fps": 30,
                "frame_skip": 1,
                "processing_resolution": [640, 480],
                "enable_gpu_acceleration": True
            },
            "privacy": {
                "privacy_mode": True,
                "anonymize_data": True,
                "auto_delete_data": True,
                "data_retention_hours": 24
            },
            "ui": {
                "show_gaze_vector": True,
                "show_eye_regions": True,
                "show_confidence_meter": True,
                "annotation_color": [0, 255, 0]
            }
        }
        
        import json
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        print(f"✓ Created default configuration: {config_file}")
    
    print("\n✅ Setup complete! You can now run the gaze detection system.")
    print("\nQuick start:")
    print("  gaze-detection              # Start GUI application")
    print("  gaze-calibrate              # Run calibration only")
    print("  gaze-test                   # Run performance test")
    print("\nFor help, run: gaze-detection --help")

if __name__ == "__main__":
    # Check if this is being run as post-install
    if len(sys.argv) > 1 and sys.argv[1] == "post_install":
        post_install()
    else:
        # Run normal setup
        setup(
            # All setup parameters defined above
        )