import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
from datetime import datetime, timedelta
import os
from pathlib import Path
import threading
import queue
import logging
from pathlib import Path as _Path

# Configure logging
# Ensure logs directory exists
_Path('logs').mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/classroom_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('VideoAnalyzer')

# Configuration management
def load_config(config_path='config.json'):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file {config_path} not found, using defaults")
        return get_default_config()
    except json.JSONDecodeError:
        print(f"Invalid config file {config_path}, using defaults")
        return get_default_config()

def get_default_config():
    """Return default configuration"""
    return {
        "model": {
            "path": "models/classroom_behavior_model.h5",
            "class_indices_path": "models/class_indices.json",
            "image_size": [224, 224],
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 50
        },
        "video_analysis": {
            "default_frame_skip": 30,
            "anomaly_threshold": 0.7,
            "max_anomaly_snapshots": 50,
            "snapshot_dir": "outputs/anomaly_snapshots"
        },
        "real_time": {
            "frame_queue_size": 50,
            "result_queue_size": 20,
            "display_fps": 30
        },
        "ui": {
            "theme": "dark",
            "max_displayed_anomalies": 9,
            "chart_height": 400
        }
    }

class VideoAnalyzer:
    def __init__(self, model_path=None, class_indices_path=None, config_path='config.json'):
        self.config = load_config(config_path)

        # Use config values or provided paths
        self.model_path = model_path or self.config['model']['path']
        self.class_indices_path = class_indices_path or self.config['model']['class_indices_path']

        self.model = None
        self.class_indices = {}
        self.classes = []
        self.img_size = tuple(self.config['model']['image_size'])
        self.anomaly_threshold = self.config['video_analysis']['anomaly_threshold']
        self.normal_class = 'Normal'
        self.snapshot_dir = self.config['video_analysis']['snapshot_dir']
        self.max_snapshots = self.config['video_analysis']['max_anomaly_snapshots']

        # Load model and classes
        self.load_model(self.model_path, self.class_indices_path)
        
    def load_model(self, model_path, class_indices_path):
        """Load the trained model and class indices"""
        try:
            # Resolve model path with fallbacks
            model_candidates = [model_path,
                                'models/classroom_behavior_model.h5',
                                'classroom_behavior_model.h5',
                                'best_classroom_model.h5']
            model_actual = next((p for p in model_candidates if os.path.exists(p)), None)
            if model_actual is None:
                raise FileNotFoundError(f"Model file not found in candidates: {model_candidates}")
            logger.info(f"Loading model from {model_actual}")
            self.model = keras.models.load_model(model_actual)
            logger.info(f"Model loaded successfully from {model_path}")

            # Resolve class indices with fallbacks
            indices_candidates = [class_indices_path,
                                   'models/class_indices.json',
                                   'class_indices.json']
            indices_actual = next((p for p in indices_candidates if os.path.exists(p)), None)
            if indices_actual is None:
                raise FileNotFoundError(f"Class indices file not found in candidates: {indices_candidates}")
            logger.info(f"Loading class indices from {indices_actual}")
            with open(indices_actual, 'r') as f:
                self.class_indices = json.load(f)

            # Create reverse mapping (index to class name)
            self.classes = list(self.class_indices.keys())
            self.idx_to_class = {v: k for k, v in self.class_indices.items()}

            logger.info(f"Classes loaded: {self.classes}")

        except FileNotFoundError as e:
            logger.error(f"Model or class indices file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def preprocess_frame(self, frame):
        """Preprocess frame for model prediction"""
        # Resize frame
        frame_resized = cv2.resize(frame, self.img_size)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        frame_batch = np.expand_dims(frame_normalized, axis=0)
        
        return frame_batch
    
    def predict_behavior(self, frame):
        """Predict behavior from a single frame"""
        if self.model is None:
            return None, 0.0
        
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Make prediction
        predictions = self.model.predict(processed_frame, verbose=0)
        
        # Get predicted class and confidence
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        predicted_class = self.idx_to_class[predicted_idx]
        
        return predicted_class, confidence
    
    def is_anomaly(self, predicted_class, confidence):
        """Determine if the prediction indicates an anomaly"""
        if predicted_class == self.normal_class:
            return False
        
        return confidence >= self.anomaly_threshold
    
    def analyze_video(self, video_path, frame_skip=None, progress_callback=None):
        """Analyze video and detect anomalies"""
        frame_skip = frame_skip or self.config['video_analysis']['default_frame_skip']

        logger.info(f"Starting video analysis: {video_path} with frame_skip={frame_skip}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Error opening video: {video_path}")
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        logger.info(f"Video info: {total_frames} frames, {fps} FPS, {duration:.2f}s duration")
        
        anomalies = []
        frame_count = 0
        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames for faster processing
            if frame_count % frame_skip == 0:
                # Calculate timestamp
                timestamp = frame_count / fps
                timestamp_str = str(timedelta(seconds=int(timestamp)))

                # Predict behavior
                predicted_class, confidence = self.predict_behavior(frame)

                # Check for anomaly
                if self.is_anomaly(predicted_class, confidence):
                    anomaly_data = {
                        'frame_number': frame_count,
                        'timestamp': timestamp_str,
                        'timestamp_seconds': timestamp,
                        'behavior': predicted_class,
                        'confidence': confidence,
                        'frame': frame.copy()
                    }
                    anomalies.append(anomaly_data)
                    logger.warning(f"Anomaly detected at {timestamp_str}: {predicted_class} ({confidence:.2f})")

                processed_frames += 1

                # Progress callback - show progress based on processed frames vs total frames to process
                if progress_callback:
                    total_frames_to_process = total_frames // frame_skip
                    progress = (processed_frames / total_frames_to_process) * 100 if total_frames_to_process > 0 else 100
                    progress_callback(min(progress, 100))

            frame_count += 1
        
        cap.release()

        logger.info(f"Analysis complete! Found {len(anomalies)} anomalies in {processed_frames} processed frames")
        return anomalies
    
    def save_anomaly_snapshots(self, anomalies, output_dir=None):
        """Save anomaly frames as images. Move previous snapshots to history before saving new ones."""
        if not anomalies:
            print("No anomalies to save")
            return []

        # Use config directory if not specified
        output_dir = output_dir or self.snapshot_dir
        Path(output_dir).mkdir(exist_ok=True)

        # Move old snapshots to history (avoid name collisions)
        history_dir = os.path.join(output_dir, 'history')
        Path(history_dir).mkdir(exist_ok=True)
        for file in Path(output_dir).glob('anomaly_*.jpg'):
            dest = Path(history_dir) / file.name
            if dest.exists():
                ts = datetime.now().strftime('%Y%m%d%H%M%S')
                base, ext = file.stem, file.suffix
                candidate = Path(history_dir) / f"{base}_{ts}{ext}"
                counter = 1
                while candidate.exists():
                    candidate = Path(history_dir) / f"{base}_{ts}_{counter}{ext}"
                    counter += 1
                dest = candidate
            file.replace(dest)

        saved_files = []
        max_snapshots = min(len(anomalies), self.max_snapshots)
        for i in range(max_snapshots):
            anomaly = anomalies[i]
            timestamp_clean = anomaly['timestamp'].replace(':', '-')
            behavior_clean = anomaly['behavior'].replace(' ', '_').replace('/', '_').replace('\\', '_')
            filename = f"anomaly_{i+1:03d}_{timestamp_clean}_{behavior_clean}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, anomaly['frame'])
            saved_files.append(filepath)
            print(f"💾 Saved: {filename}")

        if len(anomalies) > max_snapshots:
            print(f"⚠️ Limited to {max_snapshots} snapshots (out of {len(anomalies)} anomalies)")

        print(f"✅ Saved {len(saved_files)} anomaly snapshots to {output_dir}")
        return saved_files
    
    def generate_anomaly_report(self, anomalies, video_path, output_file='outputs/anomaly_report.json'):
        """Generate detailed anomaly report"""
        if not anomalies:
            print("No anomalies to report")
            return
        
        # Prepare report data
        report = {
            'video_path': video_path,
            'analysis_timestamp': datetime.now().isoformat(),
            'total_anomalies': len(anomalies),
            'anomaly_summary': {},
            'detailed_anomalies': []
        }
        
        # Count anomalies by type
        for anomaly in anomalies:
            behavior = anomaly['behavior']
            if behavior not in report['anomaly_summary']:
                report['anomaly_summary'][behavior] = 0
            report['anomaly_summary'][behavior] += 1
            
            # Add to detailed list (without frame data)
            detailed_anomaly = {
                'frame_number': anomaly['frame_number'],
                'timestamp': anomaly['timestamp'],
                'timestamp_seconds': anomaly['timestamp_seconds'],
                'behavior': anomaly['behavior'],
                'confidence': anomaly['confidence']
            }
            report['detailed_anomalies'].append(detailed_anomaly)
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📋 Anomaly report saved to {output_file}")
        
        # Print summary
        print("\n📊 ANOMALY SUMMARY:")
        print("-" * 30)
        for behavior, count in report['anomaly_summary'].items():
            print(f"{behavior}: {count} occurrences")
        
        return report

def main():
    """Test the video analyzer"""
    analyzer = VideoAnalyzer()
    
    # Test with a video file (replace with your video path)
    video_path = "test_video.mp4"
    
    if os.path.exists(video_path):
        # Analyze video
        anomalies = analyzer.analyze_video(video_path)
        
        # Save snapshots
        analyzer.save_anomaly_snapshots(anomalies)
        
        # Generate report
        analyzer.generate_anomaly_report(anomalies, video_path)
    else:
        print("No test video found. Starting real-time analysis...")
        
        # Start real-time analysis
        real_time = RealTimeAnalyzer()
        real_time.start_webcam_analysis()

if __name__ == "__main__":
    main()
