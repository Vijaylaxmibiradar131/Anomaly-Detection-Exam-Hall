# Project Description: Anomaly Detection in Exam Hall

## Overview
This project detects anomalies in exam halls using deep learning and computer vision. It analyzes exam hall videos to identify misconduct behaviors—such as copying or cheating—and produces visual snapshots and structured reports for review.

## Goals
- Enhance academic integrity by automatically detecting misconduct in recorded footage.
- Support exam administrators with evidence-based, post-exam analysis.
- Provide clear, reproducible reports and visual artifacts for auditing.

## Key Features
- CNN-based classifier for detecting predefined anomalies (e.g., copying/cheating).
- Snapshot capture of detected events with timestamps (stored in `anomaly_snapshots/`).
- Review interface to browse results and analytics.
- Model training pipeline to retrain/improve accuracy with new labeled data.

## System Architecture
1. Video Input: Exam hall video files.
2. Preprocessing: Frame extraction, resizing/normalization, optional augmentation.
3. Model Inference: CNN classifies frames or segments as normal vs. anomalous.
4. Anomaly Handler: Saves snapshots, aggregates counts, writes `anomaly_report.json`.
5. Reporting & Review: Artifacts (images, JSON, confusion matrix) and UI for browsing.

## Components in This Repository
- `video_analyzer.py`: Processes videos; runs inference and generates snapshots/reports.
- `model_trainer.py`: Training script to produce/update `.h5` model weights and evaluation reports.
- `streamlit_app.py`: Optional review interface to browse anomalies and analytics (not a live monitor).
- `best_classroom_model.h5` / `classroom_behavior_model.h5`: Trained model weights.
- `class_indices.json`: Mapping from class labels to indices used by the model.
- `classification_report.txt`: Training/evaluation metrics.
- `confusion_matrix.png`: Visual performance summary.
- `anomaly_report.json`: Structured anomaly log for detected events.
- `anomaly_snapshots/`: Evidence images captured at anomaly timestamps.
- `requirements.txt`: Dependencies for reproducible setup.

## Data & Model
- Dataset: Labeled classroom/exam hall footage for behaviors (normal, copying, cheating).
- Model: Convolutional Neural Network (CNN) trained for frame-level or segment-level classification.
- Artifacts: Saved models (`.h5`), classification reports, confusion matrix, and anomaly snapshots.

## Typical Workflow
1. Prepare video(s) from the exam hall.
2. Run analysis: `run_project.bat analyze --input your_video.mp4`.
3. Review results: Inspect `anomaly_snapshots/` and `anomaly_report.json`; open `confusion_matrix.png` and `classification_report.txt` for performance context.
4. Open the review UI: `run_project.bat app` to browse artifacts.
5. Retrain the model if needed: `run_project.bat train --epochs 10` to improve accuracy.

## Requirements
- Python 3.8+ with virtual environment (`classroom_monitor_env`).
- Libraries: TensorFlow/Keras, OpenCV, Streamlit (optional for review), NumPy, Pandas, Matplotlib.
- Recorded exam hall videos; optional GPU for faster inference/training.

## Benefits & Considerations
- Benefits: Objective, repeatable post-exam analysis; clear evidence artifacts; scalable across multiple videos.
- Considerations: No real-time alerts; privacy/compliance for recorded footage; accuracy depends on labeled training data.

## Future Enhancements
- Improve behavior taxonomy and temporal modeling (sequence-based detection).
- Add privacy-preserving methods (face blurring, access controls) for reviewed artifacts.
- Batch processing for large video sets and summary dashboards per exam session.
- Containerized packaging for reproducible, portable offline runs.
