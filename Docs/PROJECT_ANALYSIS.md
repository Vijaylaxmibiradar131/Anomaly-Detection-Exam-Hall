## AI-Powered Classroom Behavior Monitoring — Project Analysis

### 1. Overview
- **Purpose**: Detect and monitor classroom behaviors (Normal, Discussing, Peeking, cheat passing, copying, showing answer, suspicious, using copy cheat, using mobile) using a CNN and provide analytics via a Streamlit app.
- **Key Components**: `model_trainer.py` (training pipeline), `video_analyzer.py` (inference and reporting), `streamlit_app.py` (UI), artifacts (`classroom_behavior_model.h5`, `best_classroom_model.h5`, `class_indices.json`, `classification_report.txt`, `confusion_matrix.png`, `anomaly_report.json`, `anomaly_snapshots/`).

### 2. Dataset & Classes
- Dataset path: `CNN_Dataset/` with 9 class folders matching `ClassroomBehaviorTrainer.classes`.
- Image preprocessing: resize to 224×224, rescale 1/255, augmentation (rotation/shift/flip/zoom).
- Split: Keras ImageDataGenerator with `validation_split=0.2`.

### 3. Model Architecture (Training)
- Sequential CNN with 4 conv blocks (32, 64, 128, 256 filters) + BatchNorm + MaxPool + Dropout.
- GlobalAveragePooling2D → Dense(512) → Dense(256) with BN + Dropout → Dense(9, softmax).
- Optimizer: Adam(lr=0.001). Loss: categorical_crossentropy. Metrics: accuracy.
- Callbacks: EarlyStopping(monitor=val_accuracy, patience=10, restore_best_weights), ReduceLROnPlateau(monitor=val_loss), ModelCheckpoint(`best_classroom_model.h5`, monitor=val_accuracy).
- Trained with `epochs=50` by default; adjustable via Streamlit UI.

### 4. Evaluation Results
- Classification report (from `classification_report.txt`, support=736, val split):
  - High scores: copying (P/R/F1=1.00), showing answer (1.00), peeking (F1=0.94), discussing (F1=0.96), normal (R=1.00, F1=0.80).
  - Challenging classes: cheat passing (F1=0.47), using copy cheat (F1=0.43), using mobile (F1=0.61), suspicious (F1=0.64).
  - Overall accuracy: 0.77; macro avg F1≈0.76; weighted avg F1≈0.76.
- Confusion matrix plotted to `confusion_matrix.png`.

### 5. Inference & Reporting (`video_analyzer.py`)
- Loads `classroom_behavior_model.h5` and `class_indices.json`.
- Preprocess: BGR→RGB, resize 224×224, normalize, batch dim.
- Per-frame prediction; anomaly if class != "Normal" and confidence ≥ threshold (default 0.7, configurable via UI).
- Video analysis with frame skipping (default 30) to trade accuracy vs speed; collects anomalies with timestamps and confidences.
- Exports:
  - Snapshots: `anomaly_snapshots/anomaly_###_HH-MM-SS_class.jpg`.
  - JSON report: `anomaly_report.json` with summary and detailed list.
- Real-time analyzer prototype with OpenCV window, threaded queues; not integrated in Streamlit yet (placeholder in UI mentions WebRTC).

### 6. Streamlit Application (`streamlit_app.py`)
- Pages: Dashboard, Model Training, Video Analysis, Live Monitoring (demo placeholder).
- Training page: selects epochs, batch size, image size, val split; shows progress and metrics; saves model and plots.
- Video Analysis: upload a video, set frame skip and threshold; runs analysis, saves snapshots and JSON, visualizes timeline (Plotly), pie chart, and thumbnails.
- Status cards: model availability, alert status, anomaly count, monitoring status.
- Session state used for anomalies and alert toggling.

### 7. Software/Hardware Requirements
- Requirements: TensorFlow 2.15, OpenCV, Streamlit, NumPy, Pandas, Plotly, scikit-learn, etc. (see `requirements.txt`).
- Hardware: CPU okay; GPU (CUDA) recommended for faster training/inference; camera source for live analysis.

### 8. Architecture Diagram (Textual)
- Data Layer: `CNN_Dataset/` → ImageDataGenerator (train/val splits, augmentations).
- Model Layer: CNN training → checkpoints `best_classroom_model.h5` → final `classroom_behavior_model.h5` + `class_indices.json`.
- Inference Layer: `VideoAnalyzer.predict_behavior()` per frame → anomaly check.
- Application Layer: Streamlit UI for training, video upload analysis, dashboards.
- Outputs: `classification_report.txt`, `confusion_matrix.png`, `anomaly_report.json`, `anomaly_snapshots/`.

### 9. Strengths
- End-to-end pipeline: training → evaluation → inference → UI visualizations.
- Clear class taxonomy aligned with dataset folders.
- Useful reporting artifacts and visual insights (timeline, distribution, snapshots).
- Modular design: separation between training, inference, and UI.

### 10. Limitations / Risks
- Class imbalance or visual similarity may hurt classes like "cheat passing" and "using copy cheat" (low F1).
- Fixed 224×224 and simple CNN may limit performance vs transfer learning backbones.
- Anomaly definition equates any non-Normal with threshold ≥ τ; may under-detect low-confidence true positives or overflag noisy predictions.
- Real-time Streamlit webcam page is a placeholder; not using `streamlit-webrtc` yet.
- No explicit privacy/face blurring, consent management, or data retention policy enforcement.
- No calibration set/threshold tuning per class; single global threshold.

### 11. Recommendations / Next Steps
- Model improvements:
  - Adopt transfer learning (e.g., EfficientNetB0/B3, MobileNetV3) with fine-tuning; mixed precision if GPU.
  - Add class-weighting or focal loss to address hard classes; perform stratified splits.
  - Data curation: increase samples for low-F1 classes; hard negative mining; stronger augmentations relevant to classroom context.
  - Log per-class ROC/PR curves; tune per-class thresholds.
- Inference/UX:
  - Implement `streamlit-webrtc` for actual real-time webcam streaming; on-frame overlays in the app.
  - Temporal smoothing (e.g., sliding window majority vote) to reduce flicker and false positives.
  - Batch frame inference for throughput; optional GPU acceleration with TensorRT/ONNX.
  - Add privacy features: optional face blurring, on-device processing, retention settings.
- Ops & quality:
  - Save `best_classroom_model.h5` as the deployed model by default; version artifacts.
  - Add unit tests for preprocessing/predict/thresholding; e2e test on sample videos.
  - Log metrics over time; add configuration file for thresholds and paths.

### 12. How to Run
- Training: `python model_trainer.py` or via Streamlit page “Model Training”. Ensure `CNN_Dataset/` exists.
- App: `streamlit run streamlit_app.py`
- Video analysis: Use the Streamlit page “Video Analysis”, upload a file, adjust frame skip/threshold, view results, and check `anomaly_snapshots/` and `anomaly_report.json`.

### 13. Notable Files
- `model_trainer.py`: training pipeline, evaluation, saving artifacts.
- `video_analyzer.py`: frame preprocessing, prediction, anomaly detection, reporting, snapshots, real-time prototype.
- `streamlit_app.py`: multi-page dashboard, training UI, video analysis and visualization, alerts.
- `requirements.txt`: pinned dependencies.


