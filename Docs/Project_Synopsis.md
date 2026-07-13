# Project Synopsis: Anomaly Detection in Exam Hall

## 1. Group Name & Project Title
- Group Name: <Your Group Name>
- Members:
  - <Name 1> — Enrollment No., Branch/Sem, Mobile, College Email
  - <Name 2> — Enrollment No., Branch/Sem, Mobile, College Email
  - <Name 3> — Enrollment No., Branch/Sem, Mobile, College Email
- Project Title: Anomaly Detection in Exam Hall

## 2. Problem Statement
Examinations require strict academic integrity, yet manual proctoring is limited by scale, fatigue, and subjectivity. As class sizes and the number of parallel exam sessions grow, a few human invigilators cannot maintain consistent attention across all students and vantage points. Incidents may go unnoticed in real time, and post-exam review of hours-long footage is laborious and error-prone. Institutions need an objective, evidence-based workflow that ingests exam hall video, detects suspicious behaviors (e.g., copying, cheating, signaling), and produces verifiable artifacts (snapshots, logs) that streamline audits and support fair decision-making while respecting privacy norms.

## 3. Objectives and Scope
**Objectives:**
- Detect anomalous behaviors from exam hall videos using a CNN-based classifier with well-defined classes and thresholds.
- Generate timestamped snapshots and structured anomaly logs for auditing that can be linked back to original segments.
- Provide a review interface to browse incidents, visualize metrics, and export summaries for committee review.
- Enable continuous model improvement via retraining on new labeled data, supporting domain shift and class expansion.

**Scope:**
- End-to-end pipeline: Video ingestion, frame/segment preprocessing, inference, anomaly handling, evidence generation (snapshots), and reporting (JSON logs, standardized summaries). 
- Modular components: `video_analyzer.py` (analysis engine), `model_trainer.py` (training pipeline), optional `streamlit_app.py` (review UI) for transparent evaluation and oversight.
- Measurable artifacts: `anomaly_report.json` (machine-readable incidents), `anomaly_snapshots/` (visual evidence), `classification_report.txt` (precision/recall/F1), `confusion_matrix.png` (error patterns), enabling traceability and reproducibility.

## 4. Methodology (Process Description)
- Data Preparation: Curate labeled footage distinguishing normal behavior from copying/cheating; define annotation protocols, inter-rater agreement, and ethical handling of data. Partition datasets into training/validation/test splits to avoid leakage and ensure generalization.
- Model: Convolutional Neural Network (TensorFlow/Keras) for frame/segment classification; consider transfer learning (e.g., MobileNetV2/ResNet50 backbones) to improve convergence and robustness with limited data.
- Flow: Ingest video → extract frames or time windows → normalize and resize → batch inference → post-process predictions with temporal smoothing and confidence thresholds → flag anomalies → persist snapshots/logs with metadata (video timecode, detector version).
- Evaluation: Use precision, recall, F1-score, ROC-AUC where applicable; analyze confusion matrix to identify common misclassifications and guide dataset augmentation. Perform ablations (lighting, occlusions, seating density) to assess resilience.
- Diagrams (recommended): DFDs / Flowcharts / UML (Use Case, Activity, Sequence) to depict data/control flow, component boundaries, and stakeholder interactions.

## 5. Hardware & Software Used
**Hardware:** Desktop/Laptop with 8GB+ RAM; GPU recommended (e.g., NVIDIA GTX 1060+) for faster inference/training; adequate storage and I/O bandwidth for long-duration video files; optionally, external drives for archival.

**Software:** Python 3.8+, TensorFlow/Keras, OpenCV, NumPy, Pandas, Matplotlib, Streamlit (review UI), virtual environment `classroom_monitor_env`; optional tools for data labeling and augmentation; logging and serialization libraries for robust artifact management.

## 6. Application and Future Scope
**Applications:** Post-exam auditing (evidence-backed incident review), institutional integrity checks (policy enforcement and transparency), incident documentation (standardized reports for committees), edtech research (benchmarking CV models in real-world exam scenarios), and training of proctoring staff with curated case studies.

**Future Scope:** Expanded behavior taxonomy (signaling, device usage, unauthorized material), improved temporal modeling (sequence-based detectors, transformers), privacy-preserving features (face blurring, edge anonymization, role-based access), batch processing and per-exam dashboards (session-level summaries, trend analysis), containerized deployment for portability and CI/CD integration.

## 7. Project Timeline (Indicative Gantt)
- Week 1–2: Requirement analysis, literature survey, risk assessment, dataset planning (sources, consent, governance). Define labels and annotation guidelines.
- Week 3–4: Data collection/labeling, preprocessing pipeline (frame extraction, normalization, storage schema). Establish baseline metrics and experiment tracking.
- Week 5–6: Model selection and baseline training (transfer learning vs. from-scratch), hyperparameter tuning, early stopping strategies.
- Week 7–8: Validation, metrics, confusion matrix analysis, targeted augmentation to address failure modes (lighting, occlusions, crowding). Draft reporting formats.
- Week 9: Video analyzer integration (I/O handling, temporal smoothing), artifact generation (snapshots, JSON logs), metadata and versioning.
- Week 10: Review UI (Streamlit) and reporting (interactive filtering, charts), export to PDF/CSV for committees.
- Week 11: Documentation, final testing, packaging (batch scripts, README), optional containerization, and handoff.

## 8. References / Bibliography
- [1] I. Goodfellow, Y. Bengio, A. Courville, Deep Learning, MIT Press, 2016.
- [2] R. Szeliski, Computer Vision: Algorithms and Applications, Springer, 2010.
- [3] F. Chollet, Deep Learning with Python, Manning, 2018.
- [4] OpenCV Documentation: https://docs.opencv.org/
- [5] TensorFlow/Keras Guides: https://www.tensorflow.org/ and https://keras.io/
- [6] D. P. Kingma and J. Ba, Adam: A Method for Stochastic Optimization, ICLR, 2015.
- [7] K. He, X. Zhang, S. Ren, J. Sun, Deep Residual Learning for Image Recognition, CVPR, 2016.

---

## Additional Notes per Guidelines
**Resources and Limitations:**
- Resources: Labeled video data (with consent/governance), compute (CPU/GPU), storage, annotation tools, experiment tracking. Clear SOPs for data handling and deletion policies.
- Limitations: Accuracy depends on data quality/diversity; potential false positives/negatives; domain shift between institutions; privacy/compliance considerations and the need for human-in-the-loop verification.

**Conclusion:**
This synopsis presents a feasible, extensible plan for exam hall anomaly detection using CNNs, structured evidence generation, and review tooling. The methodology, resources, and timeline support delivery within a semester while laying the groundwork for robust governance, continuous improvement, and cross-institutional reuse. With clear artifacts and standardized reports, the solution strengthens trust in assessments and enables efficient, fair incident resolution.

**Q&A Preparation:**
- Strengths: Objective detection, reproducible pipeline, clear evidence artifacts, scalable across sessions, transparent metrics, and modular design that supports upgrading models and taxonomies.
- Limitations: Data-dependent accuracy, possible false alarms, domain shift across venues, and the need for strict governance and human oversight to prevent misuse.
- Positive Impact (≥100 words):
  The project promotes fairness and integrity in examinations by providing objective, evidence-based detection of misconduct, complementing human invigilation with consistent coverage. It reduces manual overhead for staff, standardizes auditing procedures with machine-readable logs and visual evidence, and supports policy enforcement through traceable artifacts. Academically, it offers a hands-on platform for students to explore responsible computer vision, annotation practices, and model evaluation in high-stakes environments. For the college and broader society, it builds trust in assessment outcomes by ensuring transparent and repeatable reviews. For industry and edtech, the pipeline informs proctoring solutions and compliance tooling, demonstrating how ML-driven systems can augment human oversight while maintaining accountability and privacy safeguards.
