import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
import tempfile
from pathlib import Path
import threading
import queue
import time
from PIL import Image
import base64

# Import our custom modules
from model_trainer import ClassroomBehaviorTrainer
from video_analyzer import VideoAnalyzer

# Page configuration
st.set_page_config(
    page_title="🎯 Anomaly Detection in Exam Hall",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional styling
st.markdown(
    """
<style>
    /* Global Styles */
    * { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }

    /* Main Header */
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
        text-shadow: none;
    }

    /* Alert Boxes */
    .alert-box {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 6px solid;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        animation: pulse 2s infinite;
        position: relative;
        overflow: hidden;
    }
    .alert-danger { background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-color: #dc3545; color: #7f1d1d; }
    .alert-warning { background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-color: #f59e0b; color: #92400e; }
    .alert-success { background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border-color: #10b981; color: #065f46; }
    .alert-info { background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border-color: #3b82f6; color: #1e40af; }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover { transform: translateY(-5px); box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4); }
    .metric-card h3 { margin: 0; font-size: 0.9rem; opacity: 0.9; font-weight: 500; }
    .metric-card p { margin: 0.5rem 0 0 0; font-size: 1.8rem; font-weight: 700; }

    /* Anomaly Snapshots */
    .anomaly-snapshot {
        border: 3px solid #dc3545;
        border-radius: 15px;
        padding: 15px;
        margin: 10px;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        box-shadow: 0 6px 20px rgba(220, 53, 69, 0.1);
        transition: all 0.3s ease;
    }
    .anomaly-snapshot:hover { transform: scale(1.02); box-shadow: 0 10px 30px rgba(220, 53, 69, 0.2); }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover { transform: translateY(-3px); box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4); background: linear-gradient(135deg, #5a6fd8 0%, #6b4190 100%); }
    .stButton > button:active { transform: translateY(-1px); }

    /* Sidebar */
    .sidebar .sidebar-content { background: linear-gradient(180deg, #667eea 0%, #764ba2 100%); padding: 2rem 1rem; }

    /* Progress Bars */
    .stProgress > div > div > div { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); }

    /* File Uploader */
    .uploadedFile { border: 2px dashed #667eea; border-radius: 10px; padding: 1rem; background: rgba(102, 126, 234, 0.05); }

    /* DataFrames */
    .dataframe { border-radius: 10px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }

    /* Animations */
    @keyframes pulse { 0% { opacity: 1; transform: scale(1); } 50% { opacity: 0.9; transform: scale(1.01); } 100% { opacity: 1; transform: scale(1); } }
    @keyframes shimmer { 0% { transform: translateX(-100%); } 100% { transform: translateX(100%); } }

    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header { font-size: 2.5rem; }
        .metric-card { padding: 1rem; }
        .metric-card p { font-size: 1.5rem; }
        .stButton > button { padding: 0.6rem 1.5rem; font-size: 0.9rem; }
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 10px; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: linear-gradient(135deg, #5a6fd8 0%, #6b4190 100%); }

    /* Loading Animation */
    .loading-dots::after { content: ''; animation: loading 1.5s infinite; }
    @keyframes loading { 0%, 20% { content: ''; } 40% { content: '.'; } 60% { content: '..'; } 80%, 100% { content: '...'; } }
</style>
    """,
    unsafe_allow_html=True,
)

class StreamlitApp:
    def __init__(self):
        # Initialize session state for caching
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'video_analyzer' not in st.session_state:
            st.session_state.video_analyzer = None
        if 'anomalies' not in st.session_state:
            st.session_state.anomalies = []
        if 'alert_active' not in st.session_state:
            st.session_state.alert_active = False

        # Cache references
        self.video_analyzer = st.session_state.video_analyzer

    def load_model_if_exists(self):
        """Load model if it exists and cache it"""
        if st.session_state.model_loaded and self.video_analyzer is not None:
            return True

        # Candidate model / class index paths (preferred first)
        model_candidates = [
            ('models/best_classroom_model.h5', 'models/class_indices.json'),
            ('models/classroom_behavior_model.h5', 'models/class_indices.json'),
            ('classroom_behavior_model.h5', 'class_indices.json'),
            ('best_classroom_model.h5', 'class_indices.json')
        ]

        chosen = None
        for m_path, c_path in model_candidates:
            if os.path.exists(m_path) and os.path.exists(c_path):
                chosen = (m_path, c_path)
                break

        if not chosen:
            return False

        try:
            self.video_analyzer = VideoAnalyzer(model_path=chosen[0], class_indices_path=chosen[1])
            st.session_state.video_analyzer = self.video_analyzer
            st.session_state.model_loaded = True
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.session_state.model_loaded = False
            return False
    
    def show_header(self):
        """Display main header"""
        st.markdown('<h1 class="main-header">🎯 Anomaly Detection in Exam Hall</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            model_status = "✅ Ready" if self.load_model_if_exists() else "❌ Not Trained"
            st.markdown(f'<div class="metric-card"><h3>Model Status</h3><p>{model_status}</p></div>', 
                       unsafe_allow_html=True)
        
        with col2:
            alert_status = "🚨 Active" if st.session_state.alert_active else "🟢 Normal"
            st.markdown(f'<div class="metric-card"><h3>Alert Status</h3><p>{alert_status}</p></div>', 
                       unsafe_allow_html=True)
        
        with col3:
            anomaly_count = len(st.session_state.anomalies)
            st.markdown(f'<div class="metric-card"><h3>Anomalies Today</h3><p>{anomaly_count}</p></div>', 
                       unsafe_allow_html=True)
        
        with col4:
            analysis_status = "✅ Ready" if st.session_state.model_loaded else "⚠️ Model Needed"
            st.markdown(f'<div class="metric-card"><h3>Analysis Status</h3><p>{analysis_status}</p></div>',
                       unsafe_allow_html=True)
    
    def show_alerts(self):
        """Display alert system for HODs"""
        if st.session_state.alert_active:
            st.markdown("""
            <div class="alert-box alert-danger">
                <h2>🚨 URGENT ALERT - HOD ATTENTION REQUIRED</h2>
                <p><strong>Anomalous behavior detected in classroom!</strong></p>
                <p>Immediate intervention may be required.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Auto-dismiss alert after some time
            if st.button("🔕 Acknowledge Alert"):
                st.session_state.alert_active = False
                st.rerun()
    
    def model_training_page(self):
        """Model training interface"""
        st.header("🤖 Model Training")

        # Validate dataset existence
        if not os.path.exists('CNN_Dataset'):
            st.error("❌ CNN_Dataset folder not found!")
            st.info("💡 Please ensure your dataset folder 'CNN_Dataset' is in the project root directory.")
            st.info("The dataset should contain subfolders for each behavior class.")
            return

        # Check dataset structure
        dataset_path = Path('CNN_Dataset')
        class_folders = [f for f in dataset_path.iterdir() if f.is_dir()]

        if len(class_folders) == 0:
            st.error("❌ CNN_Dataset folder is empty!")
            st.info("💡 The dataset should contain subfolders for each behavior class (Normal, Discussing, etc.)")
            return

        if len(class_folders) < 9:
            st.warning(f"⚠️ Expected 9 behavior classes, found {len(class_folders)}")
            st.info("Expected classes: Normal, Discussing, Peeking, cheat passing, copying, showing answer, suspicious, using copy cheat, using mobile")

        st.success(f"✅ Dataset found with {len(class_folders)} classes!")

        # Training parameters with validation
        col1, col2 = st.columns(2)

        with col1:
            epochs = st.slider("Training Epochs", 10, 100, 50, help="Number of training epochs (higher = better accuracy but slower)")
            batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1, help="Batch size for training (higher = faster but more memory)")

        with col2:
            img_size = st.selectbox("Image Size", [(224, 224), (256, 256)], index=0, help="Input image resolution")
            validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2, help="Portion of data used for validation")

        # Training warnings
        if epochs > 50:
            st.warning("⚠️ High epoch count may cause overfitting or long training time")

        if batch_size > 32 and not st.checkbox("I understand large batch sizes require more GPU memory"):
            st.info("💡 Large batch sizes need more memory but train faster")

        if st.button("🚀 Start Training", key="train_button"):
            # Final validation before training
            if not st.checkbox("I confirm the dataset is properly labeled and organized"):
                st.error("❌ Please confirm your dataset is ready for training")
                return

            with st.spinner("Training model... This may take a while."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Initialize trainer
                    trainer = ClassroomBehaviorTrainer(
                        dataset_path="CNN_Dataset",
                        img_size=img_size,
                        batch_size=batch_size
                    )

                    status_text.text("Loading and preprocessing data...")
                    progress_bar.progress(10)

                    # Load data
                    train_gen, val_gen = trainer.load_and_preprocess_data()

                    status_text.text("Building model architecture...")
                    progress_bar.progress(20)

                    # Build model
                    model = trainer.build_model()

                    status_text.text("Training model...")
                    progress_bar.progress(30)

                    # Train model
                    history = trainer.train_model(epochs=epochs)

                    status_text.text("Evaluating model...")
                    progress_bar.progress(80)

                    # Evaluate model
                    report, cm = trainer.evaluate_model()

                    status_text.text("Saving model...")
                    progress_bar.progress(90)

                    # Save model
                    trainer.save_model()

                    progress_bar.progress(100)
                    status_text.text("Training completed!")

                    st.success("🎉 Model training completed successfully!")
                    st.balloons()

                    # Display results
                    col1, col2 = st.columns(2)

                    with col1:
                        if 'val_accuracy' in history.history and history.history['val_accuracy']:
                            st.metric("Final Validation Accuracy", f"{max(history.history['val_accuracy']):.4f}")
                        if 'val_loss' in history.history and history.history['val_loss']:
                            st.metric("Final Validation Loss", f"{min(history.history['val_loss']):.4f}")

                    with col2:
                        st.metric("Training Epochs", f"{epochs}")
                        if model:
                            st.metric("Total Parameters", f"{model.count_params():,}")

                    st.session_state.model_loaded = True

                except MemoryError:
                    st.error("❌ Training failed: Out of memory!")
                    st.info("💡 Try reducing batch size or image resolution")
                except Exception as e:
                    st.error(f"❌ Training failed: {e}")
                    st.info("💡 Check your dataset structure and try again")
    
    def video_analysis_page(self):
        """Video analysis interface"""
        st.header("🎬 Video Analysis")

        try:
            if not self.load_model_if_exists():
                st.error("❌ Model not found! Please train the model first in the Model Training section.")
                return
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.info("💡 Try training a new model in the Model Training section.")
            return

        # Video upload
        uploaded_file = st.file_uploader(
            "Upload classroom video",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to analyze for anomalous behavior"
        )

        if uploaded_file is not None:
            try:
                # Validate file size (max 500MB)
                max_size = 500 * 1024 * 1024  # 500MB
                if uploaded_file.size > max_size:
                    st.error(f"❌ File too large! Maximum size is 500MB. Your file is {uploaded_file.size / (1024*1024):.1f}MB.")
                    return

                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    video_path = tmp_file.name

                st.success("✅ Video uploaded successfully!")

                # Analysis parameters
                col1, col2 = st.columns(2)

                with col1:
                    frame_skip = st.slider("Frame Skip (for faster processing)", 1, 60, 30)
                    confidence_threshold = st.slider("Anomaly Confidence Threshold", 0.5, 0.95, 0.7)

                with col2:
                    st.info("Higher frame skip = faster processing but may miss some anomalies")
                    st.info("Higher threshold = fewer false positives but may miss real anomalies")

                if st.button("🔍 Analyze Video", key="analyze_button"):
                    with st.spinner("Analyzing video... Please wait."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        def progress_callback(progress):
                            progress_bar.progress(progress / 100)

                        try:
                            status_text.text("Initializing analysis...")
                            # Set threshold
                            self.video_analyzer.anomaly_threshold = confidence_threshold

                            status_text.text("Processing video frames...")
                            # Analyze video
                            anomalies = self.video_analyzer.analyze_video(
                                video_path,
                                frame_skip=frame_skip,
                                progress_callback=progress_callback
                            )

                            status_text.text("Saving results...")
                            # Store anomalies in session state (without frame data to prevent memory leaks)
                            st.session_state.anomalies = [{
                                'frame_number': a['frame_number'],
                                'timestamp': a['timestamp'],
                                'timestamp_seconds': a['timestamp_seconds'],
                                'behavior': a['behavior'],
                                'confidence': a['confidence']
                            } for a in anomalies]

                            if anomalies:
                                st.session_state.alert_active = True

                                st.error(f"🚨 {len(anomalies)} anomalies detected!")

                                # Save snapshots
                                try:
                                    snapshot_files = self.video_analyzer.save_anomaly_snapshots(anomalies)
                                except Exception as e:
                                    st.warning(f"⚠️ Could not save snapshots: {e}")

                                # Generate report
                                try:
                                    report = self.video_analyzer.generate_anomaly_report(anomalies, video_path)
                                except Exception as e:
                                    st.warning(f"⚠️ Could not generate report: {e}")

                                # Display results
                                self.show_analysis_results(anomalies, report)

                            else:
                                st.success("✅ No anomalies detected! Classroom behavior appears normal.")

                            status_text.text("Analysis completed!")

                        except ValueError as e:
                            st.error(f"❌ Video processing error: {e}")
                        except Exception as e:
                            st.error(f"❌ Analysis failed: {e}")
                            st.info("💡 Try uploading a different video file or check the file format.")

                        finally:
                            # Clean up temporary file
                            try:
                                os.unlink(video_path)
                            except:
                                pass  # File might already be deleted

            except Exception as e:
                st.error(f"❌ Error processing uploaded file: {e}")
                st.info("💡 Try uploading the file again or check if it's corrupted.")
    
    def show_analysis_results(self, anomalies, report):
        """Display enhanced analysis results"""
        st.header("📊 Analysis Results")

        # Summary Cards with enhanced styling
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Anomalies</h3>
                <p>{len(anomalies)}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            unique_behaviors = len(set(a['behavior'] for a in anomalies))
            st.markdown(f"""
            <div class="metric-card">
                <h3>Unique Behaviors</h3>
                <p>{unique_behaviors}</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            avg_confidence = np.mean([a['confidence'] for a in anomalies]) if anomalies else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>Avg Confidence</h3>
                <p>{avg_confidence:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            total_duration = max([a['timestamp_seconds'] for a in anomalies]) if anomalies else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>Video Duration</h3>
                <p>{total_duration:.1f}s</p>
            </div>
            """, unsafe_allow_html=True)

        if anomalies:
            # Enhanced Anomaly Timeline
            st.subheader("📈 Anomaly Timeline")

            df = pd.DataFrame([{
                'Timestamp': pd.to_datetime(a['timestamp_seconds'], unit='s'),
                'Behavior': a['behavior'],
                'Confidence': a['confidence'],
                'Frame': a['frame_number']
            } for a in st.session_state.anomalies])

            # Interactive timeline with better styling
            fig_timeline = px.scatter(
                df,
                x='Timestamp',
                y='Confidence',
                color='Behavior',
                size='Confidence',
                title="Anomaly Detection Timeline",
                hover_data=['Frame', 'Behavior'],
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_timeline.update_layout(
                height=500,
                xaxis_title="Video Timeline",
                yaxis_title="Confidence Score",
                hovermode='closest'
            )
            fig_timeline.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
            st.plotly_chart(fig_timeline, use_container_width=True)

            # Behavior Distribution with enhanced pie chart
            st.subheader("📊 Behavior Distribution")
            # Normalize behavior labels
            df['Behavior'] = df['Behavior'].astype(str).str.strip()
            behavior_counts_series = df['Behavior'].value_counts()
            # Attempt to load full class list to include zero counts
            full_classes = []
            try:
                with open('models/class_indices.json', 'r') as f:
                    ci = json.load(f)
                    full_classes = [c for c,_ in sorted(ci.items(), key=lambda kv: kv[1])]
            except Exception:
                full_classes = list(behavior_counts_series.index)
            # Build complete counts dict
            behavior_counts = {cls: int(behavior_counts_series.get(cls, 0)) for cls in full_classes}

            col1, col2 = st.columns([2, 1])

            with col1:
                non_zero = {k:v for k,v in behavior_counts.items() if v > 0}
                if non_zero:
                    fig_pie = px.pie(
                        values=list(non_zero.values()),
                        names=list(non_zero.keys()),
                        title="Anomaly Distribution by Behavior Type",
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                        hole=0.3
                    )
                    fig_pie.update_layout(height=400)
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("No anomalous behaviors detected.")

            with col2:
                st.markdown("### 📋 Summary")
                total = sum(behavior_counts.values()) or 1
                for behavior, count in behavior_counts.items():
                    percentage = (count / total) * 100 if total else 0
                    st.markdown(f"**{behavior}**: {count} ({percentage:.1f}%)")

                st.markdown("---")
                st.markdown(f"**Total Frames Analyzed**: {max(df['Frame']) if not df.empty else 0}")

        # Enhanced Anomaly Snapshots Gallery
        st.subheader("🖼️ Anomaly Snapshots Gallery")

        if os.path.exists('outputs/anomaly_snapshots'):
            snapshot_files = sorted(list(Path('outputs/anomaly_snapshots').glob('*.jpg')),
                                   key=lambda x: x.stat().st_mtime, reverse=True)

            if snapshot_files:
                st.info(f"📸 Showing {min(len(snapshot_files), 9)} most recent anomaly snapshots")

                # Create responsive grid
                cols = st.columns(3)

                for i, snapshot_file in enumerate(snapshot_files[:9]):
                    with cols[i % 3]:
                        try:
                            image = Image.open(snapshot_file)

                            # Extract metadata from filename
                            parts = snapshot_file.stem.split('_')
                            if len(parts) >= 4:
                                anomaly_num = parts[1]
                                timestamp = parts[2].replace('-', ':')
                                behavior = ' '.join(parts[3:]).replace('_', ' ')

                                # Display image with enhanced caption
                                st.image(image, use_container_width=True)

                                st.markdown(f"""
                                <div class="anomaly-snapshot">
                                    <strong>🚨 Anomaly #{anomaly_num}</strong><br>
                                    <strong>⏰ Time:</strong> {timestamp}<br>
                                    <strong>🎭 Behavior:</strong> {behavior.title()}<br>
                                    <strong>📁 File:</strong> {snapshot_file.name}
                                </div>
                                """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error loading snapshot: {snapshot_file.name}")
            else:
                st.info("📷 No anomaly snapshots found. Try running video analysis first.")
        else:
            st.warning("📁 Anomaly snapshots directory not found.")

        # Detailed Report Section
        if report:
            with st.expander("📋 Detailed Analysis Report", expanded=False):
                st.json(report)
    
    def dashboard_page(self):
        """Main dashboard with enhanced visualizations"""
        st.header("📊 Monitoring Dashboard")

        # Key Performance Indicators
        st.subheader("🎯 Key Metrics")

        # Enhanced metrics with better visual design
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Analysis Sessions</h3>
                <p>{len(st.session_state.anomalies) if st.session_state.anomalies else 0}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            anomaly_count = len(st.session_state.anomalies)
            st.markdown(f"""
            <div class="metric-card">
                <h3>Anomalies Today</h3>
                <p>{anomaly_count}</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            model_status = "Online" if st.session_state.model_loaded else "Offline"
            status_color = "🟢" if st.session_state.model_loaded else "🔴"
            st.markdown(f"""
            <div class="metric-card">
                <h3>AI Model</h3>
                <p>{status_color} {model_status}</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            alert_status = "Active" if st.session_state.alert_active else "Normal"
            alert_icon = "🚨" if st.session_state.alert_active else "🛡️"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Alert Status</h3>
                <p>{alert_icon} {alert_status}</p>
            </div>
            """, unsafe_allow_html=True)

        # Recent Activity Section
        st.subheader("📈 Recent Activity")

        if st.session_state.anomalies:
            # Create enhanced anomaly timeline
            df = pd.DataFrame([{
                'Time': pd.to_datetime(a['timestamp_seconds'], unit='s'),
                'Behavior': a['behavior'],
                'Confidence': a['confidence']
            } for a in st.session_state.anomalies[-20:]])  # Show last 20

            # Timeline chart
            fig_timeline = px.scatter(
                df,
                x='Time',
                y='Confidence',
                color='Behavior',
                size='Confidence',
                title="Recent Anomaly Timeline",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_timeline.update_layout(
                height=400,
                xaxis_title="Time",
                yaxis_title="Confidence Score"
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

            # Behavior distribution pie chart
            behavior_counts = df['Behavior'].value_counts()
            fig_pie = px.pie(
                values=behavior_counts.values,
                names=behavior_counts.index,
                title="Anomaly Distribution by Type",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            # Empty state
            st.info("📊 No anomalies detected yet. Upload a video to analyze classroom behavior.")

        # System Health Section
        st.subheader("🖥️ System Health")

        # System status with enhanced styling
        status_data = {
            "Component": ["AI Model", "Video Processing", "Alert System", "Storage"],
            "Status": [
                "✅ Online" if st.session_state.model_loaded else "❌ Offline",
                "✅ Ready" if self.video_analyzer else "⚠️ Initializing",
                "✅ Active" if st.session_state.alert_active else "🟢 Normal",
                "✅ OK"
            ],
            "Last Check": ["Just now"] * 4,
            "Performance": ["Good", "Excellent", "Active", "Optimal"]
        }

        st.dataframe(
            pd.DataFrame(status_data),
            use_container_width=True,
            column_config={
                "Status": st.column_config.TextColumn("Status", help="Current component status"),
                "Performance": st.column_config.TextColumn("Performance", help="Performance indicator")
            }
        )

        # Quick Actions
        st.subheader("⚡ Quick Actions")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🎬 Upload Video for Analysis", use_container_width=True):
                st.session_state.selected_page = "🎬 Video Analysis"
                st.rerun()

        with col2:
            if st.button("🤖 Train New Model", use_container_width=True):
                st.session_state.selected_page = "🤖 Model Training"
                st.rerun()

        with col3:
            if st.button("📊 View Analytics", use_container_width=True):
                st.success("📈 Analytics displayed above!")
                st.balloons()
    
    def run(self):
        """Main application runner"""
        # Sidebar navigation
        st.sidebar.title("🎯 Navigation")
        
        pages = {
            "📊 Dashboard": self.dashboard_page,
            "🤖 Model Training": self.model_training_page,
            "🎬 Video Analysis": self.video_analysis_page
        }
        
        selected_page = st.sidebar.selectbox("Select Page", list(pages.keys()))
        
        # Sidebar info
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📋 Quick Info")
        st.sidebar.info("""
        **System Features:**
        - 🤖 AI-powered behavior detection
        - 🎬 Video analysis with snapshots
        - 🚨 Instant alert notifications
        - 📊 Comprehensive reporting
        - 📈 Advanced analytics dashboard
        """)
        
        # Main content
        self.show_header()
        self.show_alerts()
        
        # Run selected page
        pages[selected_page]()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 20px;">
            🎯 Anomaly Detection in Exam Hall | Built with Streamlit & TensorFlow<br>
            <small>Ensuring academic integrity through intelligent monitoring</small>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main function"""
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()
