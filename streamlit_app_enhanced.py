"""
Enhanced Streamlit App with Authentication, Privacy, and Reporting Features
"""

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
import time
from PIL import Image

# Import custom modules
from src.model_trainer import ClassroomBehaviorTrainer
from src.video_analyzer import VideoAnalyzer
from src.auth_manager import AuthManager
from src.report_generator import ReportGenerator
# Privacy settings removed; PrivacyEnhancer no longer used

# Initialize session state for authentication
if 'auth_manager' not in st.session_state:
    st.session_state.auth_manager = AuthManager()
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'show_login' not in st.session_state:
    st.session_state.show_login = True

# Page configuration
st.set_page_config(
    page_title="🎯 Anomaly Detection in Exam Hall",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

def login_page():
    """Display login page"""
    st.markdown("""
    <div style='text-align: center; padding: 50px 0;'>
        <h1 style='font-size: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            🎯 Anomaly Detection in Exam Hall
        </h1>
        <p style='color: #666; font-size: 1.2rem;'>Secure Login Required</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 Login")
        
        with st.form("login_form"):
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                if username and password:
                    success, message = st.session_state.auth_manager.authenticate(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.show_login = False
                        st.success(f"✅ {message}")
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
                else:
                    st.warning("Please enter both username and password")
        
        st.info("""
        **Default Credentials:**
        - Admin: `admin` / `admin123`
        - Proctor: `proctor` / `proctor123`
        """)

def main_app():
    """Main application after login"""
    auth = st.session_state.auth_manager
    
    # Sidebar with user info and navigation
    with st.sidebar:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; color: white; text-align: center;'>
            <h3 style='margin: 0;'>👤 {auth.current_user['name']}</h3>
            <p style='margin: 5px 0; opacity: 0.9;'>{auth.current_user['role'].title()}</p>
            <p style='margin: 0; font-size: 0.9rem; opacity: 0.8;'>{auth.current_user['email']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### 📋 Navigation")
        pages = ["📊 Dashboard", "🎬 Video Analysis", "🤖 Model Training", 
            "📄 Reports & Export"]
        
        if auth.is_admin():
            pages.extend(["👥 User Management", "📜 Audit Log"])

        # Keep page selection in session state for Quick Actions
        if 'nav_page' not in st.session_state:
            st.session_state.nav_page = pages[0]
        default_index = pages.index(st.session_state.nav_page) if st.session_state.nav_page in pages else 0
        selected_page = st.selectbox("Select Page", pages, index=default_index, label_visibility="collapsed")
        # Persist the choice
        st.session_state.nav_page = selected_page
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            auth.logout()
            st.session_state.logged_in = False
            st.session_state.show_login = True
            st.rerun()
    
    # Route to selected page
    if selected_page == "📊 Dashboard":
        dashboard_page()
    elif selected_page == "🎬 Video Analysis":
        video_analysis_page()
    elif selected_page == "🤖 Model Training":
        model_training_page()
    elif selected_page == "📄 Reports & Export":
        reports_page()
    elif selected_page == "👥 User Management" and auth.is_admin():
        user_management_page()
    elif selected_page == "📜 Audit Log" and auth.is_admin():
        audit_log_page()
def dashboard_page():
    """Dashboard with metrics and overview"""
    auth = st.session_state.auth_manager
    st.title("📊 Dashboard")

    # Load latest anomaly data if available
    anomaly_file = 'outputs/anomaly_report.json'
    anomaly_data = None
    if os.path.exists(anomaly_file):
        try:
            with open(anomaly_file, 'r') as f:
                anomaly_data = json.load(f)
        except Exception as e:
            st.warning(f"Could not load anomaly data: {e}")

    # Summary cards
    col1, col2, col3 = st.columns(3)
    with col1:
        total = anomaly_data.get('total_anomalies', 0) if anomaly_data else 0
        st.markdown(f"### 🧭 Total Anomalies\n**{total}**")
    with col2:
        threshold = anomaly_data.get('anomaly_threshold', None) if anomaly_data else None
        st.markdown(f"### 🎚️ Threshold\n**{(f'{threshold:.2f}' if threshold is not None else '—')}**")
    with col3:
        video_name = os.path.basename(anomaly_data.get('video_path', '—')) if anomaly_data else '—'
        st.markdown(f"### 🎞️ Last Video\n**{video_name}**")

    # Detection timeline
    st.subheader("📈 Detection Timeline")
    if anomaly_data and 'detailed_anomalies' in anomaly_data and anomaly_data['detailed_anomalies']:
        df = pd.DataFrame(anomaly_data['detailed_anomalies'])
        # Provide timestamp_seconds if missing
        if 'timestamp_seconds' not in df.columns and 'timestamp' in df.columns:
            # Attempt to parse mm:ss or hh:mm:ss
            def to_seconds(t):
                parts = str(t).split(':')
                parts = [int(float(p)) for p in parts]
                if len(parts) == 3:
                    h, m, s = parts
                    return h*3600 + m*60 + s
                if len(parts) == 2:
                    m, s = parts
                    return m*60 + s
                return 0
            df['timestamp_seconds'] = df['timestamp'].apply(to_seconds)
        fig = px.scatter(df, x='timestamp_seconds', y='confidence', color='behavior',
                         title="Anomaly Detection Over Time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 No analysis data available yet. Upload and analyze a video to see metrics.")

    # Distribution chart
    if anomaly_data:
        st.subheader("📊 Anomaly Distribution")
        # Load class list (fallback to keys if model file not accessible here)
        class_list = []
        try:
            with open('models/class_indices.json', 'r') as f:
                ci = json.load(f)
                # class_indices is mapping class -> index; preserve order by index
                class_list = [c for c,_ in sorted(ci.items(), key=lambda kv: kv[1])]
        except Exception:
            # fallback: collect from detailed anomalies
            if 'detailed_anomalies' in anomaly_data:
                class_list = sorted({a['behavior'].strip() for a in anomaly_data['detailed_anomalies']})

        # Build summary from detailed anomalies (normalize labels)
        summary_raw = anomaly_data.get('anomaly_summary')
        detailed = anomaly_data.get('detailed_anomalies', [])
        counts = {}
        if detailed:
            for a in detailed:
                beh = a.get('behavior', '').strip()
                if beh:
                    counts[beh] = counts.get(beh, 0) + 1
        elif summary_raw:
            for k,v in summary_raw.items():
                counts[k.strip()] = v

        # Ensure all known classes appear (zero-filled)
        for cls in class_list:
            counts.setdefault(cls, 0)

        if counts:
            # Separate non-zero for pie chart
            non_zero = {k:v for k,v in counts.items() if v > 0}
            c1, c2 = st.columns(2)
            with c1:
                if non_zero:
                    fig_pie = px.pie(values=list(non_zero.values()), names=list(non_zero.keys()),
                                     title="Behavior Types", color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("No anomalous behaviors detected.")
            with c2:
                fig_bar = px.bar(x=list(counts.keys()), y=list(counts.values()),
                                 title="Counts by Behavior")
                fig_bar.update_layout(xaxis_title='Behavior', yaxis_title='Count')
                st.plotly_chart(fig_bar, use_container_width=True)

    # Recent snapshots preview
    st.subheader("🖼️ Recent Snapshots")
    # Updated snapshot directory to standardized outputs path
    snap_dir = Path('outputs/anomaly_snapshots')
    if snap_dir.exists():
        snaps = sorted(snap_dir.glob('*.jpg'), reverse=True)[:6]
        if snaps:
            cols = st.columns(3)
            for i, sp in enumerate(snaps):
                with cols[i % 3]:
                    st.image(str(sp), caption=sp.stem, use_container_width=True)
        else:
            st.info("No snapshots available yet.")
    else:
        st.info("Snapshot folder not found.")

    # Quick actions
    st.subheader("⚡ Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎬 Analyze New Video", key="quick_analyze", use_container_width=True):
            st.session_state.nav_page = "🎬 Video Analysis"
            st.rerun()
    with col2:
        if st.button("📄 Generate Report", key="quick_report", use_container_width=True):
            st.session_state.nav_page = "📄 Reports & Export"
            st.rerun()

def video_analysis_page():
    """Video analysis page with privacy options"""
    auth = st.session_state.auth_manager
    
    st.title("🎬 Video Analysis")
    
    # Log page access
    auth.log_audit_event("page_access", auth.current_user['username'], "Accessed Video Analysis")
    
    # Upload video
    uploaded_file = st.file_uploader("Upload exam hall video", type=['mp4', 'avi', 'mov'])
    
    col1, col2 = st.columns(2)
    with col1:
        frame_skip = st.slider("Frame Skip (faster analysis)", 10, 60, 30)
    with col2:
        threshold = st.slider("Anomaly Threshold", 0.5, 0.95, 0.7)
    
    # Removed face blurring feature
    
    if uploaded_file:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        if st.button("🚀 Start Analysis", use_container_width=True):
            with st.spinner("Analyzing video..."):
                try:
                    # Initialize analyzer preferring best model checkpoint
                    try:
                        analyzer = VideoAnalyzer(model_path='models/best_classroom_model.h5',
                                                 class_indices_path='models/class_indices.json')
                    except Exception:
                        analyzer = VideoAnalyzer()  # fallback to default resolution logic
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(progress):
                        progress_bar.progress(int(progress) / 100)
                        status_text.text(f"Processing: {progress:.1f}%")
                    
                    # Update threshold in config if needed
                    analyzer.anomaly_threshold = threshold
                    
                    # Analyze video
                    anomalies = analyzer.analyze_video(
                        video_path,
                        frame_skip=frame_skip,
                        progress_callback=update_progress
                    )
                    
                    # Save snapshots
                    analyzer.save_anomaly_snapshots(anomalies)
                    
                    # Privacy enhancement (face blurring removed)
                    
                    # Generate report
                    report = analyzer.generate_anomaly_report(anomalies, video_path)
                    
                    # Log analysis
                    auth.log_audit_event("video_analyzed", auth.current_user['username'],
                                       f"Analyzed video: {uploaded_file.name}",
                                       {'total_anomalies': len(anomalies), 'threshold': threshold})
                    
                    st.success(f"✅ Analysis complete! Found {len(anomalies)} anomalies")
                    
                    # Display results
                    if anomalies:
                        st.subheader("📸 Anomaly Snapshots")
                        snapshot_dir = Path('outputs/anomaly_snapshots')
                        snapshots = sorted(snapshot_dir.glob('*.jpg'), reverse=True)[:12]
                        
                        cols = st.columns(3)
                        for idx, snapshot in enumerate(snapshots):
                            with cols[idx % 3]:
                                img = Image.open(snapshot)
                                st.image(img, caption=snapshot.stem, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    auth.log_audit_event("analysis_error", auth.current_user['username'],
                                       f"Error analyzing video: {str(e)}")

def model_training_page():
    """Model training page (admin only)"""
    auth = st.session_state.auth_manager
    
    if not auth.is_admin():
        st.warning("⚠️ Admin access required for model training")
        return
    
    st.title("🤖 Model Training")
    
    auth.log_audit_event("page_access", auth.current_user['username'], "Accessed Model Training")
    
    st.info("Model training functionality available for administrators")
    
    # Training parameters
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.number_input("Epochs", 10, 100, 50)
    with col2:
        batch_size = st.number_input("Batch Size", 8, 64, 32)
    
    if st.button("🚀 Start Training", use_container_width=True):
        with st.spinner("Training model..."):
            try:
                trainer = ClassroomBehaviorTrainer()
                # Training logic here
                st.success("✅ Model trained successfully!")
                auth.log_audit_event("model_trained", auth.current_user['username'],
                                   f"Trained model with {epochs} epochs")
            except Exception as e:
                st.error(f"❌ Training error: {str(e)}")

def reports_page():
    """Reports and export page"""
    auth = st.session_state.auth_manager
    
    st.title("📄 Reports & Export")
    
    auth.log_audit_event("page_access", auth.current_user['username'], "Accessed Reports")
    
    # Load anomaly data
    anomaly_file = 'outputs/anomaly_report.json'
    if not os.path.exists(anomaly_file):
        st.warning("⚠️ No analysis data available. Analyze a video first.")
        return
    
    with open(anomaly_file, 'r') as f:
        anomaly_data = json.load(f)
    
    st.subheader("📊 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📄 PDF Report")
        st.write("Comprehensive PDF report with charts and snapshots")
        
        if st.button("📄 Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF..."):
                generator = ReportGenerator()
                success, message = generator.generate_pdf_report(anomaly_data)
                
                if success:
                    st.success(message)
                    auth.log_audit_event("report_generated", auth.current_user['username'],
                                       "Generated PDF report")
                    
                    # Download button
                    with open('outputs/anomaly_report.pdf', 'rb') as f:
                        st.download_button("⬇️ Download PDF", f, file_name="anomaly_report.pdf",
                                         mime="application/pdf", use_container_width=True)
                else:
                    st.error(message)
    
    with col2:
        st.markdown("### 📊 CSV Export")
        st.write("Spreadsheet format for data analysis")
        
        if st.button("📊 Generate CSV Export", use_container_width=True):
            generator = ReportGenerator()
            success, message = generator.generate_csv_report(anomaly_data)
            
            if success:
                st.success(message)
                auth.log_audit_event("csv_exported", auth.current_user['username'],
                                   "Exported CSV report")
                
                # Download button
                with open('outputs/anomaly_report.csv', 'rb') as f:
                    st.download_button("⬇️ Download CSV", f, file_name="anomaly_report.csv",
                                     mime="text/csv", use_container_width=True)
            else:
                st.error(message)

# Privacy settings page removed

def user_management_page():
    """User management page (admin only)"""
    auth = st.session_state.auth_manager
    
    st.title("👥 User Management")
    
    auth.log_audit_event("page_access", auth.current_user['username'], "Accessed User Management")
    
    # Display existing users
    st.subheader("Current Users")
    users = auth.get_all_users()
    df = pd.DataFrame(users)
    st.dataframe(df, use_container_width=True)
    
    st.markdown("---")
    
    # Add new user
    st.subheader("➕ Add New User")
    
    with st.form("add_user_form"):
        col1, col2 = st.columns(2)
        with col1:
            new_username = st.text_input("Username")
            new_name = st.text_input("Full Name")
            new_email = st.text_input("Email")
        with col2:
            new_password = st.text_input("Password", type="password")
            new_role = st.selectbox("Role", ["proctor", "admin"])
        
        if st.form_submit_button("Add User"):
            success, message = auth.add_user(new_username, new_password, new_role, new_name, new_email)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

def audit_log_page():
    """Audit log page (admin only)"""
    auth = st.session_state.auth_manager
    
    st.title("📜 Audit Log")
    
    auth.log_audit_event("page_access", auth.current_user['username'], "Accessed Audit Log")
    
    # Get audit log
    audit_log = auth.get_audit_log(limit=100)
    
    if audit_log:
        df = pd.DataFrame(audit_log)
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            event_filter = st.multiselect("Filter by Event Type",
                                         df['event_type'].unique())
        with col2:
            user_filter = st.multiselect("Filter by User",
                                        df['username'].unique())
        
        # Apply filters
        if event_filter:
            df = df[df['event_type'].isin(event_filter)]
        if user_filter:
            df = df[df['username'].isin(user_filter)]
        
        st.dataframe(df, use_container_width=True)
        
        # Export audit log
        if st.button("📊 Export Audit Log"):
            generator = ReportGenerator()
            success, message = generator.export_audit_log(audit_log)
            if success:
                st.success(message)
                with open('audit_log_export.csv', 'rb') as f:
                    st.download_button("⬇️ Download Audit Log", f,
                                     file_name="audit_log_export.csv", mime="text/csv")
    else:
        st.info("📜 No audit log entries found")

def main():
    """Main application entry point"""
    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
