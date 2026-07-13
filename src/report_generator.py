"""
Report Generator for Exam Hall Monitoring System
Generates PDF and CSV reports with anomaly detection results
"""

import os
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io

class ReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
    def generate_csv_report(self, anomaly_data, output_file='outputs/anomaly_report.csv'):
        """Generate CSV report from anomaly data"""
        if not anomaly_data or 'detailed_anomalies' not in anomaly_data:
            return False, "No anomaly data to export"
        
        # Convert to DataFrame
        df = pd.DataFrame(anomaly_data['detailed_anomalies'])
        
        # Add summary information as metadata rows
        summary_df = pd.DataFrame([
            {'frame_number': 'SUMMARY', 'timestamp': '', 'timestamp_seconds': '', 'behavior': '', 'confidence': ''},
            {'frame_number': 'Video Path', 'timestamp': anomaly_data.get('video_path', 'N/A'), 'timestamp_seconds': '', 'behavior': '', 'confidence': ''},
            {'frame_number': 'Analysis Time', 'timestamp': anomaly_data.get('analysis_timestamp', 'N/A'), 'timestamp_seconds': '', 'behavior': '', 'confidence': ''},
            {'frame_number': 'Total Anomalies', 'timestamp': str(anomaly_data.get('total_anomalies', 0)), 'timestamp_seconds': '', 'behavior': '', 'confidence': ''},
            {'frame_number': '', 'timestamp': '', 'timestamp_seconds': '', 'behavior': '', 'confidence': ''},
        ])
        
        # Combine summary and detailed data
        final_df = pd.concat([summary_df, df], ignore_index=True)
        
        # Export to CSV
        final_df.to_csv(output_file, index=False)
        return True, f"CSV report saved to {output_file}"
    
    def generate_pdf_report(self, anomaly_data, snapshot_dir='outputs/anomaly_snapshots', output_file='outputs/anomaly_report.pdf'):
        """Generate comprehensive PDF report"""
        if not anomaly_data:
            return False, "No anomaly data to export"
        
        # Create PDF document
        doc = SimpleDocTemplate(output_file, pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        
        # Container for elements
        elements = []
        
        # Title
        title = Paragraph("Exam Hall Anomaly Detection Report", self.title_style)
        elements.append(title)
        elements.append(Spacer(1, 12))
        
        # Report metadata
        metadata_style = ParagraphStyle(
            'Metadata',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#666666')
        )
        
        report_info = f"""
        <b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <b>Video Path:</b> {anomaly_data.get('video_path', 'N/A')}<br/>
        <b>Analysis Timestamp:</b> {anomaly_data.get('analysis_timestamp', 'N/A')}<br/>
        <b>Total Anomalies Detected:</b> {anomaly_data.get('total_anomalies', 0)}
        """
        elements.append(Paragraph(report_info, metadata_style))
        elements.append(Spacer(1, 20))
        
        # Summary section
        section_title = ParagraphStyle(
            'SectionTitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#764ba2'),
            spaceAfter=12
        )
        
        elements.append(Paragraph("Anomaly Summary", section_title))
        
        # Summary table
        if 'anomaly_summary' in anomaly_data:
            summary_data = [['Behavior Type', 'Count', 'Percentage']]
            total = anomaly_data.get('total_anomalies', 0)
            
            for behavior, count in anomaly_data['anomaly_summary'].items():
                percentage = (count / total * 100) if total > 0 else 0
                summary_data.append([behavior, str(count), f"{percentage:.1f}%"])
            
            summary_table = Table(summary_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(summary_table)
            elements.append(Spacer(1, 20))
        
        # Detailed anomalies section
        elements.append(Paragraph("Detailed Anomaly Log", section_title))
        
        if 'detailed_anomalies' in anomaly_data:
            # Create detailed table with limited entries for PDF
            detailed_data = [['Frame #', 'Timestamp', 'Behavior', 'Confidence']]
            
            for anomaly in anomaly_data['detailed_anomalies'][:50]:  # Limit to first 50
                detailed_data.append([
                    str(anomaly.get('frame_number', 'N/A')),
                    anomaly.get('timestamp', 'N/A'),
                    anomaly.get('behavior', 'N/A'),
                    f"{anomaly.get('confidence', 0):.2%}"
                ])
            
            if len(anomaly_data['detailed_anomalies']) > 50:
                detailed_data.append(['...', '...', f'({len(anomaly_data["detailed_anomalies"]) - 50} more entries)', '...'])
            
            detailed_table = Table(detailed_data, colWidths=[1*inch, 1.5*inch, 2.5*inch, 1.5*inch])
            detailed_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTSIZE', (0, 1), (-1, -1), 8)
            ]))
            
            elements.append(detailed_table)
            elements.append(PageBreak())
        
        # Add sample snapshots if available
        elements.append(Paragraph("Sample Anomaly Snapshots", section_title))
        elements.append(Spacer(1, 12))
        
        if os.path.exists(snapshot_dir):
            snapshots = sorted(Path(snapshot_dir).glob('*.jpg'))[:6]  # First 6 snapshots
            
            if snapshots:
                # Create 2x3 grid of images
                snapshot_data = []
                row = []
                
                for i, snapshot_path in enumerate(snapshots):
                    try:
                        img = Image(str(snapshot_path), width=2*inch, height=1.5*inch)
                        row.append(img)
                        
                        if len(row) == 2:
                            snapshot_data.append(row)
                            row = []
                    except Exception as e:
                        print(f"Error adding image {snapshot_path}: {e}")
                
                if row:  # Add remaining images
                    snapshot_data.append(row)
                
                if snapshot_data:
                    snapshot_table = Table(snapshot_data, colWidths=[3*inch, 3*inch])
                    snapshot_table.setStyle(TableStyle([
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('LEFTPADDING', (0, 0), (-1, -1), 6),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                        ('TOPPADDING', (0, 0), (-1, -1), 6),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ]))
                    elements.append(snapshot_table)
            else:
                elements.append(Paragraph("No snapshots available", self.styles['Normal']))
        
        # Footer
        elements.append(Spacer(1, 30))
        footer_text = """
        <para align=center>
        <font size=8 color="#666666">
        Generated by Exam Hall Anomaly Detection System<br/>
        For official use only - Confidential
        </font>
        </para>
        """
        elements.append(Paragraph(footer_text, self.styles['Normal']))
        
        # Build PDF
        try:
            doc.build(elements)
            return True, f"PDF report saved to {output_file}"
        except Exception as e:
            return False, f"Error generating PDF: {str(e)}"
    
    def export_audit_log(self, audit_data, output_file='outputs/audit_log_export.csv'):
        """Export audit log to CSV"""
        if not audit_data:
            return False, "No audit log data to export"
        
        df = pd.DataFrame(audit_data)
        df.to_csv(output_file, index=False)
        return True, f"Audit log exported to {output_file}"
