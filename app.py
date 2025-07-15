"""
Main Streamlit application for Hair Follicle Segmentation & Triangle Detection.
Uses a modular architecture with the inference-sdk for Roboflow integration.
"""

import streamlit as st
import numpy as np
from PIL import Image
import os

# Import our modular components
from src import (
    # Configuration
    APP_TITLE, APP_DESCRIPTION, PAGE_ICON, SUPPORTED_IMAGE_FORMATS,
    ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_WORKFLOW_ID,
    CONFIDENCE_THRESHOLD, USE_CACHE,
    
    # Classes and functions
    ImageProcessor, save_temp_image,
    convert_rgb_to_bgr, convert_bgr_to_rgb, format_confidence, validate_workspace_name, validate_workflow_id, get_image_info
)


def setup_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=PAGE_ICON,
        layout="wide"
    )


def display_configuration():
    """Display the current configuration loaded from environment variables."""
    with st.sidebar:
        st.header("🔧 Configuration")
        st.info("Configuration loaded from `.env` file")
        
        st.subheader("Workflow Settings")
        st.text(f"Workspace: {ROBOFLOW_WORKSPACE}")
        st.text(f"Workflow ID: {ROBOFLOW_WORKFLOW_ID}")
        st.text(f"API Key: {'*' * 8}...{ROBOFLOW_API_KEY[-4:] if len(ROBOFLOW_API_KEY) > 4 else '****'}")
        
        st.subheader("Processing Settings")
        st.text(f"Confidence: {CONFIDENCE_THRESHOLD}")
        st.text(f"Use Cache: {USE_CACHE}")
        
        st.markdown("---")
        st.caption("To modify configuration, edit your `.env` file and restart the app.")

   


def process_uploaded_image(uploaded_file):
    """Process the uploaded image with the configured parameters from environment."""
    
    # Load and display original image
    image = Image.open(uploaded_file)
    image_array = np.array(image)
    
    # Convert to BGR for OpenCV processing
    image_bgr = convert_rgb_to_bgr(image_array)
    
    # Display original image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
        # Show image info
        img_info = get_image_info(image_bgr)
        st.caption(f"Size: {img_info['width']}×{img_info['height']} | Channels: {img_info['channels']}")
    
    # Validate configuration from environment
    if not ROBOFLOW_API_KEY:
        with col2:
            st.error("❌ ROBOFLOW_API_KEY not found in environment variables.")
            st.info("Please set ROBOFLOW_API_KEY in your .env file.")
        return
    
    if not ROBOFLOW_WORKSPACE:
        with col2:
            st.error("❌ ROBOFLOW_WORKSPACE not found in environment variables.")
            st.info("Please set ROBOFLOW_WORKSPACE in your .env file.")
        return
        
    if not ROBOFLOW_WORKFLOW_ID:
        with col2:
            st.error("❌ ROBOFLOW_WORKFLOW_ID not found in environment variables.")
            st.info("Please set ROBOFLOW_WORKFLOW_ID in your .env file.")
        return
        
    if not validate_workspace_name(ROBOFLOW_WORKSPACE):
        with col2:
            st.error("⚠️ Invalid workspace name in environment variables")
        return
        
    if not validate_workflow_id(ROBOFLOW_WORKFLOW_ID):
        with col2:
            st.error("⚠️ Invalid workflow ID in environment variables")
        return
    
    # Process button
    if st.button("🚀 Process Image", type="primary", use_container_width=True):
        try:
            with st.spinner("Processing image with Roboflow..."):
                # Save uploaded image to temporary file
                temp_path = save_temp_image(image)
                
                # Initialize image processor
                processor = ImageProcessor(api_key=ROBOFLOW_API_KEY)
                
                # Process the image
                processed_image, analysis_results, detections = processor.process_image_file(
                    temp_path, ROBOFLOW_WORKSPACE, ROBOFLOW_WORKFLOW_ID, CONFIDENCE_THRESHOLD, USE_CACHE
                )
                
                # Convert back to RGB for display
                processed_image_rgb = convert_bgr_to_rgb(processed_image)
                
                # Store results in session state for PDF generation
                img_info = get_image_info(image_bgr)
                st.session_state.analysis_data = {
                    'detections': detections,
                    'analysis_results': analysis_results,
                    'image_info': img_info,
                    'processed_image': processed_image_rgb
                }
                
                # Display processed image
                with col2:
                    st.subheader("Processed Image")
                    st.image(processed_image_rgb, use_container_width=True)
                
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.info("Please check your API key, model ID, and internet connection.")
    else:
        with col2:
            st.info("👆 Click the 'Process Image' button to analyze the image.")



def render_analysis_results(detections, analysis_results, image_info=None):
    """Render the comprehensive hair analysis report."""
    from src import generate_hair_analysis_report, generate_hair_analysis_pdf
    import datetime
    
    st.subheader("📊 Hair Follicle Analysis Report")
    
    if not detections:
        st.warning("⚠️ No hair follicles detected in the image.")
        return
    
    # Generate comprehensive report
    report = generate_hair_analysis_report(detections, analysis_results, image_info or {})
    
    if 'error' in report:
        st.error(f"❌ Error generating report: {report['error']}")
        return
    
    # Header metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Hair Follicles", report['total_count'])
    with col2:
        st.metric("Successful Triangles", report['triangle_analysis']['successful_triangles'])
    with col3:
        st.metric("Success Rate", f"{report['triangle_analysis']['success_rate']:.1f}%")
    with col4:
        avg_conf = report['confidence_stats']['overall']['average']
        st.metric("Avg Confidence", f"{avg_conf:.1%}")
    
    # PDF Download Button
    st.markdown("---")
    col_download1, col_download2, col_download3 = st.columns([1, 1, 1])
    with col_download2:
        if st.button("📄 Download PDF Report", type="primary", use_container_width=True):
            try:
                # Generate PDF
                with st.spinner("Generating PDF report..."):
                    pdf_data = generate_hair_analysis_pdf(report, image_info or {})
                
                # Create filename with timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"hair_follicle_analysis_{timestamp}.pdf"
                
                # Provide download
                st.download_button(
                    label="⬇️ Click to Download PDF",
                    data=pdf_data,
                    file_name=filename,
                    mime="application/pdf",
                    use_container_width=True
                )
                
                st.success("✅ PDF report generated successfully!")
                
            except Exception as e:
                st.error(f"❌ Error generating PDF: {str(e)}")
                st.info("Please ensure all dependencies are installed and try again.")
    
    st.markdown("---")
    
    # Hair Count Distribution (similar to the medical report)
    st.subheader("🔬 Hair Follicle Count by Strength")
    
    # Create a nice visual display similar to the medical report
    count_col1, count_col2 = st.columns([1, 1])
    
    with count_col1:
        st.markdown("### Hair Count Distribution")
        
        # Display counts with color indicators and bars
        strong_count = report['class_counts']['strong']
        medium_count = report['class_counts']['medium'] 
        weak_count = report['class_counts']['weak']
        total = report['total_count']
        
        # Strong hair follicles
        strong_pct = report['class_percentages']['strong']
        st.markdown(f"""
        <div style="background-color: #e8f5e8; padding: 10px; margin: 5px 0; border-radius: 5px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span><strong>🟢 Strong Hair Follicles</strong></span>
                <span><strong>{strong_count}</strong></span>
            </div>
            <div style="background-color: #4CAF50; height: 20px; width: {strong_pct}%; margin-top: 5px; border-radius: 3px;"></div>
            <small>{strong_pct:.1f}% of total</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Medium hair follicles
        medium_pct = report['class_percentages']['medium']
        st.markdown(f"""
        <div style="background-color: #fffbf0; padding: 10px; margin: 5px 0; border-radius: 5px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span><strong>🟡 Medium Hair Follicles</strong></span>
                <span><strong>{medium_count}</strong></span>
            </div>
            <div style="background-color: #FFC107; height: 20px; width: {medium_pct}%; margin-top: 5px; border-radius: 3px;"></div>
            <small>{medium_pct:.1f}% of total</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Weak hair follicles
        weak_pct = report['class_percentages']['weak']
        st.markdown(f"""
        <div style="background-color: #ffeaea; padding: 10px; margin: 5px 0; border-radius: 5px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span><strong>🔴 Weak Hair Follicles</strong></span>
                <span><strong>{weak_count}</strong></span>
            </div>
            <div style="background-color: #f44336; height: 20px; width: {weak_pct}%; margin-top: 5px; border-radius: 3px;"></div>
            <small>{weak_pct:.1f}% of total</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Total
        st.markdown(f"""
        <div style="background-color: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; border: 2px solid #333;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span><strong>📊 Total</strong></span>
                <span><strong>{total}</strong></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with count_col2:
        st.markdown("### Analysis Metrics")
        
        # Key ratios and metrics
        metrics_data = [
            ("Terminal-Vellus Ratio", report['ratios']['terminal_vellus']),
            ("Strong-Medium Ratio", report['ratios']['strong_medium']),
            ("Triangle Success Rate", f"{report['triangle_analysis']['success_rate']:.1f}%"),
            ("Strong Follicle %", f"{report['ratios']['strong_percentage']:.1f}%"),
            ("Medium Follicle %", f"{report['ratios']['medium_percentage']:.1f}%"),
            ("Weak Follicle %", f"{report['ratios']['weak_percentage']:.1f}%")
        ]
        
        # Display metrics using Streamlit's native table
        import pandas as pd
        
        try:
            # Create DataFrame from metrics data
            metrics_df = pd.DataFrame(metrics_data[1:], columns=metrics_data[0])  # Skip header row
            
            # Display as a clean table
            st.dataframe(
                metrics_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Metric": st.column_config.TextColumn("Metric", width="medium"),
                    "Value": st.column_config.TextColumn("Value", width="medium")
                }
            )
        except Exception as e:
            # Fallback to simple display if DataFrame fails
            st.write("**Analysis Metrics:**")
            for metric_name, metric_value in metrics_data[1:]:  # Skip header
                st.write(f"• **{metric_name}**: {metric_value}")
    
    st.markdown("---")
    
    # Confidence Statistics
    st.subheader("📈 Detection Confidence Statistics")
    
    conf_col1, conf_col2, conf_col3 = st.columns(3)
    
    with conf_col1:
        st.markdown("**🟢 Strong Hair Follicles**")
        strong_conf = report['confidence_stats']['strong']
        if strong_conf['count'] > 0:
            st.write(f"• Average: {strong_conf['average']:.1%}")
            st.write(f"• Range: {strong_conf['min']:.1%} - {strong_conf['max']:.1%}")
            st.write(f"• Count: {strong_conf['count']}")
        else:
            st.write("No strong follicles detected")
    
    with conf_col2:
        st.markdown("**🟡 Medium Hair Follicles**")
        medium_conf = report['confidence_stats']['medium']
        if medium_conf['count'] > 0:
            st.write(f"• Average: {medium_conf['average']:.1%}")
            st.write(f"• Range: {medium_conf['min']:.1%} - {medium_conf['max']:.1%}")
            st.write(f"• Count: {medium_conf['count']}")
        else:
            st.write("No medium follicles detected")
    
    with conf_col3:
        st.markdown("**🔴 Weak Hair Follicles**")
        weak_conf = report['confidence_stats']['weak']
        if weak_conf['count'] > 0:
            st.write(f"• Average: {weak_conf['average']:.1%}")
            st.write(f"• Range: {weak_conf['min']:.1%} - {weak_conf['max']:.1%}")
            st.write(f"• Count: {weak_conf['count']}")
        else:
            st.write("No weak follicles detected")
    
    # Summary note
    st.info(f"""
    📋 **Analysis Summary:**
    Detected {report['total_count']} hair follicles with {report['triangle_analysis']['successful_triangles']} successful directional analyses. 
    The dominant hair type is **{max(report['class_counts'], key=report['class_counts'].get)}** with {max(report['class_counts'].values())} follicles.
    """)
    
    # Optional: Detailed breakdown (collapsible)
    with st.expander("🔍 Detailed Detection Breakdown"):
        for i, result in enumerate(analysis_results):
            st.write(f"**Detection {i+1}:**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"- Detection ID: {result['detection_id']}")
                st.write(f"- Contour Points: {result['contour_points']}")
                if 'confidence' in result:
                    st.write(f"- Confidence: {format_confidence(result['confidence'])}")
            
            with col2:
                if 'class' in result:
                    from src import get_hair_strand_class_name
                    class_name = get_hair_strand_class_name(result.get('class_id', 0), result.get('class', 'unknown'))
                    
                    # Show color indicator for hair strand type
                    color_indicator = {
                        'strong': '🟢',
                        'medium': '🟡', 
                        'weak': '🔴'
                    }.get(class_name, '⚪')
                    
                    st.write(f"- Hair Strand: {color_indicator} {class_name}")
                if 'class_id' in result:
                    st.write(f"- Class ID: {result['class_id']}")
                if 'mask_pixels' in result:
                    st.write(f"- Mask Pixels: {result['mask_pixels']:,}")
            
            st.divider()


def main():
    """Main application function."""
    setup_page()
    
    # Header
    st.title(APP_TITLE)
    st.markdown(APP_DESCRIPTION)
    
    # Display configuration in sidebar
    display_configuration()
    
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Choose an image file", 
        type=SUPPORTED_IMAGE_FORMATS,
        help="Upload an image for hair follicle analysis"
    )
    
    # Clear session state if a new file is uploaded
    if uploaded_file is not None:
        # Check if this is a different file than what we processed before
        current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if 'last_file_id' not in st.session_state or st.session_state.last_file_id != current_file_id:
            # Clear previous analysis data for new file
            if 'analysis_data' in st.session_state:
                del st.session_state.analysis_data
            st.session_state.last_file_id = current_file_id
    
    # Process uploaded image
    if uploaded_file is not None:
        process_uploaded_image(uploaded_file)
    else:
        st.info("👆 Please upload an image to get started.")
    


if __name__ == "__main__":
    main() 