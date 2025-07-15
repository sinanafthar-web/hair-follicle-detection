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


def render_instructions():
    """Render the instructions section."""
    with st.expander("📋 Instructions"):
        st.markdown("""
        ### Setup Steps
        1. **Get Roboflow Account**: Create an account at [roboflow.com](https://roboflow.com)
        2. **Create Workflow**: Set up a workflow for hair follicle detection (or use an existing one)
        3. **Get Credentials**: 
           - Find your API key in your Roboflow account settings
           - Get your workspace name (e.g., `ranaudio`)
           - Get your workflow ID (e.g., `small-object-detection-sahi`)
        4. **Configure Environment**: Create a `.env` file in the project root with:
           ```
           ROBOFLOW_API_KEY=your_api_key_here
           ROBOFLOW_WORKSPACE=your_workspace_name
           ROBOFLOW_WORKFLOW_ID=your_workflow_id
           CONFIDENCE_THRESHOLD=0.4
           USE_CACHE=True
           ```
        5. **Upload Image**: Choose an image file to analyze
        6. **Process**: Click the "Process Image" button to run the analysis
        
        ### What This App Does
        - 🔬 Uses your trained Roboflow model to segment objects in the image
        - 📐 Finds the smallest triangle that encloses each segmented object
        - ➡️ Calculates the shortest side of each triangle and draws arrows to the opposite apex
        - 🎨 Provides visual overlays with color-coded elements:
          - **🔵 Blue triangles**: Smallest enclosing triangles
          - **🟢 Green arrows**: Direction from shortest side midpoint to apex
          - **🟡 Yellow dots**: Middle points of shortest sides
          - **🔴 Red dots**: Triangle apex points
          - **🟠 Cyan lines**: Original segmentation boundaries
        
        ### Example Usage
        ```python
        from inference_sdk import InferenceHTTPClient
        
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key="1SxAyEbpaNdwrNSbmDon"
        )
        
        result = client.run_workflow(
            workspace_name="ranaudio",
            workflow_id="small-object-detection-sahi",
            images={
                "image": "YOUR_IMAGE.jpg"
            },
            use_cache=True
        )
        ```
        """)


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
        st.image(image, use_column_width=True)
        
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
                
                # Display processed image
                with col2:
                    st.subheader("Processed Image")
                    st.image(processed_image_rgb, use_column_width=True)
                
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                # Display analysis results
                render_analysis_results(detections, analysis_results)
                
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.info("Please check your API key, model ID, and internet connection.")
    else:
        with col2:
            st.info("👆 Click the 'Process Image' button to analyze the image.")


def render_analysis_results(detections, analysis_results):
    """Render the analysis results section."""
    st.subheader("📊 Analysis Results")
    
    num_detections = len(detections)
    num_triangles = len(analysis_results)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Objects Detected", num_detections)
    with col2:
        st.metric("Triangles Generated", num_triangles)
    with col3:
        success_rate = (num_triangles / num_detections * 100) if num_detections > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    if num_detections > 0:
        st.success(f"✅ Successfully processed {num_detections} objects and generated {num_triangles} triangular analyses!")
        
        # Detailed results
        if analysis_results:
            with st.expander("🔍 Detailed Analysis Results"):
                for i, result in enumerate(analysis_results):
                    st.write(f"**Triangle {i+1}:**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"- Detection ID: {result['detection_id']}")
                        st.write(f"- Contour Points: {result['contour_points']}")
                        if 'confidence' in result:
                            st.write(f"- Confidence: {format_confidence(result['confidence'])}")
                    
                    with col2:
                        if 'class_id' in result:
                            st.write(f"- Class ID: {result['class_id']}")
                        if 'mask_pixels' in result:
                            st.write(f"- Mask Pixels: {result['mask_pixels']:,}")
                    
                    st.divider()
        
        # Additional statistics
        if detections and len(detections) > 0:
            confidences = [det.get('confidence', 0.0) for det in detections]
            if confidences:
                avg_confidence = float(np.mean(confidences))
                st.info(f"📈 Average Detection Confidence: {format_confidence(avg_confidence)}")
            
    else:
        st.warning("⚠️ No objects detected in the image. Try adjusting the confidence threshold or using a different image.")


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
    
    # Process uploaded image
    if uploaded_file is not None:
        process_uploaded_image(uploaded_file)
    else:
        st.info("👆 Please upload an image to get started.")
    
    # Instructions
    render_instructions()


if __name__ == "__main__":
    main() 