#!/usr/bin/env python3
"""
Example usage of the inference-sdk with workflows for segmentation.
This demonstrates the core approach used in the Streamlit app.
"""

import os
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
import supervision as sv
import cv2

def main():
    # Load environment variables
    load_dotenv()
    
    # Configuration from environment variables
    api_key = os.getenv("ROBOFLOW_API_KEY")
    workspace_name = os.getenv("ROBOFLOW_WORKSPACE")
    workflow_id = os.getenv("ROBOFLOW_WORKFLOW_ID")
    use_cache = os.getenv("USE_CACHE", "True").lower() == "true"
    
    # Validate configuration
    if not api_key:
        print("Error: ROBOFLOW_API_KEY not found in environment variables.")
        print("Please set ROBOFLOW_API_KEY in your .env file.")
        return
    
    if not workspace_name:
        print("Error: ROBOFLOW_WORKSPACE not found in environment variables.")
        print("Please set ROBOFLOW_WORKSPACE in your .env file.")
        return
        
    if not workflow_id:
        print("Error: ROBOFLOW_WORKFLOW_ID not found in environment variables.")
        print("Please set ROBOFLOW_WORKFLOW_ID in your .env file.")
        return
    
    # Load image (replace with your image path)
    image_file = "your_image.jpg"
    
    # Check if image exists
    try:
        image = cv2.imread(image_file)
        if image is None:
            raise FileNotFoundError(f"Could not load image from {image_file}")
    except Exception as e:
        print(f"Error loading image: {e}")
        print("Please make sure the image file exists and update the path.")
        return
    
    try:
        # Initialize the inference client
        print("Initializing Roboflow client...")
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key
        )
        
        # Run workflow
        print(f"Running workflow: {workspace_name}/{workflow_id}")
        result = client.run_workflow(
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            images={
                "image": image_file
            },
            use_cache=use_cache
        )
        
        # Load results into supervision Detections
        detections = sv.Detections.from_inference(result)
        
        print(f"Found {len(detections)} detections")
        
        # Create annotators
        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        mask_annotator = sv.MaskAnnotator()
        
        # Annotate the image
        annotated_image = image.copy()
        
        # Add masks if available
        if detections.mask is not None:
            annotated_image = mask_annotator.annotate(
                scene=annotated_image, detections=detections)
            print("Added segmentation masks")
        
        # Add bounding boxes
        annotated_image = bounding_box_annotator.annotate(
            scene=annotated_image, detections=detections)
        print("Added bounding boxes")
        
        # Add labels
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections)
        print("Added labels")
        
        # Save result
        output_path = "annotated_result.jpg"
        cv2.imwrite(output_path, annotated_image)
        print(f"Saved annotated image to: {output_path}")
        
        # Display detection information
        if len(detections) > 0:
            print("\nDetection Details:")
            for i in range(len(detections)):
                print(f"  Detection {i+1}:")
                if detections.class_id is not None and i < len(detections.class_id):
                    print(f"    Class ID: {detections.class_id[i]}")
                if detections.confidence is not None and i < len(detections.confidence):
                    print(f"    Confidence: {detections.confidence[i]:.2%}")
        
        # Display using supervision (if available)
        try:
            print("Displaying image...")
            sv.plot_image(annotated_image)
        except Exception as display_error:
            print(f"Note: Could not display image ({display_error}). Check annotated_result.jpg for output.")
            
    except Exception as e:
        print(f"Error during workflow execution: {e}")
        print("Please check:")
        print("1. Your API key is correct")
        print("2. Your workspace name exists")
        print("3. Your workflow ID exists and is accessible")
        print("4. Your internet connection")
        print("5. The workflow is properly configured")

if __name__ == "__main__":
    main() 