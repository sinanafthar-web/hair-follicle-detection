"""
Utility functions for the Hair Follicle Segmentation app.
Contains drawing functions and other helper utilities.
"""

import cv2
import numpy as np
from typing import Tuple

from .config import (
    COLORS, TRIANGLE_THICKNESS, ARROW_THICKNESS, POINT_RADIUS, 
    CONTOUR_THICKNESS, ARROW_TIP_LENGTH
)


def draw_arrow(image: np.ndarray, start_point: np.ndarray, end_point: np.ndarray, 
               color: Tuple[int, int, int] = COLORS['arrow'], 
               thickness: int = ARROW_THICKNESS) -> np.ndarray:
    """
    Draw an arrow from start_point to end_point.
    
    Args:
        image: Input image
        start_point: Starting point of the arrow
        end_point: Ending point of the arrow
        color: BGR color tuple
        thickness: Line thickness
        
    Returns:
        Image with arrow drawn
    """
    start_point = tuple(map(int, start_point))
    end_point = tuple(map(int, end_point))
    
    # Draw the arrow
    cv2.arrowedLine(image, start_point, end_point, color, thickness, tipLength=ARROW_TIP_LENGTH)
    
    return image


def draw_triangle(image: np.ndarray, triangle: np.ndarray, 
                 color: Tuple[int, int, int] = COLORS['triangle'], 
                 thickness: int = TRIANGLE_THICKNESS) -> np.ndarray:
    """
    Draw a triangle on the image.
    
    Args:
        image: Input image
        triangle: Triangle vertices as numpy array
        color: BGR color tuple
        thickness: Line thickness
        
    Returns:
        Image with triangle drawn
    """
    cv2.polylines(image, [triangle.astype(np.int32)], True, color, thickness)
    return image


def draw_point(image: np.ndarray, point: np.ndarray, 
               color: Tuple[int, int, int], radius: int = POINT_RADIUS) -> np.ndarray:
    """
    Draw a point (filled circle) on the image.
    
    Args:
        image: Input image
        point: Point coordinates
        color: BGR color tuple
        radius: Circle radius
        
    Returns:
        Image with point drawn
    """
    cv2.circle(image, tuple(map(int, point)), radius, color, -1)
    return image


def draw_contour(image: np.ndarray, contour: np.ndarray, 
                color: Tuple[int, int, int] = COLORS['contour'], 
                thickness: int = CONTOUR_THICKNESS) -> np.ndarray:
    """
    Draw a contour on the image.
    
    Args:
        image: Input image
        contour: Contour points
        color: BGR color tuple
        thickness: Line thickness
        
    Returns:
        Image with contour drawn
    """
    cv2.polylines(image, [contour], True, color, thickness)
    return image


def draw_visualization_elements(image: np.ndarray, triangle: np.ndarray, 
                               middle_point: np.ndarray, apex: np.ndarray, 
                               contour: np.ndarray) -> np.ndarray:
    """
    Draw all visualization elements for a single detection.
    
    Args:
        image: Input image
        triangle: Triangle vertices
        middle_point: Middle point of shortest side
        apex: Apex point
        contour: Original contour
        
    Returns:
        Image with all elements drawn
    """
    # Draw the triangle
    image = draw_triangle(image, triangle)
    
    # Draw arrow from middle of shortest side to apex
    image = draw_arrow(image, middle_point, apex)
    
    # Draw points for visualization
    image = draw_point(image, middle_point, COLORS['middle_point'])
    image = draw_point(image, apex, COLORS['apex_point'])
    
    # Draw original segmentation contour
    image = draw_contour(image, contour)
    
    return image


def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to RGB for display in Streamlit.
    
    Args:
        image: BGR image array
        
    Returns:
        RGB image array
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to BGR for OpenCV processing.
    
    Args:
        image: RGB image array
        
    Returns:
        BGR image array
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def format_confidence(confidence: float) -> str:
    """
    Format confidence value as percentage string.
    
    Args:
        confidence: Confidence value (0-1)
        
    Returns:
        Formatted percentage string
    """
    return f"{confidence:.1%}"


def validate_workspace_name(workspace_name: str) -> bool:
    """
    Validate that workspace name is not empty.
    
    Args:
        workspace_name: Workspace name string
        
    Returns:
        True if valid, False otherwise
    """
    return bool(workspace_name and workspace_name.strip())


def validate_workflow_id(workflow_id: str) -> bool:
    """
    Validate that workflow ID is not empty.
    
    Args:
        workflow_id: Workflow ID string
        
    Returns:
        True if valid, False otherwise
    """
    return bool(workflow_id and workflow_id.strip())


def get_image_info(image: np.ndarray) -> dict:
    """
    Get basic information about an image.
    
    Args:
        image: Input image array
        
    Returns:
        Dictionary with image information
    """
    if image is None:
        return {}
    
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1
    
    return {
        'width': width,
        'height': height,
        'channels': channels,
        'shape': image.shape,
        'dtype': str(image.dtype)
    } 