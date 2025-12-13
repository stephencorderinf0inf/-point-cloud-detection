"""
Image quality analysis module.
"""
import cv2
import numpy as np

def analyze_image_quality(frame):
    """
    Analyze image quality metrics.
    
    Args:
        frame: Input video frame
    
    Returns:
        dict: Quality metrics (sharpness, brightness, contrast, status)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate sharpness using Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness_variance = laplacian.var()
    
    # ADJUSTED: Normalize for low-quality cameras (typical range: 50-500 instead of 0-2000)
    sharpness = min(sharpness_variance / 500.0, 1.0)  # Changed from 2000 to 500
    
    # Calculate brightness (mean intensity)
    brightness = gray.mean() / 255.0
    
    # Calculate contrast (standard deviation)
    contrast = gray.std() / 128.0
    
    # Overall quality score (weighted average)
    overall_score = (sharpness * 0.4) + (brightness * 0.3) + (contrast * 0.3)
    
    # Determine status based on overall score
    if overall_score >= 0.7:
        status = "Excellent"
    elif overall_score >= 0.5:
        status = "Good"
    elif overall_score >= 0.3:
        status = "Fair"
    else:
        status = "Poor"
    
    return {
        "sharpness": sharpness,
        "brightness": brightness,
        "contrast": contrast,
        "overall_score": overall_score,
        "status": status
    }