"""Image quality assessment utilities."""

import cv2
import numpy as np
from typing import Tuple
from pathlib import Path
from PIL import Image

from ..core.models import QualityLevel


class QualityAssessor:
    """Assesses image quality for training data."""
    
    def __init__(self):
        """Initialize the quality assessor."""
        self.min_resolution = (256, 256)
        self.preferred_resolution = (512, 512)
        self.high_resolution = (1024, 1024)
    
    def calculate_quality_score(self, image_path: str) -> float:
        """
        Calculate overall image quality score (0.0 to 1.0).
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Quality score from 0.0 (poor) to 1.0 (excellent)
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return 0.0
            
            # Calculate individual quality metrics
            resolution_score = self._calculate_resolution_score(image)
            sharpness_score = self._calculate_sharpness_score(image)
            brightness_score = self._calculate_brightness_score(image)
            contrast_score = self._calculate_contrast_score(image)
            
            # Weighted average of quality metrics
            weights = {
                'resolution': 0.3,
                'sharpness': 0.4,
                'brightness': 0.15,
                'contrast': 0.15
            }
            
            quality_score = (
                resolution_score * weights['resolution'] +
                sharpness_score * weights['sharpness'] +
                brightness_score * weights['brightness'] +
                contrast_score * weights['contrast']
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            print(f"Error calculating quality for {image_path}: {e}")
            return 0.0
    
    def _calculate_resolution_score(self, image: np.ndarray) -> float:
        """Calculate resolution score based on image dimensions."""
        height, width = image.shape[:2]
        total_pixels = height * width
        
        # Define pixel thresholds
        min_pixels = self.min_resolution[0] * self.min_resolution[1]
        preferred_pixels = self.preferred_resolution[0] * self.preferred_resolution[1]
        high_pixels = self.high_resolution[0] * self.high_resolution[1]
        
        if total_pixels >= high_pixels:
            return 1.0
        elif total_pixels >= preferred_pixels:
            # Linear interpolation between preferred and high
            ratio = (total_pixels - preferred_pixels) / (high_pixels - preferred_pixels)
            return 0.8 + 0.2 * ratio
        elif total_pixels >= min_pixels:
            # Linear interpolation between min and preferred
            ratio = (total_pixels - min_pixels) / (preferred_pixels - min_pixels)
            return 0.4 + 0.4 * ratio
        else:
            # Below minimum resolution
            ratio = total_pixels / min_pixels
            return max(0.1, 0.4 * ratio)
    
    def _calculate_sharpness_score(self, image: np.ndarray) -> float:
        """Calculate sharpness score using Laplacian variance."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize laplacian variance to 0-1 score
            # These thresholds are empirically determined
            if laplacian_var >= 1000:
                return 1.0
            elif laplacian_var >= 500:
                return 0.8 + 0.2 * ((laplacian_var - 500) / 500)
            elif laplacian_var >= 100:
                return 0.4 + 0.4 * ((laplacian_var - 100) / 400)
            else:
                return max(0.1, 0.4 * (laplacian_var / 100))
                
        except Exception:
            return 0.5
    
    def _calculate_brightness_score(self, image: np.ndarray) -> float:
        """Calculate brightness score (penalize over/under exposed images)."""
        try:
            # Convert to grayscale and calculate mean brightness
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray) / 255.0
            
            # Optimal brightness is around 0.4-0.7 (100-180 on 0-255 scale)
            if 0.4 <= mean_brightness <= 0.7:
                return 1.0
            elif 0.2 <= mean_brightness < 0.4:
                # Too dark
                return 0.5 + 0.5 * ((mean_brightness - 0.2) / 0.2)
            elif 0.7 < mean_brightness <= 0.9:
                # Too bright
                return 0.5 + 0.5 * ((0.9 - mean_brightness) / 0.2)
            else:
                # Very dark or very bright
                return 0.2
                
        except Exception:
            return 0.5
    
    def _calculate_contrast_score(self, image: np.ndarray) -> float:
        """Calculate contrast score using standard deviation."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            contrast = np.std(gray) / 255.0
            
            # Good contrast is typically 0.15-0.4
            if 0.15 <= contrast <= 0.4:
                return 1.0
            elif 0.08 <= contrast < 0.15:
                # Low contrast
                return 0.4 + 0.6 * ((contrast - 0.08) / 0.07)
            elif 0.4 < contrast <= 0.6:
                # High contrast
                return 0.6 + 0.4 * ((0.6 - contrast) / 0.2)
            else:
                # Very low or very high contrast
                return 0.3
                
        except Exception:
            return 0.5
    
    def get_quality_level(self, quality_score: float) -> QualityLevel:
        """
        Convert quality score to quality level enum.
        
        Args:
            quality_score: Quality score from 0.0 to 1.0
            
        Returns:
            QualityLevel enum value
        """
        if quality_score >= 0.75:
            return QualityLevel.HIGH
        elif quality_score >= 0.5:
            return QualityLevel.MEDIUM
        else:
            return QualityLevel.LOW
    
    def is_acceptable_quality(self, image_path: str, min_score: float = 0.5) -> bool:
        """
        Check if image meets minimum quality requirements.
        
        Args:
            image_path: Path to the image
            min_score: Minimum acceptable quality score
            
        Returns:
            True if image quality is acceptable
        """
        quality_score = self.calculate_quality_score(image_path)
        return quality_score >= min_score
    
    def detect_motion_blur(self, image_path: str, threshold: float = 100.0) -> bool:
        """
        Detect if image has motion blur.
        
        Args:
            image_path: Path to the image
            threshold: Blur threshold (lower = more blurry)
            
        Returns:
            True if image is blurry
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return True
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            return laplacian_var < threshold
            
        except Exception:
            return True
    
    def get_image_dimensions(self, image_path: str) -> Tuple[int, int]:
        """
        Get image dimensions (width, height).
        
        Args:
            image_path: Path to the image
            
        Returns:
            Tuple of (width, height)
        """
        try:
            with Image.open(image_path) as img:
                return img.size  # PIL returns (width, height)
        except Exception:
            return (0, 0)
    
    def get_file_size_mb(self, image_path: str) -> float:
        """
        Get file size in megabytes.
        
        Args:
            image_path: Path to the image
            
        Returns:
            File size in MB
        """
        try:
            return Path(image_path).stat().st_size / (1024 * 1024)
        except Exception:
            return 0.0
    
    def check_image_corruption(self, image_path: str) -> bool:
        """
        Check if image file is corrupted.
        
        Args:
            image_path: Path to the image
            
        Returns:
            True if image is corrupted
        """
        try:
            # Try to load with both OpenCV and PIL
            cv_image = cv2.imread(str(image_path))
            if cv_image is None:
                return True
            
            with Image.open(image_path) as pil_image:
                pil_image.verify()
            
            return False
            
        except Exception:
            return True
