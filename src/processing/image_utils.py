"""Image processing utilities for cropping, resizing, and basic operations."""

import cv2
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
import hashlib

from ..core.models import FaceDetection


class ImageProcessor:
    """Handles basic image processing operations."""
    
    def __init__(self):
        """Initialize the image processor."""
        self.target_size = (512, 512)
        self.padding_ratio = 0.2  # 20% padding around face
    
    def crop_face(
        self, 
        image_path: str, 
        face_detection: FaceDetection,
        output_path: str, 
        size: Tuple[int, int] = (512, 512),
        padding_ratio: float = 0.2
    ) -> bool:
        """
        Crop face from image with padding.
        
        Args:
            image_path: Path to input image
            face_detection: Face detection result
            output_path: Path for output image
            size: Target size for cropped image
            padding_ratio: Padding ratio around face bbox
            
        Returns:
            True if successful
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return False
            
            height, width = image.shape[:2]
            x, y, w, h = face_detection.bbox
            
            # Calculate padding
            padding_x = int(w * padding_ratio)
            padding_y = int(h * padding_ratio)
            
            # Expand bounding box with padding
            x1 = max(0, x - padding_x)
            y1 = max(0, y - padding_y)
            x2 = min(width, x + w + padding_x)
            y2 = min(height, y + h + padding_y)
            
            # Crop face region
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return False
            
            # Resize to target size
            face_resized = cv2.resize(face_crop, size, interpolation=cv2.INTER_LANCZOS4)
            
            # Save image
            success = cv2.imwrite(str(output_path), face_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            return success
            
        except Exception as e:
            print(f"Error cropping face from {image_path}: {e}")
            return False
    
    def resize_image(
        self, 
        image_path: str, 
        output_path: str, 
        size: Tuple[int, int],
        maintain_aspect_ratio: bool = True
    ) -> bool:
        """
        Resize image to target size.
        
        Args:
            image_path: Path to input image
            output_path: Path for output image
            size: Target size (width, height)
            maintain_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            True if successful
        """
        try:
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                if maintain_aspect_ratio:
                    img.thumbnail(size, Image.Resampling.LANCZOS)

                    # Create new image with target size and paste centered
                    new_img = Image.new('RGB', size, (255, 255, 255))
                    paste_x = (size[0] - img.size[0]) // 2
                    paste_y = (size[1] - img.size[1]) // 2
                    new_img.paste(img, (paste_x, paste_y))
                    img = new_img
                else:
                    img = img.resize(size, Image.Resampling.LANCZOS)

                img.save(output_path, 'JPEG', quality=95)
                return True

        except Exception as e:
            print(f"Error resizing image {image_path}: {e}")
            return False
    
    def center_crop_square(self, image_path: str, output_path: str, size: int = 512) -> bool:
        """
        Center crop image to square.
        
        Args:
            image_path: Path to input image
            output_path: Path for output image
            size: Size of square crop
            
        Returns:
            True if successful
        """
        try:
            with Image.open(image_path) as img:
                # Calculate crop box for center square
                width, height = img.size
                min_dimension = min(width, height)
                
                left = (width - min_dimension) // 2
                top = (height - min_dimension) // 2
                right = left + min_dimension
                bottom = top + min_dimension
                
                # Crop to square
                img_cropped = img.crop((left, top, right, bottom))
                
                # Resize to target size
                img_resized = img_cropped.resize((size, size), Image.Resampling.LANCZOS)
                
                img_resized.save(output_path, 'JPEG', quality=95)
                return True
                
        except Exception as e:
            print(f"Error center cropping {image_path}: {e}")
            return False
    
    def normalize_image(self, image_path: str, output_path: str) -> bool:
        """
        Normalize image brightness and contrast.
        
        Args:
            image_path: Path to input image
            output_path: Path for output image
            
        Returns:
            True if successful
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Auto-enhance contrast and brightness
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.1)  # Slight contrast boost
                
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(1.05)  # Slight brightness boost
                
                # Auto-level
                img = ImageOps.autocontrast(img)
                
                img.save(output_path, 'JPEG', quality=95)
                return True
                
        except Exception as e:
            print(f"Error normalizing image {image_path}: {e}")
            return False
    
    def generate_image_hash(self, image_path: str) -> str:
        """
        Generate SHA-256 hash of image file.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Hexadecimal hash string
        """
        try:
            with open(image_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            return file_hash
        except Exception:
            return ""
    
    def convert_format(
        self, 
        image_path: str, 
        output_path: str, 
        format: str = 'JPEG',
        quality: int = 95
    ) -> bool:
        """
        Convert image to different format.
        
        Args:
            image_path: Path to input image
            output_path: Path for output image
            format: Target format (JPEG, PNG, etc.)
            quality: Quality for JPEG (1-100)
            
        Returns:
            True if successful
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if saving as JPEG
                if format.upper() == 'JPEG' and img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # Save with specified format
                if format.upper() == 'JPEG':
                    img.save(output_path, format, quality=quality)
                else:
                    img.save(output_path, format)
                
                return True
                
        except Exception as e:
            print(f"Error converting format for {image_path}: {e}")
            return False
    
    def create_thumbnail(
        self, 
        image_path: str, 
        output_path: str, 
        size: Tuple[int, int] = (256, 256)
    ) -> bool:
        """
        Create thumbnail of image.
        
        Args:
            image_path: Path to input image
            output_path: Path for thumbnail
            size: Thumbnail size
            
        Returns:
            True if successful
        """
        try:
            with Image.open(image_path) as img:
                img.thumbnail(size, Image.Resampling.LANCZOS)
                img.save(output_path, 'JPEG', quality=85)
                return True
                
        except Exception as e:
            print(f"Error creating thumbnail for {image_path}: {e}")
            return False
    
    def remove_background_simple(self, image_path: str, output_path: str) -> bool:
        """
        Simple background removal using edge detection (basic implementation).
        For production, use models like U²-Net or commercial APIs.
        
        Args:
            image_path: Path to input image
            output_path: Path for output image
            
        Returns:
            True if successful
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return False
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply GaussianBlur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Create mask using adaptive threshold
            mask = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Create 3-channel mask
            mask_3channel = cv2.merge([mask, mask, mask])
            
            # Apply mask to original image
            result = cv2.bitwise_and(image, mask_3channel)
            
            # Save result
            success = cv2.imwrite(str(output_path), result)
            return success
            
        except Exception as e:
            print(f"Error removing background from {image_path}: {e}")
            return False
    
    def validate_image_file(self, image_path: str) -> bool:
        """
        Validate that file is a valid image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if valid image file
        """
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    
    def get_dominant_colors(self, image_path: str, num_colors: int = 5) -> list:
        """
        Extract dominant colors from image.
        
        Args:
            image_path: Path to the image
            num_colors: Number of dominant colors to extract
            
        Returns:
            List of dominant colors as RGB tuples
        """
        try:
            from sklearn.cluster import KMeans
            
            # Load and resize image for faster processing
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (150, 150))
            
            # Reshape image to be a list of pixels
            pixels = image.reshape(-1, 3)
            
            # Use K-means to find dominant colors
            kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get the dominant colors
            colors = kmeans.cluster_centers_.astype(int)
            
            return [tuple(color) for color in colors]
            
        except Exception as e:
            print(f"Error extracting colors from {image_path}: {e}")
            return []
