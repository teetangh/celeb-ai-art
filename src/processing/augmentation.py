"""Data augmentation utilities for celebrity dataset."""

import cv2
import numpy as np
import random
from typing import List, Tuple, Dict, Any
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DataAugmenter:
    """Handles data augmentation for training images."""
    
    def __init__(self, preserve_face_quality: bool = True):
        """
        Initialize the data augmenter.
        
        Args:
            preserve_face_quality: Whether to use face-preserving augmentations
        """
        self.preserve_face_quality = preserve_face_quality
        self.setup_augmentation_pipeline()
    
    def setup_augmentation_pipeline(self):
        """Set up different augmentation pipelines."""
        # Conservative augmentation for face images (preserves identity)
        self.face_augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, 
                contrast_limit=0.1, 
                p=0.3
            ),
            A.HueSaturationValue(
                hue_shift_limit=5, 
                sat_shift_limit=10, 
                val_shift_limit=10, 
                p=0.3
            ),
            A.GaussNoise(var_limit=(10, 30), p=0.2),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2),
            A.Rotate(limit=5, p=0.3),  # Very small rotation
        ])
        
        # More aggressive augmentation for general images
        self.general_augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, 
                sat_shift_limit=20, 
                val_shift_limit=20, 
                p=0.5
            ),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.OneOf([
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0)),
                A.Blur(blur_limit=3),
                A.GaussianBlur(blur_limit=3),
            ], p=0.3),
            A.Rotate(limit=15, p=0.4),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.3
            ),
        ])
    
    def augment_image(
        self, 
        image_path: str, 
        output_dir: str, 
        num_variations: int = 3,
        use_face_mode: bool = True
    ) -> List[str]:
        """
        Generate augmented variations of an image.
        
        Args:
            image_path: Path to input image
            output_dir: Directory for output images
            num_variations: Number of variations to generate
            use_face_mode: Whether to use face-preserving augmentations
            
        Returns:
            List of paths to generated augmented images
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return []
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Choose augmentation pipeline
            augmentation = self.face_augmentation if use_face_mode else self.general_augmentation
            
            # Generate variations
            output_paths = []
            base_name = Path(image_path).stem
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(num_variations):
                # Apply augmentation
                augmented = augmentation(image=image_rgb)
                augmented_image = augmented['image']
                
                # Convert back to BGR for OpenCV
                augmented_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                
                # Generate output path
                output_path = output_dir / f"{base_name}_aug_{i+1:03d}.jpg"
                
                # Save augmented image
                success = cv2.imwrite(str(output_path), augmented_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                if success:
                    output_paths.append(str(output_path))
            
            return output_paths
            
        except Exception as e:
            print(f"Error augmenting image {image_path}: {e}")
            return []
    
    def augment_batch(
        self, 
        image_paths: List[str], 
        output_dir: str,
        variations_per_image: int = 3,
        use_face_mode: bool = True
    ) -> Dict[str, List[str]]:
        """
        Augment a batch of images.
        
        Args:
            image_paths: List of input image paths
            output_dir: Directory for output images
            variations_per_image: Number of variations per input image
            use_face_mode: Whether to use face-preserving augmentations
            
        Returns:
            Dictionary mapping input paths to list of output paths
        """
        results = {}
        
        for image_path in image_paths:
            augmented_paths = self.augment_image(
                image_path, 
                output_dir, 
                variations_per_image,
                use_face_mode
            )
            results[image_path] = augmented_paths
        
        return results
    
    def create_color_variations(
        self, 
        image_path: str, 
        output_dir: str,
        temperature_shifts: List[int] = [-200, -100, 100, 200]
    ) -> List[str]:
        """
        Create color temperature variations.
        
        Args:
            image_path: Path to input image
            output_dir: Directory for output images
            temperature_shifts: List of temperature shifts in Kelvin
            
        Returns:
            List of paths to generated images
        """
        try:
            output_paths = []
            base_name = Path(image_path).stem
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                for i, temp_shift in enumerate(temperature_shifts):
                    # Apply color temperature shift
                    shifted_img = self._apply_temperature_shift(img, temp_shift)
                    
                    # Save image
                    output_path = output_dir / f"{base_name}_temp_{temp_shift:+d}.jpg"
                    shifted_img.save(str(output_path), 'JPEG', quality=95)
                    output_paths.append(str(output_path))
            
            return output_paths
            
        except Exception as e:
            print(f"Error creating color variations for {image_path}: {e}")
            return []
    
    def _apply_temperature_shift(self, image: Image.Image, temp_shift: int) -> Image.Image:
        """Apply color temperature shift to image."""
        # Convert temperature shift to RGB multipliers
        # This is a simplified approach
        if temp_shift < 0:  # Cooler (more blue)
            r_mult = 1.0 + temp_shift / 1000.0
            g_mult = 1.0 + temp_shift / 2000.0  
            b_mult = 1.0
        else:  # Warmer (more red/yellow)
            r_mult = 1.0
            g_mult = 1.0 - temp_shift / 2000.0
            b_mult = 1.0 - temp_shift / 1000.0
        
        # Clamp multipliers
        r_mult = max(0.5, min(1.5, r_mult))
        g_mult = max(0.5, min(1.5, g_mult))
        b_mult = max(0.5, min(1.5, b_mult))
        
        # Apply color correction
        r, g, b = image.split()
        
        r = r.point(lambda x: max(0, min(255, int(x * r_mult))))
        g = g.point(lambda x: max(0, min(255, int(x * g_mult))))
        b = b.point(lambda x: max(0, min(255, int(x * b_mult))))
        
        return Image.merge('RGB', (r, g, b))
    
    def create_lighting_variations(
        self, 
        image_path: str, 
        output_dir: str,
        brightness_factors: List[float] = [0.8, 0.9, 1.1, 1.2],
        contrast_factors: List[float] = [0.9, 1.0, 1.1, 1.2]
    ) -> List[str]:
        """
        Create lighting variations.
        
        Args:
            image_path: Path to input image
            output_dir: Directory for output images
            brightness_factors: Brightness adjustment factors
            contrast_factors: Contrast adjustment factors
            
        Returns:
            List of paths to generated images
        """
        try:
            output_paths = []
            base_name = Path(image_path).stem
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                variation_count = 0
                for brightness in brightness_factors:
                    for contrast in contrast_factors:
                        if brightness == 1.0 and contrast == 1.0:
                            continue  # Skip original
                        
                        # Apply brightness adjustment
                        bright_enhancer = ImageEnhance.Brightness(img)
                        bright_img = bright_enhancer.enhance(brightness)
                        
                        # Apply contrast adjustment
                        contrast_enhancer = ImageEnhance.Contrast(bright_img)
                        final_img = contrast_enhancer.enhance(contrast)
                        
                        # Save image
                        output_path = output_dir / f"{base_name}_light_{variation_count:02d}.jpg"
                        final_img.save(str(output_path), 'JPEG', quality=95)
                        output_paths.append(str(output_path))
                        
                        variation_count += 1
            
            return output_paths
            
        except Exception as e:
            print(f"Error creating lighting variations for {image_path}: {e}")
            return []
    
    def create_pose_variations(
        self, 
        image_path: str, 
        output_dir: str,
        rotation_angles: List[int] = [-5, -2, 2, 5]
    ) -> List[str]:
        """
        Create subtle pose variations through rotation.
        
        Args:
            image_path: Path to input image
            output_dir: Directory for output images
            rotation_angles: Rotation angles in degrees
            
        Returns:
            List of paths to generated images
        """
        try:
            output_paths = []
            base_name = Path(image_path).stem
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load image with OpenCV
            image = cv2.imread(str(image_path))
            if image is None:
                return []
            
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            
            for angle in rotation_angles:
                # Create rotation matrix
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                # Apply rotation
                rotated = cv2.warpAffine(
                    image, 
                    rotation_matrix, 
                    (width, height),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_REFLECT_101
                )
                
                # Save rotated image
                output_path = output_dir / f"{base_name}_rot_{angle:+03d}.jpg"
                success = cv2.imwrite(str(output_path), rotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                if success:
                    output_paths.append(str(output_path))
            
            return output_paths
            
        except Exception as e:
            print(f"Error creating pose variations for {image_path}: {e}")
            return []
    
    def get_augmentation_config(self) -> Dict[str, Any]:
        """Get current augmentation configuration."""
        return {
            'face_mode_enabled': self.preserve_face_quality,
            'face_augmentation': {
                'horizontal_flip': 0.5,
                'brightness_contrast': {'brightness': 0.1, 'contrast': 0.1, 'prob': 0.3},
                'hue_saturation': {'hue': 5, 'saturation': 10, 'value': 10, 'prob': 0.3},
                'noise': {'variance': (10, 30), 'prob': 0.2},
                'rotation': {'limit': 5, 'prob': 0.3},
            },
            'general_augmentation': {
                'horizontal_flip': 0.5,
                'brightness_contrast': {'brightness': 0.2, 'contrast': 0.2, 'prob': 0.5},
                'hue_saturation': {'hue': 10, 'saturation': 20, 'value': 20, 'prob': 0.5},
                'noise': {'variance': (10, 50), 'prob': 0.3},
                'rotation': {'limit': 15, 'prob': 0.4},
                'shift_scale_rotate': {
                    'shift': 0.05, 
                    'scale': 0.1, 
                    'rotate': 10, 
                    'prob': 0.3
                },
            }
        }
    
    def update_config(self, config: Dict[str, Any]):
        """Update augmentation configuration and rebuild pipelines."""
        # This would update the internal configuration
        # and rebuild the augmentation pipelines
        # Implementation depends on specific requirements
        pass
