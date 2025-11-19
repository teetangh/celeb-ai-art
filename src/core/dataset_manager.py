"""Main dataset management class that orchestrates all components."""

import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import json
import numpy as np
from tqdm import tqdm

from .models import CelebrityInfo, CelebrityDataset, ImageMetadata, FaceDetection
from ..utils.dataset_organizer import DatasetOrganizer
from ..processing.face_detector import FaceDetector
from ..processing.quality_assessor import QualityAssessor
from ..processing.image_utils import ImageProcessor
from ..processing.augmentation import DataAugmenter
from ..utils.caption_generator import CaptionGenerator


class DatasetManager:
    """Comprehensive dataset manager for celebrity AI training data."""
    
    def __init__(self, base_path: str, trigger_word: str = "ohwx"):
        """
        Initialize the dataset manager.
        
        Args:
            base_path: Root directory for the dataset
            trigger_word: Trigger word for LoRA training
        """
        self.base_path = Path(base_path)
        
        # Initialize all components
        self.organizer = DatasetOrganizer(base_path)
        self.face_detector = FaceDetector()
        self.quality_assessor = QualityAssessor()
        self.image_processor = ImageProcessor()
        self.augmenter = DataAugmenter()
        self.caption_generator = CaptionGenerator(trigger_word)
    
    def add_celebrity(self, celebrity: CelebrityInfo) -> Path:
        """
        Add a new celebrity to the dataset.
        
        Args:
            celebrity: Celebrity information
            
        Returns:
            Path to the celebrity's dataset directory
        """
        celeb_path = self.organizer.create_celebrity_structure(celebrity)
        print(f"✅ Created dataset structure for {celebrity.name}")
        return celeb_path
    
    def process_raw_images(
        self,
        celebrity_id: str,
        min_quality: float = 0.6,
        generate_captions: bool = True,
        create_augmentations: bool = False,
        skip_face_detection: bool = False
    ) -> int:
        """
        Process raw images for a celebrity.

        Args:
            celebrity_id: Celebrity's unique ID
            min_quality: Minimum quality threshold
            generate_captions: Whether to generate training captions
            create_augmentations: Whether to create augmented versions
            skip_face_detection: Skip face detection and use full images

        Returns:
            Number of successfully processed images
        """
        print(f"📸 Processing raw images for {celebrity_id}...")
        if skip_face_detection:
            print("⚠️  Face detection disabled - using full images")

        # Load existing dataset
        dataset = self.organizer.load_celebrity_dataset(celebrity_id)
        if not dataset:
            print(f"❌ Could not load dataset for {celebrity_id}")
            return 0

        celeb_path = self.organizer.get_celebrity_path(celebrity_id)
        raw_dirs = ["high_quality", "medium_quality", "low_quality"]
        processed_count = 0

        for quality_dir in raw_dirs:
            raw_path = celeb_path / "raw" / quality_dir
            if not raw_path.exists():
                continue

            image_files = list(raw_path.glob("*.jpg")) + list(raw_path.glob("*.jpeg")) + list(raw_path.glob("*.png")) + list(raw_path.glob("*.webp"))

            for image_file in tqdm(image_files, desc=f"Processing {quality_dir}"):
                if self._process_single_image(
                    str(image_file),
                    celebrity_id,
                    dataset,
                    min_quality,
                    generate_captions,
                    skip_face_detection
                ):
                    processed_count += 1

        # Save updated dataset
        self.organizer.save_celebrity_dataset(celebrity_id, dataset)

        # Create augmentations if requested
        if create_augmentations and processed_count > 0:
            self._create_augmentations(celebrity_id)

        print(f"✅ Processed {processed_count} images for {celebrity_id}")
        return processed_count
    
    def _process_single_image(
        self,
        image_path: str,
        celebrity_id: str,
        dataset: CelebrityDataset,
        min_quality: float,
        generate_captions: bool,
        skip_face_detection: bool = False
    ) -> bool:
        """Process a single raw image."""
        try:
            # Check image quality
            quality_score = self.quality_assessor.calculate_quality_score(image_path)
            if quality_score < min_quality:
                return False

            # Generate output filename
            base_name = f"{celebrity_id}_{len(dataset.images):04d}"
            celeb_path = self.organizer.get_celebrity_path(celebrity_id)
            face_output = celeb_path / "processed" / "face_crops" / f"{base_name}_crop.jpg"

            largest_face = None

            if skip_face_detection:
                # Skip face detection - just resize image to 512x512
                if self.image_processor.resize_image(image_path, str(face_output), size=(512, 512)):
                    pass  # Success, continue to metadata
                else:
                    return False
            else:
                # Detect faces
                faces = self.face_detector.detect_faces_dlib(image_path)
                if not faces:
                    return False

                # Get largest face
                largest_face = self.face_detector.get_largest_face(faces)
                if not largest_face or not self.face_detector.is_face_visible(largest_face):
                    return False

                # Process face crop
                if not self.image_processor.crop_face(image_path, largest_face, str(face_output)):
                    return False

            # Generate caption if requested
            caption = ""
            if generate_captions:
                caption = self.caption_generator.generate_caption(dataset.celebrity_info)

            # Create metadata
            image_metadata = ImageMetadata(
                filename=f"{base_name}_crop.jpg",
                source_url=None,
                quality_score=quality_score,
                quality_level=self.quality_assessor.get_quality_level(quality_score),
                face_detection=largest_face,
                caption=caption,
                processed_date=datetime.now().isoformat(),
                dimensions=self.quality_assessor.get_image_dimensions(str(face_output)),
                file_size=int(self.quality_assessor.get_file_size_mb(str(face_output)) * 1024 * 1024)
            )

            dataset.images.append(image_metadata)
            return True

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

        return False
    
    def create_training_split(
        self, 
        celebrity_id: str, 
        validation_ratio: float = 0.15,
        seed: int = 42
    ) -> Dict[str, int]:
        """
        Create training/validation split.
        
        Args:
            celebrity_id: Celebrity's unique ID
            validation_ratio: Ratio of validation images
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with split statistics
        """
        random.seed(seed)
        np.random.seed(seed)
        
        celeb_path = self.organizer.get_celebrity_path(celebrity_id)
        processed_path = celeb_path / "processed" / "face_crops"
        validation_path = celeb_path / "validation" / "face_crops"
        
        # Get all processed images
        images = list(processed_path.glob("*.jpg"))
        random.shuffle(images)
        
        # Calculate split
        total_images = len(images)
        validation_count = int(total_images * validation_ratio)
        validation_images = images[:validation_count]
        
        # Move validation images
        validation_path.mkdir(parents=True, exist_ok=True)
        for img in validation_images:
            dest = validation_path / img.name
            shutil.move(str(img), str(dest))
        
        # Update dataset metadata
        dataset = self.organizer.load_celebrity_dataset(celebrity_id)
        if dataset:
            dataset.training_images = total_images - validation_count
            dataset.validation_images = validation_count
            self.organizer.save_celebrity_dataset(celebrity_id, dataset)
        
        stats = {
            'total_images': total_images,
            'training_images': total_images - validation_count,
            'validation_images': validation_count,
            'validation_ratio': validation_count / total_images if total_images > 0 else 0
        }
        
        print(f"📊 Split {celebrity_id}: {stats['training_images']} training, {stats['validation_images']} validation")
        return stats
    
    def _create_augmentations(self, celebrity_id: str) -> int:
        """Create augmented versions of processed images."""
        celeb_path = self.organizer.get_celebrity_path(celebrity_id)
        processed_path = celeb_path / "processed" / "face_crops"
        augmented_path = celeb_path / "processed" / "augmented"
        
        if not processed_path.exists():
            return 0
        
        # Get training images (not validation)
        training_images = list(processed_path.glob("*.jpg"))
        
        if len(training_images) < 20:  # Only augment if we have few images
            augmented_count = 0
            for image_path in training_images:
                variations = self.augmenter.augment_image(
                    str(image_path),
                    str(augmented_path),
                    num_variations=2,
                    use_face_mode=True
                )
                augmented_count += len(variations)
            
            print(f"🔄 Created {augmented_count} augmented images for {celebrity_id}")
            return augmented_count
        
        return 0
    
    def generate_training_config(
        self, 
        celebrity_ids: List[str], 
        output_path: str
    ) -> bool:
        """
        Generate training configuration for LoRA.
        
        Args:
            celebrity_ids: List of celebrity IDs to include
            output_path: Path for output config file
            
        Returns:
            True if successful
        """
        try:
            import yaml
            
            # Calculate total images
            total_training = 0
            total_validation = 0
            
            for celebrity_id in celebrity_ids:
                dataset = self.organizer.load_celebrity_dataset(celebrity_id)
                if dataset:
                    total_training += dataset.training_images
                    total_validation += dataset.validation_images
            
            config = {
                'dataset': {
                    'base_path': str(self.organizer.celebrities_path),
                    'celebrity_list': celebrity_ids,
                    'total_training_images': total_training,
                    'total_validation_images': total_validation,
                    'image_size': 512,
                    'batch_size': min(4, total_training // 10) if total_training > 0 else 1
                },
                'training': {
                    'trigger_word': self.caption_generator.trigger_word,
                    'learning_rate': 0.0001,
                    'max_train_steps': max(500, total_training * 20),
                    'validation_epochs': 10,
                    'save_every_n_epochs': 5,
                    'mixed_precision': 'fp16',
                    'gradient_accumulation_steps': 1
                },
                'model': {
                    'base_model': 'runwayml/stable-diffusion-v1-5',
                    'lora_rank': 4,
                    'lora_alpha': 32,
                    'lora_dropout': 0.1
                },
                'augmentation': {
                    'enable': True,
                    'horizontal_flip': 0.5,
                    'color_jitter': 0.1,
                    'rotation_degrees': 5
                }
            }
            
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            print(f"📝 Generated training config: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error generating config: {e}")
            return False
    
    def cleanup_dataset(self, celebrity_id: str) -> Dict[str, int]:
        """
        Clean up dataset by removing duplicates and low-quality images.
        
        Args:
            celebrity_id: Celebrity's unique ID
            
        Returns:
            Cleanup statistics
        """
        print(f"🧹 Cleaning up dataset for {celebrity_id}...")
        
        celeb_path = self.organizer.get_celebrity_path(celebrity_id)
        processed_path = celeb_path / "processed" / "face_crops"
        
        if not processed_path.exists():
            return {'removed': 0, 'remaining': 0}
        
        images = list(processed_path.glob("*.jpg"))
        removed_count = 0
        
        # Remove corrupted images
        for image_path in images[:]:
            if self.quality_assessor.check_image_corruption(str(image_path)):
                image_path.unlink()
                images.remove(image_path)
                removed_count += 1
        
        # Remove very low quality images
        for image_path in images[:]:
            quality_score = self.quality_assessor.calculate_quality_score(str(image_path))
            if quality_score < 0.3:
                image_path.unlink()
                images.remove(image_path)
                removed_count += 1
        
        stats = {
            'removed': removed_count,
            'remaining': len(images)
        }
        
        print(f"🧹 Cleanup complete: removed {removed_count}, {len(images)} remaining")
        return stats
    
    def get_dataset_summary(self) -> Dict:
        """Get comprehensive dataset summary."""
        summary = {
            'celebrities': {},
            'totals': {
                'celebrity_count': 0,
                'total_images': 0,
                'total_training': 0,
                'total_validation': 0,
                'avg_images_per_celebrity': 0
            }
        }
        
        celebrity_ids = self.organizer.list_celebrities()
        summary['totals']['celebrity_count'] = len(celebrity_ids)
        
        for celebrity_id in celebrity_ids:
            dataset = self.organizer.load_celebrity_dataset(celebrity_id)
            if dataset:
                celeb_summary = {
                    'name': dataset.celebrity_info.name,
                    'total_images': dataset.total_images,
                    'training_images': dataset.training_images,
                    'validation_images': dataset.validation_images,
                    'created_date': dataset.created_date,
                    'last_updated': dataset.last_updated
                }
                
                summary['celebrities'][celebrity_id] = celeb_summary
                summary['totals']['total_images'] += dataset.total_images
                summary['totals']['total_training'] += dataset.training_images
                summary['totals']['total_validation'] += dataset.validation_images
        
        if summary['totals']['celebrity_count'] > 0:
            summary['totals']['avg_images_per_celebrity'] = (
                summary['totals']['total_images'] / summary['totals']['celebrity_count']
            )
        
        return summary
    
    def bulk_process_all(self, min_quality: float = 0.6) -> Dict[str, int]:
        """
        Process all celebrities with raw images in bulk.
        
        Args:
            min_quality: Minimum quality threshold
            
        Returns:
            Dictionary with processing statistics per celebrity
        """
        celebrity_ids = self.organizer.list_celebrities()
        results = {}
        
        for celebrity_id in celebrity_ids:
            print(f"\n🎭 Processing {celebrity_id}...")
            
            # Process raw images
            processed_count = self.process_raw_images(
                celebrity_id=celebrity_id,
                min_quality=min_quality,
                generate_captions=True,
                create_augmentations=True
            )
            
            if processed_count > 0:
                # Create training split
                split_stats = self.create_training_split(
                    celebrity_id=celebrity_id,
                    validation_ratio=0.15
                )
                
                # Cleanup dataset
                cleanup_stats = self.cleanup_dataset(celebrity_id)
                
                results[celebrity_id] = {
                    'processed': processed_count,
                    'training': split_stats['training_images'],
                    'validation': split_stats['validation_images'],
                    'cleaned': cleanup_stats['removed']
                }
            else:
                results[celebrity_id] = {
                    'processed': 0,
                    'training': 0,
                    'validation': 0,
                    'cleaned': 0
                }
        
        return results
    
    def export_dataset_info(self, output_path: str) -> bool:
        """
        Export comprehensive dataset information to JSON.
        
        Args:
            output_path: Path for output JSON file
            
        Returns:
            True if successful
        """
        try:
            summary = self.get_dataset_summary()
            
            # Add additional system info
            export_data = {
                'export_date': datetime.now().isoformat(),
                'dataset_path': str(self.base_path),
                'trigger_word': self.caption_generator.trigger_word,
                'summary': summary
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"📋 Dataset info exported to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting dataset info: {e}")
            return False
