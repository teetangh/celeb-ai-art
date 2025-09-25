"""Dataset organization and folder structure management."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..core.models import CelebrityInfo, CelebrityDataset, ImageMetadata


class DatasetOrganizer:
    """Manages the folder structure and organization of celebrity datasets."""

    def __init__(self, base_path: str):
        """
        Initialize the dataset organizer.
        
        Args:
            base_path: Root directory for the celebrity dataset
        """
        self.base_path = Path(base_path)
        self.celebrities_path = self.base_path / "celebrities"
        self.base_models_path = self.base_path / "base_models"
        self.lora_models_path = self.base_path / "lora_models"
        self.configs_path = self.base_path / "configs"
        self.scripts_path = self.base_path / "scripts"
        self.legal_path = self.base_path / "legal"
        self.setup_base_directories()

    def setup_base_directories(self) -> None:
        """Create the base directory structure."""
        directories = [
            self.celebrities_path,
            self.base_models_path,
            self.lora_models_path,
            self.configs_path,
            self.scripts_path,
            self.legal_path,
            # Script subdirectories
            self.scripts_path / "collect",
            self.scripts_path / "process", 
            self.scripts_path / "label",
            self.scripts_path / "validate",
            # Legal subdirectories
            self.legal_path / "licenses",
            self.legal_path / "consent_forms",
            self.legal_path / "fair_use_documentation",
            self.legal_path / "takedown_requests"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create .gitignore files for sensitive directories
        gitignore_content = "*\n!.gitignore\n"
        for sensitive_dir in [self.legal_path / "consent_forms", self.legal_path / "takedown_requests"]:
            gitignore_file = sensitive_dir / ".gitignore"
            if not gitignore_file.exists():
                gitignore_file.write_text(gitignore_content)

    def create_celebrity_structure(self, celebrity: CelebrityInfo) -> Path:
        """
        Create the complete folder structure for a new celebrity.
        
        Args:
            celebrity: Celebrity information
            
        Returns:
            Path to the celebrity's root directory
        """
        celeb_path = self.celebrities_path / celebrity.id
        
        # Define all subdirectories
        subdirs = [
            "raw/high_quality",
            "raw/medium_quality",
            "raw/low_quality", 
            "processed/face_crops",
            "processed/full_body",
            "processed/portraits",
            "processed/augmented",
            "metadata",
            "validation/face_crops",
            "validation/full_body", 
            "validation/portraits",
            "backup"
        ]
        
        # Create all subdirectories
        for subdir in subdirs:
            (celeb_path / subdir).mkdir(parents=True, exist_ok=True)
        
        # Create initial dataset metadata
        dataset = CelebrityDataset(
            celebrity_info=celebrity,
            images=[],
            created_date=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            dataset_version="1.0"
        )
        
        # Save metadata file
        metadata_path = celeb_path / "metadata" / "annotations.json"
        dataset.save_to_file(metadata_path)
        
        # Create additional metadata files
        self._create_readme(celeb_path, celebrity)
        self._create_dataset_info(celeb_path, celebrity)
        
        return celeb_path

    def _create_readme(self, celeb_path: Path, celebrity: CelebrityInfo) -> None:
        """Create a README file for the celebrity dataset."""
        readme_content = f"""# {celebrity.name} Dataset

## Celebrity Information
- **Name**: {celebrity.name}
- **ID**: {celebrity.id}
- **Gender**: {celebrity.gender.value}
- **Ethnicity**: {celebrity.ethnicity}
- **Birth Year**: {celebrity.birth_year or 'Unknown'}
- **Profession**: {celebrity.profession or 'Unknown'}

## Dataset Structure

```
{celebrity.id}/
├── raw/                     # Original unprocessed images
│   ├── high_quality/       # 1024x1024+ resolution
│   ├── medium_quality/     # 512x512+ resolution
│   └── low_quality/        # <512x512 (for augmentation)
├── processed/              # Cleaned, cropped, aligned images
│   ├── face_crops/         # 512x512 face crops for LoRA training
│   ├── full_body/          # Full body shots
│   ├── portraits/          # Head & shoulders
│   └── augmented/          # Data augmentation results
├── validation/             # Hold-out set for testing
│   ├── face_crops/
│   ├── full_body/
│   └── portraits/
├── metadata/               # Dataset metadata and annotations
├── backup/                 # Backup copies
└── README.md              # This file
```

## Usage

This dataset is organized for LoRA training with Stable Diffusion. The key training data is in:
- `processed/face_crops/` - Main training images (512x512 face crops)
- `validation/face_crops/` - Validation set (15-20% of data)

## Captions

All training images use the trigger word format:
`"ohwx {celebrity.name}, [description], [style]"`

## Notes
{celebrity.notes or 'No additional notes.'}

---
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        readme_path = celeb_path / "README.md"
        readme_path.write_text(readme_content, encoding='utf-8')

    def _create_dataset_info(self, celeb_path: Path, celebrity: CelebrityInfo) -> None:
        """Create a dataset info YAML file."""
        import yaml
        
        dataset_info = {
            'celebrity': {
                'id': celebrity.id,
                'name': celebrity.name,
                'gender': celebrity.gender.value,
                'ethnicity': celebrity.ethnicity,
                'birth_year': celebrity.birth_year,
                'profession': celebrity.profession
            },
            'dataset': {
                'created': datetime.now().isoformat(),
                'version': '1.0',
                'purpose': 'LoRA training for Stable Diffusion',
                'target_images': 50,
                'trigger_word': 'ohwx',
                'image_size': 512,
                'formats': ['jpg', 'png']
            },
            'training': {
                'recommended_steps': 1000,
                'batch_size': 1,
                'learning_rate': 0.0001,
                'validation_split': 0.15
            }
        }
        
        info_path = celeb_path / "metadata" / "dataset_info.yaml"
        with open(info_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_info, f, default_flow_style=False, allow_unicode=True)

    def get_celebrity_path(self, celebrity_id: str) -> Path:
        """Get the path to a celebrity's dataset directory."""
        return self.celebrities_path / celebrity_id

    def list_celebrities(self) -> List[str]:
        """List all celebrity IDs in the dataset."""
        if not self.celebrities_path.exists():
            return []
        
        celebrity_dirs = []
        for item in self.celebrities_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                celebrity_dirs.append(item.name)
        
        return sorted(celebrity_dirs)

    def load_celebrity_dataset(self, celebrity_id: str) -> Optional[CelebrityDataset]:
        """
        Load a celebrity's dataset metadata.
        
        Args:
            celebrity_id: The celebrity's unique ID
            
        Returns:
            CelebrityDataset object or None if not found
        """
        metadata_path = self.get_celebrity_path(celebrity_id) / "metadata" / "annotations.json"
        
        if not metadata_path.exists():
            return None
        
        return CelebrityDataset.load_from_file(metadata_path)

    def save_celebrity_dataset(self, celebrity_id: str, dataset: CelebrityDataset) -> None:
        """
        Save a celebrity's dataset metadata.
        
        Args:
            celebrity_id: The celebrity's unique ID
            dataset: The dataset to save
        """
        dataset.last_updated = datetime.now().isoformat()
        metadata_path = self.get_celebrity_path(celebrity_id) / "metadata" / "annotations.json"
        dataset.save_to_file(metadata_path)

    def backup_celebrity_data(self, celebrity_id: str) -> Path:
        """
        Create a backup of all celebrity data.
        
        Args:
            celebrity_id: The celebrity's unique ID
            
        Returns:
            Path to the backup directory
        """
        celeb_path = self.get_celebrity_path(celebrity_id)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{celebrity_id}_backup_{timestamp}"
        backup_path = celeb_path / "backup" / backup_name
        
        # Create backup directory
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Backup key directories
        dirs_to_backup = ["processed", "metadata", "validation"]
        for dir_name in dirs_to_backup:
            source_dir = celeb_path / dir_name
            if source_dir.exists():
                dest_dir = backup_path / dir_name
                shutil.copytree(source_dir, dest_dir)
        
        return backup_path

    def get_dataset_statistics(self) -> Dict:
        """Get overall dataset statistics."""
        stats = {
            'total_celebrities': 0,
            'total_images': 0,
            'total_training_images': 0,
            'total_validation_images': 0,
            'celebrities_by_gender': {'male': 0, 'female': 0, 'non_binary': 0, 'unknown': 0},
            'average_images_per_celebrity': 0,
            'disk_usage_mb': 0
        }
        
        celebrity_ids = self.list_celebrities()
        stats['total_celebrities'] = len(celebrity_ids)
        
        for celebrity_id in celebrity_ids:
            dataset = self.load_celebrity_dataset(celebrity_id)
            if dataset:
                stats['total_images'] += dataset.total_images
                stats['total_training_images'] += dataset.training_images
                stats['total_validation_images'] += dataset.validation_images
                
                # Count by gender
                gender = dataset.celebrity_info.gender.value
                if gender in stats['celebrities_by_gender']:
                    stats['celebrities_by_gender'][gender] += 1
        
        if stats['total_celebrities'] > 0:
            stats['average_images_per_celebrity'] = stats['total_images'] / stats['total_celebrities']
        
        # Calculate disk usage
        try:
            total_size = sum(
                f.stat().st_size for f in self.base_path.rglob('*') 
                if f.is_file() and not f.name.startswith('.')
            )
            stats['disk_usage_mb'] = round(total_size / (1024 * 1024), 2)
        except Exception:
            stats['disk_usage_mb'] = 0
        
        return stats

    def cleanup_empty_directories(self) -> int:
        """Remove empty directories and return count of removed directories."""
        removed_count = 0
        
        # Walk through all directories bottom-up
        for dirpath, dirnames, filenames in self.base_path.walk(top_down=False):
            dirpath = Path(dirpath)
            
            # Skip if directory has files
            if filenames:
                continue
            
            # Skip if directory has non-empty subdirectories
            if any((dirpath / dirname).exists() and list((dirpath / dirname).iterdir()) 
                   for dirname in dirnames):
                continue
            
            # Skip important directories even if empty
            important_dirs = {
                'raw', 'processed', 'validation', 'metadata', 'backup',
                'celebrities', 'base_models', 'lora_models', 'configs', 'scripts', 'legal'
            }
            if dirpath.name in important_dirs:
                continue
            
            try:
                dirpath.rmdir()
                removed_count += 1
            except OSError:
                # Directory not empty or permission denied
                continue
        
        return removed_count

    def validate_structure(self, celebrity_id: str) -> Dict[str, bool]:
        """
        Validate the folder structure for a celebrity.
        
        Args:
            celebrity_id: The celebrity's unique ID
            
        Returns:
            Dictionary of structure validation results
        """
        celeb_path = self.get_celebrity_path(celebrity_id)
        
        required_dirs = [
            "raw/high_quality",
            "raw/medium_quality", 
            "raw/low_quality",
            "processed/face_crops",
            "processed/full_body",
            "processed/portraits",
            "metadata",
            "validation/face_crops",
            "validation/portraits"
        ]
        
        validation_results = {
            'celebrity_directory_exists': celeb_path.exists(),
            'metadata_file_exists': (celeb_path / "metadata" / "annotations.json").exists(),
            'readme_exists': (celeb_path / "README.md").exists(),
        }
        
        for required_dir in required_dirs:
            dir_path = celeb_path / required_dir
            validation_results[f"dir_{required_dir.replace('/', '_')}_exists"] = dir_path.exists()
        
        validation_results['all_valid'] = all(validation_results.values())
        
        return validation_results
