"""Basic usage example for the celebrity dataset management system."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.models import CelebrityInfo, Gender
from core.dataset_manager import DatasetManager


def main():
    """Demonstrate basic usage of the dataset management system."""
    
    print("🎭 Celebrity AI Dataset Management System")
    print("=" * 50)
    
    # Initialize dataset manager
    dataset_path = "./celebrity_dataset"
    manager = DatasetManager(dataset_path, trigger_word="ohwx")
    
    print(f"📁 Dataset initialized at: {dataset_path}")
    
    # Example 1: Add celebrities
    print("\n1. Adding Celebrities...")
    
    celebrities = [
        CelebrityInfo(
            id="brad_pitt_001",
            name="Brad Pitt", 
            gender=Gender.MALE,
            ethnicity="caucasian",
            birth_year=1963,
            profession="actor",
            notes="American actor and film producer"
        ),
        CelebrityInfo(
            id="dwayne_johnson_002",
            name="Dwayne Johnson",
            gender=Gender.MALE,
            ethnicity="mixed",
            birth_year=1972,
            profession="actor",
            notes="Actor and former professional wrestler"
        ),
        CelebrityInfo(
            id="priyanka_chopra_003",
            name="Priyanka Chopra",
            gender=Gender.FEMALE,
            ethnicity="indian",
            birth_year=1982,
            profession="actress",
            notes="Indian actress and global icon"
        )
    ]
    
    # Add celebrities to dataset
    for celebrity in celebrities:
        celeb_path = manager.add_celebrity(celebrity)
        print(f"  ✅ {celebrity.name} -> {celeb_path.name}")
    
    # Example 2: Show folder structure
    print("\n2. Folder Structure Created:")
    show_folder_structure(manager.organizer.celebrities_path / "brad_pitt_001")
    
    # Example 3: Simulate processing raw images
    print("\n3. Processing Raw Images:")
    print("📝 To process images, add raw images to:")
    
    for celebrity in celebrities:
        celeb_path = manager.organizer.get_celebrity_path(celebrity.id)
        print(f"  • {celebrity.name}:")
        print(f"    - High quality: {celeb_path / 'raw' / 'high_quality'}")
        print(f"    - Medium quality: {celeb_path / 'raw' / 'medium_quality'}")
    
    # Example 4: Show how to process images (when available)
    print("\n4. Processing Example (when you have images):")
    print("""
    # Process raw images for Brad Pitt
    processed_count = manager.process_raw_images(
        celebrity_id="brad_pitt_001",
        min_quality=0.6,
        generate_captions=True,
        create_augmentations=True
    )
    
    # Create training/validation split
    split_stats = manager.create_training_split(
        celebrity_id="brad_pitt_001",
        validation_ratio=0.15
    )
    """)
    
    # Example 5: Generate training configuration
    print("\n5. Generating Training Configuration...")
    
    config_path = Path(dataset_path) / "configs" / "training_config.yaml"
    success = manager.generate_training_config(
        celebrity_ids=[c.id for c in celebrities],
        output_path=str(config_path)
    )
    
    if success:
        print(f"  ✅ Training config saved to: {config_path}")
    
    # Example 6: Show dataset summary
    print("\n6. Dataset Summary:")
    summary = manager.get_dataset_summary()
    
    print(f"  📊 Total celebrities: {summary['totals']['celebrity_count']}")
    print(f"  📸 Total images: {summary['totals']['total_images']}")
    print(f"  🎯 Training images: {summary['totals']['total_training']}")
    print(f"  ✅ Validation images: {summary['totals']['total_validation']}")
    
    for celeb_id, celeb_info in summary['celebrities'].items():
        print(f"    • {celeb_info['name']}: {celeb_info['total_images']} images")
    
    # Example 7: Caption generation example
    print("\n7. Caption Generation Example:")
    
    celebrity = celebrities[0]  # Brad Pitt
    sample_captions = []
    
    for i in range(3):
        caption = manager.caption_generator.generate_caption(
            celebrity,
            template_type='basic' if i == 0 else ('detailed' if i == 1 else 'professional')
        )
        sample_captions.append(caption)
    
    print("  Sample captions for Brad Pitt:")
    for i, caption in enumerate(sample_captions, 1):
        print(f"    {i}. {caption}")
    
    # Example 8: Next steps
    print("\n8. Next Steps:")
    print("""
    To start training:
    
    1. Add raw images to the raw/ folders for each celebrity
    2. Run image processing:
       python -c "
       from examples.basic_usage import process_all_celebrities
       process_all_celebrities()
       "
    
    3. Train LoRA model using the generated configuration
    4. Use the trained model for generation!
    """)
    
    print("\n✨ Setup complete! Ready for celebrity AI generation.")


def show_folder_structure(path: Path, prefix: str = "", max_depth: int = 3, current_depth: int = 0):
    """Show folder structure in tree format."""
    if current_depth >= max_depth or not path.exists():
        return
    
    items = sorted([item for item in path.iterdir() if item.is_dir()], key=lambda x: x.name)
    
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{item.name}/")
        
        if current_depth < max_depth - 1:
            next_prefix = prefix + ("    " if is_last else "│   ")
            show_folder_structure(item, next_prefix, max_depth, current_depth + 1)


def process_all_celebrities():
    """Process all celebrities (call this after adding raw images)."""
    dataset_path = "./celebrity_dataset"
    manager = DatasetManager(dataset_path)
    
    celebrity_ids = manager.organizer.list_celebrities()
    
    for celebrity_id in celebrity_ids:
        print(f"\nProcessing {celebrity_id}...")
        
        # Process raw images
        processed_count = manager.process_raw_images(
            celebrity_id=celebrity_id,
            min_quality=0.6,
            generate_captions=True,
            create_augmentations=True
        )
        
        if processed_count > 0:
            # Create training split
            manager.create_training_split(
                celebrity_id=celebrity_id,
                validation_ratio=0.15
            )
            
            # Cleanup dataset
            manager.cleanup_dataset(celebrity_id)
    
    # Generate final training config
    config_path = f"{dataset_path}/configs/final_training_config.yaml"
    manager.generate_training_config(celebrity_ids, config_path)
    
    print(f"\n✅ Processing complete! Training config: {config_path}")


if __name__ == "__main__":
    main()
