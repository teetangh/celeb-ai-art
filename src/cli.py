"""Command-line interface for the celebrity dataset management system."""

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from pathlib import Path
from typing import List, Optional

from .core.models import CelebrityInfo, Gender
from .core.dataset_manager import DatasetManager

app = typer.Typer(help="Celebrity AI Art - Dataset Management System")
console = Console()


@app.command()
def init(
    path: str = typer.Argument(..., help="Dataset root directory"),
    trigger_word: str = typer.Option("ohwx", help="Trigger word for LoRA training")
):
    """Initialize a new celebrity dataset."""
    manager = DatasetManager(path, trigger_word)
    console.print(f"✅ Dataset initialized at: {path}", style="green")
    console.print(f"🎯 Trigger word: {trigger_word}", style="blue")


@app.command()
def add_celebrity(
    path: str = typer.Argument(..., help="Dataset root directory"),
    celebrity_id: str = typer.Option(..., help="Celebrity ID (e.g., brad_pitt_001)"),
    name: str = typer.Option(..., help="Celebrity name"),
    gender: str = typer.Option(..., help="Gender: male/female/non_binary"),
    ethnicity: str = typer.Option(..., help="Ethnicity"),
    birth_year: Optional[int] = typer.Option(None, help="Birth year"),
    profession: Optional[str] = typer.Option(None, help="Profession")
):
    """Add a new celebrity to the dataset."""
    try:
        gender_enum = Gender(gender.lower())
    except ValueError:
        console.print(f"❌ Invalid gender: {gender}. Use: male, female, non_binary", style="red")
        return
    
    celebrity = CelebrityInfo(
        id=celebrity_id,
        name=name,
        gender=gender_enum,
        ethnicity=ethnicity,
        birth_year=birth_year,
        profession=profession
    )
    
    manager = DatasetManager(path)
    celeb_path = manager.add_celebrity(celebrity)
    
    console.print(f"✅ Added {name} to dataset", style="green")
    console.print(f"📁 Structure created at: {celeb_path}", style="blue")


@app.command()
def process(
    path: str = typer.Argument(..., help="Dataset root directory"),
    celebrity_id: str = typer.Option(..., help="Celebrity ID to process"),
    min_quality: float = typer.Option(0.6, help="Minimum quality threshold"),
    generate_captions: bool = typer.Option(True, help="Generate training captions"),
    create_augmentations: bool = typer.Option(False, help="Create augmented versions"),
    skip_face_detection: bool = typer.Option(False, "--skip-face-detection", help="Skip face detection, use full images")
):
    """Process raw images for a celebrity."""
    manager = DatasetManager(path)

    with Progress() as progress:
        task = progress.add_task(f"Processing {celebrity_id}...", total=None)

        processed_count = manager.process_raw_images(
            celebrity_id=celebrity_id,
            min_quality=min_quality,
            generate_captions=generate_captions,
            create_augmentations=create_augmentations,
            skip_face_detection=skip_face_detection
        )

    if processed_count > 0:
        console.print(f"✅ Processed {processed_count} images for {celebrity_id}", style="green")
    else:
        console.print(f"❌ No images processed. Check raw image folders.", style="red")


@app.command()
def split(
    path: str = typer.Argument(..., help="Dataset root directory"),
    celebrity_id: str = typer.Option(..., help="Celebrity ID"),
    validation_ratio: float = typer.Option(0.15, help="Validation split ratio")
):
    """Create training/validation split."""
    manager = DatasetManager(path)
    
    stats = manager.create_training_split(
        celebrity_id=celebrity_id,
        validation_ratio=validation_ratio
    )
    
    console.print(f"📊 Split created for {celebrity_id}:", style="blue")
    console.print(f"  Training: {stats['training_images']} images")
    console.print(f"  Validation: {stats['validation_images']} images")


@app.command()
def config(
    path: str = typer.Argument(..., help="Dataset root directory"),
    output: str = typer.Option("training_config.yaml", help="Output config file"),
    celebrity_ids: List[str] = typer.Option(None, help="Celebrity IDs to include")
):
    """Generate training configuration."""
    manager = DatasetManager(path)
    
    if not celebrity_ids:
        celebrity_ids = manager.organizer.list_celebrities()
    
    if not celebrity_ids:
        console.print("❌ No celebrities found in dataset", style="red")
        return
    
    success = manager.generate_training_config(celebrity_ids, output)
    
    if success:
        console.print(f"✅ Training config saved to: {output}", style="green")
    else:
        console.print("❌ Failed to generate config", style="red")


@app.command()
def list_celebrities(
    path: str = typer.Argument(..., help="Dataset root directory")
):
    """List all celebrities in the dataset."""
    manager = DatasetManager(path)
    celebrity_ids = manager.organizer.list_celebrities()
    
    if not celebrity_ids:
        console.print("📭 No celebrities found in dataset", style="yellow")
        return
    
    table = Table(title="Celebrities in Dataset")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Total Images", justify="right")
    table.add_column("Training", justify="right")
    table.add_column("Validation", justify="right")
    
    for celebrity_id in celebrity_ids:
        dataset = manager.organizer.load_celebrity_dataset(celebrity_id)
        if dataset:
            table.add_row(
                celebrity_id,
                dataset.celebrity_info.name,
                str(dataset.total_images),
                str(dataset.training_images),
                str(dataset.validation_images)
            )
    
    console.print(table)


@app.command()
def summary(
    path: str = typer.Argument(..., help="Dataset root directory")
):
    """Show dataset summary statistics."""
    manager = DatasetManager(path)
    summary = manager.get_dataset_summary()
    
    console.print("📊 Dataset Summary", style="bold blue")
    console.print(f"  Total Celebrities: {summary['totals']['celebrity_count']}")
    console.print(f"  Total Images: {summary['totals']['total_images']}")
    console.print(f"  Training Images: {summary['totals']['total_training']}")
    console.print(f"  Validation Images: {summary['totals']['total_validation']}")
    console.print(f"  Avg per Celebrity: {summary['totals']['avg_images_per_celebrity']:.1f}")


@app.command()
def cleanup(
    path: str = typer.Argument(..., help="Dataset root directory"),
    celebrity_id: str = typer.Option(..., help="Celebrity ID to clean up")
):
    """Clean up dataset by removing low-quality and duplicate images."""
    manager = DatasetManager(path)
    
    stats = manager.cleanup_dataset(celebrity_id)
    
    console.print(f"🧹 Cleanup completed for {celebrity_id}:", style="green")
    console.print(f"  Removed: {stats['removed']} images")
    console.print(f"  Remaining: {stats['remaining']} images")


@app.command()
def validate(
    path: str = typer.Argument(..., help="Dataset root directory"),
    celebrity_id: str = typer.Option(..., help="Celebrity ID to validate")
):
    """Validate dataset structure."""
    manager = DatasetManager(path)
    
    validation = manager.organizer.validate_structure(celebrity_id)
    
    console.print(f"🔍 Validation Results for {celebrity_id}:", style="blue")
    
    for check, result in validation.items():
        if check == 'all_valid':
            continue
        
        status = "✅" if result else "❌"
        console.print(f"  {status} {check.replace('_', ' ').title()}")
    
    if validation['all_valid']:
        console.print("🎉 All validation checks passed!", style="green")
    else:
        console.print("⚠️ Some validation checks failed", style="yellow")


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
