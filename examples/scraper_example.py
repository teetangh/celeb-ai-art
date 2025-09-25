"""Example of using the Google Images scraper to collect celebrity images."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.models import CelebrityInfo, Gender
from core.dataset_manager import DatasetManager
from scrapers.google_images_scraper import GoogleImagesScraper


def scrape_celebrity_dataset():
    """Demonstrate scraping and processing a complete celebrity dataset."""
    
    print("🎬 Celebrity Image Scraping & Processing Demo")
    print("=" * 55)
    
    # Initialize dataset manager
    dataset_path = "./celebrity_dataset"
    manager = DatasetManager(dataset_path, trigger_word="ohwx")
    
    # Celebrity information
    celebrity = CelebrityInfo(
        id="brad_pitt_001",
        name="Brad Pitt",
        gender=Gender.MALE,
        ethnicity="caucasian",
        birth_year=1963,
        profession="actor"
    )
    
    print(f"👤 Working with: {celebrity.name}")
    
    # Step 1: Create dataset structure
    print("\n1. Creating dataset structure...")
    celeb_path = manager.add_celebrity(celebrity)
    raw_output_dir = celeb_path / "raw" / "high_quality"
    
    # Step 2: Scrape images
    print("\n2. Scraping images from Google...")
    
    search_terms = [
        f"{celebrity.name}",
        f"{celebrity.name} portrait",
        f"{celebrity.name} headshot", 
        f"{celebrity.name} professional photo",
        f"{celebrity.name} red carpet",
        f"{celebrity.name} movie premiere"
    ]
    
    # Configure scraper
    with GoogleImagesScraper(use_selenium=False, headless=True) as scraper:
        results = scraper.scrape_celebrity_images(
            celebrity_name=celebrity.name,
            output_dir=str(raw_output_dir),
            num_images=20,  # 20 images per search term
            search_terms=search_terms[:3]  # Use first 3 search terms
        )
        
        # Count total downloaded images
        total_downloaded = sum(len(files) for files in results.values())
        print(f"📥 Total images downloaded: {total_downloaded}")
        
        # Show breakdown
        for search_term, files in results.items():
            print(f"  • {search_term}: {len(files)} images")
    
    # Step 3: Process scraped images
    if total_downloaded > 0:
        print("\n3. Processing scraped images...")
        
        processed_count = manager.process_raw_images(
            celebrity_id=celebrity.id,
            min_quality=0.6,
            generate_captions=True,
            create_augmentations=True if total_downloaded < 30 else False
        )
        
        print(f"✅ Processed {processed_count} high-quality images")
        
        # Step 4: Create training split
        if processed_count > 5:
            print("\n4. Creating training/validation split...")
            
            split_stats = manager.create_training_split(
                celebrity_id=celebrity.id,
                validation_ratio=0.15
            )
            
            print(f"📊 Training: {split_stats['training_images']}, Validation: {split_stats['validation_images']}")
            
            # Step 5: Generate training config
            print("\n5. Generating training configuration...")
            
            config_path = Path(dataset_path) / "configs" / f"{celebrity.id}_training_config.yaml"
            success = manager.generate_training_config(
                celebrity_ids=[celebrity.id],
                output_path=str(config_path)
            )
            
            if success:
                print(f"📝 Training config saved: {config_path}")
            
            # Step 6: Show dataset summary
            print("\n6. Dataset Summary:")
            summary = manager.get_dataset_summary()
            
            celeb_info = summary['celebrities'].get(celebrity.id, {})
            print(f"  🎭 {celebrity.name}")
            print(f"  📸 Total images: {celeb_info.get('total_images', 0)}")
            print(f"  🎯 Training images: {celeb_info.get('training_images', 0)}")
            print(f"  ✅ Validation images: {celeb_info.get('validation_images', 0)}")
        
        else:
            print("⚠️ Not enough processed images for training split")
    else:
        print("❌ No images were downloaded. Check your internet connection.")
    
    print("\n🎉 Demo complete!")
    print("\nNext steps:")
    print("1. Review downloaded images in:", raw_output_dir)
    print("2. Add more images if needed")
    print("3. Use the training config to train your LoRA model")
    print("4. Generate amazing AI art!")


def scrape_multiple_celebrities():
    """Example of scraping images for multiple celebrities."""
    
    print("🎭 Multi-Celebrity Scraping Demo")
    print("=" * 40)
    
    celebrities = [
        CelebrityInfo(
            id="brad_pitt_001",
            name="Brad Pitt",
            gender=Gender.MALE,
            ethnicity="caucasian",
            birth_year=1963
        ),
        CelebrityInfo(
            id="scarlett_johansson_002", 
            name="Scarlett Johansson",
            gender=Gender.FEMALE,
            ethnicity="caucasian",
            birth_year=1984
        ),
        CelebrityInfo(
            id="denzel_washington_003",
            name="Denzel Washington", 
            gender=Gender.MALE,
            ethnicity="african_american",
            birth_year=1954
        )
    ]
    
    manager = DatasetManager("./multi_celebrity_dataset", trigger_word="ohwx")
    
    for celebrity in celebrities:
        print(f"\n🎬 Processing: {celebrity.name}")
        
        # Create structure
        celeb_path = manager.add_celebrity(celebrity)
        raw_output_dir = celeb_path / "raw" / "high_quality"
        
        # Scrape with basic search terms
        with GoogleImagesScraper() as scraper:
            results = scraper.scrape_celebrity_images(
                celebrity_name=celebrity.name,
                output_dir=str(raw_output_dir),
                num_images=15,  # Fewer images per celebrity
                search_terms=[
                    celebrity.name,
                    f"{celebrity.name} portrait"
                ]
            )
            
            total = sum(len(files) for files in results.values())
            print(f"  📥 Downloaded: {total} images")
    
    # Process all at once
    print("\n🔄 Processing all celebrities...")
    results = manager.bulk_process_all(min_quality=0.6)
    
    print("\n📊 Final Summary:")
    for celeb_id, stats in results.items():
        print(f"  • {celeb_id}: {stats['processed']} processed, {stats['training']} training")


def quick_scrape_demo():
    """Quick demonstration using the convenience function."""
    
    print("⚡ Quick Scrape Demo")
    print("=" * 25)
    
    from scrapers.google_images_scraper import quick_scrape_celebrity
    
    # Quick scrape with convenience function
    files = quick_scrape_celebrity(
        celebrity_name="Ryan Reynolds",
        output_dir="./quick_scrape_test",
        num_images=10,
        use_selenium=False
    )
    
    print(f"✅ Quick scrape complete: {len(files)} images")
    
    for i, filepath in enumerate(files[:5], 1):  # Show first 5
        print(f"  {i}. {Path(filepath).name}")
    
    if len(files) > 5:
        print(f"  ... and {len(files) - 5} more")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Celebrity Image Scraping Examples")
    parser.add_argument(
        "--demo", 
        choices=["single", "multi", "quick"],
        default="single",
        help="Demo type to run"
    )
    
    args = parser.parse_args()
    
    if args.demo == "single":
        scrape_celebrity_dataset()
    elif args.demo == "multi":
        scrape_multiple_celebrities()
    elif args.demo == "quick":
        quick_scrape_demo()
    
    print("\n💡 Tips:")
    print("- Use --demo single for complete single celebrity workflow")
    print("- Use --demo multi for multiple celebrities")  
    print("- Use --demo quick for fast testing")
    print("- Set use_selenium=True for more thorough scraping (requires ChromeDriver)")
    print("- Always respect robots.txt and rate limits!")
    print("- Be mindful of copyright and usage rights")
