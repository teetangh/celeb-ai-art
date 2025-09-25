# Celebrity AI Art - Dataset Management System

A comprehensive Python system for organizing, processing, and managing celebrity image datasets for LoRA training with Stable Diffusion.

## 🚀 Features

- **Automated Dataset Organization**: Proper folder structure for celebrity datasets
- **Image Processing Pipeline**: Face detection, quality assessment, and cropping
- **Caption Generation**: Automatic LoRA training captions with trigger words
- **Data Augmentation**: Face-preserving augmentations for training data
- **Quality Control**: Automated quality filtering and duplicate detection
- **Training Configuration**: Auto-generated configs for LoRA training

## 📁 Project Structure

```
celeb-ai-art/
├── src/                           # Core modules
│   ├── core/                      # Core dataset management
│   │   ├── models.py              # Data models and types  
│   │   └── dataset_manager.py     # Main dataset orchestrator
│   ├── processing/                # Image processing modules
│   │   ├── face_detector.py       # Face detection utilities
│   │   ├── quality_assessor.py    # Image quality assessment
│   │   ├── image_utils.py         # Image processing utilities
│   │   └── augmentation.py        # Data augmentation
│   ├── utils/                     # Utility modules
│   │   ├── dataset_organizer.py   # Folder structure management
│   │   └── caption_generator.py   # Caption generation for LoRA
│   ├── scrapers/                  # Web scraping modules
│   │   └── google_images_scraper.py # Google Images scraper
│   └── cli.py                     # Command-line interface
├── examples/
│   └── basic_usage.py             # Usage examples
├── pyproject.toml                 # Dependencies (Python 3.12+)
└── README.md                      # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.12+
- Git

### Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd celeb-ai-art

# Install dependencies
pip install -e .

# Or for development with optional dependencies
pip install -e ".[dev,training,web]"
```

## 📊 Dataset Organization

The system creates this folder structure for each celebrity:

```
dataset/celebrities/brad_pitt_001/
├── raw/                     # Original images
│   ├── high_quality/       # 1024x1024+ images
│   ├── medium_quality/     # 512x512+ images
│   └── low_quality/        # <512x512 images
├── processed/              # Processed training images
│   ├── face_crops/         # 512x512 face crops (main training data)
│   ├── full_body/          # Full body shots
│   ├── portraits/          # Head & shoulders
│   └── augmented/          # Augmented variations
├── validation/             # Validation set (15%)
│   ├── face_crops/
│   └── portraits/
├── metadata/               # Dataset metadata
│   ├── annotations.json    # Detailed metadata
│   └── dataset_info.yaml  # Dataset configuration
└── backup/                 # Automatic backups
```

## 🎯 Quick Start

### 1. Basic Setup
```python
from src.core.models import CelebrityInfo, Gender
from src.core.dataset_manager import DatasetManager

# Initialize dataset manager
manager = DatasetManager("./celebrity_dataset", trigger_word="ohwx")

# Add a celebrity
brad_pitt = CelebrityInfo(
    id="brad_pitt_001",
    name="Brad Pitt",
    gender=Gender.MALE,
    ethnicity="caucasian",
    birth_year=1963,
    profession="actor"
)

celeb_path = manager.add_celebrity(brad_pitt)
print(f"✅ Created structure for {brad_pitt.name}")
```

### 2. Scrape Images (Automated)
```python
from src.scrapers.google_images_scraper import GoogleImagesScraper

# Scrape images using Google Images
with GoogleImagesScraper() as scraper:
    downloaded_files = scraper.scrape_celebrity_images(
        celebrity_name="Brad Pitt",
        output_dir="./celebrity_dataset/celebrities/brad_pitt_001/raw/high_quality/",
        num_images=50,
        search_terms=[
            "Brad Pitt",
            "Brad Pitt portrait", 
            "Brad Pitt headshot",
            "Brad Pitt professional photo"
        ]
    )

print(f"📥 Downloaded {len(downloaded_files)} images")
```

### 2b. Add Raw Images (Manual)
Alternatively, manually add images to the `raw/` folders:
```bash
# Add your collected images
celebrity_dataset/celebrities/brad_pitt_001/raw/high_quality/image1.jpg
celebrity_dataset/celebrities/brad_pitt_001/raw/high_quality/image2.jpg
# ... (add 20-50 high-quality images)
```

### 3. Process Images
```python
# Process raw images into training data
processed_count = manager.process_raw_images(
    celebrity_id="brad_pitt_001",
    min_quality=0.6,                # Quality threshold
    generate_captions=True,         # Generate LoRA captions
    create_augmentations=True       # Create augmented versions
)

# Create training/validation split
split_stats = manager.create_training_split(
    celebrity_id="brad_pitt_001",
    validation_ratio=0.15           # 15% for validation
)
```

### 4. Generate Training Config
```python
# Generate LoRA training configuration
manager.generate_training_config(
    celebrity_ids=["brad_pitt_001"],
    output_path="./configs/training_config.yaml"
)
```

## 🖼️ Image Processing Pipeline

### 1. Quality Assessment
- **Resolution scoring**: Prioritizes high-resolution images
- **Sharpness detection**: Uses Laplacian variance to detect blur
- **Brightness/contrast**: Optimal exposure analysis
- **Corruption detection**: Validates image integrity

### 2. Face Detection
- **Multiple backends**: OpenCV (fast) and dlib (accurate)
- **Landmark extraction**: 68-point facial landmarks
- **Pose estimation**: Basic head pose analysis
- **Visibility checking**: Ensures face is sufficiently visible

### 3. Image Processing
- **Smart cropping**: Face-centered crops with padding
- **Aspect ratio preservation**: Maintains proper proportions
- **Format conversion**: Standardizes to JPEG format
- **Thumbnail generation**: Creates preview images

## 📝 Caption Generation

The system generates LoRA training captions automatically:

### Basic Template
```
"ohwx Brad Pitt, professional portrait, high quality"
```

### Detailed Template
```
"ohwx Brad Pitt, middle-aged man, serious expression, studio lighting, high quality"
```

### Professional Template
```
"ohwx Brad Pitt, confident, professional headshot, dramatic lighting, award winning photography"
```

## 🔧 Configuration

### Dataset Configuration
```yaml
dataset:
  base_path: "./dataset/celebrities"
  celebrity_list: ["brad_pitt_001", "dwayne_johnson_002"]
  image_size: 512
  batch_size: 4

training:
  trigger_word: "ohwx"
  learning_rate: 0.0001
  max_train_steps: 1000
  validation_epochs: 10

model:
  base_model: "runwayml/stable-diffusion-v1-5"
  lora_rank: 4
  lora_alpha: 32
```

## 🎨 Data Augmentation

Face-preserving augmentations:
- **Horizontal flip**: 50% probability
- **Brightness/contrast**: ±10% adjustment
- **Color variation**: Subtle hue/saturation changes
- **Rotation**: ±5 degrees maximum
- **Noise addition**: Gaussian noise for robustness

## 📊 Quality Requirements

### Recommended Dataset Sizes
- **Minimum**: 15-20 high-quality images per celebrity
- **Optimal**: 50-100 diverse images per celebrity
- **Maximum useful**: 200-300 images (diminishing returns)

### Image Quality Standards
- **Resolution**: 512x512 minimum, 1024x1024+ preferred
- **Face visibility**: >70% of face visible
- **Sharpness**: No motion blur or focus issues
- **Lighting**: Even exposure, not over/under-exposed

## 🚨 Legal Considerations

⚠️ **Important**: This system is for research and educational purposes. Consider:
- **Fair use guidelines**: Understand applicable fair use laws
- **Celebrity rights**: Respect personality and likeness rights
- **Commercial usage**: Obtain proper licenses for commercial use
- **Takedown requests**: Implement removal procedures

The system includes folders for legal documentation:
```
legal/
├── licenses/              # License agreements
├── consent_forms/         # Celebrity consent (if obtained)
├── fair_use_documentation/ # Fair use justification
└── takedown_requests/     # Handle removal requests
```

## 🔍 Usage Examples

### Complete Workflow Example
```python
# Run the basic usage example
python examples/basic_usage.py

# Process all celebrities with raw images
from examples.basic_usage import process_all_celebrities
process_all_celebrities()
```

### Advanced Usage
```python
# Get dataset statistics
summary = manager.get_dataset_summary()
print(f"Total celebrities: {summary['totals']['celebrity_count']}")

# Cleanup dataset
cleanup_stats = manager.cleanup_dataset("brad_pitt_001")
print(f"Removed {cleanup_stats['removed']} low-quality images")

# Generate color variations
color_paths = manager.augmenter.create_color_variations(
    "path/to/image.jpg",
    "output/directory/"
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Face detection fails**:
   - Ensure images have clearly visible faces
   - Check image quality and resolution
   - Try different source images

2. **Low quality scores**:
   - Use higher resolution source images
   - Ensure proper lighting in source images
   - Check for motion blur or camera shake

3. **Import errors**:
   ```bash
   pip install -e .
   # Or install missing dependencies
   pip install face-recognition opencv-python
   ```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Face detection powered by `face_recognition` library
- Image processing using `OpenCV` and `PIL`
- Data augmentation with `albumentations`
- Built for Stable Diffusion LoRA training

---

**Ready to create amazing celebrity AI art! 🎭✨**

For questions or support, please open an issue on GitHub.
