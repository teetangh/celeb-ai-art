"""Data models for celebrity dataset management."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
from pathlib import Path
import json


class Gender(str, Enum):
    """Gender enumeration."""
    MALE = "male"
    FEMALE = "female"
    NON_BINARY = "non_binary"
    UNKNOWN = "unknown"


class ImageType(str, Enum):
    """Image type enumeration."""
    PORTRAIT = "portrait"
    HEADSHOT = "headshot"
    FULL_BODY = "full_body"
    CANDID = "candid"
    RED_CARPET = "red_carpet"
    MOVIE_STILL = "movie_still"
    PROFESSIONAL = "professional"


class QualityLevel(str, Enum):
    """Quality level enumeration."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class CelebrityInfo:
    """Celebrity information data class."""
    id: str
    name: str
    gender: Gender
    ethnicity: str
    birth_year: Optional[int] = None
    profession: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "gender": self.gender.value,
            "ethnicity": self.ethnicity,
            "birth_year": self.birth_year,
            "profession": self.profession,
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CelebrityInfo":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            gender=Gender(data["gender"]),
            ethnicity=data["ethnicity"],
            birth_year=data.get("birth_year"),
            profession=data.get("profession"),
            notes=data.get("notes")
        )


@dataclass
class FaceDetection:
    """Face detection results."""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    landmarks: List[Tuple[float, float]]
    confidence: float
    pose: Optional[Dict[str, float]] = None  # yaw, pitch, roll

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "bbox": self.bbox,
            "landmarks": self.landmarks,
            "confidence": self.confidence,
            "pose": self.pose
        }


@dataclass
class ImageAttributes:
    """Image attributes for training data."""
    age_appearance: Optional[str] = None
    expression: Optional[str] = None
    lighting: Optional[str] = None
    background: Optional[str] = None
    image_type: Optional[ImageType] = None
    clothing: Optional[str] = None
    accessories: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "age_appearance": self.age_appearance,
            "expression": self.expression,
            "lighting": self.lighting,
            "background": self.background,
            "image_type": self.image_type.value if self.image_type else None,
            "clothing": self.clothing,
            "accessories": self.accessories
        }


@dataclass
class ImageMetadata:
    """Metadata for a single image."""
    filename: str
    source_url: Optional[str] = None
    license: Optional[str] = None
    quality_score: Optional[float] = None
    quality_level: Optional[QualityLevel] = None
    face_detection: Optional[FaceDetection] = None
    attributes: Optional[ImageAttributes] = None
    caption: Optional[str] = None
    tags: List[str] = None
    processed_date: Optional[str] = None
    file_size: Optional[int] = None
    dimensions: Optional[Tuple[int, int]] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "filename": self.filename,
            "source_url": self.source_url,
            "license": self.license,
            "quality_score": self.quality_score,
            "quality_level": self.quality_level.value if self.quality_level else None,
            "face_detection": self.face_detection.to_dict() if self.face_detection else None,
            "attributes": self.attributes.to_dict() if self.attributes else None,
            "caption": self.caption,
            "tags": self.tags,
            "processed_date": self.processed_date,
            "file_size": self.file_size,
            "dimensions": self.dimensions
        }


@dataclass
class CelebrityDataset:
    """Complete dataset metadata for a celebrity."""
    celebrity_info: CelebrityInfo
    images: List[ImageMetadata]
    created_date: Optional[str] = None
    last_updated: Optional[str] = None
    dataset_version: str = "1.0"
    total_images: int = 0
    training_images: int = 0
    validation_images: int = 0

    def __post_init__(self):
        """Calculate statistics."""
        self.total_images = len(self.images)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "celebrity_info": self.celebrity_info.to_dict(),
            "images": [img.to_dict() for img in self.images],
            "created_date": self.created_date,
            "last_updated": self.last_updated,
            "dataset_version": self.dataset_version,
            "total_images": self.total_images,
            "training_images": self.training_images,
            "validation_images": self.validation_images
        }

    def save_to_file(self, filepath: Path) -> None:
        """Save dataset metadata to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filepath: Path) -> "CelebrityDataset":
        """Load dataset metadata from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        celebrity_info = CelebrityInfo.from_dict(data["celebrity_info"])
        
        images = []
        for img_data in data["images"]:
            # Reconstruct complex objects
            face_detection = None
            if img_data.get("face_detection"):
                face_data = img_data["face_detection"]
                face_detection = FaceDetection(
                    bbox=tuple(face_data["bbox"]),
                    landmarks=[tuple(lm) for lm in face_data["landmarks"]],
                    confidence=face_data["confidence"],
                    pose=face_data.get("pose")
                )
            
            attributes = None
            if img_data.get("attributes"):
                attr_data = img_data["attributes"]
                attributes = ImageAttributes(
                    age_appearance=attr_data.get("age_appearance"),
                    expression=attr_data.get("expression"),
                    lighting=attr_data.get("lighting"),
                    background=attr_data.get("background"),
                    image_type=ImageType(attr_data["image_type"]) if attr_data.get("image_type") else None,
                    clothing=attr_data.get("clothing"),
                    accessories=attr_data.get("accessories")
                )
            
            quality_level = None
            if img_data.get("quality_level"):
                quality_level = QualityLevel(img_data["quality_level"])
            
            image_meta = ImageMetadata(
                filename=img_data["filename"],
                source_url=img_data.get("source_url"),
                license=img_data.get("license"),
                quality_score=img_data.get("quality_score"),
                quality_level=quality_level,
                face_detection=face_detection,
                attributes=attributes,
                caption=img_data.get("caption"),
                tags=img_data.get("tags", []),
                processed_date=img_data.get("processed_date"),
                file_size=img_data.get("file_size"),
                dimensions=tuple(img_data["dimensions"]) if img_data.get("dimensions") else None
            )
            images.append(image_meta)
        
        return cls(
            celebrity_info=celebrity_info,
            images=images,
            created_date=data.get("created_date"),
            last_updated=data.get("last_updated"),
            dataset_version=data.get("dataset_version", "1.0"),
            total_images=data.get("total_images", len(images)),
            training_images=data.get("training_images", 0),
            validation_images=data.get("validation_images", 0)
        )
