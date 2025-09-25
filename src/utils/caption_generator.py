"""Caption generation for LoRA training."""

import random
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

from ..core.models import CelebrityInfo, ImageAttributes, ImageType, Gender


class CaptionGenerator:
    """Generates training captions for LoRA fine-tuning."""
    
    def __init__(self, trigger_word: str = "ohwx"):
        """
        Initialize the caption generator.
        
        Args:
            trigger_word: Trigger word for LoRA training
        """
        self.trigger_word = trigger_word
        self.load_templates()
        self.load_descriptors()
    
    def load_templates(self):
        """Load caption templates."""
        self.templates = {
            'basic': [
                "{trigger} {name}",
                "{trigger} {name}, {description}",
                "{trigger} {name}, {description}, {style}",
            ],
            'detailed': [
                "{trigger} {name}, {age} {gender}, {description}, {style}",
                "{trigger} {name}, {expression}, {description}, {lighting}, {style}",
                "{trigger} {name} in {background}, {description}, {style}",
            ],
            'professional': [
                "{trigger} {name}, professional {image_type}, {lighting}, {style}",
                "{trigger} {name}, {expression} expression, professional photo, {style}",
                "{trigger} {name}, {clothing}, professional photography, {style}",
            ]
        }
    
    def load_descriptors(self):
        """Load descriptive terms for captions."""
        self.descriptors = {
            'expressions': {
                'neutral': ['neutral expression', 'calm', 'composed'],
                'happy': ['smiling', 'cheerful', 'happy', 'joyful', 'grinning'],
                'serious': ['serious expression', 'intense', 'focused', 'stern'],
                'contemplative': ['thoughtful', 'contemplative', 'pensive'],
                'confident': ['confident', 'self-assured', 'determined'],
            },
            'lighting': {
                'studio': ['studio lighting', 'professional lighting', 'controlled lighting'],
                'natural': ['natural lighting', 'daylight', 'outdoor lighting'],
                'dramatic': ['dramatic lighting', 'moody lighting', 'artistic lighting'],
                'soft': ['soft lighting', 'diffused lighting', 'gentle lighting'],
                'golden': ['golden hour lighting', 'warm lighting', 'sunset lighting'],
            },
            'backgrounds': {
                'clean': ['clean background', 'simple background', 'minimal background'],
                'studio': ['studio background', 'professional backdrop'],
                'outdoor': ['outdoor setting', 'natural background'],
                'urban': ['urban setting', 'city background'],
                'indoor': ['indoor setting', 'interior'],
            },
            'image_types': {
                ImageType.PORTRAIT: ['portrait', 'headshot', 'head and shoulders'],
                ImageType.HEADSHOT: ['headshot', 'close-up portrait', 'face shot'],
                ImageType.PROFESSIONAL: ['professional photo', 'business portrait', 'corporate headshot'],
                ImageType.RED_CARPET: ['red carpet photo', 'event photo', 'formal photo'],
                ImageType.CANDID: ['candid photo', 'natural shot', 'lifestyle photo'],
            },
            'styles': [
                'high quality', 'professional photography', 'sharp focus',
                'detailed', 'realistic', 'photorealistic', 'high resolution',
                '8k quality', 'masterpiece', 'best quality', 'ultra detailed',
                'professional', 'cinematic', 'award winning photography'
            ],
            'clothing': {
                'formal': ['suit', 'formal attire', 'business suit', 'tuxedo', 'formal wear'],
                'casual': ['casual shirt', 'casual clothing', 'relaxed attire'],
                'elegant': ['elegant dress', 'sophisticated attire', 'stylish outfit'],
                'professional': ['professional attire', 'business casual', 'work clothes'],
            },
            'ages': {
                'young': ['young', 'youthful'],
                'middle': ['middle-aged'],
                'mature': ['mature', 'distinguished'],
            }
        }
    
    def generate_caption(
        self,
        celebrity: CelebrityInfo,
        image_attributes: Optional[ImageAttributes] = None,
        template_type: str = 'basic',
        style_emphasis: bool = True
    ) -> str:
        """
        Generate a training caption for an image.
        
        Args:
            celebrity: Celebrity information
            image_attributes: Image attributes (if available)
            template_type: Type of template to use
            style_emphasis: Whether to emphasize style/quality terms
            
        Returns:
            Generated caption string
        """
        # Choose template
        templates = self.templates.get(template_type, self.templates['basic'])
        template = random.choice(templates)
        
        # Build caption components
        caption_data = {
            'trigger': self.trigger_word,
            'name': celebrity.name,
        }
        
        # Add gender-specific terms
        if celebrity.gender == Gender.MALE:
            caption_data['gender'] = random.choice(['man', 'male'])
        elif celebrity.gender == Gender.FEMALE:
            caption_data['gender'] = random.choice(['woman', 'female'])
        else:
            caption_data['gender'] = 'person'
        
        # Add age information if available
        if celebrity.birth_year:
            current_year = 2024  # Update as needed
            age = current_year - celebrity.birth_year
            if age < 35:
                caption_data['age'] = random.choice(self.descriptors['ages']['young'])
            elif age < 55:
                caption_data['age'] = random.choice(self.descriptors['ages']['middle'])
            else:
                caption_data['age'] = random.choice(self.descriptors['ages']['mature'])
        else:
            caption_data['age'] = ''
        
        # Add image attributes if available
        if image_attributes:
            caption_data.update(self._process_image_attributes(image_attributes))
        else:
            # Use default/random attributes
            caption_data.update(self._generate_default_attributes())
        
        # Add style terms
        if style_emphasis:
            caption_data['style'] = random.choice(self.descriptors['styles'])
        else:
            caption_data['style'] = ''
        
        # Format template
        try:
            caption = template.format(**caption_data)
            
            # Clean up caption (remove extra spaces, empty terms)
            caption = self._clean_caption(caption)
            
            return caption
            
        except KeyError as e:
            print(f"Missing template key: {e}")
            # Fallback to basic template
            return f"{self.trigger_word} {celebrity.name}, professional photo, high quality"
    
    def _process_image_attributes(self, attributes: ImageAttributes) -> Dict[str, str]:
        """Process image attributes into caption components."""
        caption_data = {}
        
        # Expression
        if attributes.expression:
            expr_key = attributes.expression.lower()
            if expr_key in self.descriptors['expressions']:
                caption_data['expression'] = random.choice(
                    self.descriptors['expressions'][expr_key]
                )
            else:
                caption_data['expression'] = attributes.expression
        else:
            caption_data['expression'] = random.choice(
                self.descriptors['expressions']['neutral']
            )
        
        # Lighting
        if attributes.lighting:
            lighting_key = attributes.lighting.lower()
            if lighting_key in self.descriptors['lighting']:
                caption_data['lighting'] = random.choice(
                    self.descriptors['lighting'][lighting_key]
                )
            else:
                caption_data['lighting'] = attributes.lighting
        else:
            caption_data['lighting'] = random.choice(
                self.descriptors['lighting']['studio']
            )
        
        # Background
        if attributes.background:
            bg_key = attributes.background.lower()
            if bg_key in self.descriptors['backgrounds']:
                caption_data['background'] = random.choice(
                    self.descriptors['backgrounds'][bg_key]
                )
            else:
                caption_data['background'] = attributes.background
        else:
            caption_data['background'] = random.choice(
                self.descriptors['backgrounds']['clean']
            )
        
        # Image type
        if attributes.image_type:
            if attributes.image_type in self.descriptors['image_types']:
                caption_data['image_type'] = random.choice(
                    self.descriptors['image_types'][attributes.image_type]
                )
            else:
                caption_data['image_type'] = 'photo'
        else:
            caption_data['image_type'] = 'portrait'
        
        # Clothing
        if attributes.clothing:
            clothing_lower = attributes.clothing.lower()
            for clothing_type, terms in self.descriptors['clothing'].items():
                if any(term in clothing_lower for term in terms):
                    caption_data['clothing'] = random.choice(terms)
                    break
            else:
                caption_data['clothing'] = attributes.clothing
        else:
            caption_data['clothing'] = ''
        
        # Description (combine multiple attributes)
        desc_parts = []
        if attributes.image_type:
            desc_parts.append(caption_data.get('image_type', ''))
        if not desc_parts:
            desc_parts.append('portrait')
        
        caption_data['description'] = ', '.join(filter(None, desc_parts))
        
        return caption_data
    
    def _generate_default_attributes(self) -> Dict[str, str]:
        """Generate default attributes when image analysis isn't available."""
        return {
            'expression': random.choice(self.descriptors['expressions']['neutral']),
            'lighting': random.choice(self.descriptors['lighting']['studio']),
            'background': random.choice(self.descriptors['backgrounds']['clean']),
            'image_type': random.choice(self.descriptors['image_types'][ImageType.PORTRAIT]),
            'clothing': '',
            'description': 'professional portrait'
        }
    
    def _clean_caption(self, caption: str) -> str:
        """Clean up caption by removing extra spaces and empty terms."""
        # Remove multiple spaces
        caption = ' '.join(caption.split())
        
        # Remove empty terms between commas
        parts = [part.strip() for part in caption.split(',')]
        parts = [part for part in parts if part and part != '']
        
        # Rejoin with proper spacing
        caption = ', '.join(parts)
        
        # Remove trailing comma
        caption = caption.rstrip(', ')
        
        return caption
    
    def generate_batch_captions(
        self,
        celebrity: CelebrityInfo,
        image_paths: List[str],
        attributes_list: Optional[List[ImageAttributes]] = None,
        variety: bool = True
    ) -> Dict[str, str]:
        """
        Generate captions for a batch of images.
        
        Args:
            celebrity: Celebrity information
            image_paths: List of image file paths
            attributes_list: List of image attributes (optional)
            variety: Whether to use variety in templates and terms
            
        Returns:
            Dictionary mapping image paths to captions
        """
        captions = {}
        template_types = ['basic', 'detailed', 'professional'] if variety else ['basic']
        
        for i, image_path in enumerate(image_paths):
            # Get attributes for this image
            attributes = None
            if attributes_list and i < len(attributes_list):
                attributes = attributes_list[i]
            
            # Choose template type with variety
            template_type = random.choice(template_types) if variety else 'basic'
            
            # Generate caption
            caption = self.generate_caption(
                celebrity, 
                attributes, 
                template_type,
                style_emphasis=True
            )
            
            captions[image_path] = caption
        
        return captions
    
    def save_captions_file(
        self, 
        captions: Dict[str, str], 
        output_path: str,
        format: str = 'txt'
    ) -> bool:
        """
        Save captions to file.
        
        Args:
            captions: Dictionary of image_path -> caption
            output_path: Output file path
            format: Output format ('txt', 'json', 'csv')
            
        Returns:
            True if successful
        """
        try:
            output_path = Path(output_path)
            
            if format == 'txt':
                with open(output_path, 'w', encoding='utf-8') as f:
                    for image_path, caption in captions.items():
                        image_name = Path(image_path).stem
                        f.write(f"{image_name}: {caption}\n")
            
            elif format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(captions, f, indent=2, ensure_ascii=False)
            
            elif format == 'csv':
                import csv
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['image_path', 'caption'])
                    for image_path, caption in captions.items():
                        writer.writerow([image_path, caption])
            
            return True
            
        except Exception as e:
            print(f"Error saving captions to {output_path}: {e}")
            return False
    
    def get_caption_statistics(self, captions: Dict[str, str]) -> Dict[str, any]:
        """Get statistics about generated captions."""
        if not captions:
            return {}
        
        caption_lengths = [len(caption.split()) for caption in captions.values()]
        
        return {
            'total_captions': len(captions),
            'avg_length_words': sum(caption_lengths) / len(caption_lengths),
            'min_length_words': min(caption_lengths),
            'max_length_words': max(caption_lengths),
            'trigger_word_usage': sum(1 for caption in captions.values() 
                                     if self.trigger_word in caption),
            'unique_captions': len(set(captions.values())),
        }
