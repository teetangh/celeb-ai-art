"""Celebrity AI Art Generation - Dataset Management System."""

from .core import DatasetManager, CelebrityInfo, Gender
from .scrapers import GoogleImagesScraper

__version__ = "0.1.0"
__all__ = [
    'DatasetManager',
    'CelebrityInfo', 
    'Gender',
    'GoogleImagesScraper'
]
