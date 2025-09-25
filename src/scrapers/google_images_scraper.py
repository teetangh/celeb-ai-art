"""Google Images scraper for celebrity photos."""

import requests
import time
import json
import re
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Set
from urllib.parse import urlencode, urlparse
from urllib.request import urlretrieve
import random
from dataclasses import dataclass

# Selenium imports (optional)
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

import warnings
warnings.filterwarnings("ignore")


@dataclass
class ImageResult:
    """Container for scraped image information."""
    url: str
    title: str
    width: Optional[int] = None
    height: Optional[int] = None
    file_size: Optional[int] = None
    source_site: Optional[str] = None
    thumbnail_url: Optional[str] = None


class GoogleImagesScraper:
    """Advanced Google Images scraper with multiple methods."""
    
    def __init__(self, use_selenium: bool = False, headless: bool = True):
        """
        Initialize the Google Images scraper.
        
        Args:
            use_selenium: Whether to use Selenium for JavaScript-heavy scraping
            headless: Run browser in headless mode (for Selenium)
        """
        self.use_selenium = use_selenium and SELENIUM_AVAILABLE
        self.headless = headless
        self.driver = None
        
        # Headers for requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # Rate limiting
        self.min_delay = 1.0
        self.max_delay = 3.0
        
        # Downloaded URLs tracking
        self.downloaded_urls: Set[str] = set()
        
    def setup_selenium_driver(self) -> None:
        """Setup Selenium Chrome driver."""
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium not available. Install with: pip install selenium")
        
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument(f'user-agent={self.headers["User-Agent"]}')
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        except Exception as e:
            print(f"❌ Failed to setup Chrome driver: {e}")
            print("💡 Make sure ChromeDriver is installed and in PATH")
            raise
    
    def search_images_requests(
        self, 
        query: str, 
        limit: int = 50,
        image_type: str = "photo",
        size: str = "medium"
    ) -> List[ImageResult]:
        """
        Search Google Images using requests (faster but limited).
        
        Args:
            query: Search query
            limit: Maximum number of images
            image_type: Type of images (photo, clipart, etc.)
            size: Image size filter (small, medium, large, xlarge)
            
        Returns:
            List of ImageResult objects
        """
        print(f"🔍 Searching Google Images for: {query} (requests method)")
        
        # Build search URL
        params = {
            'q': query,
            'tbm': 'isch',  # Image search
            'tbs': f'itp:{image_type},isz:{size}',
            'safe': 'off',
            'num': min(limit, 100)
        }
        
        url = f"https://www.google.com/search?{urlencode(params)}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Extract image data from HTML
            results = self._parse_images_from_html(response.text)
            
            # Limit results
            results = results[:limit]
            
            print(f"✅ Found {len(results)} images using requests method")
            return results
            
        except Exception as e:
            print(f"❌ Error searching with requests: {e}")
            return []
    
    def search_images_selenium(
        self, 
        query: str, 
        limit: int = 50,
        scroll_pause_time: float = 2.0
    ) -> List[ImageResult]:
        """
        Search Google Images using Selenium (slower but more comprehensive).
        
        Args:
            query: Search query
            limit: Maximum number of images
            scroll_pause_time: Pause time between scrolls
            
        Returns:
            List of ImageResult objects
        """
        if not self.use_selenium:
            return self.search_images_requests(query, limit)
        
        print(f"🔍 Searching Google Images for: {query} (Selenium method)")
        
        if not self.driver:
            self.setup_selenium_driver()
        
        # Build search URL
        search_url = f"https://www.google.com/search?q={query}&tbm=isch&safe=off"
        
        try:
            self.driver.get(search_url)
            time.sleep(2)
            
            results = []
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            
            while len(results) < limit:
                # Scroll down to load more images
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(scroll_pause_time)
                
                # Find image elements
                img_elements = self.driver.find_elements(By.CSS_SELECTOR, "img[data-src], img[src]")
                
                for img_elem in img_elements:
                    if len(results) >= limit:
                        break
                    
                    try:
                        img_url = img_elem.get_attribute('data-src') or img_elem.get_attribute('src')
                        if not img_url or img_url in [result.url for result in results]:
                            continue
                        
                        if not self._is_valid_image_url(img_url):
                            continue
                        
                        # Get additional metadata
                        title = img_elem.get_attribute('alt') or f"Image {len(results) + 1}"
                        
                        result = ImageResult(
                            url=img_url,
                            title=title,
                            width=None,
                            height=None
                        )
                        
                        results.append(result)
                        
                    except Exception as e:
                        continue
                
                # Check if we've reached the bottom
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    # Try clicking "Show more results" button
                    try:
                        show_more = WebDriverWait(self.driver, 2).until(
                            EC.element_to_be_clickable((By.XPATH, "//input[@value='Show more results']"))
                        )
                        show_more.click()
                        time.sleep(3)
                    except:
                        break
                
                last_height = new_height
            
            print(f"✅ Found {len(results)} images using Selenium method")
            return results
            
        except Exception as e:
            print(f"❌ Error searching with Selenium: {e}")
            return []
    
    def _parse_images_from_html(self, html: str) -> List[ImageResult]:
        """Parse image URLs from Google Images HTML."""
        results = []
        
        # Find JSON data containing image information
        json_pattern = r'\["(https://[^"]+)",(\d+),(\d+)\]'
        matches = re.findall(json_pattern, html)
        
        for match in matches:
            img_url, width, height = match
            
            if not self._is_valid_image_url(img_url):
                continue
            
            result = ImageResult(
                url=img_url,
                title=f"Image {len(results) + 1}",
                width=int(width) if width.isdigit() else None,
                height=int(height) if height.isdigit() else None
            )
            
            results.append(result)
        
        # Fallback: find img tags
        if not results:
            img_pattern = r'<img[^>]+src=[\'"]([^\'"]+)[\'"][^>]*>'
            img_matches = re.findall(img_pattern, html)
            
            for img_url in img_matches:
                if self._is_valid_image_url(img_url):
                    result = ImageResult(
                        url=img_url,
                        title=f"Image {len(results) + 1}"
                    )
                    results.append(result)
        
        return results
    
    def _is_valid_image_url(self, url: str) -> bool:
        """Check if URL is a valid image URL."""
        if not url or not url.startswith('http'):
            return False
        
        # Skip Google's internal URLs
        skip_patterns = [
            'encrypted-tbn0.gstatic.com',
            'googleusercontent.com',
            'data:image',
            'logo',
            'icon'
        ]
        
        for pattern in skip_patterns:
            if pattern in url:
                return False
        
        # Check file extension
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif']
        has_valid_ext = any(path.endswith(ext) for ext in valid_extensions)
        
        # Accept URLs without extension if they look like image URLs
        return has_valid_ext or 'image' in url.lower()
    
    def download_images(
        self, 
        results: List[ImageResult], 
        output_dir: str,
        celebrity_name: str,
        max_concurrent: int = 5
    ) -> List[str]:
        """
        Download images from search results.
        
        Args:
            results: List of ImageResult objects
            output_dir: Output directory
            celebrity_name: Celebrity name for filename prefix
            max_concurrent: Maximum concurrent downloads
            
        Returns:
            List of successfully downloaded file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        downloaded_files = []
        failed_count = 0
        
        print(f"📥 Downloading {len(results)} images to {output_dir}")
        
        for i, result in enumerate(results, 1):
            try:
                # Generate filename
                url_hash = hashlib.md5(result.url.encode()).hexdigest()[:8]
                safe_name = re.sub(r'[^\w\s-]', '', celebrity_name).strip().replace(' ', '_')
                filename = f"{safe_name}_{i:03d}_{url_hash}.jpg"
                filepath = output_path / filename
                
                # Skip if already downloaded
                if result.url in self.downloaded_urls:
                    continue
                
                # Download with retry logic
                success = self._download_single_image(result.url, filepath)
                
                if success:
                    downloaded_files.append(str(filepath))
                    self.downloaded_urls.add(result.url)
                    print(f"✅ Downloaded {i}/{len(results)}: {filename}")
                else:
                    failed_count += 1
                    print(f"❌ Failed {i}/{len(results)}: {result.url}")
                
                # Rate limiting
                time.sleep(random.uniform(self.min_delay, self.max_delay))
                
            except Exception as e:
                failed_count += 1
                print(f"❌ Error downloading {result.url}: {e}")
        
        print(f"🎯 Download complete: {len(downloaded_files)} success, {failed_count} failed")
        return downloaded_files
    
    def _download_single_image(self, url: str, filepath: Path, retries: int = 3) -> bool:
        """Download a single image with retries."""
        for attempt in range(retries):
            try:
                headers = self.headers.copy()
                headers['Referer'] = 'https://www.google.com/'
                
                response = requests.get(url, headers=headers, timeout=15, stream=True)
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if not any(img_type in content_type for img_type in ['image', 'jpeg', 'png', 'webp']):
                    return False
                
                # Save image
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Verify file size
                if filepath.stat().st_size < 1024:  # Less than 1KB
                    filepath.unlink()
                    return False
                
                return True
                
            except Exception as e:
                if attempt == retries - 1:
                    print(f"Final attempt failed for {url}: {e}")
                time.sleep(1)
        
        return False
    
    def scrape_celebrity_images(
        self,
        celebrity_name: str,
        output_dir: str,
        num_images: int = 50,
        search_terms: Optional[List[str]] = None,
        use_selenium: bool = False
    ) -> Dict[str, List[str]]:
        """
        Comprehensive celebrity image scraping.
        
        Args:
            celebrity_name: Celebrity name
            output_dir: Output directory
            num_images: Number of images per search term
            search_terms: Additional search terms
            use_selenium: Whether to use Selenium method
            
        Returns:
            Dictionary mapping search terms to downloaded files
        """
        # Set scraping method
        self.use_selenium = use_selenium and SELENIUM_AVAILABLE
        
        # Default search terms
        if not search_terms:
            search_terms = [
                celebrity_name,
                f"{celebrity_name} portrait",
                f"{celebrity_name} headshot", 
                f"{celebrity_name} professional photo",
                f"{celebrity_name} red carpet",
                f"{celebrity_name} event photo"
            ]
        
        results = {}
        
        for search_term in search_terms:
            print(f"\n🔍 Searching: {search_term}")
            
            # Search for images
            if self.use_selenium:
                image_results = self.search_images_selenium(search_term, num_images)
            else:
                image_results = self.search_images_requests(search_term, num_images)
            
            if image_results:
                # Create subdirectory for this search term
                safe_term = re.sub(r'[^\w\s-]', '', search_term).strip().replace(' ', '_')
                term_output_dir = Path(output_dir) / safe_term
                
                # Download images
                downloaded_files = self.download_images(
                    image_results,
                    str(term_output_dir),
                    celebrity_name
                )
                
                results[search_term] = downloaded_files
            else:
                results[search_term] = []
        
        return results
    
    def close(self):
        """Clean up resources."""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience functions
def quick_scrape_celebrity(
    celebrity_name: str,
    output_dir: str,
    num_images: int = 30,
    use_selenium: bool = False
) -> List[str]:
    """
    Quick function to scrape celebrity images.
    
    Args:
        celebrity_name: Celebrity name
        output_dir: Output directory
        num_images: Number of images to scrape
        use_selenium: Whether to use Selenium
        
    Returns:
        List of downloaded image paths
    """
    with GoogleImagesScraper(use_selenium=use_selenium) as scraper:
        results = scraper.scrape_celebrity_images(
            celebrity_name=celebrity_name,
            output_dir=output_dir,
            num_images=num_images
        )
        
        all_files = []
        for files in results.values():
            all_files.extend(files)
        
        return all_files


if __name__ == "__main__":
    # Example usage
    celebrity = "Brad Pitt"
    output = "./scraped_images"
    
    print(f"🎬 Scraping images for {celebrity}")
    
    files = quick_scrape_celebrity(
        celebrity_name=celebrity,
        output_dir=output,
        num_images=20,
        use_selenium=False  # Set to True for more thorough scraping
    )
    
    print(f"🎯 Downloaded {len(files)} images total")
    print("Files saved to:", output)
