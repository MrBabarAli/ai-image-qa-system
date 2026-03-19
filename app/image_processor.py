"""
Image Processor Module
Handles image upload, validation, and preprocessing
"""

from PIL import Image
import io
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
import os

class ImageProcessor:
    """Process and prepare images for vision models"""
    
    def __init__(self, target_size: Tuple[int, int] = (384, 384)):
        """
        Initialize image processor
        
        Args:
            target_size: Target size for model input (width, height)
        """
        self.target_size = target_size
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        print(f"✅ ImageProcessor initialized (target size: {target_size})")
    
    def validate_image(self, image_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Validate uploaded image
        
        Args:
            image_bytes: Raw image bytes
            filename: Original filename
            
        Returns:
            Dictionary with validation results and metadata
        """
        result = {
            'valid': False,
            'format': None,
            'size': None,
            'mode': None,
            'error': None
        }
        
        # Check file extension
        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.supported_formats:
            result['error'] = f"Unsupported format: {ext}. Supported: {self.supported_formats}"
            return result
        
        try:
            # Open image with PIL
            img = Image.open(io.BytesIO(image_bytes))
            
            # Get image info
            result['valid'] = True
            result['format'] = img.format
            result['size'] = img.size
            result['mode'] = img.mode
            result['width'], result['height'] = img.size
            
        except Exception as e:
            result['error'] = f"Error validating image: {str(e)}"
        
        return result
    
    def preprocess_for_model(self, image_bytes: bytes) -> Optional[Image.Image]:
        """
        Preprocess image for vision model
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Preprocessed PIL Image or None if error
        """
        try:
            # Open image
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Resize while maintaining aspect ratio
            img.thumbnail(self.target_size, Image.Resampling.LANCZOS)
            
            # Create new image with target size and paste resized image
            new_img = Image.new('RGB', self.target_size, (255, 255, 255))
            paste_x = (self.target_size[0] - img.size[0]) // 2
            paste_y = (self.target_size[1] - img.size[1]) // 2
            new_img.paste(img, (paste_x, paste_y))
            
            return new_img
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def extract_image_features(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract basic features from image (for quick analysis)
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with image features
        """
        # Convert PIL to numpy for OpenCV processing
        img_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(img_np.shape) == 3:
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_cv = img_np
        
        features = {
            'width': image.width,
            'height': image.height,
            'aspect_ratio': round(image.width / image.height, 2),
            'mode': image.mode,
            'format': image.format,
            'estimated_size_kb': len(image.tobytes()) // 1024
        }
        
        # Calculate basic color statistics if color image
        if len(img_np.shape) == 3:
            # Average color
            avg_color = np.mean(img_np, axis=(0, 1))
            features['avg_color_rgb'] = [int(c) for c in avg_color]
            
            # Color variance
            color_std = np.std(img_np, axis=(0, 1))
            features['color_variation'] = [int(c) for c in color_std]
        
        return features
    
    def get_image_info(self, image_bytes: bytes) -> str:
        """
        Generate a text description of image metadata
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Formatted string with image information
        """
        img = Image.open(io.BytesIO(image_bytes))
        features = self.extract_image_features(img)
        
        info = f"📷 Image Information:\n"
        info += f"   - Dimensions: {features['width']} x {features['height']} pixels\n"
        info += f"   - Aspect Ratio: {features['aspect_ratio']}\n"
        info += f"   - Format: {features['format']}\n"
        info += f"   - Color Mode: {features['mode']}\n"
        info += f"   - Size: ~{features['estimated_size_kb']} KB\n"
        
        if 'avg_color_rgb' in features:
            info += f"   - Average Color: RGB{tuple(features['avg_color_rgb'])}\n"
        
        return info

# Test the processor
if __name__ == "__main__":
    print("🔧 Testing ImageProcessor...")
    
    # Create processor
    processor = ImageProcessor()
    
    # Create a simple test image
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (200, 150), color='lightblue')
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 30, 150, 120], fill='red')
    draw.text((70, 70), "TEST", fill='white')
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    
    # Test validation
    print("\n📊 Testing validation...")
    result = processor.validate_image(img_bytes, "test_image.png")
    print(f"   Valid: {result['valid']}")
    print(f"   Size: {result['size']}")
    
    # Test preprocessing
    print("\n📊 Testing preprocessing...")
    processed = processor.preprocess_for_model(img_bytes)
    print(f"   Processed size: {processed.size}")
    
    # Test feature extraction
    print("\n📊 Testing feature extraction...")
    features = processor.extract_image_features(Image.open(io.BytesIO(img_bytes)))
    print(f"   Features: {features}")
    
    # Test info generation
    print("\n📊 Testing info generation...")
    info = processor.get_image_info(img_bytes)
    print(info)
    
    print("\n✅ ImageProcessor is working!")