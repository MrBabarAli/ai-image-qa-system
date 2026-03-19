# app/vqa_engine.py
"""
VQA Engine - Wrapper for vision models
"""

from app.vision_model import VisionModel
from PIL import Image
import time
from typing import Dict, Any

class VQAEngine:
    """Main engine for image question answering"""
    
    def __init__(self):
        """Initialize the VQA engine with vision model"""
        print("🚀 Initializing VQA Engine...")
        self.model = None
        print("✅ VQA Engine ready (model will load on first use)")
    
    def _load_model(self):
        """Lazy load the vision model"""
        if self.model is None:
            print("   Loading vision model (first time only)...")
            start_time = time.time()
            self.model = VisionModel()
            elapsed = time.time() - start_time
            print(f"   ✅ Model loaded in {elapsed:.1f} seconds")
    
    def answer_question(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """Answer a question about an image"""
        self._load_model()
        return self.model.answer_question(image, question)
    
    def generate_caption(self, image: Image.Image) -> Dict[str, Any]:
        """Generate a caption for the image"""
        self._load_model()
        return self.model.generate_caption(image)
    
    def get_image_description(self, image: Image.Image) -> Dict[str, Any]:
        """Get comprehensive description"""
        self._load_model()
        return self.model.get_image_description(image)