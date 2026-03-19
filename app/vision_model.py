"""
Vision Model Module
Uses Hugging Face transformers for image understanding
"""

from transformers import (
    BlipProcessor, 
    BlipForQuestionAnswering,
    BlipForConditionalGeneration,
)
from PIL import Image
import torch
from typing import Optional, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class VisionModel:
    """Handle vision models for image understanding"""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize vision models
        
        Args:
            device: 'cuda' for GPU, 'cpu' for CPU, None for auto-detect
        """
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"🚀 Initializing VisionModel on {self.device.upper()}...")
        
        # Initialize models (lazy loading - only load when needed)
        self.blip_processor = None
        self.blip_vqa_model = None
        self.blip_caption_model = None
        
        print("✅ VisionModel initialized (models will load on demand)")
    
    def _load_vqa_model(self):
        """Load Visual Question Answering model"""
        if self.blip_vqa_model is None:
            print("   Loading BLIP VQA model (this will download ~1.1GB)...")
            
            # Load processor first if not already loaded
            if self.blip_processor is None:
                print("   Loading BLIP processor...")
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
            
            # Load VQA model
            self.blip_vqa_model = BlipForQuestionAnswering.from_pretrained(
                "Salesforce/blip-vqa-base"
            ).to(self.device)
            
            print("   ✅ BLIP VQA model loaded")
    
    def _load_caption_model(self):
        """Load Image Captioning model"""
        if self.blip_caption_model is None:
            print("   Loading BLIP caption model (this will download ~990MB)...")
            
            # Load processor if not already loaded
            if self.blip_processor is None:
                print("   Loading BLIP processor...")
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            
            # Load caption model
            self.blip_caption_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(self.device)
            
            print("   ✅ BLIP caption model loaded")
    
    def answer_question(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """
        Answer a question about an image using VQA
        
        Args:
            image: PIL Image
            question: Question about the image
            
        Returns:
            Dictionary with answer and confidence
        """
        # Make sure VQA model is loaded
        if self.blip_vqa_model is None:
            self._load_vqa_model()
        
        try:
            # Ensure processor is loaded
            if self.blip_processor is None:
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
            
            # Prepare inputs
            inputs = self.blip_processor(image, question, return_tensors="pt").to(self.device)
            
            # Generate answer
            with torch.no_grad():
                out = self.blip_vqa_model.generate(**inputs, max_length=20)
            
            # Decode answer
            answer = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            # Simple confidence estimation (based on answer length and generation)
            confidence = min(0.5 + (len(answer) / 50), 0.95) if answer else 0.5
            
            return {
                'answer': answer if answer else "I'm not sure",
                'confidence': confidence,
                'question': question,
                'model': 'BLIP-VQA'
            }
            
        except Exception as e:
            print(f"   VQA Error details: {str(e)}")
            return {
                'answer': f"Could not answer: {str(e)}",
                'confidence': 0,
                'question': question,
                'error': str(e)
            }
    
    def generate_caption(self, image: Image.Image) -> Dict[str, Any]:
        """
        Generate a caption describing the image
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with caption and details
        """
        # Make sure caption model is loaded
        if self.blip_caption_model is None:
            self._load_caption_model()
        
        try:
            # Ensure processor is loaded
            if self.blip_processor is None:
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            
            # Prepare inputs
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                out = self.blip_caption_model.generate(**inputs, max_length=50)
            
            # Decode caption
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            
            return {
                'caption': caption if caption else "No caption generated",
                'model': 'BLIP-Captioning',
                'status': 'success'
            }
            
        except Exception as e:
            print(f"   Caption Error details: {str(e)}")
            return {
                'caption': f"Could not generate caption: {str(e)}",
                'model': 'BLIP-Captioning',
                'status': 'error',
                'error': str(e)
            }
    
    def get_image_description(self, image: Image.Image) -> Dict[str, Any]:
        """
        Get comprehensive image description (caption + objects)
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with multiple descriptions
        """
        # Generate caption
        caption_result = self.generate_caption(image)
        
        # Try to answer some common questions
        common_questions = [
            "What is in this image?",
            "What colors are in this image?",
            "Is this indoors or outdoors?"
        ]
        
        answers = {}
        for question in common_questions:
            vqa_result = self.answer_question(image, question)
            answers[question] = vqa_result['answer']
        
        return {
            'caption': caption_result['caption'],
            'answers': answers,
            'image_size': image.size
        }
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        info = {
            'device': self.device,
            'models_loaded': {
                'vqa': self.blip_vqa_model is not None,
                'caption': self.blip_caption_model is not None
            }
        }
        return info

# Test the vision model
if __name__ == "__main__":
    print("🔧 Testing VisionModel...")
    print("=" * 60)
    
    # Create test image
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (400, 300), color='lightblue')
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 50, 300, 200], fill='red')
    draw.ellipse([150, 100, 250, 150], fill='yellow')
    draw.text((180, 120), "TEST", fill='black')
    
    # Initialize model
    vision = VisionModel()
    
    # Test captioning
    print("\n📝 Testing Image Captioning...")
    caption_result = vision.generate_caption(img)
    print(f"   Caption: {caption_result['caption']}")
    
    # Test VQA (this will now load the VQA model)
    print("\n❓ Testing Visual Question Answering...")
    questions = [
        "What is the main color?",
        "What shape is in the image?",
        "What is written in the image?"
    ]
    
    for question in questions:
        result = vision.answer_question(img, question)
        print(f"   Q: {question}")
        print(f"   A: {result['answer']} (confidence: {result['confidence']:.2f})")
    
    # Test comprehensive description
    print("\n📊 Testing Comprehensive Description...")
    description = vision.get_image_description(img)
    print(f"   Caption: {description['caption']}")
    for q, a in description['answers'].items():
        print(f"   {q}: {a}")
    
    print("\n" + "=" * 60)
    print("✅ VisionModel test complete!")
    print("=" * 60)