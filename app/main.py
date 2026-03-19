# app/main.py
"""
Streamlit Web Interface for AI Image Question Answering System
"""

import streamlit as st
from PIL import Image
import io
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.image_processor import ImageProcessor
from app.vqa_engine import VQAEngine

# Page configuration
st.set_page_config(
    page_title="AI Image QA System",
    page_icon="🖼️",
    layout="wide"
)

# Initialize session state
if 'vqa_engine' not in st.session_state:
    st.session_state.vqa_engine = VQAEngine()
    st.session_state.processor = ImageProcessor()
    st.session_state.current_image = None
    st.session_state.current_image_bytes = None
    st.session_state.chat_history = []
    st.session_state.image_uploaded = False

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        color: #4a5568;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-box {
        border: 2px dashed #4299e1;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f7fafc;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .answer-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border-left: 5px solid #4299e1;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        animation: fadeIn 0.5s;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .user-message {
        background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
        color: white;
        margin-left: 20%;
        border-radius: 15px 15px 0 15px;
    }
    .bot-message {
        background-color: #edf2f7;
        color: #2d3748;
        margin-right: 20%;
        border-radius: 15px 15px 15px 0;
    }
    .stButton button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .confidence-bar {
        margin-top: 0.5rem;
        padding: 0.2rem;
        background-color: #e2e8f0;
        border-radius: 10px;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        text-align: right;
        font-size: 0.8rem;
    }
    .image-info {
        background-color: #ebf8ff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #bee3f8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🖼️ AI Image Question Answering System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload an image and ask questions about what you see. The AI will analyze and answer your questions!</p>', unsafe_allow_html=True)

# Create two columns for layout
left_col, right_col = st.columns([1, 2])

with left_col:
    st.markdown("### 📁 Upload Image")
    
    # File uploader with styling
    with st.container():
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload an image to analyze (JPG, PNG, BMP, TIFF)"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Read image
        image_bytes = uploaded_file.read()
        st.session_state.current_image_bytes = image_bytes
        
        # Validate image
        with st.spinner("Validating image..."):
            validation = st.session_state.processor.validate_image(image_bytes, uploaded_file.name)
        
        if validation['valid']:
            # Display image
            image = Image.open(io.BytesIO(image_bytes))
            st.session_state.current_image = image
            st.session_state.image_uploaded = True
            
            # Show image with caption - FIXED: use_column_width instead of use_container_width
            st.image(image, caption=f"📌 {uploaded_file.name}", use_column_width=True)
            
            # Image information in expander
            with st.expander("📊 Image Details", expanded=False):
                info = st.session_state.processor.get_image_info(image_bytes)
                st.markdown(f"<div class='image-info'>{info}</div>", unsafe_allow_html=True)
            
            # Quick action buttons
            st.markdown("### ⚡ Quick Actions")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📝 Generate Caption", use_container_width=True, type="primary"):
                    with st.spinner("🤖 AI is generating caption..."):
                        try:
                            result = st.session_state.vqa_engine.generate_caption(image)
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"**📝 Image Caption:**\n\n{result['caption']}",
                                "type": "caption"
                            })
                            st.success("✅ Caption generated!")
                        except Exception as e:
                            st.error(f"Error generating caption: {str(e)}")
            
            with col2:
                if st.button("🔍 Describe Image", use_container_width=True, type="primary"):
                    with st.spinner("🤖 AI is analyzing image..."):
                        try:
                            desc = st.session_state.vqa_engine.get_image_description(image)
                            response = f"**📋 Image Description:**\n\n"
                            response += f"**Caption:** {desc['caption']}\n\n"
                            response += "**Quick Answers:**\n"
                            for q, a in desc['answers'].items():
                                response += f"• *{q}* → **{a}**\n"
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": response,
                                "type": "description"
                            })
                            st.success("✅ Description generated!")
                        except Exception as e:
                            st.error(f"Error generating description: {str(e)}")
            
            # Clear chat button
            if st.button("🧹 Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        else:
            st.error(f"❌ {validation['error']}")
            st.session_state.image_uploaded = False
    else:
        # Show placeholder when no image is uploaded
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background-color: #f7fafc; border-radius: 10px;">
            <p style="color: #718096;">👆 Please upload an image to begin</p>
            <p style="color: #a0aec0; font-size: 0.9rem;">Supported formats: JPG, PNG, BMP, TIFF</p>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.image_uploaded = False

with right_col:
    st.markdown("### 💬 Chat with AI about your image")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input (only if image is uploaded)
        if st.session_state.image_uploaded:
            # Create a form for question input
            with st.form(key="question_form", clear_on_submit=True):
                question = st.text_input(
                    "Ask a question about the image:",
                    placeholder="e.g., What color is the object? How many people? What is happening?",
                    key="question_input"
                )
                submit_button = st.form_submit_button("Send Question", type="primary", use_container_width=True)
            
            if submit_button and question:
                # Add user message
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": f"**You:** {question}"
                })
                
                with st.chat_message("user"):
                    st.markdown(f"**You:** {question}")
                
                # Generate answer
                with st.chat_message("assistant"):
                    with st.spinner("🤖 Analyzing image and thinking..."):
                        try:
                            start_time = time.time()
                            result = st.session_state.vqa_engine.answer_question(
                                st.session_state.current_image, 
                                question
                            )
                            elapsed = time.time() - start_time
                            
                            # Display answer with styling
                            answer_text = f"**Answer:** {result['answer']}\n\n"
                            answer_text += f"*Confidence: {result['confidence']:.2f}*  \n"
                            answer_text += f"*Response time: {elapsed:.1f} seconds*"
                            
                            st.markdown(answer_text)
                            
                            # Show confidence as progress bar
                            st.progress(result['confidence'])
                            
                            # Add to history
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"**AI:** {result['answer']} (confidence: {result['confidence']:.2f})",
                                "type": "answer",
                                "confidence": result['confidence']
                            })
                        except Exception as e:
                            error_msg = f"Error analyzing image: {str(e)}"
                            st.error(error_msg)
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"**AI:** {error_msg}",
                                "type": "error"
                            })
        else:
            st.info("👆 Please upload an image first to start asking questions")
    
    # Tips and examples section
    with st.expander("💡 Tips & Example Questions", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🎯 Good questions to ask:**
            • What is in this image?
            • What colors are present?
            • Is this indoors or outdoors?
            • How many objects do you see?
            • What is the person doing?
            • What is the main subject?
            """)
        with col2:
            st.markdown("""
            **📝 Quick tips:**
            • Be specific in your questions
            • The AI understands natural language
            • First question may be slower (model loading)
            • Higher confidence = more reliable answer
            """)
    
    # Model information
    with st.expander("🔧 System Information", expanded=False):
        if st.button("Show Model Info"):
            try:
                info = st.session_state.vqa_engine.get_model_info()
                st.json(info)
            except:
                st.info("Model not loaded yet. Ask a question first to load the model.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #718096; padding: 1rem;'>
        <p>Powered by <strong>BLIP (Bootstrapping Language-Image Pre-training)</strong> • Built with Streamlit</p>
        <p style='font-size: 0.8rem;'>© 2025 AI Image Question Answering System</p>
    </div>
    """,
    unsafe_allow_html=True
)