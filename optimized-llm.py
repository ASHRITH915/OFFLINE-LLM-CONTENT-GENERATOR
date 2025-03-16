import streamlit as st
import base64
import time
import os
import json
import queue
import sounddevice as sd
import vosk
import torch
import gc
from langchain.llms import CTransformers
from langchain import PromptTemplate
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

class BlogGeneratorApp:
    def __init__(self):
        st.set_page_config(
            page_title="Content Generator",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        self.local_css()
        self.model_path = "vosk-model-small-en-us-0.15" 
        # Path to Vosk model
        self.q = queue.Queue()
        
        # Initialize session state
        if "recognized_text" not in st.session_state:
            st.session_state["recognized_text"] = ""
            
        # Check for GPU availability and memory
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
            st.sidebar.success(f"✅ GPU detected: {gpu_name} ({gpu_memory:.2f} GB)")
            
            # Check if likely a low VRAM GPU
            if gpu_memory < 6:
                st.sidebar.info("🔧 Low VRAM mode enabled automatically")
                if "low_vram_mode" not in st.session_state:
                    st.session_state["low_vram_mode"] = True
        else:
            st.sidebar.warning("⚠️ No GPU detected. Running on CPU.")
            if "low_vram_mode" not in st.session_state:
                st.session_state["low_vram_mode"] = False

    def local_css(self):
        st.markdown("""
        <style>
        .stApp {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;}
        .main-container {background: rgba(255, 255, 255, 0.2); backdrop-filter: blur(10px); border-radius: 15px;}
        .stTextInput > div > div > input, .stSelectbox > div > div > select {background: rgba(255, 255, 255, 0.1); color: white;}
        .stButton > button {background: linear-gradient(45deg, #FF6B6B, #4ECDC4); color: white; border-radius: 15px;}
        .generated-text {background: rgba(255, 255, 255, 0.1); border-radius: 15px; padding: 20px;}
        .gpu-info {background: rgba(0, 0, 0, 0.2); padding: 10px; border-radius: 10px; margin-bottom: 15px;}
        </style>
        """, unsafe_allow_html=True)

    def callback(self, indata, frames, time, status):
        if status:
            print(status, flush=True)
        self.q.put(bytes(indata))

    def speech_to_text(self):
        if not os.path.exists(self.model_path):
            st.error("Vosk model not found! Place 'vosk-model-small-en-us-0.15' in the project directory.")
            return ""
        
        model = vosk.Model(self.model_path)
        recognizer = vosk.KaldiRecognizer(model, 16000)
        recognizer.SetWords(True)
        
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=self.callback):
            st.info("Listening... Speak now!")
            while True:
                data = self.q.get()
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    return result.get("text", "")

    def clear_cuda_cache(self):
        """Explicitly clear CUDA cache to free up GPU memory"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    def generate_llama_blog(self, input_text, no_words, blog_style, temperature=0.7, max_tokens=512, low_vram_mode=False):
        try:
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Clear GPU memory before loading model
            self.clear_cuda_cache()
            
            # Configure GPU acceleration based on available VRAM
            if low_vram_mode:
                # Low VRAM configuration (4GB cards like GTX 1650)
                gpu_config = {
                    'max_new_tokens': max_tokens,
                    'temperature': temperature,
                    #'gpu_layers': 16,  # Reduced number of layers on GPU
                    'context_length': 1024,  # Reduced context window
                    'batch_size': 1,  # Minimum batch size
                    #'threads': 4  # Control CPU thread usage
                }
            else:
                # Regular configuration for higher VRAM GPUs
                gpu_config = {
                    'max_new_tokens': max_tokens, 
                    'temperature': temperature,
                    #'gpu_layers': 32,  # Moderate number of layers to GPU
                    'context_length': 2048  # Standard context window
                }
            
            if self.device == "cpu":
                # Fall back to CPU configuration if no GPU is available
                gpu_config.pop('gpu_layers', None)
            
            # For smallest models, specify a smaller quantized model
            model_path = 'llama-2-7b-chat.ggmlv3.q4_K_M.bin' if low_vram_mode else 'llama-2-7b-chat.ggmlv3.q8_0.bin'
            
            llm = CTransformers(
                model=model_path,
                model_type='llama',
                config=gpu_config
            )
            
            template = """
            Write a blog for {blog_style} job profile for a topic {input_text}
            within {no_words} words.
            """
            prompt = PromptTemplate(
                input_variables=["blog_style", "input_text", 'no_words'],
                template=template
            )
            
            response = llm(prompt.format(
                blog_style=blog_style,
                input_text=input_text,
                no_words=no_words
            ))
            
            # Clear GPU memory after generating content
            self.clear_cuda_cache()
            
            return response
        except Exception as e:
            st.error(f"Error generating blog: {str(e)}")
            # Try to clean up in case of error
            self.clear_cuda_cache()
            return None

    def create_blog_metrics_visualization(self):
        metrics_data = {'Blog Style': ['Technical', 'Professional', 'Casual', 'Academic', 'Creative'],
                        'Average Words': [350, 400, 300, 450, 250],
                        'Popularity': [85, 90, 75, 65, 95]}
        df = pd.DataFrame(metrics_data)
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.bar(df, x='Blog Style', y='Average Words', title='Average Blog Length by Style', color='Blog Style')
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = go.Figure(data=go.Scatterpolar(r=df['Popularity'], theta=df['Blog Style'], fill='toself'))
            fig2.update_layout(title='Blog Style Popularity')
            st.plotly_chart(fig2, use_container_width=True)

    def get_download_link(self, text, filename):
        b64 = base64.b64encode(text.encode()).decode()
        return f'<a href="data:file/txt;base64,{b64}" download="{filename}">📥 Download Blog</a>'

    #def display_gpu_info(self):
        if self.device == "cuda":
            st.markdown('<div class="gpu-info">', unsafe_allow_html=True)
            st.subheader("🖥️ GPU Information")
            
            cols = st.columns(3)
            with cols[0]:
                st.metric("GPU Model", torch.cuda.get_device_name(0))
            with cols[1]:
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
                st.metric("Memory Used", f"{memory_allocated:.2f} MB")
            with cols[2]:
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
                memory_percent = (memory_allocated / memory_total) * 100
                st.metric("Memory Usage", f"{memory_percent:.1f}%")
                
            st.markdown('</div>', unsafe_allow_html=True)

    def run(self):
        st.markdown("""<h1 style='text-align: center; color: white;'>🚀 AI Content Generator</h1>""", unsafe_allow_html=True)
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        # Display GPU information if available
        #self.display_gpu_info()
        
        with st.sidebar:
            st.header("🧠 Blog Generation Settings")
            
            # VRAM optimization settings
            if self.device == "cuda":
                low_vram_mode = st.checkbox("Low VRAM Mode (4GB GPUs)", 
                                           value=st.session_state.get("low_vram_mode", False),
                                           help="Enable for GTX 1650, RTX 3050 or other 4GB VRAM GPUs")
                st.session_state["low_vram_mode"] = low_vram_mode
                
                if low_vram_mode:
                    st.info("✅ Using 4-bit quantized model and reduced context")
                    model_choice = "llama-2-7b-chat.ggmlv3.q4_K_M.bin"
                else:
                    model_choice = st.selectbox("Model Precision", 
                                              ["llama-2-7b-chat.ggmlv3.q4_K_M.bin", 
                                               "llama-2-7b-chat.ggmlv3.q5_K_M.bin",
                                               "llama-2-7b-chat.ggmlv3.q8_0.bin"],
                                              index=0,
                                              help="Lower precision (q4) uses less VRAM")
            else:
                low_vram_mode = False
                model_choice = "llama-2-7b-chat.ggmlv3.q4_K_M.bin"
                
            advanced_mode = st.checkbox("Advanced Mode")
            if advanced_mode:
                temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.7, 0.1)
                max_tokens = st.number_input("Maximum Tokens", 100, 1024 if low_vram_mode else 2048, 
                                            512 if not low_vram_mode else 256)
                
                if self.device == "cuda" and not low_vram_mode:
                    gpu_layers = st.slider("GPU Layers", 1, 32, 16, 
                                         help="More layers use more VRAM but process faster")
                else:
                    gpu_layers = 16 if self.device == "cuda" else 0
            else:
                temperature = 0.7
                max_tokens = 256 if low_vram_mode else 512
                gpu_layers = 16 if self.device == "cuda" else 0

        col1, col2 = st.columns(2)
        
        with col1:
            # Display and update the input field with recognized text
            input_text = st.text_input("Content Topic", st.session_state["recognized_text"])
            
            if st.button("🎤 Speak Topic"):
                recognized_text = self.speech_to_text()
                st.session_state["recognized_text"] = recognized_text  # Update session state
                st.experimental_rerun()  # Refresh UI to reflect new input
        
        with col2:
            blog_styles = {'🖥️ Technical': 'technical', '💼 Professional': 'professional', '😎 Casual': 'casual', '🎓 Academic': 'academic', '🎨 Creative': 'creative'}
            blog_style = st.selectbox("Blog Style", list(blog_styles.keys()))
            blog_style = blog_styles[blog_style]

        no_words = st.slider("📏 Word Count", 100, 1000, 300, 50)

        if st.button("✨ Generate Magical Content ✨"):
            if not input_text:
                st.warning("🚨 Please enter a topic")
            else:
                # Capture start time for performance measurement
                start_time = time.time()
                
                generated_blog = self.generate_llama_blog(
                    input_text, 
                    no_words, 
                    blog_style, 
                    temperature=temperature, 
                    max_tokens=max_tokens,
                    low_vram_mode=low_vram_mode
                )
                
                # Calculate and display generation time
                generation_time = time.time() - start_time
                
                if generated_blog:
                    st.markdown('<div class="generated-text">', unsafe_allow_html=True)
                    st.write(generated_blog)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.success(f"✅ Content generated in {generation_time:.2f} seconds")
                    st.markdown(self.get_download_link(generated_blog, f"{input_text.replace(' ', '_')}_blog.txt"), unsafe_allow_html=True)

        # Display memory usage after operations if on GPU
        #if self.device == "cuda":
            #memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
            #memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            #memory_percent = (memory_allocated / memory_total) * 100
            #st.sidebar.metric("Current VRAM Usage", f"{memory_percent:.1f}%", f"{memory_allocated:.1f} MB")

        st.markdown('</div>', unsafe_allow_html=True)
        
        # Only show visualizations if not in low VRAM mode to save resources
        if not low_vram_mode or self.device == "cpu":
            st.header("📊 Content Generation Insights")
            self.create_blog_metrics_visualization()

# Run app
def main():
    app = BlogGeneratorApp()
    app.run()

if __name__ == "__main__":
    main()
