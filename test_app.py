import streamlit as st
import os
import gdown
import librosa
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, pipeline

# --- 1. MODEL DEFINITIONS ---
class ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)

# --- 2. SETUP & DOWNLOAD ---
@st.cache_resource
def setup_models():
    # 1. SETUP DIRECTORIES
    base_dir = "Models"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
        # Fetching Secret ID from Streamlit Cloud Dashboard
        folder_id = st.secrets["drive_ids"]["models_folder"]
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        gdown.download_folder(url=url, output=base_dir, quiet=False)
    
    # 2. PATH DISCOVERY
    paths = {
        "processor": os.path.join(base_dir, "processor"),
        "age_model": os.path.join(base_dir, "age_model"),
        "gender_model": os.path.join(base_dir, "gender_model"),
        "emotion_model": os.path.join(base_dir, "emotion_model")
    }
    
    # 3. INITIALIZE MODELS
    # Added local_files_only=True to prevent Hugging Face Hub connectivity errors
    processor = Wav2Vec2Processor.from_pretrained(paths["processor"], local_files_only=True)
    
    # Load custom age model and ensure CPU compatibility
    age_model = torch.load(os.path.join(paths["age_model"], "model.pth"), map_location=torch.device('cpu'))
    age_model.eval()
            
    # Initialize pipelines with local override
    gender_pipe = pipeline("audio-classification", model=paths["gender_model"], local_files_only=True)
    emotion_pipe = pipeline("audio-classification", model=paths["emotion_model"], local_files_only=True)
            
    return processor, age_model, gender_pipe, emotion_pipe

# --- 3. STREAMLIT INTERFACE ---
st.title("Voice Age & Emotion Detector")
processor, age_model, gender_pipe, emotion_pipe = setup_models()

uploaded_file = st.file_uploader("Upload voice note", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')
    y, sr = librosa.load(uploaded_file, sr=16000)
    
    # Gender check
    gender_results = gender_pipe(y)
    gender_label = gender_results[0]['label'].lower()
    
    if 'female' in gender_label:
        st.error("Upload a male voice note.")
    else:
        # Age Prediction
        inputs = processor(y, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(torch.float32)
        
        with torch.no_grad():
            logits_age = age_model(input_values)
            # Handle model output structure
            if isinstance(logits_age, tuple):
                logits_age = logits_age[0]
        
        age = int(logits_age.item() * 100)
        
        # Logic orchestration
        if age <= 0:
            st.warning("Could not clearly detect age.")
        elif age > 60:
            emotion_results = emotion_pipe(y)
            emotion = emotion_results[0]['label']
            st.success(f"Detected Age: {age} (Senior Citizen)")
            st.info(f"Detected Emotion: {emotion}")
        else:
            st.success(f"Detected Age: {age}")
