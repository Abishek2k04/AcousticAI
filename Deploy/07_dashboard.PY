import streamlit as st
import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ATM Security AI",
    page_icon="🛡️",
    layout="wide"
)

# --- CONSTANTS (Must match training) ---
MODEL_PATH = "4_models/atm_crnn_enhanced.h5"
SAMPLE_RATE = 22050
DURATION = 4
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
EXPECTED_FRAMES = int((SAMPLE_RATE * DURATION) / HOP_LENGTH) + 1

# --- FUNCTIONS ---

@st.cache_resource  # Cache the model so it loads only once
def load_trained_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except:
        return None

def extract_features_from_upload(uploaded_file):
    """Processes the uploaded audio file for the AI."""
    try:
        # Load directly from the uploaded file object
        signal, sr = librosa.load(uploaded_file, sr=SAMPLE_RATE, mono=True, duration=DURATION)
        
        # Pad/Truncate
        target_len = SAMPLE_RATE * DURATION
        if len(signal) < target_len:
            signal = np.pad(signal, (0, target_len - len(signal)), mode='constant')
        else:
            signal = signal[:target_len]

        # MFCC
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        
        # Pad Frames
        if mfcc.shape[1] < EXPECTED_FRAMES:
            mfcc = np.pad(mfcc, ((0, 0), (0, EXPECTED_FRAMES - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :EXPECTED_FRAMES]
            
        return mfcc, signal
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None

# --- DASHBOARD UI ---

# Sidebar
st.sidebar.title("🛡️ Control Panel")
st.sidebar.info("ATM Acoustic Surveillance System v1.0")
st.sidebar.markdown("---")
st.sidebar.write("**System Status:** 🟢 Online")
st.sidebar.write("**Model:** Hybrid CRNN (Augmented)")

# Main Content
st.title("🏦 ATM Threat Detection System")
st.markdown("Upload audio from the ATM environment to detect potential attacks (Drilling, Sawing, etc).")

st.divider()

# 1. File Uploader
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Input Feed")
    uploaded_file = st.file_uploader("Upload Audio Clip (.wav)", type=["wav", "mp3"])

# Logic
if uploaded_file is not None:
    # Load Model
    model = load_trained_model()
    
    if model is None:
        st.error("❌ Model not found! Please run '04_train_enhanced_model.py' first.")
    else:
        # 2. Processing
        with col2:
            st.subheader("2. Audio Analysis")
            features, signal = extract_features_from_upload(uploaded_file)
            
            if features is not None:
                # Draw Waveform
                fig, ax = plt.subplots(figsize=(10, 2))
                librosa.display.waveshow(signal, sr=SAMPLE_RATE, axis='time', color='#3498db')
                plt.title("Waveform Visualizer")
                st.pyplot(fig)
                
                # Play Audio
                st.audio(uploaded_file)

        # 3. Prediction
        st.divider()
        st.subheader("3. Security Assessment")
        
        # Prepare input for model
        X_input = features[np.newaxis, ..., np.newaxis]
        
        with st.spinner('Analyzing acoustic signature...'):
            prediction_prob = model.predict(X_input, verbose=0)[0][0]
        
        # 4. The Verdict Card
        result_col1, result_col2 = st.columns([1, 3])
        
        if prediction_prob > 0.5:
            # ANOMALY
            with result_col1:
                st.error("## 🚨 THREAT DETECTED")
            with result_col2:
                confidence = prediction_prob * 100
                st.metric(label="Threat Confidence", value=f"{confidence:.2f}%", delta="CRITICAL ALERT")
                st.write("Type: **Abnormal Acoustic Activity (Drill/Saw)**")
                st.write("Action: **Notify Security / Lock Cash Dispenser**")
        else:
            # NORMAL
            with result_col1:
                st.success("## ✅ SECURE")
            with result_col2:
                confidence = (1 - prediction_prob) * 100
                st.metric(label="Safety Confidence", value=f"{confidence:.2f}%", delta="Normal State")
                st.write("Type: **Normal Environment (Traffic/AC)**")
                st.write("Action: **Continue Monitoring**")

else:
    # Placeholder when no file is uploaded
    with col2:
        st.info("Waiting for audio input stream...")