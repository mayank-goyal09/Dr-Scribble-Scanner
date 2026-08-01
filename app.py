import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os
import xgboost as xgb

# Set up page config
st.set_page_config(
    page_title="Dr. Scribble Scanner - ICD-9 Diagnostic Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a highly premium glassmorphism dark-mode look
st.markdown("""
<style>
    /* Main body styling */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styles */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Premium card container styling */
    .premium-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }
    
    /* Diagnostic Result Cards */
    .result-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-left: 6px solid #38bdf8;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        border-left-width: 10px;
        box-shadow: 0 4px 20px 0 rgba(56, 189, 248, 0.15);
    }
    
    .result-code {
        font-size: 1.3rem;
        font-weight: 700;
        color: #38bdf8;
    }
    
    .result-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-top: 4px;
    }
    
    .result-desc {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    
    /* Custom Progress Bar */
    .progress-bg {
        background-color: #334155;
        border-radius: 4px;
        height: 8px;
        width: 100%;
        margin-top: 10px;
        overflow: hidden;
    }
    
    .progress-bar {
        background: linear-gradient(90deg, #38bdf8 0%, #a855f7 100%);
        height: 100%;
        border-radius: 4px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #0b0f19 !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to clean incoming medical text
def clean_medical_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\[\*\*.*?\*\*\]', '', text) # Remove de-identified brackets
    text = re.sub(r'[^a-zA-Z\s]', '', text)     # Remove numbers/special chars
    text = re.sub(r'\s+', ' ', text).strip()    # Remove extra whitespace
    return text

# Load trained model artifacts
@st.cache_resource
def load_ml_pipeline():
    model_path = 'models/classifier.pkl'
    vectorizer_path = 'models/vectorizer.pkl'
    encoder_path = 'models/label_encoder.pkl'
    
    if not (os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(encoder_path)):
        return None, None, None
        
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    return model, vectorizer, encoder

# Load ICD-9 description dictionary
@st.cache_data
def load_dictionary():
    dict_path = 'data/D_ICD_DIAGNOSES.csv'
    if os.path.exists(dict_path):
        df_dict = pd.read_csv(dict_path)
        # Ensure code columns are strings and cleaned
        df_dict['ICD9_CODE'] = df_dict['ICD9_CODE'].astype(str).str.strip()
        return df_dict.set_index('ICD9_CODE')
    return None

# Pre-defined sample transcriptions for demonstration
SAMPLES = {
    "Select a sample note...": "",
    "Sample 1: Cardiovascular Disease & Hypertension": (
        "Patient is a 65-year-old male presenting with shortness of breath and chest tightness. "
        "Past history is notable for chronic diastolic congestive heart failure (CHF) and atrial fibrillation. "
        "EKG shows atrial fibrillation with rapid ventricular response. Blood pressure was elevated at 170/95. "
        "Diagnosed with acute on chronic systolic congestive heart failure and uncontrolled essential hypertension."
    ),
    "Sample 2: Community-Acquired Pneumonia": (
        "72-year-old female admitted with persistent productive cough, high fever, and altered mental status. "
        "Chest X-ray reveals right lower lobe consolidation consistent with acute lobar pneumonia. "
        "Oxygen saturation is 88% on room air. Treated with IV antibiotics (ceftriaxone and azithromycin) "
        "for community-acquired bacterial pneumonia and acute respiratory failure."
    ),
    "Sample 3: Diabetes Mellitus with Acute Renal Failure": (
        "58-year-old patient with long-standing history of insulin-dependent Type 2 diabetes mellitus presenting with "
        "severe fatigue, nausea, and decreased urine output. Lab results indicate elevated blood urea nitrogen (BUN) "
        "and serum creatinine, consistent with acute kidney failure super-imposed on diabetic nephropathy. "
        "Intravenous fluids administered, glycemic control optimized using insulin sliding scale."
    )
}

# Main Application Layout
st.markdown('<div class="main-title">🩺 Dr. Scribble Scanner</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time doctor handwriting text parsing and ICD-9 medical diagnosis classifier</div>', unsafe_allow_html=True)

# Load Pipeline and Dictionary
model, vectorizer, encoder = load_ml_pipeline()
df_dict = load_dictionary()

# Check if model exists
if model is None:
    st.error("⚠️ **Machine Learning Model files not found!**")
    st.info("The classifier needs to be trained first. Please execute the training script by running `python train.py` in your terminal. Once completed, this page will automatically load the model.")
    st.stop()

# Sidebar Info
with st.sidebar:
    st.markdown("### About Dr. Scribble Scanner")
    st.write("This app parses unstructured clinical notes and predicts the top ICD-9 diagnostic codes using an XGBoost Classifier trained on discharge summaries.")
    
    st.markdown("---")
    st.markdown("### System Statistics")
    if df_dict is not None:
        st.metric(label="ICD-9 Dictionary size", value=f"{len(df_dict):,}")
    
    st.markdown("---")
    st.markdown("### OCR Options")
    st.info("Local OCR (Image Upload) requires the Tesseract-OCR binary installed on your OS. In its absence, you can use our sample presets or paste raw text below.")

# Split UI into two columns
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("### 📥 Input Clinical Note")
    
    # Pre-defined sample dropdown
    selected_sample = st.selectbox("Load Sample Presets:", list(SAMPLES.keys()))
    preset_text = SAMPLES[selected_sample]
    
    # File uploader (OCR demonstration)
    uploaded_image = st.file_uploader("Or Upload Doctor's Handwriting Image:", type=["png", "jpg", "jpeg"])
    
    # OCR fallback handling
    if uploaded_image is not None:
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(uploaded_image)
            # Try to run OCR
            with st.spinner("Extracting text from image via OCR..."):
                ocr_text = pytesseract.image_to_string(img)
            st.success("Text successfully extracted via OCR!")
            preset_text = ocr_text
        except ImportError:
            st.warning("⚠️ **Pytesseract library not installed.** Showing simulation output instead. To use live image OCR, install `pytesseract` and Tesseract OCR engine.")
            # Auto-select Sample 1 for demo purposes
            if not preset_text:
                preset_text = SAMPLES["Sample 1: Cardiovascular Disease & Hypertension"]
        except Exception as e:
            st.error(f"OCR Error: {e}")
            if not preset_text:
                preset_text = SAMPLES["Sample 1: Cardiovascular Disease & Hypertension"]
                
    # Text input area
    medical_text = st.text_area(
        "Edit or Paste Medical Text below:",
        value=preset_text,
        height=250,
        placeholder="Enter patient discharge summaries, physician notes, or diagnostic reviews here..."
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

with col_output:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Classification Results")
    
    if medical_text.strip() == "":
        st.write("Results will appear here in real-time once you paste some medical notes or select a sample preset on the left.")
    else:
        # Preprocess text
        cleaned = clean_medical_text(medical_text)
        
        # Display clean preview
        with st.expander("Show preprocessed text (normalized for ML)"):
            st.code(cleaned, language="text")
            
        if cleaned == "":
            st.warning("Text normalization resulted in an empty string. Please provide valid medical text.")
        else:
            with st.spinner("Classifying diagnostics..."):
                # Vectorize
                vec_text = vectorizer.transform([cleaned])
                
                # Predict probabilities
                probs = model.predict_proba(vec_text)[0]
                
                # Get top 5 indices sorted by highest probability
                top_indices = np.argsort(probs)[::-1][:5]
                top_probs = probs[top_indices]
                top_encoded_classes = model.classes_[top_indices]
                
                # Decode to actual ICD-9 codes
                top_icd_codes = encoder.inverse_transform(top_encoded_classes)
                
                # Output Results
                st.markdown("#### Top Predicted ICD-9 Diagnoses:")
                
                for idx, (code, prob) in enumerate(zip(top_icd_codes, top_probs)):
                    # Format lookup code to match MIMIC ICD9 database formatting
                    # (e.g. converting 4019 to 401.9, 25000 to 250.00, etc.)
                    formatted_code = str(code).strip()
                    
                    # Try to fetch description
                    short_title = "Unknown Diagnosis"
                    long_title = "No descriptive information available in ICD-9 dictionary."
                    
                    if df_dict is not None:
                        # Attempt direct match
                        if formatted_code in df_dict.index:
                            row = df_dict.loc[formatted_code]
                            short_title = row['SHORT_TITLE']
                            long_title = row['LONG_TITLE']
                        # Attempt matching without decimals if dictionary matches have decimals
                        elif formatted_code.replace('.', '') in df_dict.index:
                            row = df_dict.loc[formatted_code.replace('.', '')]
                            short_title = row['SHORT_TITLE']
                            long_title = row['LONG_TITLE']
                    
                    confidence_percent = prob * 100
                    
                    # HTML representation of results with premium styles and links
                    st.markdown(f"""
                    <div class="result-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span class="result-code">ICD-9: {code}</span>
                            <span style="font-weight: 700; color: #a855f7;">{confidence_percent:.1f}% Confidence</span>
                        </div>
                        <div class="result-title">{short_title}</div>
                        <div class="result-desc"><b>Details:</b> {long_title}</div>
                        <div style="margin-top: 8px; font-size: 0.85rem;">
                            <a href="https://www.icd9data.com/getICD9Code.ashx?icd9={code}" target="_blank" style="color: #38bdf8; text-decoration: none;">🔍 Reference Lookup →</a>
                        </div>
                        <div class="progress-bg">
                            <div class="progress-bar" style="width: {confidence_percent}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
    st.markdown('</div>', unsafe_allow_html=True)
