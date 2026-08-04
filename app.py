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
        background: linear-gradient(135deg, #0b0f19 0%, #0f172a 50%, #1e1b4b 100%);
        color: #f8fafc;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header styles */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.8rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090d16 0%, #0f172a 60%, #1e1b4b 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    
    .sidebar-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 16px;
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: all 0.3s ease;
    }
    
    .sidebar-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
        box-shadow: 0 6px 24px rgba(56, 189, 248, 0.12);
    }
    
    .sidebar-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .sidebar-text {
        font-size: 0.88rem;
        color: #94a3b8;
        line-height: 1.55;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: #4ade80;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    .status-dot {
        width: 7px;
        height: 7px;
        background-color: #4ade80;
        border-radius: 50%;
        box-shadow: 0 0 8px #4ade80;
    }
    
    .stat-number {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        margin: 4px 0;
    }
    
    .ocr-pill {
        display: inline-block;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.25);
        color: #38bdf8;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 4px;
        margin-bottom: 4px;
    }
    
    /* Premium card container styling */
    .premium-card {
        background: rgba(30, 41, 59, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }
    
    /* Section Divider Tag */
    .or-divider {
        display: flex;
        align-items: center;
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1px;
        margin: 16px 0;
    }
    
    .or-divider::before, .or-divider::after {
        content: '';
        flex: 1;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .or-divider span {
        padding: 0 12px;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
    }
    
    /* Custom Streamlit Selectbox Styling */
    div[data-testid="stSelectbox"] > label {
        font-weight: 600 !important;
        color: #38bdf8 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.2px;
    }
    
    div[data-baseweb="select"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        border-radius: 10px !important;
        color: #f8fafc !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-baseweb="select"]:hover {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.25) !important;
    }
    
    /* Custom Streamlit File Uploader Styling */
    div[data-testid="stFileUploader"] > label {
        font-weight: 600 !important;
        color: #c084fc !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.2px;
    }
    
    section[data-testid="stFileUploadDropzone"] {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 2px dashed rgba(192, 132, 252, 0.4) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        transition: all 0.3s ease !important;
    }
    
    section[data-testid="stFileUploadDropzone"]:hover {
        border-color: #c084fc !important;
        background: rgba(192, 132, 252, 0.08) !important;
        box-shadow: 0 0 20px rgba(192, 132, 252, 0.2) !important;
    }
    
    /* Custom Streamlit Textarea Styling */
    div[data-testid="stTextArea"] > label {
        font-weight: 600 !important;
        color: #f8fafc !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.2px;
    }
    
    div[data-testid="stTextArea"] textarea {
        background-color: rgba(15, 23, 42, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        color: #f1f5f9 !important;
        font-family: 'Fira Code', 'Courier New', monospace !important;
        font-size: 0.92rem !important;
        line-height: 1.6 !important;
        padding: 14px !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 18px rgba(56, 189, 248, 0.3) !important;
        background-color: rgba(15, 23, 42, 0.95) !important;
    }
    
    /* Diagnostic Result Cards */
    .result-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-left: 6px solid #38bdf8;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 14px;
        backdrop-filter: blur(8px);
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        border-left-width: 8px;
        border-color: rgba(56, 189, 248, 0.5);
        box-shadow: 0 8px 24px 0 rgba(56, 189, 248, 0.18);
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
        margin-top: 6px;
    }
    
    .result-desc {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 6px;
        line-height: 1.5;
    }
    
    /* Custom Progress Bar */
    .progress-bg {
        background-color: #334155;
        border-radius: 6px;
        height: 8px;
        width: 100%;
        margin-top: 12px;
        overflow: hidden;
    }
    
    .progress-bar {
        background: linear-gradient(90deg, #38bdf8 0%, #a855f7 100%);
        height: 100%;
        border-radius: 6px;
    }
    
    .meta-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 6px;
        padding: 0 4px;
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

# Redesigned Glassmorphism Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-header">🩺 About Scanner</div>
        <div class="sidebar-text">
            Parses unstructured clinical notes and predicts the top <b>ICD-9 diagnostic codes</b> using an <b>XGBoost Classifier</b> trained on MIMIC discharge summaries.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-header">📊 System Statistics</div>
        <div class="status-badge">
            <span class="status-dot"></span> System Ready
        </div>
        <div class="sidebar-text" style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">ICD-9 Dictionary Size</div>
    """, unsafe_allow_html=True)
    
    if df_dict is not None:
        st.markdown(f'<div class="stat-number">{len(df_dict):,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-text" style="font-size: 0.78rem;">Indexed medical diagnoses & terms</div>', unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-header">📷 OCR Engine & Modes</div>
        <div style="margin-bottom: 10px;">
            <span class="ocr-pill">Image Upload</span>
            <span class="ocr-pill">Preset Notes</span>
            <span class="ocr-pill">Manual Text</span>
        </div>
        <div class="sidebar-text">
            Local OCR requires the <b>Tesseract-OCR</b> binary installed on your OS. In its absence, you can use our sample presets or paste raw text into the editor.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Split UI into two columns
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-bottom: 16px; color: #f8fafc;'>📥 Input Clinical Note</h3>", unsafe_allow_html=True)
    
    # Pre-defined sample dropdown
    selected_sample = st.selectbox("💡 Step 1: Select Sample Preset Note", list(SAMPLES.keys()))
    preset_text = SAMPLES[selected_sample]
    
    st.markdown('<div class="or-divider"><span>OR UPLOAD IMAGE</span></div>', unsafe_allow_html=True)
    
    # File uploader (OCR demonstration)
    uploaded_image = st.file_uploader("🖼️ Step 2: Upload Doctor's Handwriting Image", type=["png", "jpg", "jpeg"])
    
    # OCR fallback handling
    if uploaded_image is not None:
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(uploaded_image)
            with st.spinner("Extracting text from image via OCR..."):
                ocr_text = pytesseract.image_to_string(img)
            st.success("✨ Text successfully extracted via OCR!")
            preset_text = ocr_text
        except ImportError:
            st.warning("⚠️ **Pytesseract library not installed.** Displaying sample clinical transcript instead. Install `pytesseract` for live OCR.")
            if not preset_text:
                preset_text = SAMPLES["Sample 1: Cardiovascular Disease & Hypertension"]
        except Exception as e:
            st.error(f"OCR Error: {e}")
            if not preset_text:
                preset_text = SAMPLES["Sample 1: Cardiovascular Disease & Hypertension"]
                
    st.markdown('<div class="or-divider"><span>EDIT & PARSE</span></div>', unsafe_allow_html=True)
    
    # Text input area
    medical_text = st.text_area(
        "📝 Step 3: Medical Text Editor (Real-Time Parsing)",
        value=preset_text,
        height=220,
        placeholder="Enter patient discharge summaries, physician notes, or diagnostic reviews here..."
    )
    
    # Character / Word count display bar
    char_count = len(medical_text)
    word_count = len(medical_text.split()) if medical_text.strip() else 0
    st.markdown(f"""
    <div class="meta-bar">
        <span><b>Words:</b> {word_count:,} | <b>Chars:</b> {char_count:,}</span>
        <span>⚡ Auto-classifying</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col_output:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-bottom: 16px; color: #f8fafc;'>📋 Classification Results</h3>", unsafe_allow_html=True)
    
    if medical_text.strip() == "":
        st.info("💡 **Ready for input.** Select a sample preset on the left or paste doctor notes to view predicted ICD-9 diagnostic codes in real-time.")
    else:
        # Preprocess text
        cleaned = clean_medical_text(medical_text)
        
        # Display clean preview
        with st.expander("🔍 View Preprocessed Normalized Text (ML Feature Input)"):
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
                st.markdown("<h4 style='color: #94a3b8; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 14px;'>Top Predicted ICD-9 Diagnoses</h4>", unsafe_allow_html=True)
                
                for idx, (code, prob) in enumerate(zip(top_icd_codes, top_probs)):
                    # Format lookup code to match MIMIC ICD9 database formatting
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
                            <span style="font-weight: 700; color: #a855f7; background: rgba(168, 85, 247, 0.1); border: 1px solid rgba(168, 85, 247, 0.3); padding: 3px 10px; border-radius: 20px; font-size: 0.85rem;">{confidence_percent:.1f}% Confidence</span>
                        </div>
                        <div class="result-title">{short_title}</div>
                        <div class="result-desc"><b>Details:</b> {long_title}</div>
                        <div style="margin-top: 10px; font-size: 0.85rem;">
                            <a href="https://www.icd9data.com/getICD9Code.ashx?icd9={code}" target="_blank" style="color: #38bdf8; text-decoration: none; font-weight: 600;">🔍 Reference Lookup →</a>
                        </div>
                        <div class="progress-bg">
                            <div class="progress-bar" style="width: {confidence_percent}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
    st.markdown('</div>', unsafe_allow_html=True)

