import pandas as pd
import re
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load data
print("Loading datasets...")
try:
    notes = pd.read_csv('data/NOTEEVENTS.csv')
    diagnoses = pd.read_csv('data/DIAGNOSES_ICD.csv')
    print("Datasets loaded successfully.")
except Exception as e:
    print(f"Error loading datasets: {e}")
    exit(1)

# Preprocess
print("Filtering notes for discharge summaries...")
notes_filt = notes[notes['CATEGORY'] == 'Discharge summary'][['HADM_ID', 'TEXT']]
df_combined = pd.merge(notes_filt, diagnoses[['HADM_ID', 'ICD9_CODE']], on='HADM_ID')
df_combined.dropna(subset=['TEXT', 'ICD9_CODE'], inplace=True)

# Get top 50 codes
print("Filtering to top 50 most frequent ICD-9 codes...")
top_50_codes = df_combined['ICD9_CODE'].value_counts().nlargest(50).index.tolist()
df_final = df_combined[df_combined['ICD9_CODE'].isin(top_50_codes)].copy()

# Sample for fast local training
max_samples = 20000
if len(df_final) > max_samples:
    print(f"Sampling dataset down to {max_samples} rows for rapid local training...")
    df_final = df_final.sample(n=max_samples, random_state=42)

def clean_medical_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\[\*\*.*?\*\*\]', '', text) # Remove de-identified brackets
    text = re.sub(r'[^a-zA-Z\s]', '', text)     # Remove numbers/special chars
    text = re.sub(r'\s+', ' ', text).strip()    # Remove extra whitespace
    return text

print("Cleaning medical text...")
df_final['CLEAN_TEXT'] = df_final['TEXT'].apply(clean_medical_text)
df_final = df_final[df_final['CLEAN_TEXT'] != '']

print("Extracting TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(df_final['CLEAN_TEXT'])
y = df_final['ICD9_CODE']

print("Encoding labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Splitting train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

print("Training LogisticRegression (very fast CPU text classifier)...")
model = LogisticRegression(
    max_iter=250,
    solver='lbfgs',
    n_jobs=-1,
    random_state=42
)
model.fit(X_train, y_train)

# Calculate accuracy
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Training Accuracy: {train_acc * 100:.2f}%")
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save models
print("Saving model objects to disk...")
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

with open('models/classifier.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model training completed successfully and files saved to 'models/'!")
