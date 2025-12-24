import pandas as pd
import logging
import os
from src.validator import DataValidator
from src.cleaner import MedicalTextPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# --- CONFIGURATION ---
RAW_DATA_PATH = 'data/raw/dataset.csv' # Ensure your downloaded CSV is renamed to this
PROCESSED_DATA_PATH = 'data/processed/cleaned_medical_data.csv'
REQUIRED_COLUMNS = ['label', 'text'] # Kaggle dataset usually has 'label' (disease) and 'text' (symptom)

# --- SPECIALIST MAPPING LOGIC ---
# Mapping the specific Kaggle Symptom2Disease labels to Medical Specialties
SPECIALIST_MAP = {
    'Psoriasis': 'Dermatology', 'Acne': 'Dermatology', 'Impetigo': 'Dermatology', 
    'Fungal infection': 'Dermatology', 'Chicken pox': 'Dermatology',
    'Diabetes': 'Endocrinology', 'Hypoglycemia': 'Endocrinology',
    'Hypertension': 'Cardiology', 'Heart attack': 'Cardiology', 'Varicose veins': 'Vascular Surgery',
    'Migraine': 'Neurology', 'Cervical spondylosis': 'Neurology', 'Paralysis (brain hemorrhage)': 'Neurology',
    'Jaundice': 'Gastroenterology', 'Gastroesophageal reflux disease': 'Gastroenterology', 
    'Peptic ulcer diseae': 'Gastroenterology', 'Dimorphic hemorrhoids(piles)': 'Gastroenterology',
    'Pneumonia': 'Pulmonology', 'Bronchial Asthma': 'Pulmonology', 'Common Cold': 'General Practice',
    'Urinary tract infection': 'Urology',
    'Malaria': 'Infectious Disease', 'Dengue': 'Infectious Disease', 'Typhoid': 'Infectious Disease',
    'Arthritis': 'Rheumatology', 'Osteoarthristis': 'Rheumatology',
    'Allergy': 'Immunology', 'Drug Reaction': 'Immunology'
}

def run_pipeline():
    logging.info("Starting Health Data Pipeline...")

    # 1. Ingestion
    if not os.path.exists(RAW_DATA_PATH):
        logging.error(f"File not found: {RAW_DATA_PATH}. Please place the Kaggle CSV there.")
        return

    df = pd.read_csv(RAW_DATA_PATH)
    logging.info(f"Data Loaded. Shape: {df.shape}")

    # 2. Validation
    validator = DataValidator(required_columns=REQUIRED_COLUMNS)
    if not validator.validate_schema(df):
        return
    
    df, removed = validator.remove_invalid_rows(df)
    
    # 3. Processing (NLP Cleaning)
    cleaner = MedicalTextPreprocessor()
    logging.info("Starting NLP Cleaning (this may take a moment)...")
    
    # Apply cleaning to the 'text' column
    df['cleaned_symptoms'] = df['text'].apply(cleaner.clean_text)
    
    # 4. Feature Engineering (Mapping to Specialist)
    logging.info("Mapping Symptoms to Specialists...")
    # Map the disease label to the specialist using our dictionary
    df['specialist_referral'] = df['label'].map(SPECIALIST_MAP)
    
    # Fill unknown mappings with 'General Practice'
    df['specialist_referral'] = df['specialist_referral'].fillna('General Practice')

    # Add urgency flag (Example logic)
    high_urgency = ['Heart attack', 'Paralysis (brain hemorrhage)', 'Pneumonia']
    df['urgency'] = df['label'].apply(lambda x: 'High' if x in high_urgency else 'Normal')

    # 5. Load (Save to Processed)
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    logging.info(f"Pipeline Finished. Data saved to {PROCESSED_DATA_PATH}")
    logging.info(f"Preview:\n{df[['label', 'specialist_referral', 'urgency']].head()}")

if __name__ == "__main__":
    run_pipeline()
