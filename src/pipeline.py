import pandas as pd
import logging
import os
from src.validator import DataValidator
from src.cleaner import MedicalTextPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Configuration
RAW_DATA_PATH = 'data/raw/dataset.csv'
PROCESSED_DATA_PATH = 'data/processed/cleaned_medical_data.csv'
REQUIRED_COLUMNS = ['diseases']

# Disease to specialist mapping
SPECIALIST_MAP = {
    'psoriasis': 'Dermatology', 'acne': 'Dermatology', 'impetigo': 'Dermatology', 
    'fungal infection': 'Dermatology', 'chicken pox': 'Dermatology',
    'diabetes': 'Endocrinology', 'hypoglycemia': 'Endocrinology', 'hypothyroidism': 'Endocrinology',
    'hyperthyroidism': 'Endocrinology',
    'hypertension': 'Cardiology', 'heart attack': 'Cardiology', 'varicose veins': 'Vascular Surgery',
    'heart failure': 'Cardiology', 'coronary artery disease': 'Cardiology',
    'migraine': 'Neurology', 'cervical spondylosis': 'Neurology', 'paralysis (brain hemorrhage)': 'Neurology',
    'epilepsy': 'Neurology', 'stroke': 'Neurology', 'alzheimer': 'Neurology', 'parkinson': 'Neurology',
    'jaundice': 'Gastroenterology', 'gastroesophageal reflux disease': 'Gastroenterology', 
    'peptic ulcer diseae': 'Gastroenterology', 'dimorphic hemorrhoids(piles)': 'Gastroenterology',
    'hepatitis': 'Gastroenterology', 'cirrhosis': 'Gastroenterology',
    'pneumonia': 'Pulmonology', 'bronchial asthma': 'Pulmonology', 'tuberculosis': 'Pulmonology',
    'copd': 'Pulmonology', 'asthma': 'Pulmonology',
    'common cold': 'General Practice', 'flu': 'General Practice', 'allergy': 'Immunology',
    'urinary tract infection': 'Urology', 'kidney stone': 'Urology',
    'malaria': 'Infectious Disease', 'dengue': 'Infectious Disease', 'typhoid': 'Infectious Disease',
    'aids': 'Infectious Disease', 'hepatitis a': 'Infectious Disease', 'hepatitis b': 'Infectious Disease',
    'hepatitis c': 'Infectious Disease', 'hepatitis d': 'Infectious Disease', 'hepatitis e': 'Infectious Disease',
    'arthritis': 'Rheumatology', 'osteoarthristis': 'Rheumatology', 'rheumatoid arthritis': 'Rheumatology',
    'panic disorder': 'Psychiatry', 'depression': 'Psychiatry', 'anxiety': 'Psychiatry',
    'bipolar disorder': 'Psychiatry', 'schizophrenia': 'Psychiatry',
    'drug reaction': 'Immunology', 'anemia': 'Hematology', 'leukemia': 'Oncology',
}

def convert_binary_symptoms_to_text(row):
    """Convert binary symptom columns to text string"""
    symptom_columns = [col for col in row.index if col != 'diseases']
    active_symptoms = [col for col in symptom_columns if row[col] == 1]
    return ' '.join(active_symptoms) if active_symptoms else 'no symptoms reported'


def run_pipeline():
    logging.info("Starting pipeline...")

    # Load data
    if not os.path.exists(RAW_DATA_PATH):
        logging.error(f"File not found: {RAW_DATA_PATH}")
        return

    df = pd.read_csv(RAW_DATA_PATH)
    logging.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Transform binary symptoms to text
    logging.info("Converting symptoms to text...")
    df['text'] = df.apply(convert_binary_symptoms_to_text, axis=1)
    df['label'] = df['diseases']

    # Validate
    validator = DataValidator(required_columns=['diseases'])
    if not validator.validate_schema(df):
        return
    
    # Remove rows without symptoms
    initial_count = len(df)
    df = df[df['text'] != 'no symptoms reported']
    removed = initial_count - len(df)
    if removed > 0:
        logging.warning(f"Removed {removed} rows with no symptoms")
    
    # NLP cleaning
    cleaner = MedicalTextPreprocessor()
    logging.info("Applying NLP cleaning...")
    df['cleaned_symptoms'] = df['text'].apply(cleaner.clean_text)
    
    # Map to specialists
    logging.info("Mapping to specialists...")
    df['specialist_referral'] = df['label'].str.lower().map(SPECIALIST_MAP)
    df['specialist_referral'] = df['specialist_referral'].fillna('General Practice')

    # Add urgency
    high_urgency = ['heart attack', 'stroke', 'paralysis (brain hemorrhage)', 'pneumonia', 'sepsis']
    df['urgency'] = df['label'].str.lower().apply(lambda x: 'High' if x in high_urgency else 'Normal')

    # Save output
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    output_columns = ['label', 'text', 'cleaned_symptoms', 'specialist_referral', 'urgency']
    df_output = df[output_columns]
    
    df_output.to_csv(PROCESSED_DATA_PATH, index=False)
    logging.info(f"Saved to {PROCESSED_DATA_PATH}")
    logging.info(f"Processed {len(df_output)} records")
    logging.info(f"Unique diseases: {df_output['label'].nunique()}")
    logging.info(f"\nSpecialist distribution:\n{df_output['specialist_referral'].value_counts()}")

if __name__ == "__main__":
    run_pipeline()
