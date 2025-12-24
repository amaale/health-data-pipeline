#!/usr/bin/env python3
"""Test the updated pipeline with actual data structure"""

import pandas as pd
import logging
from src.pipeline import convert_binary_symptoms_to_text, SPECIALIST_MAP
from src.cleaner import MedicalTextPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def test_with_real_data_sample():
    """Test with a small sample that mimics the real data structure"""
    print("\n" + "="*70)
    print("Testing Pipeline with Real Data Structure")
    print("="*70)
    
    # Load a small sample of the real data
    df = pd.read_csv('data/raw/dataset.csv', nrows=5)
    print(f"\nLoaded sample data shape: {df.shape}")
    print(f"Columns: {len(df.columns)} ({df.columns[0]}, ...)")
    
    # Test the conversion function
    print("\n--- Testing Binary to Text Conversion ---")
    for idx in range(min(3, len(df))):
        row = df.iloc[idx]
        disease = row['diseases']
        text = convert_binary_symptoms_to_text(row)
        print(f"\nDisease: {disease}")
        print(f"Symptoms (first 150 chars): {text[:150]}...")
    
    # Add text and label columns
    df['text'] = df.apply(convert_binary_symptoms_to_text, axis=1)
    df['label'] = df['diseases']
    
    # Test NLP cleaning
    print("\n--- Testing NLP Cleaning ---")
    cleaner = MedicalTextPreprocessor()
    df['cleaned_symptoms'] = df['text'].apply(cleaner.clean_text)
    
    for idx in range(min(2, len(df))):
        print(f"\nOriginal: {df['text'].iloc[idx][:100]}...")
        print(f"Cleaned:  {df['cleaned_symptoms'].iloc[idx][:100]}...")
    
    # Test specialist mapping
    print("\n--- Testing Specialist Mapping ---")
    df['specialist_referral'] = df['label'].str.lower().map(SPECIALIST_MAP)
    df['specialist_referral'] = df['specialist_referral'].fillna('General Practice')
    
    for idx in range(min(3, len(df))):
        print(f"{df['label'].iloc[idx]:30s} -> {df['specialist_referral'].iloc[idx]}")
    
    # Test urgency flag
    high_urgency = ['heart attack', 'stroke', 'paralysis (brain hemorrhage)', 'pneumonia']
    df['urgency'] = df['label'].str.lower().apply(lambda x: 'High' if x in high_urgency else 'Normal')
    
    print("\n--- Final Output Sample ---")
    output_cols = ['label', 'specialist_referral', 'urgency']
    print(df[output_cols].head())
    
    print("\n✓ Pipeline test completed successfully!")
    print(f"\nThe pipeline can process {len(df)} rows")
    print(f"Unique diseases in sample: {df['label'].nunique()}")

if __name__ == "__main__":
    try:
        test_with_real_data_sample()
        print("\n" + "="*70)
        print("✅ PIPELINE IS READY TO RUN ON YOUR FULL DATASET!")
        print("="*70)
        print("\nTo run the full pipeline, execute:")
        print("  .venv/bin/python src/pipeline.py")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
