# health-data-pipeline
Medical symptom analysis and specialist recommendation system

## Overview

ETL pipeline for processing medical symptom data. Converts raw symptom data into cleaned text, then uses machine learning to predict appropriate medical specialists.

**Tech Stack:** Python, Pandas, spaCy, scikit-learn, Pytest

## Features

- Binary symptom data conversion
- NLP text preprocessing (lemmatization, stopword removal)
- ML-based specialist classification
- Data validation and quality checks

## Project Structure

```
health-data-pipeline/
├── data/
│   ├── raw/              # Input data
│   └── processed/        # Output data
├── src/
│   ├── validator.py      # Data validation
│   ├── cleaner.py        # Text preprocessing
│   ├── model.py          # ML classifier
│   └── pipeline.py       # Main pipeline
└── tests/                # Unit tests
```

## Setup

1. **Install dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. **Prepare data**
Place your dataset in `data/raw/dataset.csv`

3. **Run pipeline**
```bash
python src/pipeline.py
```

## How It Works

1. Loads binary symptom data (378 columns)
2. Converts binary values to text descriptions
3. Cleans text using spaCy NLP
4. Maps diseases to medical specialists
5. Trains classifier (TF-IDF + Logistic Regression)
6. Saves trained model and processed data

## Output

- `data/processed/cleaned_medical_data.csv` - Processed dataset
- `data/model.pkl` - Trained classifier

## Results

Trained on 246,945 medical cases with 97% validation accuracy.

**Dataset Distribution:**
- General Practice: 92.7% (228,999 cases)
- Specialty cases: 7.3% (17,946 cases across 14 specialties)

**Model Performance:**
- Strong accuracy on high-frequency specialties (Psychiatry: 62%, Endocrinology: 93%)
- Lower recall on rare conditions due to class imbalance
- Best suited for general triage and common condition routing

**Key Findings:**
- Class imbalance affects specialty prediction (model favors majority class)
- Pipeline successfully processes large-scale medical data
- NLP preprocessing improves text quality for ML
- System could benefit from balanced sampling or weighted training

## Testing

```bash
pytest tests/
```
