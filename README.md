# health-data-pipeline
🏥 AI-Powered Symptom-to-Specialist Mapper

Project Overview

This project is an ETL (Extract, Transform, Load) Data Pipeline designed to process unstructured medical text data. It ingests raw symptom descriptions, cleans them using NLP (spaCy), and maps them to the appropriate medical specialist (e.g., Chest Pain $\rightarrow$ Cardiology).

Key Tech Stack: Python, Pandas, spaCy (NLP), Docker, Pytest.

📂 Project Structure

health-data-pipeline/
├── data/               # Data storage (Raw vs Processed)
├── src/
│   ├── validator.py    # Schema validation & Data Quality checks
│   ├── cleaner.py      # NLP Preprocessing (Lemmatization, Stopwords)
│   └── pipeline.py     # Main ETL Orchestrator
├── tests/              # Unit tests for CI/CD
└── Dockerfile          # Containerization for deployment


🚀 How to Run

1. Prerequisite

Place your Kaggle CSV file in data/raw/dataset.csv.
(Target Dataset: Symptom2Disease or similar)

2. Run Locally

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Execute the pipeline
python -m src.pipeline


3. Run with Docker

docker build -t medical-pipeline .
docker run -v $(pwd)/data:/app/data medical-pipeline


4. Run Tests

pytest tests/


🛠️ Engineering Highlights

Data Validation: The pipeline rejects schemas that do not meet strict requirements, preventing downstream errors.

NLP Normalization: Uses spaCy to lemmatize clinical terms (e.g., "coughing" $\rightarrow$ "cough") for better classification.

Testing: Includes pytest coverage to ensure the cleaning logic handles edge cases (empty strings, special characters).
