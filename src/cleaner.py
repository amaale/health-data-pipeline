import spacy
import re
import logging

class MedicalTextPreprocessor:
    """Handles NLP text cleaning using spaCy"""
    
    def __init__(self, model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
            logging.info(f"Loaded spaCy model: {model}")
        except OSError:
            logging.error(f"spaCy model '{model}' not found. Please run: python -m spacy download {model}")
            raise

    def clean_text(self, text: str) -> str:
        """Clean and normalize text using lemmatization and stopword removal"""
        if not isinstance(text, str):
            return ""

        # Basic normalization
        text = text.lower().strip()
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # NLP processing
        doc = self.nlp(text)
        
        # Extract lemmas, remove stopwords and short tokens
        cleaned_tokens = [
            token.lemma_ for token in doc 
            if not token.is_stop and not token.is_punct and len(token.text) > 2
        ]
        
        return " ".join(cleaned_tokens)
