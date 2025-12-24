import spacy
import re
import logging

class MedicalTextPreprocessor:
    """
    Handles the NLP cleaning pipeline using spaCy.
    """
    def __init__(self, model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
            logging.info(f"Loaded spaCy model: {model}")
        except OSError:
            logging.error(f"spaCy model '{model}' not found. Please run: python -m spacy download {model}")
            raise

    def clean_text(self, text: str) -> str:
        """
        Applies a pipeline of cleaning operations:
        1. Lowercasing
        2. Removing special characters
        3. Lemmatization (running -> run)
        4. Stopword removal (the, is, and)
        """
        if not isinstance(text, str):
            return ""

        # 1. Basic normalization
        text = text.lower().strip()
        text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove punctuation/numbers

        # 2. Advanced NLP processing
        doc = self.nlp(text)
        
        # 3. Extract lemmas for non-stop words
        cleaned_tokens = [
            token.lemma_ for token in doc 
            if not token.is_stop and not token.is_punct and len(token.text) > 2
        ]
        
        return " ".join(cleaned_tokens)
