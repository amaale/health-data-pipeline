import pandas as pd
import logging
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class SpecialistClassifier:
    """Machine learning model for specialist prediction"""
    
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('classifier', LogisticRegression(max_iter=1000))
        ])
        self.is_trained = False

    def train(self, X_text: pd.Series, y_specialist: pd.Series):
        """Train the model on symptom text and specialist labels"""
        logging.info("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X_text, y_specialist, test_size=0.2, random_state=42)
        
        logging.info("Training model...")
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True
        
        accuracy = self.pipeline.score(X_test, y_test)
        logging.info(f"Training complete. Validation accuracy: {accuracy:.2f}")
        
        predictions = self.pipeline.predict(X_test)
        logging.info(f"\n{classification_report(y_test, predictions)}")

    def predict(self, text_list: list):
        """Predict specialists for new symptom descriptions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.pipeline.predict(text_list)
        
    def save_model(self, filepath='data/model.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)
        logging.info(f"Model saved to {filepath}")
