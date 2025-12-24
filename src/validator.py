import pandas as pd
import logging
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataValidator:
    """
    Responsible for checking the integrity of raw medical data before processing.
    """
    
    def __init__(self, required_columns: list):
        self.required_columns = required_columns

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Checks if the dataframe contains the necessary columns."""
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            logging.error(f"Schema Validation Failed. Missing columns: {missing_cols}")
            return False
        logging.info("Schema Validation Passed.")
        return True

    def remove_invalid_rows(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Removes rows with null values in critical columns.
        Returns the cleaned dataframe and the count of removed rows.
        """
        initial_count = len(df)
        # Drop rows where any of the required columns are NaN or empty strings
        clean_df = df.dropna(subset=self.required_columns)
        
        # Filter out empty strings if any exist
        for col in self.required_columns:
            clean_df = clean_df[clean_df[col].astype(str).str.strip() != '']

        removed_count = initial_count - len(clean_df)
        
        if removed_count > 0:
            logging.warning(f"Data Quality Warning: Removed {removed_count} rows due to missing/empty values.")
        else:
            logging.info("Data Quality Check: No invalid rows found.")
            
        return clean_df, removed_count
