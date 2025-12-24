import pandas as pd
import logging
from typing import Optional

class DataBalancer:
    """Handles class balancing using undersampling"""
    
    def __init__(self, strategy: str = 'moderate', random_state: int = 42):
        """
        Args:
            strategy: 'aggressive', 'moderate', or 'conservative'
            random_state: Random seed for reproducibility
        """
        self.strategy = strategy
        self.random_state = random_state
        self.strategy_params = {
            'aggressive': {'max_ratio': 2.0, 'min_samples': 50},
            'moderate': {'max_ratio': 5.0, 'min_samples': 100},
            'conservative': {'max_ratio': 10.0, 'min_samples': 200}
        }
        
    def balance_dataset(self, df: pd.DataFrame, target_col: str = 'specialist') -> pd.DataFrame:
        """
        Balance dataset by undersampling majority classes
        
        Args:
            df: Input dataframe
            target_col: Column containing class labels
            
        Returns:
            Balanced dataframe
        """
        if target_col not in df.columns:
            logging.error(f"Target column '{target_col}' not found")
            return df
            
        original_counts = df[target_col].value_counts()
        logging.info(f"Original distribution: {len(df)} total samples")
        
        # Get strategy parameters
        params = self.strategy_params[self.strategy]
        max_ratio = params['max_ratio']
        min_samples = params['min_samples']
        
        # Find minority class size (excluding very rare classes)
        class_counts = original_counts[original_counts >= min_samples]
        if len(class_counts) == 0:
            logging.warning("No classes with sufficient samples, using all data")
            return df
            
        minority_size = class_counts.min()
        target_size = int(minority_size * max_ratio)
        
        logging.info(f"Using {self.strategy} strategy:")
        logging.info(f"  Minority class size: {minority_size}")
        logging.info(f"  Target max size per class: {target_size}")
        
        # Sample from each class
        balanced_dfs = []
        for class_label in df[target_col].unique():
            class_df = df[df[target_col] == class_label]
            class_size = len(class_df)
            
            if class_size <= target_size:
                # Keep all samples for small classes
                sampled = class_df
            else:
                # Undersample large classes
                sampled = class_df.sample(n=target_size, random_state=self.random_state)
                
            balanced_dfs.append(sampled)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        # Log results
        new_counts = balanced_df[target_col].value_counts()
        logging.info(f"\nBalanced distribution: {len(balanced_df)} total samples")
        logging.info(f"Reduction: {len(df)} -> {len(balanced_df)} ({len(balanced_df)/len(df)*100:.1f}%)")
        
        for class_label in sorted(new_counts.index):
            original = original_counts.get(class_label, 0)
            new = new_counts[class_label]
            logging.info(f"  {class_label}: {original} -> {new}")
        
        return balanced_df
