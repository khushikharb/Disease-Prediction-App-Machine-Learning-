import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def prepare_data(file_path):
    try:
        # Read the uploaded file
        df = pd.read_csv(file_path)
        
        # Separate features and target
        X = df.iloc[:, :-1]  # All columns except last
        y = df.iloc[:, -1]   # Last column
        
        # Encode the target variable
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test, le, X.columns.tolist(), df
    
    except Exception as e:
        raise ValueError(f"Error processing file: {str(e)}")