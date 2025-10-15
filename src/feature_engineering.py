import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the King County dataset.
    
    Steps:
    - Create new features such as 'sqft_basement' if missing
    - Add house age or other derived metrics
    - Encode or scale selected features if needed
    """
    
    df = df.copy()
    
    # 1️⃣ Recalculate 'sqft_basement' if columns exist
    if {'sqft_living', 'sqft_above'}.issubset(df.columns):
        df['sqft_basement'] = df['sqft_living'] - df['sqft_above']
    
    # 2️⃣ Add 'house_age' feature
    if {'yr_built', 'yr_sold'}.issubset(df.columns):
        df['house_age'] = df['yr_sold'] - df['yr_built']
    elif 'yr_built' in df.columns:
        df['house_age'] = 2025 - df['yr_built']  # fallback for missing sale year
    
    # 3️⃣ Encode binary renovation feature
    if 'yr_renovated' in df.columns:
        df['was_renovated'] = df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
    
    # 4️⃣ Scale numeric columns (optional — you can adapt)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df
