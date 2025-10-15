import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the King County dataset.
    
    Steps:
    - Remove obvious outliers or invalid rows
    - Replace '?' with NaN
    - Convert numeric columns to appropriate dtypes
    """
    
    df = df.copy()  # avoid modifying original
    
    # 1️⃣ Remove the outlier row (bedrooms = 33)
    if 'bedrooms' in df.columns:
        df = df[df['bedrooms'] != 33]
    
    # 2️⃣ Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)
    
    # 3️⃣ Convert 'sqft_basement' to float (after replacing '?')
    if 'sqft_basement' in df.columns:
        df['sqft_basement'] = df['sqft_basement'].astype(float)
    
    # 4️⃣ (Optional) Drop rows with missing critical values
    df.dropna(subset=['price', 'sqft_living'], inplace=True)
    
    # 5️⃣ Reset index
    df.reset_index(drop=True, inplace=True)
    
    return df
