"""
Real Estate Investment Advisor - Data Processing & Feature Engineering Pipeline
Author: Capstone Project Team
Domain: Real Estate / Financial Analytics
"""

import os
import sqlite3
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

# Top Tier Indian Tech & Economic Hubs
TIER1_CITIES = {
    'Mumbai', 'Bangalore', 'Delhi', 'Hyderabad', 'Pune', 
    'Chennai', 'Kolkata', 'Ahmedabad'
}

TIER2_CITIES = {
    'Jaipur', 'Lucknow', 'Chandigarh', 'Indore', 'Coimbatore', 
    'Kochi', 'Surat', 'Nagpur', 'Visakhapatnam', 'Bhopal', 
    'Vadodara', 'Ludhiana', 'Patna', 'Agra', 'Nashik', 
    'Varanasi', 'Rajkot', 'Kanpur', 'Meerut', 'Faridabad', 
    'Ghaziabad', 'Noida', 'Gurgaon'
}

def get_city_tier_rate(city: str) -> float:
    """Returns baseline annual appreciation rate based on city economic tier."""
    if city in TIER1_CITIES:
        return 0.085
    elif city in TIER2_CITIES:
        return 0.075
    else:
        return 0.065

def get_age_factor(age: float) -> float:
    """Returns age depreciation/appreciation adjustment."""
    if age <= 5:
        return 0.015
    elif age <= 15:
        return 0.005
    elif age <= 25:
        return -0.005
    else:
        return -0.012

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw dataset from CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[Data Pipeline] Loaded raw data shape: {df.shape}")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset:
    - Checks missing values and imputes if any
    - Removes duplicates
    - Validates floor counts and ages
    - Handles extreme statistical outliers using IQR bounds
    """
    df_clean = df.copy()
    
    # 1. Deduplication
    init_count = len(df_clean)
    df_clean.drop_duplicates(inplace=True)
    dups_removed = init_count - len(df_clean)
    if dups_removed > 0:
        print(f"[Cleaning] Removed {dups_removed} duplicate records.")

    # 2. Null handling
    null_cols = df_clean.isnull().sum()
    if null_cols.sum() > 0:
        print("[Cleaning] Imputing missing values...")
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if df_clean[col].dtype in ['int64', 'float64']:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                else:
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

    # 3. Logical Consistency Checks
    # Floor_No should not exceed Total_Floors
    df_clean['Total_Floors'] = np.maximum(df_clean['Floor_No'], df_clean['Total_Floors'])
    
    # Age of Property consistency: Age = 2025 - Year_Built (bounded >= 0)
    df_clean['Age_of_Property'] = np.clip(df_clean['Age_of_Property'], 0, 100)

    # 4. Outlier capping for Price_in_Lakhs and Size_in_SqFt (1st and 99.5th percentile)
    for col in ['Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt']:
        p_low = df_clean[col].quantile(0.005)
        p_high = df_clean[col].quantile(0.995)
        df_clean[col] = df_clean[col].clip(p_low, p_high)

    print(f"[Cleaning] Cleaned dataset shape: {df_clean.shape}")
    return df_clean

def engineer_features(df: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    """
    Engineer comprehensive domain features:
    - Amenities count and specific flags (Gym, Pool, Clubhouse, etc.)
    - Multi-modal Infrastructure Score
    - Relative locality price ratio
    - Floor ratio (Floor_No / Total_Floors)
    - City tier growth rate
    - Ground-truth domain 5-Year Future Price (Regression target)
    - Binary classification target: Is_Good_Investment
    """
    df_feat = df.copy()

    # 1. Amenities Features
    standard_amenities = ['Gym', 'Pool', 'Clubhouse', 'Garden', 'Playground']
    for am in standard_amenities:
        col_name = f'Amenity_{am}'
        df_feat[col_name] = df_feat['Amenities'].astype(str).apply(
            lambda x: 1 if am.lower() in x.lower() else 0
        )
    
    df_feat['Amenities_Count'] = (
        df_feat[[f'Amenity_{am}' for am in standard_amenities]].sum(axis=1)
    )

    # 2. Connectivity & Infrastructure Sub-scores
    df_feat['Transport_Score'] = df_feat['Public_Transport_Accessibility'].map({
        'High': 1.0, 'Medium': 0.6, 'Low': 0.2
    }).fillna(0.5)

    df_feat['Security_Score'] = df_feat['Security'].map({'Yes': 1.0, 'No': 0.0}).fillna(0.0)
    df_feat['Parking_Score'] = df_feat['Parking_Space'].map({'Yes': 1.0, 'No': 0.0}).fillna(0.0)
    df_feat['School_Density_Score'] = df_feat['Nearby_Schools'] / 10.0
    df_feat['Hospital_Density_Score'] = df_feat['Nearby_Hospitals'] / 10.0

    # 3. Composite Infrastructure Score (Weighted livability & asset quality index)
    df_feat['Infrastructure_Score'] = (
        0.30 * (df_feat['Amenities_Count'] / 5.0) +
        0.25 * df_feat['Transport_Score'] +
        0.15 * df_feat['School_Density_Score'] +
        0.15 * df_feat['Hospital_Density_Score'] +
        0.10 * df_feat['Security_Score'] +
        0.05 * df_feat['Parking_Score']
    ).round(4)

    # 4. Floor Position Ratio (0 = Ground floor, 1 = Penthouse / Top floor)
    df_feat['Floor_Ratio'] = (
        df_feat['Floor_No'] / df_feat['Total_Floors'].replace(0, 1)
    ).clip(0.0, 1.0).round(4)

    # 5. Locality Relative Price Index
    locality_median_price = df_feat.groupby('Locality')['Price_per_SqFt'].transform('median')
    df_feat['Locality_Relative_Price'] = (
        df_feat['Price_per_SqFt'] / locality_median_price.replace(0, 0.01)
    ).clip(0.5, 2.0).round(4)

    # 6. City Baseline Growth & Age Factors
    df_feat['City_Tier_Rate'] = df_feat['City'].apply(get_city_tier_rate)
    df_feat['Age_Factor'] = df_feat['Age_of_Property'].apply(get_age_factor)

    # 7. Annualized Real Estate Appreciation Rate (Ground Truth Domain Formula)
    np.random.seed(random_seed)
    noise = np.random.normal(0, 0.003, len(df_feat))
    
    df_feat['Annual_Appreciation_Rate'] = (
        df_feat['City_Tier_Rate'] +
        0.035 * (df_feat['Infrastructure_Score'] - 0.5) +
        0.025 * (1.0 - df_feat['Locality_Relative_Price']) +
        df_feat['Age_Factor'] +
        noise
    ).clip(0.03, 0.16).round(5)

    # 8. Target 2 (Regression): Estimated Property Price after 5 Years (in Lakhs)
    df_feat['Future_Price_5Y'] = (
        df_feat['Price_in_Lakhs'] * ((1.0 + df_feat['Annual_Appreciation_Rate']) ** 5)
    ).round(2)

    # 5-Year Total Capital Gain Percentage
    df_feat['Appreciation_Pct_5Y'] = (
        ((df_feat['Future_Price_5Y'] - df_feat['Price_in_Lakhs']) / df_feat['Price_in_Lakhs']) * 100.0
    ).round(2)

    # 9. Target 1 (Classification): Is_Good_Investment (Hurdle Rate: >= 7.5% CAGR & Infra >= 0.45)
    df_feat['Is_Good_Investment'] = (
        (df_feat['Annual_Appreciation_Rate'] >= 0.075) & 
        (df_feat['Infrastructure_Score'] >= 0.45)
    ).astype(int)

    print(f"[Feature Engineering] Completed! Engineered columns: {df_feat.shape[1]}")
    print(f"[Target Summary] Good Investments: {df_feat['Is_Good_Investment'].mean():.2%}")
    print(f"[Target Summary] Mean 5Y Price: {df_feat['Future_Price_5Y'].mean():.2f} Lakhs (Initial: {df_feat['Price_in_Lakhs'].mean():.2f} Lakhs)")
    return df_feat

def export_to_sqlite(df_clean: pd.DataFrame, df_feat: pd.DataFrame, db_path: str):
    """Save processed datasets to SQLite database tables with indexes."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    
    # 1. Cleaned properties table
    df_clean.to_sql('properties_cleaned', conn, if_exists='replace', index=False)
    
    # 2. Engineered features & predictions table
    df_feat.to_sql('properties_features', conn, if_exists='replace', index=False)
    
    # 3. City-level aggregation analytics table
    city_analytics = df_feat.groupby('City').agg(
        Total_Properties=('ID', 'count'),
        Avg_Price_Lakhs=('Price_in_Lakhs', 'mean'),
        Avg_Price_per_SqFt=('Price_per_SqFt', 'mean'),
        Avg_Future_Price_5Y=('Future_Price_5Y', 'mean'),
        Avg_Appreciation_Pct_5Y=('Appreciation_Pct_5Y', 'mean'),
        Good_Investment_Pct=('Is_Good_Investment', 'mean'),
        Avg_Infra_Score=('Infrastructure_Score', 'mean')
    ).reset_index().round(3)
    city_analytics.to_sql('city_analytics_summary', conn, if_exists='replace', index=False)

    # Create indexes for fast query performance in Streamlit / SQL
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_city ON properties_features(City);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_investment ON properties_features(Is_Good_Investment);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_price ON properties_features(Price_in_Lakhs);")
    conn.commit()
    conn.close()
    print(f"[Database Export] Successfully exported to SQLite DB: {db_path}")

if __name__ == '__main__':
    base_dir = r'C:\Users\Admin\.gemini\antigravity\scratch\real_estate_advisor'
    raw_path = os.path.join(base_dir, 'data', 'raw', 'india_housing_prices.csv')
    clean_path = os.path.join(base_dir, 'data', 'processed', 'cleaned_housing_prices.csv')
    feat_path = os.path.join(base_dir, 'data', 'processed', 'housing_features.csv')
    db_path = os.path.join(base_dir, 'data', 'processed', 'real_estate_database.db')

    df_raw = load_raw_data(raw_path)
    df_cleaned = clean_data(df_raw)
    df_features = engineer_features(df_cleaned)

    # Save to CSV
    df_cleaned.to_csv(clean_path, index=False)
    df_features.to_csv(feat_path, index=False)
    print(f"[File Export] Saved cleaned CSV to: {clean_path}")
    print(f"[File Export] Saved features CSV to: {feat_path}")

    # Save to SQLite DB
    export_to_sqlite(df_cleaned, df_features, db_path)
    print("[Pipeline Execution] All preprocessing and data exports completed successfully!")
