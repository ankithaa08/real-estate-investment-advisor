"""
Feature Engineering Utilities for Real Estate Investment Advisor
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

def compute_infrastructure_score(amenities_count: int, transport_accessibility: str, 
                                 schools_count: int, hospitals_count: int,
                                 security: str, parking_space: str) -> float:
    """Calculates weighted composite infrastructure livability score (0.0 to 1.0)."""
    t_map = {'High': 1.0, 'Medium': 0.6, 'Low': 0.2}
    t_score = t_map.get(transport_accessibility, 0.5)
    sec_score = 1.0 if security.lower() == 'yes' else 0.0
    park_score = 1.0 if parking_space.lower() == 'yes' else 0.0
    school_score = np.clip(schools_count / 10.0, 0.0, 1.0)
    hosp_score = np.clip(hospitals_count / 10.0, 0.0, 1.0)
    amenities_score = np.clip(amenities_count / 5.0, 0.0, 1.0)

    score = (
        0.30 * amenities_score +
        0.25 * t_score +
        0.15 * school_score +
        0.15 * hosp_score +
        0.10 * sec_score +
        0.05 * park_score
    )
    return round(float(score), 4)

def prepare_input_dataframe(data_dict: Dict[str, Any]) -> pd.DataFrame:
    """Formats single prediction dictionary into a valid model input DataFrame."""
    price_lakhs = float(data_dict['Price_in_Lakhs'])
    size_sqft = float(data_dict['Size_in_SqFt'])
    price_per_sqft = price_lakhs / max(size_sqft, 1.0)
    
    amenities_count = int(data_dict.get('Amenities_Count', 0))
    infra_score = compute_infrastructure_score(
        amenities_count,
        data_dict.get('Public_Transport_Accessibility', 'Medium'),
        int(data_dict.get('Nearby_Schools', 5)),
        int(data_dict.get('Nearby_Hospitals', 5)),
        data_dict.get('Security', 'No'),
        data_dict.get('Parking_Space', 'No')
    )

    floor_no = float(data_dict.get('Floor_No', 1))
    total_floors = float(data_dict.get('Total_Floors', 1))
    floor_ratio = round(min(1.0, floor_no / max(1.0, total_floors)), 4)

    row = {
        'BHK': [int(data_dict['BHK'])],
        'Size_in_SqFt': [size_sqft],
        'Price_in_Lakhs': [price_lakhs],
        'Price_per_SqFt': [price_per_sqft],
        'Floor_No': [floor_no],
        'Total_Floors': [max(floor_no, total_floors)],
        'Age_of_Property': [float(data_dict.get('Age_of_Property', 5))],
        'Nearby_Schools': [int(data_dict.get('Nearby_Schools', 5))],
        'Nearby_Hospitals': [int(data_dict.get('Nearby_Hospitals', 5))],
        'Amenities_Count': [amenities_count],
        'Infrastructure_Score': [infra_score],
        'Floor_Ratio': [floor_ratio],
        'Locality_Relative_Price': [float(data_dict.get('Locality_Relative_Price', 1.0))],
        'Property_Type': [str(data_dict.get('Property_Type', 'Apartment'))],
        'Furnished_Status': [str(data_dict.get('Furnished_Status', 'Semi-furnished'))],
        'Public_Transport_Accessibility': [str(data_dict.get('Public_Transport_Accessibility', 'Medium'))],
        'Parking_Space': [str(data_dict.get('Parking_Space', 'Yes'))],
        'Security': [str(data_dict.get('Security', 'Yes'))],
        'Facing': [str(data_dict.get('Facing', 'East'))],
        'Owner_Type': [str(data_dict.get('Owner_Type', 'Owner'))],
        'Availability_Status': [str(data_dict.get('Availability_Status', 'Ready_to_Move'))],
        'City': [str(data_dict.get('City', 'Mumbai'))]
    }
    return pd.DataFrame(row)
