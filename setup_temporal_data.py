#!/usr/bin/env python3
"""
Script to set up temporal data for time-aware prediction.
Creates temporal sequences from ADNI data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import os

def parse_date(date_str: str) -> datetime:
    """Parse date string from TADPOLE_COMPLETE format."""
    try:
        # Try common formats
        for fmt in ['%m/%d/%Y', '%m/%d/%y', '%Y-%m-%d', '%d/%m/%Y']:
            try:
                return datetime.strptime(str(date_str).strip(), fmt)
            except ValueError:
                continue
        
        # If none work, raise error
        raise ValueError(f"Cannot parse date: {date_str}")
    except Exception as e:
        print(f"Warning: Could not parse date '{date_str}': {e}")
        return None

def months_between_dates(date1: datetime, date2: datetime) -> float:
    """Calculate months between two dates."""
    if date1 is None or date2 is None:
        return np.nan
    
    # Calculate total days and convert to months (approximate)
    days_diff = (date2 - date1).days
    months_diff = days_diff / 30.44  # Average days per month
    return round(months_diff, 1)

def create_temporal_dataset(complete_path: str, simplified_path: str) -> pd.DataFrame:
    """Create temporal sequences with actual dates."""
    print("="*60)
    print("CREATING TEMPORAL DATASET")
    print("="*60)
    
    print("Loading TADPOLE_COMPLETE...")
    df_complete = pd.read_csv(complete_path)
    
    print("Loading TADPOLE_Simplified for labels...")
    df_simplified = pd.read_csv(simplified_path)
    
    # Clean subject IDs (remove underscores in complete, keep in simplified)
    df_complete['Subject'] = df_complete['Subject'].str.replace('_', '', regex=False)
    
    # Use all data from complete file - no fMRI filtering needed
    fmri_data = df_complete.copy()
    
    print(f"TADPOLE_COMPLETE: {len(df_complete)} rows, {df_complete['Subject'].nunique()} subjects")
    print(f"TADPOLE_Simplified: {len(df_simplified)} rows, {df_simplified['Subject'].nunique()} subjects")
    print(f"Processing data: {len(fmri_data)} rows, {fmri_data['Subject'].nunique()} subjects")
    
    # Parse acquisition dates
    print("Parsing acquisition dates...")
    fmri_data['Acq_Date_Parsed'] = fmri_data['Acq Date'].apply(parse_date)
    fmri_data = fmri_data[fmri_data['Acq_Date_Parsed'].notna()]
    
    print(f"Data with valid dates: {len(fmri_data)} rows")
    
    # Create temporal sequences
    print("Creating temporal sequences...")
    temporal_data = []
    
    for subject in fmri_data['Subject'].unique():
        subject_data = fmri_data[fmri_data['Subject'] == subject].copy()
        
        # Remove duplicates based on Subject, Visit, and Acq Date
        subject_data = subject_data.drop_duplicates(subset=['Subject', 'Visit', 'Acq Date'], keep='first')
        
        subject_data = subject_data.sort_values('Acq_Date_Parsed')
        
        # Get label for this subject from simplified data
        simplified_subject = df_simplified[df_simplified['Subject'] == subject]
        if len(simplified_subject) == 0:
            continue
        
        # Use the label from the simplified dataset
        subject_label = simplified_subject['Label_CS_Num'].iloc[0]
        
        # Get baseline date (first visit) for this subject
        baseline_date = subject_data.iloc[0]['Acq_Date_Parsed']
        
        for idx, (_, row) in enumerate(subject_data.iterrows()):
            # Calculate months from baseline
            months_from_baseline = months_between_dates(baseline_date, row['Acq_Date_Parsed'])
            if months_from_baseline is None or np.isnan(months_from_baseline):
                months_from_baseline = 0
            
            # Calculate time to next visit
            if idx < len(subject_data) - 1:
                next_date = subject_data.iloc[idx + 1]['Acq_Date_Parsed']
                months_to_next = months_between_dates(row['Acq_Date_Parsed'], next_date)
            else:
                months_to_next = np.nan
            
            # Add to temporal data
            temporal_entry = {
                'Subject': subject,
                'Visit': row['Visit'],
                'Acq_Date': row['Acq Date'],
                'Months_From_Baseline': months_from_baseline,
                'Months_To_Next': months_to_next,
                'Label_CS_Num': subject_label,
                'Visit_Order': idx + 1,
                'Total_Visits': len(subject_data)
            }
            
            # Add other relevant columns that exist in TADPOLE_COMPLETE
            for col in ['Age', 'Sex', 'Group']:
                if col in row:
                    temporal_entry[col] = row[col]
            
            temporal_data.append(temporal_entry)
    
    df_temporal = pd.DataFrame(temporal_data)
    print(f"Temporal dataset created: {len(df_temporal)} rows, {df_temporal['Subject'].nunique()} subjects")
    
    return df_temporal

def analyze_prediction_pairs(df: pd.DataFrame):
    """Analyze prediction opportunities in the temporal dataset."""
    print("\n" + "="*60)  
    print("PREDICTION PAIRS ANALYSIS")
    print("="*60)
    
    # Count visits per subject
    visits_per_subject = df.groupby('Subject').size()
    multi_visit_subjects = visits_per_subject[visits_per_subject > 1]
    
    print(f"Subjects available:")
    print(f"  Single visit: {len(visits_per_subject[visits_per_subject == 1])}")
    print(f"  Multi-visit: {len(multi_visit_subjects)}")
    print(f"  Total prediction pairs: {sum(multi_visit_subjects - 1)}")
    
    # Analyze temporal gaps for prediction horizons
    valid_gaps = df['Months_To_Next'].dropna()
    if len(valid_gaps) > 0:
        # Define prediction horizon bins
        bins = [0, 6, 12, 24, float('inf')]
        labels = ['0-6m', '6-12m', '12-24m', '24m+']
        
        print(f"\nPrediction horizons:")
        for i in range(len(bins)-1):
            count = len(valid_gaps[(valid_gaps >= bins[i]) & (valid_gaps < bins[i+1])])
            print(f"  {labels[i]:8s}: {count:3d} prediction opportunities")

if __name__ == "__main__":
    # File paths
    complete_path = 'data/TADPOLE_COMPLETE.csv'
    simplified_path = 'data/TADPOLE_Simplified.csv'
    
    # Check if input files exist
    if not os.path.exists(complete_path):
        print(f"Error: {complete_path} not found!")
        exit(1)
    
    if not os.path.exists(simplified_path):
        print(f"Error: {simplified_path} not found!")
        exit(1)
    
    print("SETTING UP TEMPORAL DATA FOR TIME-AWARE PREDICTION")
    print("="*60)
    
    # Create temporal dataset
    df_temporal = create_temporal_dataset(complete_path, simplified_path)
    
    # Analyze prediction pairs
    analyze_prediction_pairs(df_temporal)
    
    # Save results
    output_file = 'data/TADPOLE_TEMPORAL.csv'
    print(f"\nSAVING RESULTS...")
    df_temporal.to_csv(output_file, index=False)
    
    print(f"File created:")
    print(f"   {output_file}")
    print(f"\nReady for time-aware training!")