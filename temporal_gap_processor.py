import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional

def parse_viscode_to_months(viscode: str) -> int:
    viscode = str(viscode).lower().strip()
    
    # Comprehensive VISCODE to months mapping
    viscode_mapping = {
        # Baseline visits (0 months)
        'sc': 0,
        'scmri': 0,
        'bl': 0,
        'init': 0,
        'v02': 0,  # ADNI-2 screening + MRI baseline
        'v03': 0,  # ADNI-2 baseline in-clinic
        'v06': 0,  # ADNI-2 baseline for roll-overs from ADNI-GO
        
        # ADNI-2 timeline (v-codes with specific months)
        'v04': 3,   # Month 3 MRI
        'v05': 6,   # Month 6
        'v11': 12,  # Year 1
        'v21': 24,  # Year 2
        'v31': 36,  # Year 3
        'v41': 48,  # Year 4
        'v51': 60,  # Year 5
        
        # ADNI-3 timeline (year codes)
        'y1': 12,
        'y2': 24,
        'y3': 36,
        'y4': 48,
        'y5': 60,
    }
    
    # Check direct mapping first
    if viscode in viscode_mapping:
        return viscode_mapping[viscode]
    
    # Handle ADNI-1/GO month codes (m03, m06, m12, m18, m48, etc.)
    if viscode.startswith('m') and len(viscode) > 1:
        try:
            months = int(viscode[1:])
            return months
        except ValueError:
            print(f"Warning: Unable to parse month code '{viscode}', defaulting to 0")
            return 0
    
    # Handle special/undefined codes
    if viscode in ['nv', 'uns1'] or viscode.startswith('tel'):
        print(f"Warning: Special/undefined visit code '{viscode}', defaulting to -1")
        return -1
    
    print(f"Warning: Unknown VISCODE '{viscode}', defaulting to 0")
    return 0


def calculate_temporal_gaps(df: pd.DataFrame, 
                           subject_col: str = 'Subject',
                           visit_col: str = 'Visit') -> pd.DataFrame:
    df = df.copy()
    
    # Parse visit codes to months
    df['visit_months'] = df[visit_col].apply(parse_viscode_to_months)
    
    # Initialize new columns
    df['months_to_next'] = np.nan
    df['visit_order'] = 0
    
    # Process each subject
    for subject_id in df[subject_col].unique():
        subject_mask = df[subject_col] == subject_id
        subject_data = df.loc[subject_mask].copy()
        
        # Sort by visit months (handles out-of-order visits)
        subject_data = subject_data.sort_values('visit_months')
        
        # Calculate months to next visit
        visit_months = subject_data['visit_months'].values
        months_to_next = np.diff(visit_months)
        
        # Assign months_to_next (last visit gets NaN)
        subject_indices = subject_data.index
        if len(months_to_next) > 0:
            df.loc[subject_indices[:-1], 'months_to_next'] = months_to_next
        
        # Assign visit order
        df.loc[subject_indices, 'visit_order'] = range(1, len(subject_data) + 1)
    
    return df


def normalize_time_gaps(gaps: np.ndarray, 
                       method: str = 'log',
                       max_months: float = 60.0) -> np.ndarray:
    gaps = np.array(gaps, dtype=np.float32)
    
    if method == 'log':
        # Log scale: log(1 + months/12) - good for capturing both short and long-term
        return np.log1p(gaps / 12.0)
    
    elif method == 'minmax':
        # Min-max scaling to [0, 1]
        return np.clip(gaps / max_months, 0, 1)
    
    elif method == 'buckets':
        # Categorical buckets for different time horizons
        buckets = np.zeros_like(gaps)
        buckets[(gaps > 0) & (gaps <= 6)] = 0.25   # 0-6 months
        buckets[(gaps > 6) & (gaps <= 12)] = 0.5   # 6-12 months
        buckets[(gaps > 12) & (gaps <= 24)] = 0.75 # 12-24 months
        buckets[gaps > 24] = 1.0                    # 24+ months
        return buckets
    
    elif method == 'raw':
        # Raw months divided by 12 (in years)
        return gaps / 12.0
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_temporal_features(visit_sequence: List[str],
                            normalization: str = 'log') -> Tuple[List[int], List[float]]:
    # Convert visit codes to months
    visit_months = [parse_viscode_to_months(v) for v in visit_sequence]
    
    # Calculate gaps to next visit
    gaps_to_next = []
    for i in range(len(visit_months) - 1):
        gap = visit_months[i + 1] - visit_months[i]
        gaps_to_next.append(gap)
    
    # Last visit has no "next" - use special value
    gaps_to_next.append(-1)  # Will be handled specially in the model
    
    # Normalize gaps (excluding the last special value)
    if len(gaps_to_next) > 1:
        normalized_gaps = normalize_time_gaps(gaps_to_next[:-1], method=normalization)
        normalized_gaps = normalized_gaps.tolist()
        normalized_gaps.append(-1)  # Keep special value for last visit
    else:
        normalized_gaps = gaps_to_next
    
    return visit_months, normalized_gaps


def analyze_temporal_distribution(df: pd.DataFrame,
                                 visit_col: str = 'Visit',
                                 output_stats: bool = True) -> Dict:
    # Parse all visit codes
    visit_months = df[visit_col].apply(parse_viscode_to_months).values
    valid_months = visit_months[visit_months >= 0]  # Exclude special codes
    
    stats = {
        'total_visits': len(df),
        'valid_visits': len(valid_months),
        'unique_timepoints': len(np.unique(valid_months)),
        'min_months': np.min(valid_months) if len(valid_months) > 0 else 0,
        'max_months': np.max(valid_months) if len(valid_months) > 0 else 0,
        'mean_months': np.mean(valid_months) if len(valid_months) > 0 else 0,
        'median_months': np.median(valid_months) if len(valid_months) > 0 else 0,
        'baseline_visits': np.sum(valid_months == 0),
        'visits_0_6m': np.sum((valid_months > 0) & (valid_months <= 6)),
        'visits_6_12m': np.sum((valid_months > 6) & (valid_months <= 12)),
        'visits_12_24m': np.sum((valid_months > 12) & (valid_months <= 24)),
        'visits_24m_plus': np.sum(valid_months > 24)
    }
    
    if output_stats:
        print("\n=== Temporal Distribution Analysis ===")
        print(f"Total visits: {stats['total_visits']}")
        print(f"Valid temporal visits: {stats['valid_visits']}")
        print(f"Unique timepoints: {stats['unique_timepoints']}")
        print(f"Time range: {stats['min_months']:.0f} - {stats['max_months']:.0f} months")
        print(f"Mean: {stats['mean_months']:.1f} months, Median: {stats['median_months']:.0f} months")
        print(f"\nVisit distribution by time horizon:")
        print(f"  Baseline: {stats['baseline_visits']} visits")
        print(f"  0-6 months: {stats['visits_0_6m']} visits")
        print(f"  6-12 months: {stats['visits_6_12m']} visits")
        print(f"  12-24 months: {stats['visits_12_24m']} visits")
        print(f"  24+ months: {stats['visits_24m_plus']} visits")
    
    return stats


if __name__ == "__main__":
    # Test VISCODE parsing
    print("Testing VISCODE parsing:")
    test_codes = ['init', 'scmri', 'v02', 'v04', 'v05', 'v06', 'v11', 'v21', 
                  'v31', 'v41', 'v51', 'm03', 'm06', 'm48', 'y1', 'y2', 'y3', 'y4', 'y5']
    
    for code in test_codes:
        months = parse_viscode_to_months(code)
        print(f"  {code:8s} → {months:3d} months")
    
    # Test with actual data if available
    try:
        df = pd.read_csv('/media/volume/ADNI-Data/git/TabGNN/FinalDeliverables/data/TADPOLE_Simplified.csv')
        print("\n" + "="*50)
        print("Analyzing TADPOLE dataset temporal distribution:")
        stats = analyze_temporal_distribution(df)
        
        # Add temporal features to dataframe
        df_with_temporal = calculate_temporal_gaps(df)
        print("\n" + "="*50)
        print("Sample of data with temporal features:")
        print(df_with_temporal[['Subject', 'Visit', 'visit_months', 'months_to_next', 'visit_order']].head(10))
        
    except FileNotFoundError:
        print("\nTADPOLE_Simplified.csv not found - skipping real data test")