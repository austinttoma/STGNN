import pandas as pd
import numpy as np
from collections import defaultdict

def get_subject_conversion_type(subject_id, label_df):
    # Filter visits for this subject
    subject_visits = label_df[label_df['Subject'] == subject_id].sort_values('Visit_idx')
    
    if len(subject_visits) == 0:
        return 'Unknown'
    
    # Get baseline group and conversion label
    baseline_group = subject_visits.iloc[0]['Group']
    is_converter = subject_visits.iloc[0]['Label_CS_Num'] == 1
    
    # Simple logic based on baseline group and converter status
    if baseline_group == 'CN':
        if is_converter:
            return 'CN->MCI'  # CN converters become MCI
        else:
            return 'CN-Stable'
    elif baseline_group == 'MCI':
        if is_converter:
            return 'MCI->AD'  # MCI converters become AD
        else:
            return 'MCI-Stable'
    elif baseline_group == 'AD':
        # Include AD subjects in breakdown but they're typically stable
        return 'AD-Stable'
    
    return 'Unknown'

def analyze_conversion_predictions(test_subjects, predictions, targets, label_csv_path):
    # Load label data
    label_df = pd.read_csv(label_csv_path)
    label_df['Subject'] = label_df['Subject'].str.replace('_', '', regex=False)
    
    # Initialize results dictionary
    results = defaultdict(lambda: {'correct': 0, 'total': 0, 'predictions': [], 'targets': []})
    
    # Analyze each test subject
    for i, subject_id in enumerate(test_subjects):
        conversion_type = get_subject_conversion_type(subject_id, label_df)
        
        pred = predictions[i]
        target = targets[i]
        
        # Store results by conversion type
        results[conversion_type]['total'] += 1
        results[conversion_type]['predictions'].append(pred)
        results[conversion_type]['targets'].append(target)
        
        if pred == target:
            results[conversion_type]['correct'] += 1
    
    # Calculate accuracy for each conversion type
    final_results = {}
    for conv_type, data in results.items():
        if data['total'] > 0:
            accuracy = data['correct'] / data['total']
            
            # Break down by stable vs converter predictions
            stable_correct = sum(1 for p, t in zip(data['predictions'], data['targets']) if p == 0 and t == 0)
            stable_total = sum(1 for t in data['targets'] if t == 0)
            converter_correct = sum(1 for p, t in zip(data['predictions'], data['targets']) if p == 1 and t == 1)
            converter_total = sum(1 for t in data['targets'] if t == 1)
            
            final_results[conv_type] = {
                'overall_accuracy': accuracy,
                'total_subjects': data['total'],
                'correct_predictions': data['correct'],
                'stable_correct': stable_correct,
                'stable_total': stable_total,
                'stable_accuracy': stable_correct / stable_total if stable_total > 0 else 0,
                'converter_correct': converter_correct,
                'converter_total': converter_total,
                'converter_accuracy': converter_correct / converter_total if converter_total > 0 else 0,
                'predictions': data['predictions'],
                'targets': data['targets']
            }
    
    return final_results

def print_conversion_accuracy_report(conversion_results):
    # Show total test subjects in this fold
    total_subjects = sum(result['total_subjects'] for result in conversion_results.values())
    print(f"\nTotal test subjects in this fold: {total_subjects}")
    print()
    
    # Sort conversion types for consistent reporting
    conversion_order = ['CN-Stable', 'CN->MCI', 'MCI-Stable', 'MCI->AD', 'AD-Stable']
    
    for conv_type in conversion_order:
        if conv_type in conversion_results:
            result = conversion_results[conv_type]
            print(f"\n{conv_type}:")
            print(f"  Overall: {result['correct_predictions']}/{result['total_subjects']} correct ({result['overall_accuracy']:.3f})")
            
            if result['stable_total'] > 0:
                print(f"  Stable predictions: {result['stable_correct']}/{result['stable_total']} correct ({result['stable_accuracy']:.3f})")
            
            if result['converter_total'] > 0:
                print(f"  Converter predictions: {result['converter_correct']}/{result['converter_total']} correct ({result['converter_accuracy']:.3f})")
    
    # Print any other conversion types not in the standard order
    for conv_type, result in conversion_results.items():
        if conv_type not in conversion_order:
            print(f"\n{conv_type}:")
            print(f"  Overall: {result['correct_predictions']}/{result['total_subjects']} correct ({result['overall_accuracy']:.3f})")
            
            if result['stable_total'] > 0:
                print(f"  Stable predictions: {result['stable_correct']}/{result['stable_total']} correct ({result['stable_accuracy']:.3f})")
            
            if result['converter_total'] > 0:
                print(f"  Converter predictions: {result['converter_correct']}/{result['converter_total']} correct ({result['converter_accuracy']:.3f})")

def aggregate_conversion_results(fold_conversion_results):
    aggregated = defaultdict(lambda: {
        'total_subjects': 0,
        'correct_predictions': 0,
        'stable_correct': 0,
        'stable_total': 0,
        'converter_correct': 0,
        'converter_total': 0
    })
    
    # Aggregate results across folds
    for fold_results in fold_conversion_results:
        for conv_type, result in fold_results.items():
            agg = aggregated[conv_type]
            agg['total_subjects'] += result['total_subjects']
            agg['correct_predictions'] += result['correct_predictions']
            agg['stable_correct'] += result['stable_correct']
            agg['stable_total'] += result['stable_total']
            agg['converter_correct'] += result['converter_correct']
            agg['converter_total'] += result['converter_total']
    
    # Calculate final accuracies
    final_aggregated = {}
    for conv_type, agg in aggregated.items():
        final_aggregated[conv_type] = {
            'overall_accuracy': agg['correct_predictions'] / agg['total_subjects'] if agg['total_subjects'] > 0 else 0,
            'total_subjects': agg['total_subjects'],
            'correct_predictions': agg['correct_predictions'],
            'stable_correct': agg['stable_correct'],
            'stable_total': agg['stable_total'],
            'stable_accuracy': agg['stable_correct'] / agg['stable_total'] if agg['stable_total'] > 0 else 0,
            'converter_correct': agg['converter_correct'],
            'converter_total': agg['converter_total'],
            'converter_accuracy': agg['converter_correct'] / agg['converter_total'] if agg['converter_total'] > 0 else 0
        }
    
    return final_aggregated