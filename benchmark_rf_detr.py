import pandas as pd
import numpy as np

def calculate_comprehensive_box_count_metrics(ground_truth_file, predicted_file):
    # Read the CSV files
    ground_truth_df = pd.read_csv(ground_truth_file)
    predicted_df = pd.read_csv(predicted_file)
    
    # Ensure the dataframes are sorted by image name for accurate comparison
    ground_truth_df = ground_truth_df.sort_values('Image Name').reset_index(drop=True)
    predicted_df = predicted_df.sort_values('Image Name').reset_index(drop=True)
    
    
    
    # Verify that image names match
    if not ground_truth_df['Image Name'].equals(predicted_df['Image Name']):
        raise ValueError("Image names in ground truth and predicted files do not match!")
    
    # Extract Box Count 1 columns
    ground_truth_counts = ground_truth_df['Box Count 1']
    predicted_counts = predicted_df['Box Count 1']
    
    # Initialize metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    
    # Calculate metrics for each image
    for gt_count, pred_count in zip(ground_truth_counts, predicted_counts):
        # Case 1: Both have boxes
        if gt_count > 0 and pred_count > 0:
            tp = min(gt_count, pred_count)
            fp = max(0, pred_count - gt_count)
            fn = max(0, gt_count - pred_count)
            tn = 0
        
        # Case 2: Ground truth has boxes, prediction has none
        elif gt_count > 0 and pred_count == 0:
            tp = 0
            fp = 0
            fn = gt_count
            tn = 0
        
        # Case 3: No boxes in ground truth, prediction has boxes
        elif gt_count == 0 and pred_count > 0:
            tp = 0
            fp = pred_count
            fn = 0
            tn = 0
        
        # Case 4: No boxes in both ground truth and prediction
        else:
            tp = 0
            fp = 0
            fn = 0
            tn = 1
        
        # Accumulate metrics
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
    
    # Total samples
    total_samples = len(ground_truth_counts)
    
    # Calculate performance metrics using standard formula
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print detailed results
    print("Comprehensive Box Count Comparison Metrics:")
    print(f"Total Images: {total_samples}")
    print("\nConfusion Matrix:")
    print(f"True Positives (TP): {total_tp}")
    print(f"False Positives (FP): {total_fp}")
    print(f"False Negatives (FN): {total_fn}")
    print(f"True Negatives (TN): {total_tn}")
    
    print("\nPerformance Metrics:")
    print(f"Total Ground Truth Boxes: {sum(ground_truth_counts)}")
    print(f"Total Predicted Boxes: {sum(predicted_counts)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    
    # Optional: Detailed comparison of counts
    print("\nDetailed Box Count Comparison:")
    for img_name, gt_count, pred_count in zip(
        ground_truth_df['Image Name'], 
        ground_truth_counts, 
        predicted_counts
    ):
        local_tp = min(gt_count, pred_count) if gt_count > 0 and pred_count > 0 else 0
        local_fp = max(0, pred_count - gt_count) if pred_count > 0 else 0
        local_fn = max(0, gt_count - pred_count) if gt_count > 0 else 0
        local_tn = 1 if gt_count == 0 and pred_count == 0 else 0
        
        print(f"{img_name}: GT={gt_count}, Pred={pred_count}, TP={local_tp}, FP={local_fp}, FN={local_fn}, TN={local_tn}")

# Usage
ground_truth_file = 'box_counts_with_categories_human.csv'
predicted_file = 'valid_box_counts_with_categories.csv'

calculate_comprehensive_box_count_metrics(ground_truth_file, predicted_file)

