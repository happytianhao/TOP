"""
Evaluation Metrics for Traffic Accident Anticipation

This script demonstrates the key evaluation metrics used in the TOP framework.
The metrics are designed to assess both the accuracy and timeliness of accident predictions.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_auc_at_fpr(y_true, y_scores, fpr_max=0.1):
    """
    Compute Area Under ROC Curve (AUC) up to a maximum False Positive Rate.

    Args:
        y_true (np.ndarray): Ground truth binary labels (1 for accident, 0 for normal)
        y_scores (np.ndarray): Predicted scores
        fpr_max (float): Maximum FPR to consider (default: 0.1 for 10% FPR)

    Returns:
        float: AUC value normalized by fpr_max
        list: FPR values
        list: TPR values
    """
    # Sort by prediction scores (descending)
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]

    # Count positive and negative samples
    P = np.sum(y_true == 1)  # Total positives
    N = np.sum(y_true == 0)  # Total negatives

    # Initialize TPR and FPR
    TPR = [0]
    FPR = [0]
    TP, FP = 0, 0

    # Compute ROC curve
    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            TP += 1
        else:
            FP += 1
        TPR.append(TP / P)
        FPR.append(FP / N)

    # Compute AUC using trapezoidal rule up to fpr_max
    auc = 0
    for i in range(1, len(FPR)):
        if FPR[i] > fpr_max:
            break
        dx = FPR[i] - FPR[i - 1]
        dy = TPR[i] + TPR[i - 1]
        auc += dx * dy / 2

    return auc / fpr_max, FPR, TPR


def compute_average_precision(y_true, y_scores):
    """
    Compute Average Precision (AP).

    Args:
        y_true (np.ndarray): Ground truth binary labels
        y_scores (np.ndarray): Predicted scores

    Returns:
        float: Average Precision score
    """
    return average_precision_score(y_true, y_scores)


def compute_time_to_accident(predictions, accident_frame_idx, threshold=0.5, fps=10.0):
    """
    Compute Time-to-Accident (TTA) - how early the model detects an accident.

    Args:
        predictions (np.ndarray): Predicted scores for each frame
        accident_frame_idx (int): Index of the actual accident frame
        threshold (float): Prediction threshold for detection
        fps (float): Frames per second

    Returns:
        float: Time-to-accident in seconds (0 if not detected before accident)
    """
    # Find first frame where prediction exceeds threshold
    detected_frames = np.where(predictions[:accident_frame_idx + 1] >= threshold)[0]

    if len(detected_frames) > 0:
        first_detection = detected_frames[0]
        tta = (accident_frame_idx - first_detection) / fps
        return tta
    else:
        return 0.0


def compute_mean_tta(predictions_list, accident_indices, abnormal_start_indices,
                     thresholds, fps=10.0):
    """
    Compute Mean Time-to-Accident (mTTA) across multiple videos and thresholds.

    Args:
        predictions_list (list): List of prediction arrays for each video
        accident_indices (list): List of accident frame indices
        abnormal_start_indices (list): List of abnormal start frame indices
        thresholds (np.ndarray): Array of thresholds to evaluate
        fps (float): Frames per second

    Returns:
        np.ndarray: Mean TTA for each threshold
    """
    all_ttas = []

    for pred, accident_idx, abnormal_idx in zip(predictions_list, accident_indices,
                                                  abnormal_start_indices):
        # Only consider frames between abnormal start and accident
        relevant_pred = pred[abnormal_idx:accident_idx + 1]

        # For each threshold, find when prediction first exceeds it
        ttas_per_threshold = []
        for threshold in thresholds:
            detected = np.where(relevant_pred >= threshold)[0]
            if len(detected) > 0:
                frames_before_accident = len(relevant_pred) - 1 - detected[0]
                tta = frames_before_accident / fps
            else:
                tta = 0.0
            ttas_per_threshold.append(tta)

        all_ttas.append(ttas_per_threshold)

    # Average across all videos
    mean_ttas = np.mean(all_ttas, axis=0)
    return mean_ttas


def compute_temporal_metrics(predictions_pos, predictions_neg,
                             accident_indices, abnormal_start_indices,
                             time_windows=[0.0, 0.5, 1.0, 1.5], fps=10.0):
    """
    Compute temporal evaluation metrics at different time windows before accident.

    This is the core evaluation metric for TOP framework.

    Args:
        predictions_pos (list): Predictions for positive (accident) videos
        predictions_neg (list): Predictions for negative (normal) videos
        accident_indices (list): Accident frame indices for positive videos
        abnormal_start_indices (list): Abnormal start frame indices
        time_windows (list): Time windows in seconds before accident
        fps (float): Frames per second

    Returns:
        dict: Dictionary containing mAP, mAUC, and TTA metrics
    """
    results = {}

    # For each time window, extract max prediction score
    for t in time_windows:
        frame_offset = int(t * fps)

        # Extract scores from positive videos
        scores_pos = []
        for pred, acc_idx in zip(predictions_pos, accident_indices):
            if acc_idx >= frame_offset + 5:
                # Take max score in 0.5s window before the target time
                window_start = max(0, acc_idx - frame_offset - 5)
                window_end = acc_idx - frame_offset
                if window_end > window_start:
                    scores_pos.append(np.max(pred[window_start:window_end]))

        # Extract scores from negative videos (take max from last 0.5s)
        scores_neg = [np.max(pred[-5:]) for pred in predictions_neg]

        # Combine into binary classification problem
        y_true = np.array([1] * len(scores_pos) + [0] * len(scores_neg))
        y_scores = np.array(scores_pos + scores_neg)

        # Compute metrics
        auc_partial, _, _ = compute_auc_at_fpr(y_true, y_scores, fpr_max=0.1)
        auc_full = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)

        results[f'AUC@{t}s'] = auc_partial
        results[f'AUC_full@{t}s'] = auc_full
        results[f'AP@{t}s'] = ap

    # Compute mean metrics (excluding t=0.0s)
    results['mAUC'] = np.mean([results[f'AUC@{t}s'] for t in time_windows[1:]])
    results['mAP'] = np.mean([results[f'AP@{t}s'] for t in time_windows[1:]])

    # Compute Time-to-Accident metrics
    threshold = 0.5
    ttas = [compute_time_to_accident(pred, acc_idx, threshold, fps)
            for pred, acc_idx in zip(predictions_pos, accident_indices)]
    results['TTA@0.5'] = np.mean(ttas)

    return results


# Example usage
if __name__ == "__main__":
    # Simulate predictions for demonstration
    np.random.seed(42)

    # Positive videos (accidents)
    n_pos = 50
    predictions_pos = [np.random.rand(100) * 0.5 + np.linspace(0, 0.5, 100)
                       for _ in range(n_pos)]
    accident_indices = [95] * n_pos
    abnormal_start_indices = [70] * n_pos

    # Negative videos (normal)
    n_neg = 100
    predictions_neg = [np.random.rand(100) * 0.3 for _ in range(n_neg)]

    # Compute metrics
    metrics = compute_temporal_metrics(
        predictions_pos, predictions_neg,
        accident_indices, abnormal_start_indices,
        time_windows=[0.0, 0.5, 1.0, 1.5],
        fps=10.0
    )

    print("=" * 60)
    print("TOP Framework Evaluation Metrics")
    print("=" * 60)
    for metric_name, value in metrics.items():
        print(f"{metric_name:20s}: {value:.4f}")
    print("=" * 60)
