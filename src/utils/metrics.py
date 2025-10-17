import numpy as np
from typing import Union, List, Tuple
import logging

logger = logging.getLogger(__name__)


def mean_squared_error(y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List]) -> float:
    """
    Calculate Mean Squared Error (MSE)
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MSE value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    squared_errors = (y_true - y_pred) ** 2
    return np.mean(squared_errors)


def root_mean_squared_error(y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List]) -> float:
    """
    Calculate Root Mean Squared Error (RMSE)
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        RMSE value
    """
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


def mean_absolute_error(y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List]) -> float:
    """
    Calculate Mean Absolute Error (MAE)
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MAE value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    absolute_errors = np.abs(y_true - y_pred)
    return np.mean(absolute_errors)


def r2_score(y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List]) -> float:
    """
    Calculate R-squared (coefficient of determination)
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        R² score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)


def accuracy_score(y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List]) -> float:
    """
    Calculate accuracy for classification
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Accuracy score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    return np.mean(y_true == y_pred)


def precision_score(y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List], 
                   average: str = 'binary') -> float:
    """
    Calculate precision score
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy ('binary', 'micro', 'macro', 'weighted')
    
    Returns:
        Precision score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if average == 'binary':
        # For binary classification
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        
        if tp + fp == 0:
            return 0.0
        
        return tp / (tp + fp)
    
    # For multiclass, calculate per-class precision
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    precisions = []
    
    for label in unique_labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        
        if tp + fp == 0:
            precisions.append(0.0)
        else:
            precisions.append(tp / (tp + fp))
    
    if average == 'macro':
        return np.mean(precisions)
    elif average == 'micro':
        # Micro-averaged precision is same as accuracy for multiclass
        return accuracy_score(y_true, y_pred)
    else:
        # Weighted average
        label_counts = np.bincount(y_true)
        weights = label_counts / len(y_true)
        return np.average(precisions, weights=weights)


def recall_score(y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List], 
                average: str = 'binary') -> float:
    """
    Calculate recall score
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy ('binary', 'micro', 'macro', 'weighted')
    
    Returns:
        Recall score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if average == 'binary':
        # For binary classification
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        if tp + fn == 0:
            return 0.0
        
        return tp / (tp + fn)
    
    # For multiclass, calculate per-class recall
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    recalls = []
    
    for label in unique_labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        
        if tp + fn == 0:
            recalls.append(0.0)
        else:
            recalls.append(tp / (tp + fn))
    
    if average == 'macro':
        return np.mean(recalls)
    elif average == 'micro':
        # Micro-averaged recall is same as accuracy for multiclass
        return accuracy_score(y_true, y_pred)
    else:
        # Weighted average
        label_counts = np.bincount(y_true)
        weights = label_counts / len(y_true)
        return np.average(recalls, weights=weights)


def f1_score(y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List], 
             average: str = 'binary') -> float:
    """
    Calculate F1 score
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy ('binary', 'micro', 'macro', 'weighted')
    
    Returns:
        F1 score
    """
    precision = precision_score(y_true, y_pred, average)
    recall = recall_score(y_true, y_pred, average)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def confusion_matrix(y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List]) -> np.ndarray:
    """
    Calculate confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Confusion matrix
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    n_labels = len(unique_labels)
    
    cm = np.zeros((n_labels, n_labels), dtype=int)
    
    for i, true_label in enumerate(unique_labels):
        for j, pred_label in enumerate(unique_labels):
            cm[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    
    return cm


def classification_report(y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List]) -> dict:
    """
    Generate classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Dictionary with precision, recall, f1-score for each class
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    
    report = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_avg': {
            'precision': precision_score(y_true, y_pred, 'macro'),
            'recall': recall_score(y_true, y_pred, 'macro'),
            'f1_score': f1_score(y_true, y_pred, 'macro')
        },
        'weighted_avg': {
            'precision': precision_score(y_true, y_pred, 'weighted'),
            'recall': recall_score(y_true, y_pred, 'weighted'),
            'f1_score': f1_score(y_true, y_pred, 'weighted')
        },
        'per_class': {}
    }
    
    for label in unique_labels:
        report['per_class'][f'class_{label}'] = {
            'precision': precision_score(y_true == label, y_pred == label, 'binary'),
            'recall': recall_score(y_true == label, y_pred == label, 'binary'),
            'f1_score': f1_score(y_true == label, y_pred == label, 'binary')
        }
    
    return report


def evaluate_model(y_true: Union[np.ndarray, List], y_pred: Union[np.ndarray, List], 
                  task_type: str = 'regression') -> dict:
    """
    Comprehensive model evaluation
    
    Args:
        y_true: True values/labels
        y_pred: Predicted values/labels
        task_type: 'regression' or 'classification'
    
    Returns:
        Dictionary with all relevant metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if task_type == 'regression':
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': root_mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred)
        }
    elif task_type == 'classification':
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, 'macro'),
            'recall': recall_score(y_true, y_pred, 'macro'),
            'f1_score': f1_score(y_true, y_pred, 'macro'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred)
        }
    else:
        raise ValueError("task_type must be 'regression' or 'classification'")


# Example usage and testing
if __name__ == "__main__":
    # Test regression metrics
    y_true_reg = [1, 2, 3, 4, 5]
    y_pred_reg = [1.1, 2.2, 2.8, 4.1, 4.9]
    
    print("Regression Metrics:")
    print(f"MSE: {mean_squared_error(y_true_reg, y_pred_reg):.4f}")
    print(f"RMSE: {root_mean_squared_error(y_true_reg, y_pred_reg):.4f}")
    print(f"MAE: {mean_absolute_error(y_true_reg, y_pred_reg):.4f}")
    print(f"R²: {r2_score(y_true_reg, y_pred_reg):.4f}")
    
    # Test classification metrics
    y_true_clf = [0, 1, 0, 1, 1, 0, 1, 0]
    y_pred_clf = [0, 1, 1, 1, 0, 0, 1, 0]
    
    print("\nClassification Metrics:")
    print(f"Accuracy: {accuracy_score(y_true_clf, y_pred_clf):.4f}")
    print(f"Precision: {precision_score(y_true_clf, y_pred_clf):.4f}")
    print(f"Recall: {recall_score(y_true_clf, y_pred_clf):.4f}")
    print(f"F1-Score: {f1_score(y_true_clf, y_pred_clf):.4f}")
    
    # Comprehensive evaluation
    print("\nComprehensive Evaluation:")
    reg_results = evaluate_model(y_true_reg, y_pred_reg, 'regression')
    clf_results = evaluate_model(y_true_clf, y_pred_clf, 'classification')
    
    print("Regression:", reg_results)
    print("Classification:", clf_results)
