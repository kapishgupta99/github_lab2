"""
Model Evaluation Script - Iris Classification
==============================================

This script evaluates the trained Random Forest model on the Iris dataset
and saves comprehensive evaluation metrics for tracking model performance.

Metrics tracked:
- Accuracy
- F1 Score (weighted & macro)
- Precision (weighted)
- Recall (weighted)
- Confusion Matrix
- Classification Report

Author: Kapish Gupta
Course: IE 7374 - MLOps, Northeastern University
"""

import os
import json
import joblib
import numpy as np
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)


def load_model_and_scaler(model_dir="models"):
    """
    Load the latest trained model and scaler.
    
    Args:
        model_dir: Directory containing saved models
    
    Returns:
        model: Loaded model
        scaler: Loaded scaler
    """
    print("=" * 60)
    print("LOADING MODEL AND SCALER")
    print("=" * 60)
    
    model_path = os.path.join(model_dir, "iris_rf_model_latest.joblib")
    scaler_path = os.path.join(model_dir, "scaler_latest.joblib")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_model.py first.")
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Run train_model.py first.")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    print(f"Model loaded from: {model_path}")
    print(f"Scaler loaded from: {scaler_path}")
    
    return model, scaler


def prepare_test_data():
    """
    Prepare test data from Iris dataset.
    Uses same split as training for consistency.
    
    Returns:
        X_test: Test features
        y_test: Test labels
        target_names: Class names
    """
    print("\n" + "=" * 60)
    print("PREPARING TEST DATA")
    print("=" * 60)
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Use same split parameters as training
    _, X_test, _, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"Test set prepared: {len(X_test)} samples")
    print(f"Class distribution in test set: {np.bincount(y_test)}")
    
    return X_test, y_test, iris.target_names


def evaluate_model(model, scaler, X_test, y_test, target_names):
    """
    Evaluate the model with comprehensive metrics.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        X_test: Test features
        y_test: Test labels
        target_names: Class names
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    print("\n" + "=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=target_names)
    
    metrics = {
        "accuracy": accuracy,
        "f1_score_weighted": f1_weighted,
        "f1_score_macro": f1_macro,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": class_report
    }
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 40)
    print(f"Accuracy:            {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1 Score (weighted): {f1_weighted:.4f}")
    print(f"F1 Score (macro):    {f1_macro:.4f}")
    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    
    print("\nConfusion Matrix:")
    print("-" * 40)
    print(f"Classes: {list(target_names)}")
    print(np.array(conf_matrix))
    
    print("\nDetailed Classification Report:")
    print("-" * 40)
    print(class_report)
    
    return metrics


def save_metrics(metrics, target_names, metrics_dir="metrics"):
    """
    Save evaluation metrics to JSON and text files.
    
    Args:
        metrics: Dictionary of evaluation metrics
        target_names: Class names
        metrics_dir: Directory to save metrics
    
    Returns:
        metrics_path: Path to saved metrics JSON
    """
    print("\n" + "=" * 60)
    print("SAVING METRICS")
    print("=" * 60)
    
    os.makedirs(metrics_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare metrics for JSON (exclude long string report)
    metrics_json = {
        "timestamp": timestamp,
        "dataset": "Iris",
        "model": "RandomForestClassifier",
        "modification": "Using Iris dataset instead of synthetic data",
        "accuracy": round(metrics["accuracy"], 4),
        "f1_score_weighted": round(metrics["f1_score_weighted"], 4),
        "f1_score_macro": round(metrics["f1_score_macro"], 4),
        "precision_weighted": round(metrics["precision_weighted"], 4),
        "recall_weighted": round(metrics["recall_weighted"], 4),
        "confusion_matrix": metrics["confusion_matrix"],
        "class_names": list(target_names)
    }
    
    # Save versioned metrics
    metrics_filename = f"metrics_{timestamp}.json"
    metrics_path = os.path.join(metrics_dir, metrics_filename)
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=4)
    print(f"Versioned metrics saved: {metrics_path}")
    
    # Save as latest
    latest_path = os.path.join(metrics_dir, "metrics_latest.json")
    with open(latest_path, 'w') as f:
        json.dump(metrics_json, f, indent=4)
    print(f"Latest metrics saved:    {latest_path}")
    
    # Save detailed classification report
    report_path = os.path.join(metrics_dir, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("IRIS CLASSIFICATION - MODEL EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset: Iris (3-class flower classification)\n")
        f.write(f"Model: Random Forest Classifier\n")
        f.write(f"Modification: Using Iris dataset instead of synthetic data\n\n")
        f.write("-" * 40 + "\n")
        f.write("METRICS SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy:            {metrics['accuracy']:.4f}\n")
        f.write(f"F1 Score (weighted): {metrics['f1_score_weighted']:.4f}\n")
        f.write(f"F1 Score (macro):    {metrics['f1_score_macro']:.4f}\n")
        f.write(f"Precision:           {metrics['precision_weighted']:.4f}\n")
        f.write(f"Recall:              {metrics['recall_weighted']:.4f}\n\n")
        f.write("-" * 40 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("-" * 40 + "\n")
        f.write(metrics['classification_report'])
        f.write("\n" + "-" * 40 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("-" * 40 + "\n")
        f.write(f"Classes: {list(target_names)}\n")
        f.write(str(np.array(metrics['confusion_matrix'])))
        f.write("\n")
    print(f"Classification report:   {report_path}")
    
    return metrics_path


def main():
    """
    Main function to orchestrate model evaluation.
    """
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + "   IRIS CLASSIFICATION - MODEL EVALUATION   ".center(58) + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    print("\n")
    
    # Step 1: Load model and scaler
    model, scaler = load_model_and_scaler()
    
    # Step 2: Prepare test data
    X_test, y_test, target_names = prepare_test_data()
    
    # Step 3: Evaluate model
    metrics = evaluate_model(model, scaler, X_test, y_test, target_names)
    
    # Step 4: Save metrics
    metrics_path = save_metrics(metrics, target_names)
    
    # Summary
    print("\n")
    print("*" * 60)
    print("*" + " EVALUATION COMPLETED SUCCESSFULLY ".center(58) + "*")
    print("*" * 60)
    print(f"\n  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['f1_score_weighted']:.4f}")
    print(f"  Metrics saved to: {metrics_path}")
    print("\n")
    
    return metrics


if __name__ == "__main__":
    main()
