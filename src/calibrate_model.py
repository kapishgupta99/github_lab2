"""
Model Calibration Script - Iris Classification
===============================================

This script calibrates the trained Random Forest model's predicted
probabilities using Platt scaling (sigmoid calibration).

Why Calibration?
- Ensures predicted probabilities match actual likelihood of outcomes
- Critical for reliable decision-making in production
- Random Forest often produces overconfident predictions

Author: Kapish Gupta
Course: IE 7374 - MLOps, Northeastern University
"""

import os
import joblib
import numpy as np
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


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
    print("LOADING ORIGINAL MODEL")
    print("=" * 60)
    
    model_path = os.path.join(model_dir, "iris_rf_model_latest.joblib")
    scaler_path = os.path.join(model_dir, "scaler_latest.joblib")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_model.py first.")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    print(f"Model loaded from: {model_path}")
    print(f"Scaler loaded from: {scaler_path}")
    
    return model, scaler


def prepare_calibration_data():
    """
    Prepare data for calibration.
    Splits data into train, calibration, and test sets.
    
    Returns:
        X_train, X_cal, X_test: Feature sets
        y_train, y_cal, y_test: Label sets
        target_names: Class names
    """
    print("\n" + "=" * 60)
    print("PREPARING CALIBRATION DATA")
    print("=" * 60)
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Second split: separate calibration set from training
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"Data split completed:")
    print(f"  - Training samples:    {len(X_train)}")
    print(f"  - Calibration samples: {len(X_cal)}")
    print(f"  - Test samples:        {len(X_test)}")
    
    return X_train, X_cal, X_test, y_train, y_cal, y_test, iris.target_names


def calibrate_model(model, X_cal, y_cal):
    """
    Calibrate model using CalibratedClassifierCV.
    Uses sigmoid (Platt scaling) method.
    
    Args:
        model: Original trained model
        X_cal: Calibration features (scaled)
        y_cal: Calibration labels
    
    Returns:
        calibrated_model: Calibrated model
    """
    print("\n" + "=" * 60)
    print("CALIBRATING MODEL PROBABILITIES")
    print("=" * 60)
    
    print("Calibration method: Platt Scaling (Sigmoid)")
    print("Fitting calibration layer...")
    
    # Create calibrated classifier
    # cv='prefit' means the base model is already fitted
    calibrated_model = CalibratedClassifierCV(
        estimator=model,
        method='sigmoid',  # Platt scaling
        cv='prefit'
    )
    
    # Fit calibration on calibration set
    calibrated_model.fit(X_cal, y_cal)
    
    print("Model calibration completed!")
    
    return calibrated_model


def evaluate_calibration(original_model, calibrated_model, X_test, y_test, target_names):
    """
    Compare calibration between original and calibrated models.
    Uses Brier score (lower is better) to measure calibration quality.
    
    Args:
        original_model: Original model
        calibrated_model: Calibrated model
        X_test: Test features (scaled)
        y_test: Test labels
        target_names: Class names
    
    Returns:
        calibration_results: Dictionary of Brier scores
    """
    print("\n" + "=" * 60)
    print("EVALUATING CALIBRATION IMPROVEMENT")
    print("=" * 60)
    
    # Get predicted probabilities
    original_proba = original_model.predict_proba(X_test)
    calibrated_proba = calibrated_model.predict_proba(X_test)
    
    print("\nBrier Score Comparison (lower is better):")
    print("-" * 50)
    print(f"{'Class':<15} {'Original':<15} {'Calibrated':<15} {'Improvement':<15}")
    print("-" * 50)
    
    calibration_results = {}
    
    for class_idx, class_name in enumerate(target_names):
        # Create binary labels for this class
        y_binary = (y_test == class_idx).astype(int)
        
        # Calculate Brier scores
        original_brier = brier_score_loss(y_binary, original_proba[:, class_idx])
        calibrated_brier = brier_score_loss(y_binary, calibrated_proba[:, class_idx])
        
        # Calculate improvement
        if original_brier > 0:
            improvement = ((original_brier - calibrated_brier) / original_brier) * 100
        else:
            improvement = 0
        
        calibration_results[class_name] = {
            "original_brier": original_brier,
            "calibrated_brier": calibrated_brier,
            "improvement_percent": improvement
        }
        
        print(f"{class_name:<15} {original_brier:<15.4f} {calibrated_brier:<15.4f} {improvement:>+.2f}%")
    
    print("-" * 50)
    
    # Calculate average improvement
    avg_improvement = np.mean([r["improvement_percent"] for r in calibration_results.values()])
    print(f"\nAverage Calibration Improvement: {avg_improvement:+.2f}%")
    
    return calibration_results


def save_calibrated_model(calibrated_model, model_dir="models"):
    """
    Save the calibrated model with timestamp.
    
    Args:
        calibrated_model: Calibrated model
        model_dir: Directory to save models
    
    Returns:
        model_path: Path to saved calibrated model
        timestamp: Version timestamp
    """
    print("\n" + "=" * 60)
    print("SAVING CALIBRATED MODEL")
    print("=" * 60)
    
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save versioned calibrated model
    model_filename = f"iris_rf_calibrated_{timestamp}.joblib"
    model_path = os.path.join(model_dir, model_filename)
    joblib.dump(calibrated_model, model_path)
    print(f"Versioned calibrated model: {model_path}")
    
    # Save as latest
    latest_path = os.path.join(model_dir, "iris_rf_calibrated_latest.joblib")
    joblib.dump(calibrated_model, latest_path)
    print(f"Latest calibrated model:    {latest_path}")
    
    return model_path, timestamp


def main():
    """
    Main function to orchestrate model calibration.
    """
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + "   IRIS CLASSIFICATION - MODEL CALIBRATION   ".center(58) + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    print("\n")
    
    # Step 1: Load original model
    original_model, scaler = load_model_and_scaler()
    
    # Step 2: Prepare calibration data
    X_train, X_cal, X_test, y_train, y_cal, y_test, target_names = prepare_calibration_data()
    
    # Scale the data
    X_cal_scaled = scaler.transform(X_cal)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 3: Calibrate model
    calibrated_model = calibrate_model(original_model, X_cal_scaled, y_cal)
    
    # Step 4: Evaluate calibration
    calibration_results = evaluate_calibration(
        original_model, calibrated_model, X_test_scaled, y_test, target_names
    )
    
    # Step 5: Save calibrated model
    model_path, timestamp = save_calibrated_model(calibrated_model)
    
    # Summary
    print("\n")
    print("*" * 60)
    print("*" + " CALIBRATION COMPLETED SUCCESSFULLY ".center(58) + "*")
    print("*" * 60)
    print(f"\n  Calibration Version: {timestamp}")
    print(f"  Model Path: {model_path}")
    print("\n")
    
    return calibrated_model, calibration_results


if __name__ == "__main__":
    main()
