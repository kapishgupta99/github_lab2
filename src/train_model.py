"""
Train Model Script - Iris Dataset Classification
=================================================

MODIFICATION FROM ORIGINAL LAB:
- Original: Used synthetic/randomly generated data
- Modified: Uses the Iris dataset from scikit-learn

This script trains a Random Forest classifier on the Iris dataset
and saves the trained model with timestamp-based versioning.

Author: Kapish Gupta
Course: IE 7374 - MLOps, Northeastern University
"""

import os
import joblib
import numpy as np
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def load_iris_data():
    """
    Load the Iris dataset from scikit-learn.
    
    MODIFICATION: Using real Iris dataset instead of synthetic data.
    
    The Iris dataset contains:
    - 150 samples of iris flowers
    - 4 features: Sepal length, Sepal width, Petal length, Petal width
    - 3 classes: Setosa (0), Versicolor (1), Virginica (2)
    
    Returns:
        X: Feature matrix
        y: Target labels
        feature_names: Names of features
        target_names: Names of target classes
    """
    print("=" * 60)
    print("LOADING IRIS DATASET")
    print("=" * 60)
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    print(f"Dataset loaded successfully!")
    print(f"  - Total samples: {X.shape[0]}")
    print(f"  - Number of features: {X.shape[1]}")
    print(f"  - Feature names: {list(iris.feature_names)}")
    print(f"  - Number of classes: {len(np.unique(y))}")
    print(f"  - Class names: {list(iris.target_names)}")
    print(f"  - Class distribution: {np.bincount(y)}")
    
    return X, y, iris.feature_names, iris.target_names


def preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    Preprocess the data: split and scale.
    
    Args:
        X: Feature matrix
        y: Target labels
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
    
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING DATA")
    print("=" * 60)
    
    # Split data with stratification to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print(f"Train/Test split completed:")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Test samples: {len(X_test)}")
    
    # Scale features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Feature scaling applied (StandardScaler)")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Scaled training features
        y_train: Training labels
    
    Returns:
        Trained model
    """
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print("=" * 60)
    
    # Initialize Random Forest with specified hyperparameters
    model = RandomForestClassifier(
        n_estimators=100,      # Number of trees
        max_depth=5,           # Maximum depth of trees
        random_state=42,       # For reproducibility
        n_jobs=-1              # Use all CPU cores
    )
    
    print(f"Model configuration:")
    print(f"  - Algorithm: Random Forest Classifier")
    print(f"  - n_estimators: 100")
    print(f"  - max_depth: 5")
    print(f"  - random_state: 42")
    
    # Train the model
    print(f"\nTraining model...")
    model.fit(X_train, y_train)
    print(f"Training completed!")
    
    return model


def evaluate_quick(model, X_train, X_test, y_train, y_test):
    """
    Quick evaluation of model performance.
    """
    print("\n" + "=" * 60)
    print("QUICK EVALUATION")
    print("=" * 60)
    
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Test Accuracy:     {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    return train_accuracy, test_accuracy


def save_model(model, scaler, model_dir="models"):
    """
    Save the trained model and scaler with timestamp versioning.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        model_dir: Directory to save models
    
    Returns:
        model_path: Path to saved model
        timestamp: Version timestamp
    """
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save versioned model
    model_filename = f"iris_rf_model_{timestamp}.joblib"
    model_path = os.path.join(model_dir, model_filename)
    joblib.dump(model, model_path)
    print(f"Versioned model saved: {model_path}")
    
    # Save as 'latest' for easy access
    latest_model_path = os.path.join(model_dir, "iris_rf_model_latest.joblib")
    joblib.dump(model, latest_model_path)
    print(f"Latest model saved:    {latest_model_path}")
    
    # Save the scaler
    scaler_path = os.path.join(model_dir, "scaler_latest.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved:          {scaler_path}")
    
    # Save timestamp to file for workflow reference
    with open("timestamp.txt", "w") as f:
        f.write(timestamp)
    print(f"Timestamp saved:       timestamp.txt")
    
    return model_path, timestamp


def main():
    """
    Main function to orchestrate the training pipeline.
    """
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + "   IRIS CLASSIFICATION - MODEL TRAINING PIPELINE   ".center(58) + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    print("\n")
    
    # Step 1: Load data
    X, y, feature_names, target_names = load_iris_data()
    
    # Step 2: Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    # Step 3: Train model
    model = train_random_forest(X_train, y_train)
    
    # Step 4: Quick evaluation
    train_acc, test_acc = evaluate_quick(model, X_train, X_test, y_train, y_test)
    
    # Step 5: Save model
    model_path, timestamp = save_model(model, scaler)
    
    # Summary
    print("\n")
    print("*" * 60)
    print("*" + " TRAINING COMPLETED SUCCESSFULLY ".center(58) + "*")
    print("*" * 60)
    print(f"\n  Model Version: {timestamp}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Model Path:    {model_path}")
    print("\n")
    
    return model, scaler, timestamp


if __name__ == "__main__":
    main()
