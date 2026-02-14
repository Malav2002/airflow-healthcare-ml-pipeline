import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import json

def load_data():
    """
    Load cardiovascular disease dataset.
    Returns:
        pd.DataFrame: Loaded dataset
    """
    # Path relative to Airflow DAGs folder
    data_path = "/opt/airflow/data/cardiovascular_data.csv"
    
    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path, delimiter=';')
    
    print(f"Dataset loaded successfully. Shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    
    return data

def data_preprocessing(data):
    """
    Preprocess the cardiovascular disease data.
    Args:
        data: Raw dataset
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print("Starting data preprocessing...")
    
    # Create a copy to avoid modifying original
    df = data.copy()
    
    # Convert age from days to years
    df['age'] = (df['age'] / 365).round().astype(int)
    
    # Calculate BMI (weight in kg, height in cm)
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    
    # Create blood pressure categories
    df['bp_category'] = pd.cut(df['ap_hi'], 
                                bins=[0, 120, 140, 180, 300],
                                labels=['Normal', 'Elevated', 'High', 'Very High'])
    
    # Convert categorical to numeric
    df['bp_category'] = df['bp_category'].cat.codes
    
    # Remove outliers
    # Age should be reasonable (30-100 years)
    df = df[(df['age'] >= 30) & (df['age'] <= 100)]
    
    # Blood pressure should be positive and reasonable
    df = df[(df['ap_hi'] > 0) & (df['ap_hi'] < 250)]
    df = df[(df['ap_lo'] > 0) & (df['ap_lo'] < 200)]
    df = df[df['ap_hi'] > df['ap_lo']]  # Systolic > Diastolic
    
    # Height and weight should be reasonable
    df = df[(df['height'] > 120) & (df['height'] < 220)]
    df = df[(df['weight'] > 30) & (df['weight'] < 200)]
    
    print(f"After outlier removal. Shape: {df.shape}")
    
    # Select features
    feature_columns = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                      'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi', 'bp_category']
    
    X = df[feature_columns]
    y = df['cardio']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Target distribution - Train: {y_train.value_counts().to_dict()}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance using SMOTE
    print("Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"After SMOTE - Train set size: {X_train_balanced.shape[0]}")
    print(f"Target distribution after SMOTE: {pd.Series(y_train_balanced).value_counts().to_dict()}")
    
    # Save scaler for future use
    model_dir = "/opt/airflow/model"
    os.makedirs(model_dir, exist_ok=True)
    
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")
    
    return X_train_balanced, X_test_scaled, y_train_balanced, y_test.values

def build_model(data, filename):
    """
    Build and train Random Forest model.
    Args:
        data: Tuple of (X_train, X_test, y_train, y_test)
        filename: Name to save the model
    """
    X_train, X_test, y_train, y_test = data
    
    print("Training Random Forest model...")
    
    # Create Random Forest classifier with tuned parameters
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    # Train the model
    rf_model.fit(X_train, y_train)
    
    print("Model training completed!")
    
    # Save the model
    model_dir = "/opt/airflow/model"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, filename)
    with open(model_path, 'wb') as f:
        pickle.dump(rf_model, f)
    
    print(f"Model saved to: {model_path}")

def evaluate_model(data, filename):
    """
    Load and evaluate the trained model.
    Args:
        data: Tuple of (X_train, X_test, y_train, y_test)
        filename: Name of the saved model
    Returns:
        dict: Evaluation metrics
    """
    X_train, X_test, y_train, y_test = data
    
    # Load the model
    model_path = os.path.join("/opt/airflow/model", filename)
    print(f"Loading model from: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Feature importance
    feature_names = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                    'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi', 'bp_category']
    feature_importance = dict(zip(feature_names, model.feature_importances_))
    top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])
    
    # Prepare results
    results = {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'confusion_matrix': cm.tolist(),
        'top_features': {k: round(v, 4) for k, v in top_features.items()}
    }
    
    # Print results
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {results['accuracy']:.2%}")
    print(f"Precision: {results['precision']:.2%}")
    print(f"Recall:    {results['recall']:.2%}")
    print(f"F1-Score:  {results['f1_score']:.2%}")
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0][0]:5d} | FP: {cm[0][1]:5d}")
    print(f"FN: {cm[1][0]:5d} | TP: {cm[1][1]:5d}")
    print("\nTop 5 Important Features:")
    for feature, importance in top_features.items():
        print(f"  {feature:15s}: {importance:.4f}")
    print("="*50 + "\n")
    
    # Save results to JSON
    results_path = os.path.join("/opt/airflow/model", "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved to: {results_path}")
    
    return results

if __name__ == '__main__':
    # For testing locally
    data = load_data()
    processed_data = data_preprocessing(data)
    build_model(processed_data, 'cardiovascular_rf_model.pkl')
    evaluate_model(processed_data, 'cardiovascular_rf_model.pkl')