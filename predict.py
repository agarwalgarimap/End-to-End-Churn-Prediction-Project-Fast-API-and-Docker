import joblib
import pandas as pd
import numpy as np

# ================== LOAD MODEL ==================
model = joblib.load("models/best_xgb_model.pkl")
threshold = joblib.load("models/best_threshold.pkl")
feature_names = joblib.load("models/feature_names.pkl")

print("✅ Model loaded!")

# ================== PREPROCESSING FUNCTION ==================
def preprocess_customer_data(data_dict):
    """
    Apply the SAME preprocessing steps used during training
    """
    df = pd.DataFrame([data_dict])

    # Step 0: Drop customerID
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Step 1: Handle Missing Values
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        if df['TotalCharges'].isnull().any():
            df['TotalCharges'] = df['MonthlyCharges'] * df['tenure']

    # Step 2: Binary Encoding
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Step 3: One-Hot Encoding
    all_categories = {
        'MultipleLines': ['No', 'No phone service', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['No', 'No internet service', 'Yes'],
        'OnlineBackup': ['No', 'No internet service', 'Yes'],
        'DeviceProtection': ['No', 'No internet service', 'Yes'],
        'TechSupport': ['No', 'No internet service', 'Yes'],
        'StreamingTV': ['No', 'No internet service', 'Yes'],
        'StreamingMovies': ['No', 'No internet service', 'Yes'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaymentMethod': ['Bank transfer (automatic)', 'Credit card (automatic)',
                         'Electronic check', 'Mailed check']
    }

    for col, categories in all_categories.items():
        if col in df.columns:
            current_value = df[col].iloc[0]
            for category in categories[1:]:
                col_name = f"{col}_{category}"
                df[col_name] = 1 if current_value == category else 0
            df.drop(col, axis=1, inplace=True)

    # Step 4: Feature Engineering
    df['ChargeRatio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    
    df['TenureGroup'] = pd.cut(
        df['tenure'],
        bins=[-1, 12, 24, 60, 1000],
        labels=[0, 1, 2, 3]
    ).astype(int)
    
    df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['CLV'] = df['MonthlyCharges'] * df['tenure']

    service_cols = []
    if 'PhoneService' in df.columns:
        service_cols.append('PhoneService')
    if 'StreamingTV_Yes' in df.columns:
        service_cols.append('StreamingTV_Yes')
    if 'StreamingMovies_Yes' in df.columns:
        service_cols.append('StreamingMovies_Yes')
    df['TotalServices'] = df[service_cols].sum(axis=1) if service_cols else 0

    df['IsNewCustomer'] = (df['tenure'] <= 6).astype(int)
    df['ChargePerService'] = df['MonthlyCharges'] / (df['TotalServices'] + 1)

    tech_col = 'TechSupport_Yes' if 'TechSupport_Yes' in df.columns else None
    sec_col = 'OnlineSecurity_Yes' if 'OnlineSecurity_Yes' in df.columns else None
    if tech_col and sec_col:
        df['SupportSecurity'] = df[tech_col] + df[sec_col]
    elif tech_col:
        df['SupportSecurity'] = df[tech_col]
    elif sec_col:
        df['SupportSecurity'] = df[sec_col]
    else:
        df['SupportSecurity'] = 0

    return df


# ================== PREDICTION FUNCTION ==================
def predict_churn(data: dict):
    """
    Preprocess and predict churn
    """
    # Save customerID before preprocessing
    customer_id = data.get('customerID', None)
    
    # Preprocess
    df = preprocess_customer_data(data)
    
    # Reorder columns to match training
    df = df[feature_names]
    
    # Predict
    churn_prob = model.predict_proba(df)[0, 1]
    churn_pred = int(churn_prob > threshold)
    
    # Risk level
    if churn_prob < 0.3:
        risk_level = "Low"
    elif churn_prob < 0.6:
        risk_level = "Medium"
    else:
        risk_level = "High"
    
    # Message
    if churn_pred == 1:
        message = f"⚠️ Customer likely to CHURN! Risk: {risk_level}"
    else:
        message = f"✅ Customer likely to STAY. Risk: {risk_level}"
    
    return {
        "customer_id": customer_id,
        "churn_prediction": churn_pred,
        "churn_probability": round(float(churn_prob), 4),
        "risk_level": risk_level,
        "message": message
    }