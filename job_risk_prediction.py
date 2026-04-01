import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Data Loading ---
file_path = '/content/AI_Impact_on_Jobs_2030.csv'
df = pd.read_csv(file_path)

print("Original DataFrame head:")
print(df.head())

# --- 2. Feature Engineering: Experience_Level_Group ---
def categorize_experience(years):
    if 1 <= years <= 10:
        return 'Junior'
    elif 11 <= years <= 20:
        return 'Mid-level'
    elif 20 < years <= 30:
        return 'Senior'
    else:
        return 'out of range'

df['Experience_Level_Group'] = df['Years_Experience'].apply(categorize_experience)

# Reorder columns to place Experience_Level_Group after Years_Experience
columns = df.columns.tolist()
columns.remove('Experience_Level_Group')
columns.insert(3, 'Experience_Level_Group') # Insert at index 3, after Years_Experience
df = df[columns]

print("
DataFrame head after Experience_Level_Group creation:")
print(df.head())

# --- 3. Data Type Conversion: Numerical to float ---
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    df[col] = df[col].astype(float)

print("
DataFrame info after converting numerical columns to float:")
df.info()

# --- 4. One-Hot Encoding for Categorical Features ---
# Identify categorical columns for one-hot encoding
categorical_features = ['Job_Title', 'Education_Level', 'Experience_Level_Group', 'Risk_Category']

# Apply one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=False)

print("
DataFrame head after One-Hot Encoding:")
print(df_encoded.head())

# --- 5. Addressing Data Leakage & Deterministic Link ---
# Define the target columns for Risk_Category and the deterministically linked column
target_risk_columns = ['Risk_Category_Low', 'Risk_Category_Medium', 'Risk_Category_High']
deterministic_feature = 'Automation_Probability_2030'

# Filter out target columns that exist after one-hot encoding
actual_target_columns_in_df = [col for col in target_risk_columns if col in df_encoded.columns]

# Define X by dropping actual target columns and the deterministic feature
X = df_encoded.drop(columns=actual_target_columns_in_df + [deterministic_feature], errors='ignore')

# Redefine y for Risk_Category_Low and Risk_Category_Medium
y_low = df_encoded['Risk_Category_Low']
y_medium = df_encoded['Risk_Category_Medium']

print(f"
Shape of X (features) after removing target columns and '{deterministic_feature}': {X.shape}")
print(f"Shape of y_low (target for Low Risk): {y_low.shape}")
print(f"Shape of y_medium (target for Medium Risk): {y_medium.shape}")

# --- 6. Train-Test Split ---
# Split data into training and testing sets, stratifying by target variables
X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(X, y_low, test_size=0.2, random_state=42, stratify=y_low)
X_train_medium, X_test_medium, y_train_medium, y_test_medium = train_test_split(X, y_medium, test_size=0.2, random_state=42, stratify=y_medium)

print("
Shape of training and testing sets:")
print(f"X_train_low: {X_train_low.shape}, X_test_low: {X_test_low.shape}")
print(f"X_train_medium: {X_train_medium.shape}, X_test_medium: {X_test_medium.shape}")

# --- 7. Scaling (Standardization) ---
# Identify continuous numerical columns to scale
continuous_numerical_features = [
    'Average_Salary', 'AI_Exposure_Index', 'Tech_Growth_Factor',
    'Skill_1', 'Skill_2', 'Skill_3', 'Skill_4', 'Skill_5', 'Skill_6',
    'Skill_7', 'Skill_8', 'Skill_9', 'Skill_10', 'Years_Experience'
]

# Initialize scaler
scaler_standard = StandardScaler()

# For Risk_Category_Low model
X_train_low_scaled = X_train_low.copy()
X_test_low_scaled = X_test_low.copy()
X_train_low_scaled[continuous_numerical_features] = scaler_standard.fit_transform(X_train_low_scaled[continuous_numerical_features])
X_test_low_scaled[continuous_numerical_features] = scaler_standard.transform(X_test_low_scaled[continuous_numerical_features])

# For Risk_Category_Medium model
X_train_medium_scaled = X_train_medium.copy()
X_test_medium_scaled = X_test_medium.copy()
X_train_medium_scaled[continuous_numerical_features] = scaler_standard.fit_transform(X_train_medium_scaled[continuous_numerical_features])
X_test_medium_scaled[continuous_numerical_features] = scaler_standard.transform(X_test_medium_scaled[continuous_numerical_features])

print("
First 5 rows of scaled X_train_low (numerical features):")
print(X_train_low_scaled[continuous_numerical_features].head())

# --- 8. XGBoost Model Training ---
model_low_xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
model_medium_xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

print("
Training XGBoost models...")
model_low_xgb.fit(X_train_low_scaled, y_train_low)
print("XGBClassifier for Risk_Category_Low trained.")
model_medium_xgb.fit(X_train_medium_scaled, y_train_medium)
print("XGBClassifier for Risk_Category_Medium trained.")

# --- 9. XGBoost Model Evaluation ---
y_pred_low_xgb = model_low_xgb.predict(X_test_low_scaled)
y_pred_medium_xgb = model_medium_xgb.predict(X_test_medium_scaled)

print("
--- Evaluation for Risk_Category_Low (XGBoost) ---")
print(f"Accuracy: {accuracy_score(y_test_low, y_pred_low_xgb):.4f}")
print("Classification Report:")
print(classification_report(y_test_low, y_pred_low_xgb))

print("
--- Evaluation for Risk_Category_Medium (XGBoost) ---")
print(f"Accuracy: {accuracy_score(y_test_medium, y_pred_medium_xgb):.4f}")
print("Classification Report:")
print(classification_report(y_test_medium, y_pred_medium_xgb))
