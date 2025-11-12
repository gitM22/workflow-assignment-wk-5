import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --- Part 2: Case Study Application ---

print("--- AI Hospital Readmission Demo ---")

# --- 1. Data Strategy (Hypothetical Data) ---
# In a real project, this data would come from EHRs, demographics, etc.
data = {
    'age': [55, 72, 45, 68, 81, 59, 76, 60, 49, 85],
    'gender': ['Male', 'Female', 'Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female'],
    'previous_admissions': [1, 3, 0, 2, 4, 1, 3, 0, 1, 5],
    'lab_results_abnormal': [1, 0, 1, 1, 0, 0, 1, 1, 0, 1], # 1 for abnormal, 0 for normal
    'primary_diagnosis': ['Heart', 'Diabetes', 'Trauma', 'Heart', 'Kidney', 'Diabetes', 'Heart', 'Trauma', 'Kidney', 'Diabetes'],
    'length_of_stay_days': [5, 10, 3, 7, 14, 6, 9, 2, 4, 18],
    'readmitted_within_30_days': [1, 1, 0, 0, 1, 0, 1, 0, 0, 1] # Target variable
}
df = pd.DataFrame(data)
print(f"Loaded hypothetical data with {df.shape[0]} patients.\n")

# --- 2. Data Strategy (Preprocessing Pipeline) ---

# Define features (X) and target (y)
X = df.drop('readmitted_within_30_days', axis=1)
y = df['readmitted_within_30_days']

# Identify numerical and categorical features
numerical_features = ['age', 'previous_admissions', 'length_of_stay_days']
categorical_features = ['gender', 'primary_diagnosis', 'lab_results_abnormal']

# Create preprocessing pipelines for both types
# Note: This handles tasks from Part 1 (missing data, normalization)
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Handle missing data
    ('scaler', StandardScaler())                   # Normalization/Scaling
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Handle missing data
    ('onehot', OneHotEncoder(handle_unknown='ignore'))     # Convert categories to numbers
])

# Combine pipelines using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- 3. Model Development ---

# Split data (as required in Part 1)
# Using a 70/30 split for this small demo dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Justification (for Part 2): We choose Logistic Regression.
# Why? It is highly interpretable (you can see *which* features
# contribute to risk), fast to train, and a strong baseline
# for binary classification problems in healthcare.
model = LogisticRegression(random_state=42)

# Create the full pipeline: Preprocess data, then train the model
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', model)])

# Train the model
clf.fit(X_train, y_train)
print("Model training complete.\n")

# --- 4. Evaluation (Addresses Part 1 & 2) ---

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Hypothetical Confusion Matrix (for Part 2)
#          Predicted 0   Predicted 1
# Actual 0    [TN]          [FP]
# Actual 1    [FN]          [TP]
cm = confusion_matrix(y_test, y_pred)
print("--- Model Evaluation ---")
print("Confusion Matrix:")
print(cm)

# Calculate Precision & Recall (for Part 2)
# Precision: Of all patients we *predicted* would be readmitted, how many actually were?
# (TP / (TP + FP))
precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.4f}")

# Recall: Of all patients who *actually* were readmitted, how many did we catch?
# (TP / (TP + FN))
recall = recall_score(y_test, y_pred)
print(f"Recall: {recall:.4f}")

print("\nFull Classification Report:")
print(classification_report(y_test, y_pred))

# --- 5. Optimization (Address Overfitting - Part 2) ---
# LogisticRegression has a 'C' hyperparameter for regularization.
# A smaller 'C' value adds a stronger L2 penalty, which helps
# prevent overfitting by shrinking feature weights.
# Example: model = LogisticRegression(C=0.1, penalty='l2')

print("\n--- Demo Finished ---")