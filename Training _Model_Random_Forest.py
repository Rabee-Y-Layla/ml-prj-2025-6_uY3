import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load data
#df = pd.read_csv('nested_features_dataset.csv')
df = pd.read_csv('comprehensive_fault_features_light.csv') 


# Step 1: Data Exploration
print("=== Data Exploration ===")
print(f"Data dimensions: {df.shape}")
print(f"Number of samples: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")
print("\nFault types:")
print(df['fault_type'].value_counts())
print("\nMain categories:")
print(df['main_category'].value_counts())

# Step 2: Data Cleaning
# Remove unwanted columns (file name and path)
df_clean = df.drop(['file_name', 'file_path'], axis=1)

# Check for missing values
print(f"\nMissing values: {df_clean.isnull().sum().sum()}")

# Step 3: Data Preprocessing
# Encode categorical variables
le_fault = LabelEncoder()
le_main = LabelEncoder()
df_clean['fault_type_encoded'] = le_fault.fit_transform(df_clean['fault_type'])
df_clean['main_category_encoded'] = le_main.fit_transform(df_clean['main_category'])

# Separate features and target variables
X = df_clean.drop(['fault_type', 'main_category', 'fault_type_encoded', 'main_category_encoded'], axis=1)
y_fault = df_clean['fault_type_encoded']  # For detailed fault type classification
y_main = df_clean['main_category_encoded']  # For general category classification

print(f"\nNumber of features after cleaning: {X.shape[1]}")
print(f"Number of fault classes: {len(le_fault.classes_)}")
print(f"Fault classes: {le_fault.classes_}")

# Step 4: Use ALL features (removed feature selection)
print(f"\nUsing ALL {X.shape[1]} features for training")

# Step 5: Data Splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y_fault, test_size=0.2, random_state=42, stratify=y_fault
)

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set shape: {X_train_scaled.shape}")
print(f"Test set shape: {X_test_scaled.shape}")

# Step 6: Model Building and Training
models = {
    ''' RandomForestClassifier(
    n_estimators=1000,
    max_depth=35,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,        # لمتابعة الأداء
    n_jobs=-1,
    random_state=42,
    verbose=1
),'''
    'Random Forest':RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

results = {}

print("\n=== Training Results ===")
for name, model in models.items():
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"{name}: {accuracy:.4f}")
    
    # Detailed classification report
    print(f"\nClassification Report - {name}:")
    print(classification_report(y_test, y_pred, target_names=le_fault.classes_))

# Step 7: Model Optimization
print("\n=== Model Optimization ===")
best_model_name = max(results, key=results.get)
print(f"Best model: {best_model_name}")

if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    best_model = RandomForestClassifier(random_state=42)
    
elif best_model_name == 'SVM':
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf', 'poly']
    }
    best_model = SVC(random_state=42)
    
else:  # Logistic Regression
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs', 'sag'],
        'penalty': ['l2', 'none']
    }
    best_model = LogisticRegression(random_state=42, max_iter=2000)

# Hyperparameter tuning
grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# Final model
final_model = grid_search.best_estimator_
final_model.fit(X_train_scaled, y_train)

# Evaluate final model
y_pred_final = final_model.predict(X_test_scaled)
final_accuracy = accuracy_score(y_test, y_pred_final)
print(f"\nFinal model test accuracy: {final_accuracy:.4f}")

# Step 8: Visualization
plt.figure(figsize=(15, 5))

# Plot 1: Confusion Matrix
plt.subplot(1, 3, 1)
cm = confusion_matrix(y_test, y_pred_final)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le_fault.classes_, 
            yticklabels=le_fault.classes_)
plt.title('Confusion Matrix - Final Model\n(All Features)')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Plot 2: Model Accuracy Comparison
plt.subplot(1, 3, 2)
models_comparison = list(results.keys())
accuracy_scores = list(results.values())
plt.bar(models_comparison, accuracy_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
plt.title('Model Accuracy Comparison\n(All Features)')
plt.ylim(0, 1)
plt.ylabel('Accuracy')

# Plot 3: Feature Importance (for tree-based models only)
plt.subplot(1, 3, 3)
if hasattr(final_model, 'feature_importances_'):
    feature_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    plt.barh(feature_imp['feature'], feature_imp['importance'])
    plt.title('Top 15 Most Important Features')
    plt.xlabel('Importance')
else:
    # If model doesn't have feature importance, show correlation instead
    plt.text(0.5, 0.5, 'Feature Importance\nNot Available', 
             horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes, fontsize=12)
    plt.title('Feature Importance Not Available')

plt.tight_layout()
plt.show()

# Step 9: Cross-Validation
print("\n=== Cross-Validation ===")
cv_scores = cross_val_score(final_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation results (5-fold): {cv_scores}")
print(f"Average cross-validation: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Step 10: Save Model for Future Use
import joblib

model_data = {
    'model': final_model,
    'scaler': scaler,
    'label_encoder_fault': le_fault,
    'label_encoder_main': le_main,
    'feature_names': X.columns.tolist(),
    'used_all_features': True
}

joblib.dump(model_data, 'fault_detection_model_all_features.pkl')
print("\nModel saved to 'fault_detection_model_all_features.pkl'")

# Step 11: Prediction Example
print("\n=== Prediction Example ===")
sample_idx = 0
sample_data = X_test_scaled[sample_idx].reshape(1, -1)
prediction = final_model.predict(sample_data)

# Use predict_proba only if model supports it
if hasattr(final_model, 'predict_proba'):
    prediction_proba = final_model.predict_proba(sample_data)
    print(f"Prediction probabilities: {dict(zip(le_fault.classes_, prediction_proba[0]))}")

print(f"Actual sample: {le_fault.inverse_transform([y_test.iloc[sample_idx]])[0]}")
print(f"Prediction: {le_fault.inverse_transform(prediction)[0]}")

# Additional: Compare with feature selection approach
print("\n" + "="*50)
print("COMPARISON: Using All Features vs Feature Selection")
print("="*50)
print(f"Number of features used: {X.shape[1]}")
print(f"Final model accuracy: {final_accuracy:.4f}")
print(f"Best model: {best_model_name}")
print(f"Best parameters: {grid_search.best_params_}")