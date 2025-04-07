import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

# Load dataset
df = pd.read_csv("E:/PCCOE/Semesters/6th/ML/Mini Project/Dataset/student_data.csv")

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

# Define features and labels
X = df[['study_hours', 'attendance_(%)', 'assignment_score', 'last_sem_percentage', 'mobile_screen_time', 'sleep_hours']]
y_regression = df['final_score_(%)']
y_classification = df['pass/fail']

# ---------------------- REGRESSION ----------------------

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)

# Train model
regression_model = LinearRegression()
regression_model.fit(X_train_reg, y_train_reg)

# Predict and evaluate
y_pred_reg = regression_model.predict(X_test_reg)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print("Regression Performance:")
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

# Plot predicted vs actual
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test_reg, y=y_pred_reg, color='blue')
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--')
plt.xlabel('Actual Final Score (%)')
plt.ylabel('Predicted Final Score (%)')
plt.title('Regression: Actual vs Predicted Final Scores')
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------- CLASSIFICATION ----------------------

# Split data
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_classification, test_size=0.2, random_state=42)

# Train model
classification_model = RandomForestClassifier(n_estimators=100, random_state=42)
classification_model.fit(X_train_clf, y_train_clf)

# Predict and evaluate
y_pred_clf = classification_model.predict(X_test_clf)
acc = accuracy_score(y_test_clf, y_pred_clf)
report = classification_report(y_test_clf, y_pred_clf)
print("\nClassification Performance:")
print("Accuracy:", acc)
print("Classification Report:\n", report)

# Plot confusion matrix (handles single-class error)
labels = np.unique(y_classification)  # ensures all labels appear
cm = confusion_matrix(y_test_clf, y_pred_clf, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

plt.figure(figsize=(6, 5))
disp.plot(cmap='Blues', values_format='d')
plt.title("Classification: Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.show()

# ---------------------- SAVE MODELS ----------------------

pickle.dump(regression_model, open("E:/PCCOE/Semesters/6th/ML/Mini Project/Dataset/regression_model.pkl", "wb"))
pickle.dump(classification_model, open("E:/PCCOE/Semesters/6th/ML/Mini Project/Dataset/classification_model.pkl", "wb"))

print("Models trained, evaluated, and saved successfully!")
