# Import necessary libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('physionet2017.csv')

# Check the first few rows and columns
print(data.head())

# Extract features and labels
X = data.iloc[:, :-2].values  # Features (all columns except the last two)
y = data.iloc[:, -1].values   # Labels (last column)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
# Evaluate the model
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))  # Add zero_division parameter
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model for future use
import joblib
joblib.dump(model, 'ecg_classifier_model.pkl')

# Plot a sample ECG signal
sample_ecg = X[0]  # Assuming X contains ECG signals
plt.plot(sample_ecg)
plt.title("ECG Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

# Plot histogram of labels
plt.hist(y, bins=4)  # Assuming y contains the labels
plt.title("Distribution of ECG Classes")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.xticks(np.arange(4), ['Normal', 'AF', 'Other', 'Noisy'])
plt.show()

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Plot feature importances
feature_importances = model.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]
plt.bar(range(len(sorted_indices)), feature_importances[sorted_indices])
plt.title('Feature Importances')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.show()
