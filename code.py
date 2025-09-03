# =============================================================================
# ChargeAI Project Submission: Airline Passenger Satisfaction Prediction
# =============================================================================

# 1. Import Necessary Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Libraries imported successfully.")

# 2. Load and Combine the Datasets
try:
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    df = pd.concat([df_train, df_test], ignore_index=True)
    print("Datasets loaded and combined successfully.")
    print(f"Total dataset shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'train.csv' or 'test.csv' not found.")
    print("Please download the dataset from Kaggle and place it in the same directory.")
    exit()

# 3. Data Preprocessing: Cleaning and Transformation
print("\nStarting data preprocessing...")

# [cite_start]Drop unnecessary columns [cite: 49, 309]
df.drop(['Unnamed: 0', 'id'], axis=1, inplace=True)

# [cite_start]Handle missing values by imputing with the median [cite: 44]
df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].median(), inplace=True)

# Encode the binary target variable 'satisfaction'
df['satisfaction'] = df['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)

# [cite_start]Encode other binary categorical features [cite: 57, 93]
df['Customer Type'] = df['Customer Type'].apply(lambda x: 1 if x == 'Loyal Customer' else 0)
df['Type of Travel'] = df['Type of Travel'].apply(lambda x: 1 if x == 'Business travel' else 0)
df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

# [cite_start]One-Hot Encode the 'Class' column to handle nominal categories [cite: 56, 94]
df = pd.get_dummies(df, columns=['Class'], drop_first=True)

print("Data preprocessing complete.")
print("Columns after encoding:", df.columns.tolist())

# 4. Model Building and Training
print("\nPreparing data for the model...")

# Define features (X) and target (y)
X = df.drop('satisfaction', axis=1)
y = df['satisfaction']

# [cite_start]Split the data into training and testing sets (80/20 split) [cite: 116]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# [cite_start]Initialize and train the Random Forest Classifier [cite: 103]
print("\nTraining the Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Model training complete!")

# 5. Model Evaluation
print("\nEvaluating model performance...")

# Make predictions on the test set
y_pred = model.predict(X_test)

# [cite_start]Calculate and print model accuracy [cite: 111]
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# [cite_start]Generate and print the classification report (Precision, Recall, F1-Score) [cite: 112, 113, 114]
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Neutral or Dissatisfied', 'Satisfied']))

# Generate and plot the confusion matrix for better visualization
print("\nGenerating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Neutral or Dissatisfied', 'Satisfied'],
            yticklabels=['Neutral or Dissatisfied', 'Satisfied'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("\nProject script finished.")
