import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report

# Load the dataset
dataset = pd.read_csv("Dataset.csv")

# Check for missing values
dataset.info()
dataset.isnull().sum()

# Drop missing values
dataset.dropna(inplace=True)

# Create a count plot for the 'class' column
sns.set(style="darkgrid")
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='class', data=dataset, palette="Set3")
plt.title("Count Plot")
plt.xlabel("Categories")
plt.ylabel("Count")

# Annotate each bar with its count value
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')
plt.show()

# Label Encoding for the 'class' column
le = LabelEncoder()
dataset['class'] = le.fit_transform(dataset['class'])

# Splitting features and labels
X = dataset.iloc[:, 0:170]
y = dataset.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Labels for classification
labels = ['POSITIVE', 'NEGATIVE']

# Global variables to store metrics
precision = []
recall = []
fscore = []
accuracy = []

# Function to calculate metrics
def calculateMetrics(algorithm, predict, testY):
    testY = testY.astype('int')
    predict = predict.astype('int')
    p = precision_score(testY, predict, average='macro') * 100
    r = recall_score(testY, predict, average='macro') * 100
    f = f1_score(testY, predict, average='macro') * 100
    a = accuracy_score(testY, predict) * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    print(f"{algorithm} Accuracy    : {a}")
    print(f"{algorithm} Precision   : {p}")
    print(f"{algorithm} Recall      : {r}")
    print(f"{algorithm} F-Score     : {f}")
    
    report = classification_report(testY, predict, target_names=labels)
    print(f"\n{algorithm} classification report\n{report}")
    
    conf_matrix = confusion_matrix(testY, predict)
    plt.figure(figsize=(5, 5))
    ax = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="Blues", fmt="g")
    ax.set_ylim([0, len(labels)])
    plt.title(f"{algorithm} Confusion Matrix")
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.show()

# Logistic Regression model
if os.path.exists('Logistic Regression.pkl'):
    clf = joblib.load('Logistic Regression.pkl')
    print("Logistic Regression model loaded successfully.")
else:
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    joblib.dump(clf, 'Logistic Regression.pkl')
    print("Logistic Regression model saved successfully.")

# Prediction for Logistic Regression
predict = clf.predict(X_test)
calculateMetrics("Logistic Regression", predict, y_test)

# XGBoost Classifier model
if os.path.exists('XGBClassifier.pkl'):
    clf = joblib.load('XGBClassifier.pkl')
    print("XGBoost model loaded successfully.")
else:
    clf = XGBClassifier(max_depth=100, random_state=0)
    clf.fit(X_train, y_train)
    joblib.dump(clf, 'XGBClassifier.pkl')
    print("XGBoost model saved successfully.")

# Prediction for XGBoost Classifier
predict = clf.predict(X_test)
calculateMetrics("XGBoost Classifier", predict, y_test)

# Showing all algorithms' performance values
columns = ["Algorithm Name", "Accuracy", "Precision", "Recall", "F-Score"]
values = []
algorithm_names = ["Logistic Regression", "XGBoostClassifier"]
for i in range(len(algorithm_names)):
    values.append([algorithm_names[i], accuracy[i], precision[i], recall[i], fscore[i]])

performance_df = pd.DataFrame(values, columns=columns)
print(performance_df)

# Load test data
test = pd.read_csv("test.csv")

# Make predictions on the test data
predict = clf.predict(test)

# Loop through each prediction and print the corresponding row
for i, p in enumerate(predict):
    if p == 0:
        print(test.iloc[i])
        print(f"Row {i}: ************************************************** POSITIVE")
    else:
        print(test.iloc[i])
        print(f"Row {i}: ************************************************** NEGATIVE")

# Adding predicted values to the test data
test['Predicted'] = predict
print(test)
