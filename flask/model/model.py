import pandas as pd
df=pd.read_csv('./my_merge.csv')

df.head()

# Prepare training data for the machine learning model
# Define independent variables
ind_col = [col for col in df.columns if col!='Class_info']
ind_col=[col for col in ind_col if col!='subject_id']
#ind_col=[col for col in df.columns if col!='Subject_id']
# Define dependent variable
dep_col = 'Class_info'
print(ind_col)
print(dep_col)

X = df[ind_col]
y = df[dep_col]

import matplotlib.pyplot as plt
import seaborn as sns

# Import necessary libraries
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from collections import Counter

# Print the original distribution
print("Original dataset shape:", Counter(y))

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Print the resampled distribution
print("Resampled dataset shape:", Counter(y_resampled))

# Visualize the resampled data distribution
fig, ax = plt.subplots(1, 2)
fig.suptitle('Resampled Status', fontsize=16, color='navy')
plt.style.use('seaborn-v0_8-bright')

plt.subplot(1, 2, 1)
y_resampled.value_counts().plot(kind='bar', color=sns.color_palette("husl"))
plt.title('Balanced Class Distribution')

plt.subplot(1, 2, 2)
y_resampled.value_counts().plot(kind='pie', autopct="%.2f%%")
plt.title('Balanced Class Proportion')

plt.show()

smote_df = pd.concat([X_resampled, y_resampled], axis=1)

# Print the original columns to verify
print("Original columns:", smote_df.columns.tolist())

# Define the exact column names to drop
columns_to_drop = ['subject_id']

# Drop the columns if they exist in the DataFrame
smote_df.drop(columns=[col for col in columns_to_drop if col in smote_df.columns], inplace=True)

# Verify the remaining columns
print("Remaining columns:", smote_df.columns.tolist())

# Prepare training data for the machine learning model
# Define independent variables
ind_col = [col for col in smote_df.columns if col!='Class_info']
# Define dependent variable
dep_col = 'Class_info'

X = smote_df[ind_col]
y = smote_df[dep_col]

X.columns

y.shape

from sklearn.model_selection import train_test_split

# Divide the data set into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

from sklearn.preprocessing import StandardScaler

# Features normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import joblib

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

"""  XG BOOST"""

from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=3,
    gamma=5,
    subsample=0.7,
    colsample_bytree=0.8,
    lambda_=5,
    alpha=2,
    random_state=42
)

xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

print(f'Training Accuracy of XGBoost is {accuracy_score(y_train, xgb.predict(X_train))}\n')

xgb_accuracy = accuracy_score(y_test, y_pred)
xgb_precision = precision_score(y_test, y_pred, average='macro')
xgb_recall = recall_score(y_test, y_pred, average='macro')
xgb_f1_score = f1_score(y_test, y_pred, average='macro')

#print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}\n")
print(f"Test Accuracy of XGBoost is {xgb_accuracy} \n")
print(f"Test Precision of XGBoost is {xgb_precision} \n")
print(f"Test Recall of XGBoost is {xgb_recall} \n")
print(f"Test F1_score of XGBoost is {xgb_f1_score} \n")
print(f"Classification Report: \n{classification_report(y_test, y_pred)}")
cm = confusion_matrix(y_test, y_pred)
sns.set(rc={"figure.figsize": (4, 3)})
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')

"""KNN"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Create KNN classifier
knn = KNeighborsClassifier()
# Train KNN classifier
knn.fit(X_train, y_train)
# Use the trained model for prediction
y_pred = knn.predict(X_test)

print(f'Training Accuracy of KNN is {accuracy_score(y_train, knn.predict(X_train))}\n')

knn_accuracy = accuracy_score(y_test, y_pred)
knn_precision = precision_score(y_test, y_pred, average='macro')
knn_recall = recall_score(y_test, y_pred, average='macro')
knn_f1_score = f1_score(y_test, y_pred, average='macro')

print(f"Test Accuracy of KNN is {knn_accuracy} \n")
print(f"Test Precision of KNN is {knn_precision} \n")
print(f"Test Recall of KNN is {knn_recall} \n")
print(f"Test F1_score of KNN is {knn_f1_score} \n")
print(f"Classification Report: \n {classification_report(y_test, y_pred)}")

print("confusion matrix ")
cm = confusion_matrix(y_test, y_pred)
sns.set(rc={"figure.figsize": (4, 3)})
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')

"""Random forest classifier"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

rfc = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    max_features="sqrt",
    min_samples_split=10,
    min_samples_leaf=5,
    bootstrap=True,  # Enable bootstrapping
    max_samples=0.8,  # Use only 80% of training data
    random_state=42
)

# Fit the model
rfc.fit(X_train, y_train)

# Predicting on test data
y_pred = rfc.predict(X_test)

# Training accuracy
print(f'Training Accuracy of Random Forest is {accuracy_score(y_train, rfc.predict(X_train))}\n')

# Model evaluation metrics
rfc_accuracy = accuracy_score(y_test, y_pred)
rfc_precision = precision_score(y_test, y_pred, average='macro')
rfc_recall = recall_score(y_test, y_pred, average='macro')
rfc_f1_score = f1_score(y_test, y_pred, average='macro')

print(f"Test Accuracy of Random Forest is {rfc_accuracy} \n")
print(f"Test Precision of Random Forest is {rfc_precision} \n")
print(f"Test Recall of Random Forest is {rfc_recall} \n")
print(f"Test F1_score of Random Forest is {rfc_f1_score} \n")
print(f"Classification Report: \n{classification_report(y_test, y_pred)}")

# Confusion matrix
print("Confusion matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.set(rc={"figure.figsize": (4, 3)})
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.title("Confusion Matrix")
plt.show()

"""Ada Boost"""

from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier()
ada.fit(X_train, y_train)
y_pred = ada.predict(X_test)

print(f'Training Accuracy of AdaBoost is {accuracy_score(y_train, ada.predict(X_train))}\n')

ada_accuracy = accuracy_score(y_test, y_pred)
ada_precision = precision_score(y_test, y_pred, average='macro')
ada_recall = recall_score(y_test, y_pred, average='macro')
ada_f1_score = f1_score(y_test, y_pred, average='macro')

print(f"Test Accuracy of AdaBoost is {ada_accuracy} \n")
print(f"Test Precision of AdaBoost is {ada_precision} \n")
print(f"Test Recall of AdaBoost is {ada_recall} \n")
print(f"Test F1_score of AdaBoost is {ada_f1_score} \n")
print(f"Classification Report: \n{classification_report(y_test, y_pred)}")

print("confusion matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.set(rc={"figure.figsize": (4, 3)})
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')

"""Logistic Regression"""

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print(f'Training Accuracy of Logistic Regression is {accuracy_score(y_train, logreg.predict(X_train))}\n')

logreg_accuracy = accuracy_score(y_test, y_pred)
logreg_precision = precision_score(y_test, y_pred, average='macro')
logreg_recall = recall_score(y_test, y_pred, average='macro')
logreg_f1_score = f1_score(y_test, y_pred, average='macro')

print(f"Test Accuracy of Logistic Regression is {logreg_accuracy} \n")
print(f"Test Precision of Logistic Regression is {logreg_precision} \n")
print(f"Test Recall of Logistic Regression is {logreg_recall} \n")
print(f"Test F1_score of Logistic Regression is {logreg_f1_score} \n")
print(f"Classification Report: \n{classification_report(y_test, y_pred)}")

print("confusion matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.set(rc={"figure.figsize": (4, 3)})
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')

"""MLP classifier"""

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

mlp = MLPClassifier(random_state=0)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

print(f'Training Accuracy of MLP is {accuracy_score(y_train, mlp.predict(X_train))}\n')

mlp_accuracy = accuracy_score(y_test, y_pred)
mlp_precision = precision_score(y_test, y_pred, average='macro')
mlp_recall = recall_score(y_test, y_pred, average='macro')
mlp_f1_score = f1_score(y_test, y_pred, average='macro')

print(f"Test Accuracy of MLP is {mlp_accuracy} \n")
print(f"Test Precision of MLP is {mlp_precision} \n")
print(f"Test Recall of MLP is {mlp_recall} \n")
print(f"Test F1_score of MLP is {mlp_f1_score} \n")
print(f"Classification Report: \n{classification_report(y_test, y_pred)}")


print("confusion matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.set(rc={"figure.figsize": (5, 3)})
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')

"""MLP using deep neural networks"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define an MLP with Fully Connected Layers
model = Sequential([
    Dense(16, activation='relu', input_shape=(11,)),  # Input Layer (12 features) -> 16 neurons
    Dense(8, activation='relu'),  # Hidden Layer with 8 neurons
    Dense(1, activation='sigmoid')  # Output Layer (1 neuron, sigmoid for binary classification)
])
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of Model Architecture
model.summary()

print(X_train.shape)
print(X_test.shape)

history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

import matplotlib.pyplot as plt

# Plotting Training and Validation Accuracy
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o', linestyle='-')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='x', linestyle='--')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', marker='o', linestyle='-')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='x', linestyle='--')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test loss:{test_loss:.4f}")

# Predict Probabilities
y_pred_prob = model.predict(X_test)

# Convert Probabilities to Binary Labels (Threshold = 0.5)
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert True/False to 1/0

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("confusion matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.set(rc={"figure.figsize": (4, 3)})
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')

"""CAT boos"""


from catboost import CatBoostClassifier
cat_model = CatBoostClassifier(
    iterations=500,   # Reduce total trees (default is 1000+)
    depth=5,          # Reduce depth (default 6-10)
    l2_leaf_reg=8,    # Increase L2 regularization
    learning_rate=0.05,  # Reduce learning rate
    random_seed=42,
    verbose=100
)
cat_model.fit(X_train, y_train)
y_pred = cat_model.predict(X_test)

print(f'Training Accuracy of CatBoost is {accuracy_score(y_train, cat_model.predict(X_train))}\n')

cat_accuracy = accuracy_score(y_test, y_pred)
cat_precision = precision_score(y_test, y_pred, average='macro')
cat_recall = recall_score(y_test, y_pred, average='macro')
cat_f1_score = f1_score(y_test, y_pred, average='macro')

#print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}\n")
print(f"Test Accuracy of CatBoost is {cat_accuracy} \n")
print(f"Test Precision of CatBoost is {cat_precision} \n")
print(f"Test Recall of CatBoost is {cat_recall} \n")
print(f"Test F1_score of CatBoost is {cat_f1_score} \n")
print(f"Classification Report: \n{classification_report(y_test, y_pred)}")
cm = confusion_matrix(y_test, y_pred)
sns.set(rc={"figure.figsize": (4, 3)})
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')

"""SVM"""

from sklearn.svm import SVC

svm = SVC(probability=True)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

print(f'Training Accuracy of SVM is {accuracy_score(y_train, svm.predict(X_train))}\n')

svm_accuracy = accuracy_score(y_test, y_pred)
svm_precision = precision_score(y_test, y_pred, average='macro')
svm_recall = recall_score(y_test, y_pred, average='macro')
svm_f1_score = f1_score(y_test, y_pred, average='macro')

print(f"Test Accuracy of SVM is {svm_accuracy} \n")
print(f"Test Precision of SVM is {svm_precision} \n")
print(f"Test Recall of SVM is {svm_recall} \n")
print(f"Test F1_score of SVM is {svm_f1_score} \n")
print(f"Classification Report: \n{classification_report(y_test, y_pred)}")
cm = confusion_matrix(y_test, y_pred)
sns.set(rc={"figure.figsize": (4, 3)})
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')

"""accuracy ranking"""

import plotly.express as px

accuracy = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 'SVM', 'Random Forest', 'AdaBoost', 'XGBoost', 'CatBoost', 'MLP'],
    'Score': [knn_accuracy, logreg_accuracy, svm_accuracy, rfc_accuracy, ada_accuracy, xgb_accuracy, cat_accuracy,mlp_accuracy]})
accuracy_sorted = accuracy.sort_values(by='Score', ascending=False)
fig = px.bar(data_frame=accuracy_sorted, x='Model', y='Score', color='Score',
             title='Accuracy Comparison', text='Score')
fig.update_layout(width=800, height=600)
fig.show()

"""STACKING CLASSIFIER

Random Forest as meta learner
"""

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# Calibrated SVM to improve probability estimates
svm_calibrated = CalibratedClassifierCV(svm, method='sigmoid')
svm_calibrated.fit(X_train, y_train)

# Define base learners
base_learners = [
    ('Logistic Regression', logreg),
    ('Random Forest', rfc),
    ('XGBoost', xgb),
    ('CatBoost', cat_model),
    ('SVM', svm_calibrated),
    ('KNN', knn),
    ('MLP', mlp)
]

# Meta-classifier
meta_classifier = XGBClassifier()

# Stacking Classifier
stacking_clf6 = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_classifier,
    stack_method='predict_proba',  # Use probabilities as input to meta-classifier
    n_jobs=-1  # Utilize all available CPU cores
)

# Train the stacking classifier
stacking_clf6.fit(X_train, y_train)

# Make predictions
y_pred = stacking_clf6.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Stacking Classifier Accuracy: {accuracy:.4f}")
print(f"Stacking Classifier Precision: {precision:.4f}")
print(f"Stacking Classifier Recall: {recall:.4f}")
print(f"Stacking Classifier F1 Score: {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.set(rc={"figure.figsize": (4, 3)})
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')


model.save('parkinsons_model.h5')

'''joblib.dump(model, 'parkinsons_model.pkl')
print('Model saved as parkinsons_model.pkl')'''