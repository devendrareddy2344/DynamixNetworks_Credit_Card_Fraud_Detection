import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import SMOTE
df = pd.read_csv("creditcard.csv")  
X = df.drop("Class", axis=1)
y = df["Class"]
print("Original class distribution:")
print(y.value_counts())
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
smote = SMOTE(random_state=42, k_neighbors=1)

X_train_res, y_train_res = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("After SMOTE class distribution:")
print(pd.Series(y_train_res).value_counts())
extra_model = ExtraTreesClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

extra_model.fit(X_train_res, y_train_res)
y_train_pred = extra_model.predict(X_train_res)
y_test_pred = extra_model.predict(X_test_scaled)

y_test_prob = extra_model.predict_proba(X_test_scaled)[:, 1]
print("Training Accuracy:", accuracy_score(y_train_res, y_train_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Classification Report:\n")
print(classification_report(y_test, y_test_pred))
cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Extra Trees")
plt.show()
roc_auc = roc_auc_score(y_test, y_test_prob)
print("ROC-AUC Score:", roc_auc)
with open("credit_fraud_pipeline.pkl", "wb") as f:
    pickle.dump(
        {
            "model": extra_model,
            "scaler": scaler
        },
        f
    )
print("Model and scaler saved to credit_fraud_pipeline.pkl")