import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import shap
import matplotlib.pyplot as plt

# Load datasets from Parquet
X_train = pd.read_parquet("../data/X_train.parquet")
y_train = pd.read_parquet("../data/y_train.parquet")["label"]
X_val = pd.read_parquet("../data/X_val.parquet")
y_val = pd.read_parquet("../data/y_val.parquet")["label"]

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
start_time = time.time()
y_pred = clf.predict(X_val)
end_time = time.time()
total_time = end_time - start_time
time_per_account = total_time / len(y_val)

print(f"\n🕒 Total prediction time: {total_time:.4f} seconds")
print(f"⏱️ Average time per account: {time_per_account:.6f} seconds")
print("🔍 Final Classification Report:")
print(classification_report(y_val, y_pred))

# SHAP explainer
explainer = shap.Explainer(clf, X_train)
shap_values = explainer(X_val)



# SHAP values plots
plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values[..., 1], X_val, plot_type="dot", show=False)
plt.title("Class 1 (Bot)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values[..., 0], X_val, plot_type="dot", show=False)
plt.title("Class 0 (Human)")
plt.tight_layout()
plt.show()

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Human", "Bot"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
