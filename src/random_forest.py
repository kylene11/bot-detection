import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import shap
import matplotlib.pyplot as plt

# Load datasets
X_train = pd.read_csv("../data/X_train.csv")
y_train = pd.read_csv("../data/y_train.csv").squeeze()
X_val = pd.read_csv("../data/X_val.csv")
y_val = pd.read_csv("../data/y_val.csv").squeeze()

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
print(classification_report(y_val, y_pred))

# SHAP explainer
explainer = shap.Explainer(clf, X_train)
shap_values = explainer(X_val)

# Check shape
print("shap_values shape:", shap_values.values.shape)  # (139, 7, 2)


## SHAP values (more positive -> more likely to be said class)
# the feature’s impact on the model’s output


# Class 1 (bot)
plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values[..., 1], X_val, plot_type="dot", show=False)
plt.title("Class 1 (Bot)")
plt.tight_layout()
plt.show()

# Class 0 (human)
plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values[..., 0], X_val, plot_type="dot", show=False)
plt.title("Class 0 (Human)")
plt.tight_layout()
plt.show()


# Compute and plot confusion matrix
cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Human", "Bot"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()