import os
import socket
# Patch hostname resolution
socket.gethostname = lambda: "localhost"
socket.getfqdn = lambda: "localhost"

# Set environment for XGBoost tracker
os.environ["DMLC_TRACKER_URI"] = "127.0.0.1"
os.environ["DMLC_TRACKER_PORT"] = "9091"


from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
import time


def main():
    # Start a local Dask cluster
    cluster = LocalCluster(host="127.0.0.1", n_workers=4, threads_per_worker=2)

    client = Client(cluster)
    print("✅ Dask cluster started")

    # Load dataset (Parquet is efficient for large datasets)
    X_train = dd.read_parquet("../data/X_train.parquet")
    y_train = dd.read_parquet("../data/y_train.parquet")["label"]
    X_val = dd.read_parquet("../data/X_val.parquet")
    y_val = dd.read_parquet("../data/y_val.parquet")["label"]
    
    # Convert to DaskDMatrix
    dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)
    dval = xgb.dask.DaskDMatrix(client, X_val, y_val)

    # Define XGBoost parameters
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 2,
        "eta": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.7,
        "tree_method": "hist",  # or "approx" if GPU is available: use "gpu_hist"
    }

    # Train
    
    output = xgb.dask.train(
    client,
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dval, "validation")],
    
)

    

    # Predict
    start_time = time.time()
    preds = xgb.dask.predict(client, output["booster"], X_val).compute()
    preds_binary = (preds > 0.5).astype(int)
    y_val_true = y_val.compute()
    end_time = time.time()
    total_time = end_time - start_time
    time_per_account = total_time / len(y_val)

    # Evaluation
    print(f"\n🕒 Total prediction time: {total_time:.4f} seconds")
    print(f"⏱️ Average time per account: {time_per_account:.6f} seconds")
    print("🔍 Final Classification Report:")
    print(classification_report(y_val_true, preds_binary))
    cm = confusion_matrix(y_val_true, preds_binary)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Human", "Bot"]).plot(cmap="Blues")
    plt.show()

if __name__ == "__main__":
    main()
