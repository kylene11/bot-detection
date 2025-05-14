from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
from lightgbm.dask import DaskLGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import itertools
import time
import shap

def main():
    
    # Start Dask cluster
    cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit="4GB")
    client = Client(cluster)
    print("✅ Dask cluster started")

    # Load training data

    X_train = dd.read_parquet("../data/X_train.parquet")
    y_train = dd.read_parquet("../data/y_train.parquet")["label"]

    

    # ## TESTING PARAMS
    # # Define grid
    # param_grid = {
    #     "n_estimators": [100, 200],
    #     "learning_rate": [0.05, 0.1],
    #     "max_depth": [3, 5],
    #     "num_leaves": [15, 31],
    # }

    # # Create all combinations
    # param_combinations = list(itertools.product(
    #     param_grid["n_estimators"],
    #     param_grid["learning_rate"],
    #     param_grid["max_depth"],
    #     param_grid["num_leaves"]
    # ))

    # best_score = 0
    # best_model = None
    # best_params = None

    # for n_estimators, learning_rate, max_depth, num_leaves in param_combinations:
    #     clf = DaskLGBMClassifier(
    #         objective="binary",
    #         tree_learner="data",
    #         n_estimators=n_estimators,
    #         learning_rate=learning_rate,
    #         max_depth=max_depth,
    #         num_leaves=num_leaves
    #     )
    #     clf.fit(X_train, y_train)


    #     # Evaluate on validation set
    #     X_val = dd.read_csv("../data/X_val.csv")
    #     y_val = pd.read_csv("../data/y_val.csv").squeeze()
    #     y_pred = clf.predict(X_val).compute()

    #     score = classification_report(y_val, y_pred, output_dict=True)["accuracy"]
    #     print(f"✅ Tried: {n_estimators}, {learning_rate}, {max_depth}, {num_leaves} → Accuracy: {score:.4f}")

    #     if score > best_score:
    #         best_score = score
    #         best_model = clf
    #         best_params = {
    #             "n_estimators": n_estimators,
    #             "learning_rate": learning_rate,
    #             "max_depth": max_depth,
    #             "num_leaves": num_leaves
    #         }

    # print("\n✅ Best params:", best_params)
    # Use best known parameters
    best_model = DaskLGBMClassifier(
        objective="binary",
        tree_learner="data",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=15
    )
    
    best_model.fit(X_train, y_train)

    # Evaluate
    X_val = dd.read_parquet("../data/X_val.parquet")
    y_val = pd.read_parquet("../data/y_val.parquet")["label"]

    start_time = time.time()

    y_pred = best_model.predict(X_val).compute()

    end_time = time.time()
    total_time = end_time - start_time
    time_per_account = total_time / len(y_val)

    print(f"\n🕒 Total prediction time: {total_time:.4f} seconds")
    print(f"⏱️ Average time per account: {time_per_account:.6f} seconds")

    print("🔍 Final Classification Report:")
    print(classification_report(y_val, y_pred))

    # Feature importance that influences the model's decision
    booster = best_model.to_local().booster_
    lgb.plot_importance(booster, max_num_features=10, importance_type='gain')
    plt.tight_layout()
    plt.show()
    


    # Compute and plot confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Human", "Bot"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

     # ==== SHAP EXPLANATIONS ====
    X_val_pd = X_val.compute()

    # Use SHAP TreeExplainer with local LightGBM booster
    explainer = shap.Explainer(booster)
    shap_values = explainer(X_val_pd)

 
    # SHAP Summary Dot Plot
    shap.summary_plot(shap_values, X_val_pd, plot_type="dot", show=False)
    plt.title("SHAP Summary – Feature Impact (Dot)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
