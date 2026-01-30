import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.base import clone
import matplotlib.pyplot as plt
import seaborn as sns


def _evaluate_partition(classifier, X, y, partition_idx):
    """
    Evaluate a single bootstrap partition for any sklearn classifier.
    Handles predict_proba, decision_function, or fallback to predict.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=partition_idx, stratify=y
    )

    model = clone(classifier)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Probabilities / scores for ROC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
        # normalize decision_function to [0,1]
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
    else:
        # fallback: use predicted classes as scores (not ideal for ROC, but works)
        y_prob = y_pred

    return {
        "partition": partition_idx + 1,
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred, average='binary', pos_label=0),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred, pos_label=0),
        "Recall": recall_score(y_test, y_pred, pos_label=0)
    }


def evaluate_classifier_parallel(classifier, X, y, n_partitions=100, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_evaluate_partition)(classifier, X, y, i)
        for i in range(n_partitions)
    )
    return pd.DataFrame(results)


def evaluate_all_models_parallel(models_dict, X_dict, y_dict, n_partitions=100, n_jobs=-1):
    all_results = []
    for name, model in models_dict.items():
        print(f"Evaluating {name}...")
        X, y = X_dict[name], y_dict[name]
        df = evaluate_classifier_parallel(model, X, y, n_partitions=n_partitions, n_jobs=n_jobs)
        df["Model"] = name
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)
    long_df = combined.melt(
        id_vars=["partition", "Model"],
        value_vars=["Accuracy", "F1", "AUC", "Precision", "Recall"],
        var_name="Metric",
        value_name="Score"
    )
    return long_df


def plot_grouped_violin(df):
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red",
              "tab:purple", "tab:brown", "tab:pink")
    models = df["Model"].unique()
    color_map = {m: colors[i % len(colors)] for i, m in enumerate(models)}

    plt.figure(figsize=(17.5, 10)) 
    sns.set(style="whitegrid", font_scale=1.2)

    ax = sns.violinplot(
        data=df,
        x="Metric",
        y="Score",
        hue="Model",
        palette=color_map,
        inner="quartile",
        cut=0,
        linewidth=1,
        scale="width",
        bw=0.3,
    )

    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Score", fontsize=24, labelpad=16)
    ax.set_xlabel("Metric", fontsize=24, labelpad=16)
    ax.set_title("Bootstrapped Performance Metrics by Model", fontsize=30, pad=20)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.xticks(rotation=0, fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()
