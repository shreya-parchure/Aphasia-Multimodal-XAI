import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
from sklearn.base import clone
from joblib import Parallel, delayed

def _run_single_bootstrap(
    clf,
    X,
    y,
    classifier_name,
    partition_idx,
    test_size,
):
    """
    Runs one bootstrap partition for one classifier.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
    )

    clf = clone(clf)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    return [
        {
            "classifier": classifier_name,
            "metric": "accuracy",
            "value": accuracy_score(y_test, y_pred),
        },
        {
            "classifier": classifier_name,
            "metric": "f1_score",
            "value": f1_score(y_test, y_pred, pos_label=0),
        },
        {
            "classifier": classifier_name,
            "metric": "auc",
            "value": roc_auc_score(y_test, y_prob),
        },
        {
            "classifier": classifier_name,
            "metric": "precision_category_0",
            "value": precision_score(y_test, y_pred, pos_label=0),
        },
        {
            "classifier": classifier_name,
            "metric": "recall_category_0",
            "value": recall_score(y_test, y_pred, pos_label=0),
        },
    ]

def bootstrap_and_plot_violin(
    rf_models: dict,
    X_dict: dict,
    y_dict: dict,
    n_partitions: int = 500,
    test_size: float = 0.3,
    n_jobs: int = -1,
):
    """
    Parallelized bootstrap evaluation with violin plot visualization.
    """

    jobs = []

    for name, clf in rf_models.items():
        X = X_dict[name]
        y = y_dict[name]

        for i in range(n_partitions):
            jobs.append(
                delayed(_run_single_bootstrap)(
                    clf=clf,
                    X=X,
                    y=y,
                    classifier_name=name,
                    partition_idx=i,
                    test_size=test_size,
                )
            )

    results = Parallel(n_jobs=n_jobs, verbose=5)(jobs)

    # Flatten list of lists
    flat_results = [item for sublist in results for item in sublist]
    df_all_results = pd.DataFrame(flat_results)

    # -------------------
    # Plotting
    # -------------------
    plt.figure(figsize=(14, 10))
    sns.set_palette("tab10")

    sns.violinplot(
        x="metric",
        y="value",
        hue="classifier",
        data=df_all_results,
        inner="quart",
        scale="width",
        linewidth=1,
    )

    plt.ylim(0.5, 1)
    plt.grid(axis="y", alpha=0.3, linewidth=0.5)

    plt.title(
        "Bootstrapped Distributions of Classifier Performance Metrics",
        fontsize=16,
        pad=20,
    )
    plt.xlabel("Performance Metric", fontsize=14)
    plt.ylabel("Score (0 to 1)", fontsize=14)

    plt.legend(
        title="Classifier",
        bbox_to_anchor=(0.5, 1.12),
        loc="center",
        ncol=5,
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=10,
        title_fontsize=12,
    )

    plt.gca().set_facecolor("#fafafa")
    plt.tight_layout()
    plt.show()

    return df_all_results

