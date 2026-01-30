import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, roc_curve
from sklearn.base import clone

def plot_mean_rf_roc_sem(
    models_dict,
    X_dict,
    y_dict,
    n_splits=5,
    title="Mean ROC curve ± SEM",
):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(10, 10))

    colors = ("tab:pink", "#b947b1", 
              "#4d1c91")

    for idx, key in enumerate(models_dict):
        model = models_dict[key]
        X = X_dict[key]
        y = y_dict[key]

        # safety checks
        assert isinstance(y, pd.Series)
        assert X.shape[0] == y.shape[0]
        assert X.index.equals(y.index)

        tprs = []
        aucs = []

        for train, test in cv.split(X, y):
            clf = clone(model)
            clf.fit(X.iloc[train], y.iloc[train])

            y_proba = clf.predict_proba(X.iloc[test])[:, 1]
            fpr, tpr, _ = roc_curve(y.iloc[test], y_proba)

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc(fpr, tpr))

        tprs = np.array(tprs)
        mean_tpr = tprs.mean(axis=0)
        mean_tpr[-1] = 1.0

        mean_auc = auc(mean_fpr, mean_tpr)
        sem_auc = np.std(aucs, ddof=1) / np.sqrt(n_splits)

        sem_tpr = tprs.std(axis=0, ddof=1) / np.sqrt(n_splits)
        tprs_upper = np.minimum(mean_tpr + sem_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - sem_tpr, 0)

        color = colors[idx % len(colors)]

        ax.plot(
            mean_fpr,
            mean_tpr,
            color=color,
            lw=2,
            label=f"{key} (AUC = {mean_auc:.3f} ± {sem_auc:.3f})",
        )

        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color=color,
            alpha=0.2,
        )

    ax.plot([0, 1], [0, 1], linestyle="--", lw=1, color="black")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=title,
    )
    ax.legend(loc="lower right", fontsize=18)
    ax.title.set_fontsize(20)
    ax.xaxis.label.set_fontsize(18)
    ax.yaxis.label.set_fontsize(18)
    plt.grid(alpha = 0.4)
    plt.tight_layout()
    plt.show()
