import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
import numpy as np

def plot_mean_rf_pr_sem(models_dict, X_dict, y_dict, n_splits=10, test_size=0.1, random_state=42):
    np.random.seed(random_state)
    plt.figure(figsize=(10, 10))

    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red", 
              "tab:purple", "tab:brown", "tab:pink")

    for idx, key in enumerate(models_dict):
        model = models_dict[key]
        X = X_dict[key]
        y = y_dict[key]

        precisions = []
        aps = []
        base_recall = np.linspace(0, 1, 101)

        for _ in range(n_splits):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=None
            )

            y_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_proba)

            # Interpolate precision as a function of recall
            precision_interp = np.interp(base_recall, recall[::-1], precision[::-1])
            precisions.append(precision_interp)
            aps.append(average_precision_score(y_test, y_proba))

        precisions = np.array(precisions)
        mean_precision = precisions.mean(axis=0)
        sem_precision = precisions.std(axis=0) / np.sqrt(n_splits)
        mean_ap = np.mean(aps)
        std_ap = np.std(aps)

        color = colors[idx % len(colors)]

        plt.plot(
            base_recall, mean_precision, color=color, linewidth=2,
            label=f'{key} (AP = {mean_ap:.3f} ± {std_ap:.3f})'
        )
        plt.fill_between(base_recall, mean_precision - sem_precision, 
                         mean_precision + sem_precision, color=color, alpha=0.2)

    plt.xlabel('Recall', fontsize=24)
    plt.ylabel('Precision', fontsize=24)
    plt.title('Mean Precision–Recall Curves ± SEM', fontsize=30)
    plt.legend(loc='lower left', fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
