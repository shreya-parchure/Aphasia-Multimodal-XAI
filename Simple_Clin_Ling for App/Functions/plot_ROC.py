import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import numpy as np

def plot_mean_roc_sem(models_dict, X_dict, y_dict, n_splits=10, test_size=0.1, random_state=42):
    np.random.seed(random_state)
    plt.figure(figsize=(10, 10))
    
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
              "tab:brown", "tab:pink")
    
    for idx, key in enumerate(models_dict):
        model = models_dict[key]
        X = X_dict[key]
        y = y_dict[key]
        
        tprs = []
        aucs = []
        base_fpr = np.linspace(0, 1, 101)
        
        for _ in range(n_splits):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=None
            )

            # --- SAFE probability extraction ---
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            else:
                # fallback to decision_function
                y_dec = model.decision_function(X_test)
                # normalize scores to 0–1 to mimic probability scale
                y_score = (y_dec - y_dec.min()) / (y_dec.max() - y_dec.min())
            # ------------------------------------

            fpr, tpr, _ = roc_curve(y_test, y_score)
            aucs.append(auc(fpr, tpr))

            tpr_interp = np.interp(base_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)
        
        tprs = np.array(tprs)
        mean_tpr = tprs.mean(axis=0)
        sem_tpr = tprs.std(axis=0) / np.sqrt(n_splits)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        
        color = colors[idx % len(colors)]
        
        plt.plot(
            base_fpr, mean_tpr, color=color, linewidth=2,
            label=f"{key} (AUC = {mean_auc:.3f} ± {std_auc:.3f})"
        )
        plt.fill_between(base_fpr, mean_tpr - sem_tpr, mean_tpr + sem_tpr,
                         color=color, alpha=0.2)
    
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate", fontsize=24)
    plt.ylabel("True Positive Rate", fontsize=24)
    plt.title("Mean ROC Curves ± SEM", fontsize=30)
    plt.legend(loc="lower right", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
