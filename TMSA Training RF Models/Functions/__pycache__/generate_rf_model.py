import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    auc
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed


def train_model(X, y, name):
    seed = 42
    # param_dist = {
    #     'n_estimators': [200, 300, 400],
    #     'max_depth': [8, 10, 12],
    #     'max_features': ['sqrt', 'log2', 0.5],
    #     'min_samples_split': [0.01, 0.025, 0.05],
    #     'min_samples_leaf': [0.01, 0.015, 0.02]
    # }
    param_dist = {
        'n_estimators': [200, 300, 400],
        'max_depth': [5, 8, 10],
        'min_samples_leaf': [0.01, 0.025, 0.05, 0.1],
        'min_samples_split': [0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
        'bootstrap': [True, False]
    }


    # Jitter function
    def jitter_X_train(X_train):
        X_train = X_train.copy()  
        jitter_config = {
            'Lesion_Volume': 1000, 'Age': 1, 'Yrs Edu': 1, 'CCRSA': 1,
            'Avg_WAB_AQ ': 1, 'SS_WAB_Avg': 0.5, 'AVC_WAB_Avg': 0.5,
            'NWF_WAB_Avg': 0.5, 'Noun_Freq': 0.5, 'Noun_NA': 0.5,
            'Syllables_avg (SyllaPy)': 0.5, 'Phonemes_avg (CMUDict)': 0.5, 'Morphemes': 0.5
        }
        for col, scale in jitter_config.items():
            if col in X_train.columns:
                X_train[col] += np.random.uniform(-1, 1, size=len(X_train)) * scale
        return X_train

    # Train-validation-test split
    X_temp, X_val, y_temp, y_val = train_test_split(
        X, y, train_size=0.9, random_state=seed, stratify=y
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_temp, y_temp, train_size=0.9, random_state=seed, stratify=y_temp
    )
    X_train = jitter_X_train(X_train)

    # Random Forest base model (no oob_score, class_weight balanced)
    rf = RandomForestClassifier(
        random_state=seed,
        class_weight="balanced"
    )

    random_search_model = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=100,
        cv=5,
        verbose=2,
        random_state=seed,
        n_jobs=-1,
        error_score="raise"  # fail loudly if something breaks
    )
    random_search_model.fit(X_train, y_train)
    best_rf_model = random_search_model.best_estimator_

    # Save CV results
    model_rf_results = random_search_model.cv_results_
    param_results = pd.DataFrame(model_rf_results['params'])
    param_results['mean_test_score'] = model_rf_results['mean_test_score']
    param_results['std_test_score'] = model_rf_results['std_test_score']
    os.makedirs('Models', exist_ok=True)
    param_results.to_csv(f'Models/{name}_rf_results.csv', index=False)
    joblib.dump(best_rf_model, f'Models/best_rf_{name}.pkl')

    # Training & test accuracy
    print(f"Train Accuracy: {best_rf_model.score(X_train, y_train):.4f}")
    print(f"Test Accuracy: {best_rf_model.score(X_test, y_test):.4f}")
    preds = best_rf_model.predict(X_test)
    print(classification_report(y_test, preds))

    # Confusion matrix
    disp = ConfusionMatrixDisplay.from_estimator(
        best_rf_model, X_test, y_test,
        display_labels=np.unique(y_test),
        cmap=plt.cm.Blues,
        normalize="true"
    )
    disp.ax_.set_title("Normalized Confusion Matrix")
    plt.show()

    # -------------------
    # Parallel k-fold ROC
    # -------------------
    n_splits = 10
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    X_np = X.to_numpy() if isinstance(X, pd.DataFrame) else X
    y_np = y.to_numpy() if isinstance(y, (pd.Series, pd.DataFrame)) else y
    y_np = y_np.squeeze()
    mean_fpr = np.linspace(0, 1, 100)

    def fit_and_roc(train_idx, test_idx):
        model = RandomForestClassifier(**best_rf_model.get_params())
        model.fit(X_np[train_idx], y_np[train_idx])
        viz = RocCurveDisplay.from_estimator(
            model, X_np[test_idx], y_np[test_idx], alpha=0.3, lw=1
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        return interp_tpr, viz.roc_auc

    results = Parallel(n_jobs=-1)(
        delayed(fit_and_roc)(train, test) for train, test in cv.split(X_np, y_np)
    )
    tprs, aucs = zip(*results)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(mean_fpr, mean_tpr, color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2, alpha=0.8)
    ax.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0),
                    np.minimum(mean_tpr + std_tpr, 1),
                    color="grey", alpha=0.2, label=r"$\pm$ 1 std. dev.")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title=f"Mean ROC curve with variability")
    ax.legend(loc="lower right", prop={'size': 8})
    plt.show()

    return best_rf_model, X_val, y_val
