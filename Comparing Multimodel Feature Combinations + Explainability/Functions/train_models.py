import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from scipy.stats import loguniform, randint
from sklearn.utils import resample
from sklearn.base import clone
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import joblib, os

# ------------------------------
# Model Training Functions
# ------------------------------

def train_logistic_regression_cv(X_train, y_train, random_state=42):
    """
    Trains a LogisticRegressionCV model with built-in cross-validation to tune regularization strength (C).
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegressionCV(
            Cs=20,                     # number of regularization strengths to test
            cv=5,                      # 5-fold CV
            max_iter=5000,
            penalty='l2',
            solver='lbfgs',
            scoring='accuracy',
            n_jobs=-1,
            random_state=random_state
        ))
    ])
    pipe.fit(X_train, y_train)
    return pipe, None


def train_elasticnet(X_train, y_train, param_search=True, random_state=42):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            solver='saga',
            max_iter=5000,
            penalty='elasticnet',
            random_state=random_state
        ))
    ])
    if param_search:
        param_dist = {
            "model__C": loguniform(1e-3, 1e2),
            "model__l1_ratio": np.linspace(0.0, 1.0, 6)
        }
        search = RandomizedSearchCV(
            pipe, param_dist, n_iter=10, cv=3, n_jobs=1,
            scoring="accuracy", error_score='raise', random_state=random_state
        )
        search.fit(X_train, y_train)
        return search.best_estimator_, search.best_score_
    else:
        pipe.fit(X_train, y_train)
        return pipe, None


def train_svc(X_train, y_train, param_search=True, random_state=42):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True, random_state=random_state))
    ])
    if param_search:
        param_dist = {
            "model__C": loguniform(1e-3, 1e2),
            "model__gamma": loguniform(1e-4, 1e1),
            "model__kernel": ["rbf"]
        }
        search = RandomizedSearchCV(pipe, param_dist, n_iter=10, cv=3, n_jobs=1, 
                                   scoring="accuracy", error_score='raise', random_state=random_state)
        search.fit(X_train, y_train)
        return search.best_estimator_, search.best_score_
    else:
        pipe.fit(X_train, y_train)
        return pipe, None


def train_random_forest(X_train, y_train, param_search=True, random_state=42):
    model = RandomForestClassifier(random_state=random_state)
    n_feats = X_train.shape[1]
    max_features = int(0.25 * n_feats)
    if param_search:
        param_dist = {
            "n_estimators": randint(50, 200),
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": randint(2, 5),
            "min_samples_leaf": randint(1, 3),
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False]
        }
        search = RandomizedSearchCV(model, param_dist, n_iter=200, cv=5, n_jobs=-1, 
                                   scoring="accuracy", error_score='raise', random_state=random_state)
        search.fit(X_train, y_train)
        return search.best_estimator_, search.best_score_
    else:
        model.fit(X_train, y_train)
        return model, None


def train_mlp(X_train, y_train, param_search=True, random_state=42):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(max_iter=1000, early_stopping=True, random_state=random_state))
    ])
    if param_search:
        param_dist = {
            "model__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32)],
            "model__activation": ["relu", "tanh"],
            "model__alpha": loguniform(1e-5, 1e-1),
            "model__learning_rate_init": loguniform(1e-4, 1e-2)
        }
        search = RandomizedSearchCV(pipe, param_dist, n_iter=10, cv=3, n_jobs=1, 
                                   scoring="accuracy", error_score='raise', random_state=random_state)
        search.fit(X_train, y_train)
        return search.best_estimator_, search.best_score_
    else:
        pipe.fit(X_train, y_train)
        return pipe, None


# ------------------------------
# Jitter Helper
# ------------------------------

def apply_jitter_safe(X_df):
    X = X_df.copy().astype(np.float64)
    rng = np.random.default_rng()
    
    jitter_specs = {
        "Lesion_Volume": 0.25,
        "MPO": 0.15,
        "Age": 0.15,
        "CCRSA": 0.1,
        "SS_WAB_Avg": 0.1,
        "AVC_WAB_Avg": 0.1,
        "NWF_WAB_Avg": 0.1,
        "Avg_WAB_AQ": 0.1,
        "Noun_Freq": 0.1,
        "Noun_NA": 0.1,
        "Syllables_avg (SyllaPy)": 0.1,
        "Phonemes_avg (CMUDict)": 0.1,
        "Morphemes": 0.1
    }
    
    for col, scale in jitter_specs.items():
        if col in X.columns:
            factors = 1 + rng.uniform(-scale, scale, size=len(X))
            X[col] = X[col].values * factors
            X[col] = np.where(np.abs(X[col]) < 1e-10, 1e-10, X[col])
    
    return np.nan_to_num(X.values, nan=0.0, posinf=1e6, neginf=-1e6)


# ------------------------------
# Main Training & Bootstrap ROC
# ------------------------------
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.base import clone

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.base import clone
import matplotlib.pyplot as plt
import numpy as np
import os, joblib

def train_and_kfold_validate(X_, y_, model_type='rf', name='model',
                              param_search=True, n_splits=5, test_size=0.2,
                              random_state=42):
    """
    Train a model with k-fold CV on the training set, and keep a separate held-out test set.
    Returns:
        - best_model (trained on full training set)
        - metrics (dict with CV and test metrics)
        - X_test, y_test (held-out test set)
    """

    np.random.seed(random_state)

    # Ensure DataFrame/Series
    if not isinstance(X_, pd.DataFrame):
        X_ = pd.DataFrame(X_)
    if not isinstance(y_, pd.Series):
        y_ = pd.Series(y_)

    # Split out a held-out test set
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_, y_, test_size=test_size, stratify=y_, random_state=random_state
    )

    # Map model_type to training function
    model_funcs = {
        'logregcv': train_logistic_regression_cv,
        'elasticnet': train_elasticnet,
        'svc': train_svc,
        'rf': train_random_forest,
        'mlp': train_mlp
    }
    if model_type not in model_funcs:
        raise ValueError(f"Unknown model_type {model_type}. Choose from {list(model_funcs.keys())}")

    # Train best model on full training set
    best_model, _ = model_funcs[model_type](X_train_full, y_train_full, param_search=param_search)

    # K-Fold CV on training set
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accuracies = []
    base_fpr = np.linspace(0, 1, 101)
    tprs = []

    for train_idx, val_idx in skf.split(X_train_full, y_train_full):
        X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        # Optional: jitter training features
        X_train_jittered = apply_jitter_safe(X_train)

        model_clone = clone(best_model)
        model_clone.fit(X_train_jittered, y_train)

        # Accuracy
        y_pred = model_clone.predict(X_val)
        accuracies.append(accuracy_score(y_val, y_pred))

        # ROC
        if hasattr(model_clone, "predict_proba"):
            y_prob = model_clone.predict_proba(X_val)[:, 1]
        else:
            y_prob = model_clone.decision_function(X_val)
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())

        fpr, tpr, _ = roc_curve(y_val, y_prob)
        tpr_interp = np.interp(base_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

    # Aggregate CV results
    tprs = np.array(tprs)
    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(base_fpr, mean_tpr)

    cv_metrics = {
        'cv_mean_accuracy': np.mean(accuracies),
        'cv_std_accuracy': np.std(accuracies),
        'cv_95ci_accuracy': np.percentile(accuracies, [2.5, 97.5]),
        'cv_mean_tpr': mean_tpr,
        'cv_std_tpr': std_tpr,
        'base_fpr': base_fpr,
        'cv_mean_auc': mean_auc
    }

    # Evaluate on separate test set
    if hasattr(best_model, "predict_proba"):
        y_test_prob = best_model.predict_proba(X_test)[:, 1]
    else:
        y_test_prob = best_model.decision_function(X_test)
        y_test_prob = (y_test_prob - y_test_prob.min()) / (y_test_prob.max() - y_test_prob.min())

    y_test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
    test_auc = auc(fpr_test, tpr_test)

    test_metrics = {
        'test_accuracy': test_acc,
        'test_fpr': fpr_test,
        'test_tpr': tpr_test,
        'test_auc': test_auc
    }

    # Combine metrics
    metrics = {**cv_metrics, **test_metrics}

    # Plot CV ROC
    plt.figure(figsize=(7,6))
    plt.plot(base_fpr, mean_tpr, color='b', label=f'{name} | {model_type} | CV Mean ROC (AUC={mean_auc:.3f})')
    plt.fill_between(base_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='b', alpha=0.2, label='±1 std. dev.')
    plt.plot([0,1], [0,1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} ({model_type}) | CV ROC")
    plt.legend(loc="lower right")
    plt.show()

    # Save model
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name}_{model_type}_best_model.joblib")
    joblib.dump(best_model, save_path)
    print(f"✅ Model saved to: {save_path}")
    print(f"{name} | {model_type} | Test accuracy: {test_acc:.3f}, Test AUC: {test_auc:.3f}")

    return best_model, metrics, X_test, y_test

