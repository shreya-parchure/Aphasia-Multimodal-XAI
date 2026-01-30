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
            pipe, param_dist, n_iter=200, cv=5, n_jobs=1,
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
        search = RandomizedSearchCV(pipe, param_dist, n_iter=200, cv=5, n_jobs=1, 
                                   scoring="accuracy", error_score='raise', random_state=random_state)
        search.fit(X_train, y_train)
        return search.best_estimator_, search.best_score_
    else:
        pipe.fit(X_train, y_train)
        return pipe, None


def train_random_forest(X_train, y_train, param_search=True, random_state=42):
    model = RandomForestClassifier(random_state=random_state)
    if param_search:
        param_dist = {
            "n_estimators": randint(50, 200),
            "max_depth": [None, 5, 10],
            "min_samples_split": randint(2, 5),
            "min_samples_leaf": randint(1, 3),
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False]
        }
        search = RandomizedSearchCV(model, param_dist, n_iter=200, cv=5, n_jobs=1, 
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

def train_and_bootstrap_with_roc(X_, y_, model_type='logregcv', name='model',
                                 param_search=True, n_bootstraps=100, test_size=0.2,
                                 n_jobs=-1, random_state=42):

    np.random.seed(random_state)

    # Ensure DataFrame/Series
    if not isinstance(X_, pd.DataFrame):
        X_ = pd.DataFrame(X_)
    if not isinstance(y_, pd.Series):
        y_ = pd.Series(y_)

    # Split into train/test
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
    best_model, _ = model_funcs[model_type](X_train_full, y_train_full)

    base_fpr = np.linspace(0, 1, 101)

    def _bootstrap_iteration(seed):
        rng = np.random.default_rng(seed)
        X_bs, y_bs = resample(X_train_full, y_train_full,
                              replace=True, random_state=rng.integers(0, 1e6))

        X_bs_jittered = apply_jitter_safe(X_bs)
        model_clone = clone(best_model)
        model_clone.fit(X_bs_jittered, y_bs)

        if hasattr(model_clone, "predict_proba"):
            y_prob = model_clone.predict_proba(X_test)[:, 1]
        else:
            y_prob = model_clone.decision_function(X_test)
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())

        y_pred = model_clone.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        tpr_interp = np.interp(base_fpr, fpr, tpr)
        tpr_interp[0] = 0.0

        return acc, tpr_interp

    seeds = np.random.randint(0, 1e6, size=n_bootstraps)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_bootstrap_iteration)(seed) for seed in seeds
    )

    bootstrap_scores, tprs = zip(*results)
    bootstrap_scores = np.array(bootstrap_scores)
    tprs = np.array(tprs)

    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(base_fpr, mean_tpr)

    metrics = {
        'mean_accuracy': bootstrap_scores.mean(),
        'std_accuracy': bootstrap_scores.std(),
        '95ci_accuracy': np.percentile(bootstrap_scores, [2.5, 97.5]),
        'all_scores': bootstrap_scores,
        'mean_tpr': mean_tpr,
        'std_tpr': std_tpr,
        'base_fpr': base_fpr,
        'mean_auc': mean_auc
    }

    # Plot mean ROC
    plt.figure(figsize=(7,6))
    plt.plot(base_fpr, mean_tpr, color='b',
             label=f'{name} | {model_type} | Mean ROC (AUC={mean_auc:.3f})')
    plt.fill_between(base_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                     color='b', alpha=0.2, label='±1 std. dev.')
    plt.plot([0,1], [0,1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Bootstrap ROC for {name} ({model_type})")
    plt.legend(loc="lower right")
    plt.show()

    print(f"{name} | {model_type} | Bootstrap accuracy: {metrics['mean_accuracy']:.3f} ± {metrics['std_accuracy']:.3f}")

    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name}_{model_type}_best_model.joblib")
    joblib.dump(best_model, save_path)
    print(f"✅ Model saved to: {save_path}")

    return best_model, metrics, X_test, y_test
