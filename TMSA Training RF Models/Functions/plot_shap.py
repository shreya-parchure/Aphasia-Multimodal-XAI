import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline

# ------------------------------
# Universal SHAP Computation
# ------------------------------

def get_shap_values(X, model, class_index=1):
    """
    Compute SHAP values for any sklearn model or pipeline.
    Handles tree, linear, kernel, and neural models.
    """
    model_name = model.__class__.__name__.lower()

    # === Unwrap sklearn Pipeline if present ===
    if isinstance(model, Pipeline):
        print(f"Detected sklearn Pipeline — using final estimator: {model.named_steps[list(model.named_steps.keys())[-1]]}")
        model = model.named_steps[list(model.named_steps.keys())[-1]]

    # Convert DataFrame to numpy if needed
    X_values = X.values if isinstance(X, pd.DataFrame) else np.array(X)

    try:
        # === Tree-based models ===
        if any(x in model_name for x in ["forest", "xgb", "gbm", "lgbm", "tree"]):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_values)

        # === Linear models (LogisticRegression, ElasticNet, etc.) ===
        elif any(x in model_name for x in ["logistic", "elastic", "linear"]):
            explainer = shap.Explainer(model.predict_proba, X_values)
            shap_values = explainer(X_values).values

        # === Kernel / Neural / Other black-box models (SVC, MLP, etc.) ===
        else:
            # If predict_proba not available, fallback to decision_function or predict
            if hasattr(model, "predict_proba"):
                predict_fn = model.predict_proba
            elif hasattr(model, "decision_function"):
                predict_fn = model.decision_function
            else:
                predict_fn = model.predict

            explainer = shap.Explainer(predict_fn, X_values)
            shap_values = explainer(X_values).values

        # === Handle multiclass or binary output ===
        if isinstance(shap_values, list):
            shap_vals = shap_values[class_index]
        elif shap_values.ndim == 3 and shap_values.shape[2] > 1:
            shap_vals = shap_values[:, :, class_index]
        elif shap_values.ndim == 2:
            shap_vals = shap_values
        else:
            raise ValueError(f"Unexpected SHAP shape: {shap_values.shape}")

        return shap_vals, explainer

    except Exception as e:
        raise RuntimeError(f"Failed to compute SHAP for {model_name}: {e}")


# ------------------------------
# Feature Importance Extraction
# ------------------------------

def get_top_shap_features(X, model, n_features=25, class_index=1, model_name=None):
    shap_vals, explainer = get_shap_values(X, model, class_index)
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    feature_importance = pd.Series(mean_abs_shap, index=X.columns).sort_values(ascending=False)

    top_features = feature_importance.head(n_features)
    title = f"Top {n_features} Features by Mean |SHAP| Value"
    if model_name:
        title += f" — {model_name}"
    print(f"\n{title}")
    print("=" * len(title))
    print(top_features.to_string(float_format="%.4f"))

    return top_features.index.tolist(), shap_vals, explainer


# ------------------------------
# Dependence Plots (Single & Batch)
# ------------------------------

def _plot_single_shap_dependence(feature, X, shap_vals, model_name, save_dir, cmap):
    plt.figure(figsize=(4, 3))
    shap.dependence_plot(
        feature,
        shap_vals,
        X,
        show=False,
        alpha=0.7,
        dot_size=10,
        cmap=cmap
    )
    plt.title(f"{model_name}\n{feature}", fontsize=10, pad=4)
    plt.xlabel(feature, fontsize=9)
    plt.ylabel("SHAP value", fontsize=9)
    plt.grid(False)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{model_name.replace(' ', '_')}_{feature}.png")
        plt.savefig(out_path, dpi=250, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_shap_dependence_for_features(
    X,
    shap_vals,
    features,
    model_name="Model",
    save_dir=None,
    n_jobs=-1
):
    cmap = plt.get_cmap("coolwarm")
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    print(f"\nGenerating {len(features)} SHAP dependence plots for {model_name} in parallel...")

    Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_plot_single_shap_dependence)(
            feature, X, shap_vals, model_name, save_dir, cmap
        )
        for feature in features
    )

    print(f"Finished all SHAP plots for {model_name} — saved to {save_dir if save_dir else 'screen'}")


# ------------------------------
# Beeswarm Summary Plot (Top Features)
# ------------------------------

def plot_shap(
    X,
    shap_vals,
    top_features=None,
    model_name="Model",
    save_path=None
):
    """
    Plots a SHAP beeswarm summary with auto-scaled figure height
    based on number of displayed features.
    """
    print(f"\nGenerating SHAP beeswarm for {model_name}...")

    # Restrict to top features if provided
    if top_features is not None:
        X_plot = X[top_features]
        shap_vals_plot = shap_vals[:, [X.columns.get_loc(f) for f in top_features]]
    else:
        X_plot = X
        shap_vals_plot = shap_vals

    num_features = len(X_plot.columns)

    # --- Dynamic height calculation ---
    base_per_feature = 0.45    # inches per feature
    min_height = 4
    max_height = 20
    fig_height = min(max(num_features * base_per_feature, min_height), max_height)

    plt.figure(figsize=(8, fig_height))

    shap.summary_plot(
        shap_vals_plot,
        X_plot,
        show=False,
        plot_size=(8, fig_height),
        color_bar=True,
        max_display=num_features,
        alpha=0.8
    )

    plt.title(f"{model_name} — SHAP Beeswarm (Top Features)", fontsize=13, pad=12)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Beeswarm saved to {save_path}")
    else:
        plt.show()
        plt.close()


def get_top_shap_features_with_direction(X, model, n_features=25, class_index=1):

    shap_vals, explainer = get_shap_values(X, model, class_index)
    
    abs_sum = np.abs(shap_vals).sum(axis=0)
    signed_sum = shap_vals.sum(axis=0)
    
    top_idx = np.argsort(-abs_sum)[:n_features]  # top N by absolute sum
    top_features = pd.DataFrame({
        'abs_shap_sum': abs_sum[top_idx],
        'signed_shap_sum': signed_sum[top_idx],
        'direction': np.where(signed_sum[top_idx] >= 0, 'positive', 'negative')
    }, index=X.columns[top_idx])
    
    print(f"\nTop {n_features} SHAP features (absolute sum + direction):")
    print(top_features.to_string(float_format="%.4f"))
    
    return top_features, shap_vals, explainer
