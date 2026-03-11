import argparse
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.dataset.dataset import load_data, prepare_data, downsample_group
from src.models.baseline import train_model, evaluate_model
from src.fairness.reduction import train_fairness_model


def baseline_shap(model, X_train, X_test, feature_names):
    """
    SHAP for baseline logistic regression
    """
    explainer = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    return explainer, shap_values, shap_df


def fairness_model_predict_proba(fairness_model, X):
    """
    Weighted average probability for ExponentiatedGradient fairness model
    """
    predictors = fairness_model.predictors_
    weights = np.asarray(fairness_model.weights_)

    probs = []
    for pred in predictors:
        p = pred.predict_proba(X)[:, 1]
        probs.append(p)

    probs = np.vstack(probs)  # shape: [n_predictors, n_samples]
    weighted_probs = np.average(probs, axis=0, weights=weights)
    return weighted_probs


def fairness_shap(fairness_model, X_train, X_test, feature_names, background_size=50, test_size=None):
    """
    SHAP for fairness model using KernelExplainer
    """
    background = shap.sample(X_train, nsamples=background_size, random_state=1234)

    if test_size is None:
        X_test_sample = X_test
    else:
        X_test_sample = shap.sample(X_test, nsamples=test_size, random_state=1234)

    def model_fn(x):
        return fairness_model_predict_proba(fairness_model, x)

    explainer = shap.KernelExplainer(model_fn, background)
    shap_values = explainer.shap_values(X_test_sample)

    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    return explainer, shap_values, shap_df, X_test_sample


def plot_summary(shap_values, X_test, feature_names, title="SHAP summary"):
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        show=False
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_local(explainer, shap_values, X_test, feature_names, sample_idx=0, title="SHAP local explanation"):
    """
    Local explanation for a single sample using waterfall plot
    """
    if hasattr(X_test, "toarray"):
        X_test_dense = X_test.toarray()
    else:
        X_test_dense = X_test

    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value,
        data=X_test_dense[sample_idx],
        feature_names=feature_names
    )

    plt.figure()
    shap.plots.waterfall(explanation, max_display=25, show=False)
    plt.title(f"{title} (sample {sample_idx})")
    plt.tight_layout()
    plt.show()


def print_local_prediction(model, X_test, y_test, sample_idx=0):
    """
    Print prediction details for one sample
    """
    y_pred = model.predict(X_test)[sample_idx]

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[sample_idx, 1]
        print(f"Sample {sample_idx} - true label: {y_test.iloc[sample_idx] if hasattr(y_test, 'iloc') else y_test[sample_idx]}")
        print(f"Sample {sample_idx} - predicted label: {y_pred}")
        print(f"Sample {sample_idx} - predicted probability (loan=1): {y_prob:.4f}")
    else:
        print(f"Sample {sample_idx} - true label: {y_test.iloc[sample_idx] if hasattr(y_test, 'iloc') else y_test[sample_idx]}")
        print(f"Sample {sample_idx} - predicted label: {y_pred}")


def print_fairness_local_prediction(fairness_model, X_test_sample, sample_idx=0):
    """
    Print prediction details for one sample for fairness model
    """
    y_pred = fairness_model.predict(X_test_sample)[sample_idx]
    y_prob = fairness_model_predict_proba(fairness_model, X_test_sample)[sample_idx]

    print(f"Sample {sample_idx} - predicted label: {y_pred}")
    print(f"Sample {sample_idx} - weighted probability (loan=1): {y_prob:.4f}")


def run_original(X_train_raw, X_test_raw, y_train, y_test, s_test, preprocessor, local_idx=0):
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    feature_names = preprocessor.get_feature_names_out()

    print("=== Original dataset ===")
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, s_test)

    explainer, shap_values, shap_df = baseline_shap(model, X_train, X_test, feature_names)

    plot_summary(
        shap_values,
        X_test,
        feature_names,
        title="SHAP summary - Original dataset"
    )

    print_local_prediction(model, X_test, y_test, sample_idx=local_idx)
    plot_local(
        explainer,
        shap_values,
        X_test,
        feature_names,
        sample_idx=local_idx,
        title="SHAP local explanation - Original dataset"
    )

    return model, explainer, shap_values, shap_df


def run_downsampled(
    X_train_raw,
    X_test_raw,
    y_train,
    y_test,
    s_train,
    s_test,
    preprocessor,
    group_value="female",
    label_value=1,
    keep_frac=0.6,
    local_idx=0
):
    X_train_ds_raw, y_train_ds, s_train_ds = downsample_group(
        X_train_raw,
        y_train,
        s_train,
        group_value=group_value,
        label_value=label_value,
        keep_frac=keep_frac
    )

    X_train_ds = preprocessor.fit_transform(X_train_ds_raw)
    X_test_ds = preprocessor.transform(X_test_raw)
    feature_names = preprocessor.get_feature_names_out()

    print("=== Downsampled dataset ===")
    print(f"Group: {group_value}, label: {label_value}, keep_frac: {keep_frac}")

    model_ds = train_model(X_train_ds, y_train_ds)
    evaluate_model(model_ds, X_test_ds, y_test, s_test)

    explainer_ds, shap_values_ds, shap_df_ds = baseline_shap(
        model_ds,
        X_train_ds,
        X_test_ds,
        feature_names
    )

    plot_summary(
        shap_values_ds,
        X_test_ds,
        feature_names,
        title="SHAP summary - Downsampled dataset"
    )

    print_local_prediction(model_ds, X_test_ds, y_test, sample_idx=local_idx)
    plot_local(
        explainer_ds,
        shap_values_ds,
        X_test_ds,
        feature_names,
        sample_idx=local_idx,
        title="SHAP local explanation - Downsampled dataset"
    )

    return model_ds, explainer_ds, shap_values_ds, shap_df_ds


def run_fairness(
    X_train_raw,
    X_test_raw,
    y_train,
    y_test,
    s_train,
    s_test,
    preprocessor,
    constraint_name="equalized_odds",
    group_value="female",
    label_value=1,
    keep_frac=0.6,
    local_idx=0
):
    """
    Train/evaluate/SHAP only for fairness model
    """
    X_train_ds_raw, y_train_ds, s_train_ds = downsample_group(
        X_train_raw,
        y_train,
        s_train,
        group_value=group_value,
        label_value=label_value,
        keep_frac=keep_frac
    )

    X_train_ds = preprocessor.fit_transform(X_train_ds_raw)
    X_test_ds = preprocessor.transform(X_test_raw)
    feature_names = preprocessor.get_feature_names_out()

    print("=== Fairness model ===")
    print(f"Constraint: {constraint_name}")
    print(f"Group: {group_value}, label: {label_value}, keep_frac: {keep_frac}")

    fairness_model = train_fairness_model(
        X_train_ds,
        y_train_ds,
        s_train_ds,
        constraint_name=constraint_name
    )

    evaluate_model(fairness_model, X_test_ds, y_test, s_test)

    fairness_explainer, fairness_shap_values, fairness_shap_df, X_test_sample = fairness_shap(
        fairness_model,
        X_train_ds,
        X_test_ds,
        feature_names
    )

    plot_summary(
        fairness_shap_values,
        X_test_sample,
        feature_names,
        title=f"SHAP summary - Fairness model ({constraint_name})"
    )

    local_idx = min(local_idx, len(X_test_sample) - 1)
    print_fairness_local_prediction(fairness_model, X_test_sample, sample_idx=local_idx)
    plot_local(
        fairness_explainer,
        fairness_shap_values,
        X_test_sample,
        feature_names,
        sample_idx=local_idx,
        title=f"SHAP local explanation - Fairness model ({constraint_name})"
    )

    return fairness_model, fairness_explainer, fairness_shap_values, fairness_shap_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SHAP analysis on baseline or fairness model.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["original", "downsampled", "fairness"],
        required=True,
        help="Choose which model to analyze."
    )
    parser.add_argument(
        "--local-idx",
        type=int,
        default=0,
        help="Index of the test sample for local SHAP explanation."
    )
    parser.add_argument(
        "--constraint",
        type=str,
        choices=["demographic_parity", "equalized_odds"],
        default="equalized_odds",
        help="Constraint used for fairness mode."
    )

    args = parser.parse_args()

    DATA_PATH = "../dataset/loan_data.csv"
    sensitive_column = "person_gender"

    df = load_data(DATA_PATH)
    X, y, sensitive_features, preprocessor = prepare_data(df, sensitive_column)

    X_train_raw, X_test_raw, y_train, y_test, s_train, s_test = train_test_split(
        X,
        y,
        sensitive_features,
        test_size=0.2,
        random_state=1234,
        stratify=y
    )

    y_test = y_test.reset_index(drop=True)
    s_test = s_test.reset_index(drop=True)
    X_test_raw = X_test_raw.reset_index(drop=True)

    if args.mode == "original":
        run_original(
            X_train_raw=X_train_raw,
            X_test_raw=X_test_raw,
            y_train=y_train,
            y_test=y_test,
            s_test=s_test,
            preprocessor=preprocessor,
            local_idx=args.local_idx
        )

    elif args.mode == "downsampled":
        run_downsampled(
            X_train_raw=X_train_raw,
            X_test_raw=X_test_raw,
            y_train=y_train,
            y_test=y_test,
            s_train=s_train,
            s_test=s_test,
            preprocessor=preprocessor,
            group_value="female",
            label_value=1,
            keep_frac=0.6,
            local_idx=args.local_idx
        )

    elif args.mode == "fairness":
        run_fairness(
            X_train_raw=X_train_raw,
            X_test_raw=X_test_raw,
            y_train=y_train,
            y_test=y_test,
            s_train=s_train,
            s_test=s_test,
            preprocessor=preprocessor,
            constraint_name=args.constraint,
            group_value="female",
            label_value=1,
            keep_frac=0.6,
            local_idx=args.local_idx
        )