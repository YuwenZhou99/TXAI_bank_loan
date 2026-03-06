from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.dataset.dataset import load_data, prepare_data, downsample_group
from src.models.baseline import evaluate_model


def train_fairness_model(X_train, y_train, s_train, constraint_name="demographic_parity"):
    """
    Train a fairness-aware model using reduction method.
    constraint_name:
        - "demographic_parity"
        - "equalized_odds"
    """

    base_estimater = LogisticRegression(max_iter=1000)

    if constraint_name == "demographic_parity":
        constraint = DemographicParity()
    elif constraint_name == "equalized_odds":
        constraint = EqualizedOdds()
    else:
        raise ValueError("Constraint name must be 'demographic_parity' or 'equalized_odds'")

    mitigator = ExponentiatedGradient(
        estimator=base_estimater,
        constraints=constraint
    )

    mitigator.fit(X_train, y_train, sensitive_features=s_train)
    return mitigator


if __name__ == "__main__":
    DATA_PATH = "../dataset/loan_data.csv"
    sensitive_column = "person_gender"

    df = load_data(DATA_PATH)
    X, y, sensitive_features, preprocessor = prepare_data(df, sensitive_column)

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X,
        y,
        sensitive_features,
        test_size=0.2,
        random_state=1234,
        stratify=y
    )

    # downsampling
    X_train, y_train, s_train = downsample_group(
        X_train, y_train, s_train,
        group_value="female",
        label_value=1,
        keep_frac=0.6
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    fairness_model_dp = train_fairness_model(
        X_train, y_train, s_train,
        constraint_name="demographic_parity"
    )

    fairness_results_dp = evaluate_model(
        fairness_model_dp,
        X_test, y_test, s_test
    )

    fairness_model_eo = train_fairness_model(
        X_train, y_train, s_train,
        constraint_name="equalized_odds"
    )

    fairness_results_eo = evaluate_model(
        fairness_model_eo,
        X_test, y_test, s_test
    )
