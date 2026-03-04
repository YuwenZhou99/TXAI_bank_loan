from src.dataset.dataset import load_data, prepare_data, downsample_group
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference


def train_model(X_train, y_train, preprocessor):
    model = Pipeline(
        steps=[
            ("preprosessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, sensitive_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(
        y_test,
        y_pred,
        target_names=["not loan", "loan"],
        zero_division=0
    )

    auc = roc_auc_score(y_test, y_prob)
    # Fairness evaluation
    dp = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_test)
    eo = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_test)

    print(f"Classification report: {report}")
    print(f"AUC: {auc:.4f}")
    print(f"Demographic Parity Difference: {dp:.4f}")
    print(f"Equalized Odds Difference: {eo:.4f}")

    return report, auc, dp, eo


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

    model = train_model(X_train, y_train, preprocessor)

    results = evaluate_model(model, X_test, y_test, s_test)
