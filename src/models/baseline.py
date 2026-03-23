from src.dataset.dataset import load_data, prepare_data, downsample_group
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test, sensitive_test):
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["not loan", "loan"],
        zero_division=0
    )

    # Fairness evaluation
    dp = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_test)
    eo = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_test)

    print(f"Classification report: {report}")
    print(f"Demographic Parity Difference: {dp:.4f}")
    print(f"Equalized Odds Difference: {eo:.4f}")

    return report, dp, eo


if __name__ == "__main__":
    DATA_PATH = "src/dataset/loan_data.csv"
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

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    print("Original dataset:")
    model = train_model(X_train, y_train)
    results = evaluate_model(model, X_test, y_test, s_test)

    # downsampling
    X_train_ds, y_train_ds, s_train_ds = downsample_group(
        X_train_raw.copy(), y_train.copy(), s_train.copy(),
        group_value="female",
        label_value=1,
        keep_frac=0.6
    )

    X_train_ds = preprocessor.fit_transform(X_train_ds)
    X_test_ds = preprocessor.transform(X_test_raw)

    print("After downsampling:")
    model = train_model(X_train_ds, y_train_ds)
    results = evaluate_model(model, X_test_ds, y_test, s_test)
