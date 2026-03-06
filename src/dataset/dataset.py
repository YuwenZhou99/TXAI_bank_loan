import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_data(path):
    df = pd.read_csv(path)
    return df


def prepare_data(df, sensitive_column):
    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    sensitive_features = df[sensitive_column]

    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first"), categorical_cols)
        ]
    )

    return X, y, sensitive_features, preprocessor


def downsample_group(X, y, s, group_value, label_value = 1, keep_frac=0.6, seed=1234):
    rng = np.random.RandomState(seed)

    # Convert to pandas for easy masking
    Xp = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    yp = y if isinstance(y, pd.Series) else pd.Series(y)
    sp = s if isinstance(s, pd.Series) else pd.Series(s)

    mask_target = (sp == group_value) & (yp == label_value)
    idx_target = yp[mask_target].index.to_numpy()
    idx_other = yp[~mask_target].index.to_numpy()

    n_keep = int(np.ceil(len(idx_target) * keep_frac))
    idx_keep_target = rng.choice(idx_target, size=n_keep, replace=False)

    idx_keep = np.concatenate([idx_other, idx_keep_target])
    rng.shuffle(idx_keep)

    X_ds = Xp.loc[idx_keep]
    y_ds = yp.loc[idx_keep]
    s_ds = sp.loc[idx_keep]

    return X_ds, y_ds, s_ds
