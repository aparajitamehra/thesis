from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from utils.data import load_credit_scoring_data
from utils.preprocessing import HighVIFDropper


def create_preprocessing_pipeline(data):
    numeric_features = data.select_dtypes("number").columns
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("vif_dropper", HighVIFDropper(threshold=10)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_features = data.select_dtypes("category").columns
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def create_classifier(preprocessor, classifier):
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])

    return clf


def main():
    data = load_credit_scoring_data(
        "datasets/UK/input.txt", "datasets/UK/descriptor.csv"
    )

    y = data.pop("Censor")
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    preprocessor = create_preprocessing_pipeline(X)

    log_reg = create_classifier(
        preprocessor, LogisticRegressionCV(cv=5, max_iter=1000, class_weight="balanced")
    )

    random_forest = create_classifier(
        preprocessor,
        RandomForestClassifier(
            random_state=42,
            n_estimators=500,
            n_jobs=-1,
            oob_score=True,
            class_weight="balanced",
            max_depth=5,
        ),
    )

    adaboost = create_classifier(
        preprocessor,
        AdaBoostClassifier(
            RandomForestClassifier(
                random_state=42,
                n_estimators=500,
                n_jobs=-1,
                oob_score=True,
                class_weight="balanced",
                max_depth=5,
            ),
            n_estimators=10,
        ),
    )

    classifiers = {
        "LogisticRegressionCV": log_reg,
        "RandomForestClassifier": random_forest,
        "AdaBoostClassifier": adaboost,
    }

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        print(f"{name} score: {clf.score(X_test, y_test)}")


if __name__ == "__main__":
    main()
