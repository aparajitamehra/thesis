from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

# from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils.data import load_credit_scoring_data
from utils.preprocessing import preprocessing_pipeline_onehot


def create_classifier(preprocessor, classifier):
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
    return clf


def main(data_path, descriptor_path):
    data = load_credit_scoring_data(data_path, descriptor_path)

    y = data.pop("censor")
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    onehot_preprocessor = preprocessing_pipeline_onehot(X)
    # emb_preprocessor = preprocessing_pipeline_embedding(data)

    # log_reg = create_classifier(
    #     emb_preprocessor,
    #     LogisticRegressionCV(cv=5,
    #     max_iter=1000,
    #     class_weight="balanced",
    #     ),
    # )

    random_forest = create_classifier(
        onehot_preprocessor,
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
        onehot_preprocessor,
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
        # "LogisticRegressionCV": log_reg,
        "RandomForestClassifier": random_forest,
        "AdaBoostClassifier": adaboost,
    }

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        print(f"{name} score: {clf.score(X_test, y_test)}")


if __name__ == "__main__":
    for ds_name in ["UK", "bene1", "bene2", "german"]:
        print(ds_name)
        main(
            f"datasets/{ds_name}/input_{ds_name}.csv",
            f"datasets/{ds_name}/descriptor_{ds_name}.csv",
        )
