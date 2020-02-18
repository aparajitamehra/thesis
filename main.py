from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
# from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils.data import load_credit_scoring_data
from utils.preprocessing import preprocessing_pipeline_onehot
import numpy as np


def create_classifier(preprocessor, classifier):
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
    return clf

def rfgridsearch(classifier):


    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=11)]

    # Number of features to consider at every split
    #max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [5, 8, 15, 25, 30]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 5]
    # Method of selecting samples for training each tree
    #bootstrap = [True, False]
    # Create the random grid
    grid =      {'n_estimators': n_estimators,
                   #'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   #'bootstrap': bootstrap
                   }


    gridrf =GridSearchCV(estimator=classifier, param_grid = grid,
                                           cv=5, n_jobs=-1, verbose=1)

    #bestrf = gridrf.fit(X_train, y_train)
    return gridrf




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

    random_forest_tuned = create_classifier(
        onehot_preprocessor,

        rfgridsearch(RandomForestClassifier( random_state=42)),

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
        "RandomForestTuned" : random_forest_tuned,
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
