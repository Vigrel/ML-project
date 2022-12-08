import sys
from urllib.parse import urlparse

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_predict

from pipeline import LogAtt

RANDOM_SEED = 42


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)

    # Read the dataset from the csv file
    wine = pd.read_csv("database0001.csv")

    #drop one column by name
    wine.drop('Id', axis=1, inplace=True)

    # Stratified Split of the dataset into train and test
    split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )

    for train_index, test_index in split.split(wine, wine["quality"]):
        train_set = wine.loc[train_index]
        test_set = wine.loc[test_index]

    # Get column indices for the attributes that will be passed to the pipeline
    tsd, chl, rs, sp = [
    wine.columns.get_loc(i)
    for i in ["total sulfur dioxide", "chlorides", "residual sugar", "sulphates"]
    ]
    
    # Create a pipeline object from the LogAtt class (pipeline.py)
    pipe = LogAtt(tsd, chl, rs, sp)


    # Fit the pipeline to the training set 
    X_train_pp, Y_train_pp = pipe.transform(train_set.values)
    X_test_pp, Y_test_pp = pipe.transform(test_set.values)

    # Set initial values for the hyperparameters
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    criterion = sys.argv[2] if len(sys.argv) > 2 else "gini"
    max_depth = int(sys.argv[3]) if len(sys.argv) > 3 else None
    min_samples_split = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    min_samples_leaf = int(sys.argv[5]) if len(sys.argv) > 5 else 1

    with mlflow.start_run():
        rf_clf = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=RANDOM_SEED,
        )
        rf_clf.fit(X_train_pp, Y_train_pp)

        rf_clf_pred = cross_val_predict(rf_clf, X_train_pp, Y_train_pp, cv=5)

        acc = accuracy_score(Y_train_pp, rf_clf_pred)

        print(f"""Random Forest Classifier (
            n_estimators={n_estimators}, 
            criterion={criterion}, 
            max_depth={max_depth}, 
            min_samples_split={min_samples_split}, 
            min_samples_leaf={min_samples_leaf},
            ):

            Accuracy: {acc}
        """)

        # Log the hyperparameters and the metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("criterion", criterion)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_metric("accuracy", acc)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
     

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(rf_clf, "model", registered_model_name="RandomForestClassifierWineModel")
        else:
            mlflow.sklearn.log_model(rf_clf, "model")
