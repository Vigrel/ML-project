from urllib.parse import urlparse

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_predict
import datetime


from pipeline import LogAtt

RANDOM_SEED = 42

# TODO: perguntar do cross val com train

def objective(trial):
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

    param = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
        "criterion": trial.suggest_categorical("criterion", ['gini', 'entropy']),
        "max_depth": trial.suggest_int("max_depth", 2, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10)
    }
    #Get datetime string:
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H:%M:%S")

    with mlflow.start_run(run_name=f"trial_{trial.number}_{now_str}"):
        rf = RandomForestClassifier(**param)

        rf.fit(X_train_pp, Y_train_pp)

        preds = rf.predict(X_test_pp)
        pred_labels = np.rint(preds)
        accuracy = accuracy_score(Y_test_pp, pred_labels)
        # Log the hyperparameters and the metrics
        mlflow.log_param("n_estimators", param["n_estimators"])
        mlflow.log_param("criterion", param["criterion"])
        mlflow.log_param("max_depth", param["max_depth"])
        mlflow.log_param("min_samples_split", param["min_samples_split"])
        mlflow.log_param("min_samples_leaf", param["min_samples_leaf"])
        mlflow.log_metric("accuracy", accuracy)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
     

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(rf, "model", registered_model_name="RandomForestClassifierWineModel")
        else:
            mlflow.sklearn.log_model(rf, "model")

    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
