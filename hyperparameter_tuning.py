import optuna
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

import xgboost as xgb

import numpy as np
import shutil
import os


STATIC_PARAMS = {
    "silent": 1,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "booster": "gbtree",
}

N_STEPS = 300  # The number of training iterations
EARLY_STOPPING_ROUNDS = 30

N_TRIALS = 100  # optimization trials
N_FOLDS = 5  # Number of folds for CV


def simple_tuner(X, y):

    def train_evaluate(X, y, all_params):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=4)
        # i'll be using under the hood api of xgboost :)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_val, label=y_val)

        evals = [(dtest, "validation")]
        model = xgb.train(all_params, dtrain, num_boost_round=N_STEPS, evals=evals,
                          early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose_eval=False)
        preds = model.predict(dtest)

        score = roc_auc_score(y_val, preds)
        return score

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'eta': trial.suggest_loguniform("eta", 1e-3, 1.0),  # learning rate
            'gamma': trial.suggest_loguniform("gamma", 1e-8, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            #               'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),  # L1 reg Lasso
            'scale_pos_weight': trial.suggest_int('scale_pos_weight', 1, 90),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0)
        }

        all_params = {**params, **STATIC_PARAMS}
        return train_evaluate(X, y, all_params)


    # create the study object to set the direction of maximisation
    study = optuna.create_study(direction='maximize')  # maximize auc
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=12)

    # Run statistics
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def cv_tuner(X, y):

    def train_evaluate(X, y, params):

        xgb_model = xgb.XGBClassifier(**params)
        # define evaluation
        cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=3, random_state=1)
        # evaluate model
        scores = cross_val_score(xgb_model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        best_score = np.mean(scores)

        return best_score

    def objective(trial):

        params = {
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'eta': trial.suggest_loguniform("eta", 1e-3, 1.0),  # learning rate
            'gamma': trial.suggest_loguniform("gamma", 1e-8, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            #               'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),  # L1 reg Lasso
            'scale_pos_weight': trial.suggest_int('scale_pos_weight', 1, 90),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0)
        }

        all_params = {**params, **STATIC_PARAMS}
        return train_evaluate(X, y, all_params)

    # create the study object to set the direction of maximisation
    study = optuna.create_study(direction='maximize')  # maximize auc
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=12)

    # Run statistics
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
