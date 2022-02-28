"""
Optuna optimizes a classifier configuration using XGBoost.
Optimize the validation accuracy of positive class using XGBoost.
Function optimizes both the choice of booster model and its hyperparameters.
"""

import numpy as np
import optuna

import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb

def objective(trial):
    
    # Create XGBoost DMatrix
    dtrain = xgb.DMatrix(data, label=target)
    
    # General XGBoost Hyperparameters That Apply to All Boosters
    param = {
        "silent"          : 1,
        "verbosity"       : 0,
        "objective"       : "binary:logistic", # Objective: Binary Classification 
        "disable_default_eval_metric": 1,
        #"scale_pos_weight": some numeric, # Imbalanced data (if predicting the correct class is most important) <---
        "max_delta_step"  : trial.suggest_float("max_delta_step", 1, 10.0), # Imbalanced data (if predicting the right probability is most important) <---
        "eval_metric"     : "logloss", # Available eval metrics: 'auc', 'logloss', 'error', 'aucpr', map' 
        "tree_method"     : trial.suggest_categorical("tree_method",["approx", "hist"]), # Different methods for small dataset
        "booster"         : trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]), # Defines booster
        "lambda"          : trial.suggest_float("lambda", 1e-8, 1.0, log=True), # L2 regularization weight
        "alpha"           : trial.suggest_float("alpha", 1e-8, 1.0, log=True), # L1 regularization weight
        "subsample"       : trial.suggest_float("subsample", 0.5, 1.0), # Sampling ratio for training data **LOWER RATIOS HELP AVOID OVER-FITTING**
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0), # Sampling according to each tree. **LOWER RATIOS HELP AVOID OVER-FITTING**
    }

    # Hyperparameters That Are Specific to GBTREE and DART Boosters
    if param["booster"] in ["gbtree", "dart"]:
        param["max_depth"]        = trial.suggest_int("max_depth", 1, 5, step=1) # Max tree depth, lower values help avoid overfitting
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 5, 10) # Min child weight, higher values help avoid overfitting
        param["eta"]              = trial.suggest_float("eta", 1e-8, .5, log=True) # learning rate, lower values help avoid overfitting
        param["gamma"]            = trial.suggest_float("gamma", 1.0, 2.0, log=True) # Lagrangian multiplier, higher values help avoid overfitting
        param["grow_policy"]      = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    # Hyperparameters That Are Specific to Only DART
    if param["booster"] == "dart":
        param["sample_type"]    = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"]      = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"]      = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
    
    # Select the Metric to Prune With
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-logloss")
    
    # Cross-Validation
    history = xgb.cv(
        params                = param,
        dtrain                = dtrain,
        num_boost_round       = 100, # So this is one of the biggest, most misunderstood params for xgboost because outside of CV it is set with n_estimators like with sklearn!
        nfold                 = 3,
        metrics               = {'logloss'},
        shuffle               = True,
        stratified            = True,
        early_stopping_rounds = 10,
        seed                  = 111,
        verbose_eval          = False,
    )
    
    # Extract the best score
    eval_metric = history["test-logloss-mean"].values[-1]
    
    # Print n_estimators in the output at each call to the objective function
    print('-'*10, 'Trial {} has optimal trees: {}'.format(trial.number, str(history.shape[0])), '-'*10)
    
    # Set n_estimators as a trial attribute; Accessible via study.trials_dataframe()
    trial.set_user_attr('n_estimators', len(history))
    
    return eval_metric

if __name__ == "__main__":
    
    # Next step, fully integrate kurobako-py to fully leverage solvers and problems
    
    # Pruners
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10) # Median Pruner
    #puner  = optuna.pruners.PercentilePruner(25.0, n_startup_trials=5, n_warmup_steps=30, interval_steps=10) # Percentile Pruner
    #pruner = optuna.pruners.SuccessiveHalvingPruner() 
    #pruner = optuna.pruners.HyperbandPruner(min_resource=1, reduction_factor=3)
    
    # Samplers
    sampler = optuna.samplers.RandomSampler() # Random Sampler
    #sampler = optuna.samplers.TPESampler() # Tree-structured Parzen Estimator Sampler
    #sampler = optuna.samplers.CmaEsSampler() # CMA Sampler
    #sampler = optuna.samplers.NSGAIISampler() # Nondominated Sorting Genetic Algorithm II Sampler
    #sampler = optuna.samplers.MOTPESampler() # Multi-Objective Tree-structured Parzen Estimator Sampler
    #sampler = optuna.samplers.IntersectionSearchSpace(include_pruned=False) # Intersection Search Space Sampler
    #sampler = optuna.samplers.intersection_search_space() # Intersection Search Space Sampler II
    
    # Studies
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction='minimize')
    
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
