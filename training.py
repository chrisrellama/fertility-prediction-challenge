"""
This is an example script to train your model given the (cleaned) input dataset.

This script will not be run on the holdout data, 
but the resulting model model.joblib will be applied to the holdout data.

It is important to document your training steps here, including seed, 
number of folds, model, et cetera
"""
from xgboost import XGBClassifier
from catboost import CatBooostClassifier
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
# from sklearn.ensemble import RandomForestClassifier

def train_save_model(cleaned_df, outcome_df):
    """
    Trains a model using the cleaned dataframe and saves the model to a file.

    Parameters:
    cleaned_df (pd.DataFrame): The cleaned data from clean_df function to be used for training the model.
    outcome_df (pd.DataFrame): The data with the outcome variable (e.g., from PreFer_train_outcome.csv or PreFer_fake_outcome.csv).
    """
    
    ## This script contains a bare minimum working example
    # not useful here because logistic regression deterministic
    
    # Combine cleaned_df and outcome_df
    model_df = pd.merge(cleaned_df, outcome_df, on="nomem_encr")

    # Filter cases for whom the outcome is not available
    model_df = model_df[~model_df['new_child'].isna()]  
    
    # Logistic regression model

    # Best parameters obtained from grid search
    # best_params = {
    #     'learning_rate': 0.3,
    #     'max_depth': 7,
    #     'min_child_weight': 5,
    #     'subsample': 0.9,
    #     'n_estimators': 100
    # }

    xgb_best_params = {'booster': 'dart', 
                       'lambda': 9.537613790152782e-05, 
                       'alpha': 0.029520475365462803, 
                       'n_estimators': 169, 
                       'max_depth': 9, 
                       'learning_rate': 0.1650094305004683, 
                       'gamma': 0.9801978632408545, 
                       'colsample_bytree': 0.9114664321483119, 
                       'subsample': 0.7410766322049791, 
                       'min_child_weight': 4, 
                       'grow_policy': 'lossguide', 
                       'sample_type': 'uniform', 
                       'normalize_type': 'tree', 
                       'rate_drop': 2.094963883215845e-06, 
                       'skip_drop': 1.031056283978034e-05}

    # cb_best_params = {
    #                 'objective': 'CrossEntropy',
    #                 'colsample_bylevel': 0.059026626813230176,
    #                 'depth': 3,
    #                 'boosting_type': 'Ordered',
    #                 'bootstrap_type': 'Bernoulli',
    #                 'subsample': 0.2020107281358829}

    # lgb_best_params = {
    #                 'lambda_l1': 2.1298482220530595e-05,
    #                 'lambda_l2': 0.009111625952785245,
    #                 'num_leaves': 219,
    #                 'feature_fraction': 0.4945570495621492,
    #                 'bagging_fraction': 0.8955785257332284,
    #                 'bagging_freq': 4,
    #                 'min_child_samples': 39,
    #                 'learning_rate': 0.061717313116079406}
    
    # mlp_best_params = {
    #                 'hidden_layer_sizes': (100,),
    #                 'activation': 'logistic',
    #                 'solver': 'sgd',
    #                 'alpha': 0.0011580539894207684,
    #                 'learning_rate': 'adaptive',
    #                 'learning_rate_init': 0.022226270592867074,
    #                 'max_iter': 204,
    #                 'momentum': 0.17310540168283473,
    #                 'nesterovs_momentum': False}
    
    # lr_best_params =  {
    #                 'penalty': 'l1',
    #                 'C': 0.15023470037831235,
    #                 'solver': 'liblinear',
    #                 'max_iter': 639}

    model = XGBClassifier(xgb_best_params, random_state=1)
    # best_cat_model = CatBooostClassifier(cb_best_params, random_state=1)
    # best_lgb_model = lgb.LGBMClassifier(lgb_best_params, random_state=1)
    # best_mlp_model = MLPClassifier(mlp_best_params, random_state=1)
    # best_lr_model = LogisticRegression(lr_best_params, random_state=1)

    # model = VotingClassifier(estimators=[('xgb', best_xgb_model), 
    #                                      ('cat', best_cat_model), 
    #                                      ('lgb', best_lgb_model),
    #                                      ('mlp', best_mlp_model),
    #                                      ('lr', best_lr_model)], 
    #                                      voting='soft')

    # model = RandomForestClassifier(random_state=1)

    # Fit the model
    # model.fit(model_df[['age']], model_df['new_child'])

    X = model_df.drop(columns=['new_child'], axis=1)

    y = model_df['new_child']

    model.fit(X, y)
    joblib.dump(model, "model.joblib")

    
