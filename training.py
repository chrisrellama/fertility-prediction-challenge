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

    xgb_best_params = {'booster': 'gbtree',
                       'lambda': 9.778009956323202e-08,
                       'alpha': 3.431986541390118e-08,
                       'n_estimators': 64,
                       'max_depth': 7,
                       'learning_rate': 0.042977256477361954,
                       'gamma': 0.42608626647544,
                       'colsample_bytree': 0.8616503420589152,
                       'subsample': 0.5357180066386461}

    cb_best_params = {'objective': 'Logloss',
                      'colsample_bylevel': 0.08903806045133904,
                      'depth': 9,
                      'boosting_type': 'Ordered',
                      'bootstrap_type': 'MVS'}

    lgb_best_params = {'lambda_l1': 0.001361227520152072,
                       'lambda_l2': 3.5829946241097575e-06,
                       'num_leaves': 242,
                       'feature_fraction': 0.7419538370655974,
                       'bagging_fraction': 0.6487653994303965,
                       'bagging_freq': 6,
                       'min_child_samples': 33}

    best_xgb_model = XGBClassifier(xgb_best_params, random_state=1)
    best_cat_model = CatBooostClassifier(cb_best_params, random_state=1)
    best_lgb_model = lgb.LGBMClassifier(lgb_best_params, random_state=1)

    # X_train = model_df.drop(columns=['new_child'], axis=1)
    # y_train = model_df['new_child']

    # best_xgb_model.fit(X_train, y_train)
    # best_cat_model.fit(X_train, y_train)
    # best_lgb_model.fit(X_train, y_train)

    model = VotingClassifier(estimators=[('xgb', best_xgb_model), ('cat', best_cat_model), ('lgb', best_lgb_model)], voting='soft')

    # model = RandomForestClassifier(random_state=1)

    # Fit the model
    # model.fit(model_df[['age']], model_df['new_child'])

    X = model_df.drop(columns=['new_child'], axis=1)

    y = model_df['new_child']

    model.fit(X, y)
    joblib.dump(model, "model.joblib")

    
