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
                       'lambda': 0.00434039625174651,
                       'alpha': 0.00029313510747824243,
                       'n_estimators': 161,
                       'max_depth': 7,
                       'learning_rate': 0.018533335439747604,
                       'gamma': 0.6347219755334971,
                       'colsample_bytree': 0.5267376565133665,
                       'subsample': 0.6764250274627918,
                       'min_child_weight': 1,
                       'grow_policy': 'depthwise'}

    cb_best_params = {'objective': 'CrossEntropy',
                      'colsample_bylevel': 0.012811022397622869,
                      'depth': 10,
                      'boosting_type': 'Ordered',
                      'bootstrap_type': 'MVS'}

    lgb_best_params = {'lambda_l1': 0.0010324197101867663,
                       'lambda_l2': 7.69031152321403e-08,
                       'num_leaves': 199,
                       'feature_fraction': 0.6367914164169993,
                       'bagging_fraction': 0.8642109943467728,
                       'bagging_freq': 1,
                       'min_child_samples': 7}

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

    
