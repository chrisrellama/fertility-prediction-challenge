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
                        'lambda': 0.0002450003592351577,
                        'alpha': 9.814710792341585e-07,
                        'n_estimators': 173,
                        'max_depth': 7,
                        'learning_rate': 0.048383643801108184,
                        'gamma': 0.8737083525973144,
                        'colsample_bytree': 0.5175195766417736,
                        'subsample': 0.7902267651046265,
                        'min_child_weight': 4,
                        'grow_policy': 'depthwise',
                        'sample_type': 'uniform',
                        'normalize_type': 'tree',
                        'rate_drop': 0.009471802325413732,
                        'skip_drop': 0.004004496225724098}
    
    lgb_best_params = {'lambda_l1': 0.10711895367497702,
                        'lambda_l2': 5.991056048374842e-05,
                        'num_leaves': 237,
                        'feature_fraction': 0.8514738846025549,
                        'bagging_fraction': 0.7924121221778662,
                        'bagging_freq': 7,
                        'min_child_samples': 67,
                        'learning_rate': 0.09898408693688884}

    best_xgb_model = XGBClassifier(**xgb_best_params, random_state=1)
    best_lgb_model = lgb.LGBMClassifier(**lgb_best_params, random_state=1, verbose=-1)

    model = VotingClassifier(estimators=[('xgb', best_xgb_model),  
                                         ('lgb', best_lgb_model)], 
                                         voting='soft')

    # model = RandomForestClassifier(random_state=1)

    # Fit the model
    # model.fit(model_df[['age']], model_df['new_child'])

    X = model_df.drop(columns=['new_child'], axis=1)

    y = model_df['new_child']

    model.fit(X, y)
    joblib.dump(model, "model.joblib")

    
