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

    xgb_best_params = {'booster': 'gbtree',
                        'lambda': 0.25403323366892516,
                        'alpha': 5.9851374260924525e-05,
                        'n_estimators': 117,
                        'max_depth': 7,
                        'learning_rate': 0.07172736686641923,
                        'gamma': 0.035832922662134614,
                        'colsample_bytree': 0.702966330702086,
                        'subsample': 0.7942272197561732,
                        'min_child_weight': 5,
                        'grow_policy': 'lossguide'}

    cat_best_params = {'objective': 'CrossEntropy',
                        'colsample_bylevel': 0.055407633055084025,
                        'depth': 2,
                        'boosting_type': 'Plain',
                        'bootstrap_type': 'MVS'}
    
    lgb_best_params = {'lambda_l1': 0.022519482464091294,
                        'lambda_l2': 1.7855738045848233e-05,
                        'num_leaves': 82,
                        'feature_fraction': 0.7419316873030533,
                        'bagging_fraction': 0.9826701473496973,
                        'bagging_freq': 6,
                        'min_child_samples': 39,
                        'learning_rate': 0.06273800912224269}

    best_xgb_model = XGBClassifier(**xgb_best_params, random_state=1)
    best_cat_model = CatBoostClassifier(**cat_best_params, random_state=1, verbose=0)
    best_lgb_model = lgb.LGBMClassifier(**lgb_best_params, random_state=1, verbose=-1)

    model = VotingClassifier(estimators=[('xgb', best_xgb_model),
                                         ('cat', best_cat_model),  
                                         ('lgb', best_lgb_model)], 
                                         voting='soft')

    # model = RandomForestClassifier(random_state=1)

    # Fit the model
    # model.fit(model_df[['age']], model_df['new_child'])

    X = model_df.drop(columns=['new_child'], axis=1)

    y = model_df['new_child']

    model.fit(X, y)
    joblib.dump(model, "model.joblib")

    
