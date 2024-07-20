# Description of submission
Submission 1 
- used XGBoost for feature importance selection to reduce computational cost
- employed appropriate imputation techniques: mode imputation for categorical features and mean imputation for numerical features

Submission 2
- adopted mode imputation to handle missing values across all data types
- expanded the model ensemble by incorporating CatBoost and LightGBM alongside XGBoost
- utilized Optuna for efficient hyperparameter optimization
- leveraged VotingClassifier to combine the predictions from the three models (XGBoost, CatBoost, LightGBM) using soft voting (majority rule based on predicted probabilities)

Submission 3
- adopted mode imputation to handle missing values across all data types
- expanded the model ensemble by incorporating CatBoost,LightGBM, MLPCLassifier and LogisticRegrression alongside XGBoost
- utilized Optuna and StratifiedKFold for efficient hyperparameter optimization
- leveraged VotingClassifier to combine the predictions from the five models (XGBoost, CatBoost, LightGBM, MLPClassifier, LogisticRegression) using soft voting (majority rule based on predicted probabilities)

Submission 3.1
- found some features to fill nans with specific value isntead of mode value
- drop some features based on variance inflation factor
- found that using XGBoost only gives the highest score in train and validation data