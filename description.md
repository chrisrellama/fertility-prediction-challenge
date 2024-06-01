# Description of submission
Submission 1 
- used XGBoost for feature importance selection to reduce computational cost
- employed appropriate imputation techniques: mode imputation for categorical features and mean imputation for numerical features

Submission 2
- adopted mode imputation to handle missing values across all data types
- expanded the model ensemble by incorporating CatBoost and LightGBM alongside XGBoost
- utilized Optuna for efficient hyperparameter optimization
- leveraged VotingClassifier to combine the predictions from the three models (XGBoost, CatBoost, LightGBM) using soft voting (majority rule based on predicted probabilities)
