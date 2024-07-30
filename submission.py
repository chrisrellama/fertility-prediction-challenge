"""
This is an example script to generate the outcome variable given the input dataset.

This script should be modified to prepare your own submission that predicts 
the outcome for the benchmark challenge by changing the clean_df and predict_outcomes function.

The predict_outcomes function takes a Pandas data frame. The return value must
be a data frame with two columns: nomem_encr and outcome. The nomem_encr column
should contain the nomem_encr column from the input data frame. The outcome
column should contain the predicted outcome for each nomem_encr. The outcome
should be 0 (no child) or 1 (having a child).

clean_df should be used to clean (preprocess) the data.

run.py can be used to test your submission.
"""

# List your libraries and modules here. Don't forget to update environment.yml!
import pandas as pd
# from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json

def scaler(df):

    scaler = StandardScaler()

    numeric = df.select_dtypes(include=['int', 'float64'])
    df[numeric.columns] = scaler.fit_transform(numeric)

    return df


def clean_df(df, background_df=None):
    """
    Preprocess the input dataframe to feed the model.
    # If no cleaning is done (e.g. if all the cleaning is done in a pipeline) leave only the "return df" command

    Parameters:
    df (pd.DataFrame): The input dataframe containing the raw data (e.g., from PreFer_train_data.csv or PreFer_fake_data.csv).
    background (pd.DataFrame): Optional input dataframe containing background data (e.g., from PreFer_train_background_data.csv or PreFer_fake_background_data.csv).

    Returns:
    pd.DataFrame: The cleaned dataframe with only the necessary columns and processed variables.
    """

    ## This script contains a bare minimum working example
    # Create new variable with age
    # df["age"] = 2024 - df["birthyear_bg"]

    # with open('settings.json') as f:
    #     settings = json.load(f)

    # Use file paths from settings
    # file_path = settings['file_paths']['index']

    file_path = "index_166.txt"

    loaded_feature_names = []

    with open(file_path, "r") as file:
        for line in file:
            loaded_feature_names.append(line.strip())

    df = df.copy()

    df = df[loaded_feature_names]

    df.fillna({'cf17j029': 1988.0}, inplace=True)

    # outliers
    df['cf20m130'] = df['cf20m130'].replace(2025, 1)
    df['cs20m172'] = df['cs20m172'].replace(99, 8)
    df['cs20m169'] = df['cs20m169'].replace(99, 8)
    df['cs20m165'] = df['cs20m165'].replace(99, 8)
    df['cs20m500'] = df['cs20m500'].replace(99, 8)
    df['cs20m182'] = df['cs20m182'].replace(99, 8)
    
    df.drop('ci20m326', axis=1, inplace=True) 
    df.drop('brutohh_f_2020', axis=1, inplace=True)

    df.fillna({'cw17j033': 8.0}, inplace=True)
    df.fillna({'ch19l006': 6.0}, inplace=True)
    df.fillna({'cv19k230': 5.0}, inplace=True)  

    df.drop('outcome_available', axis=1, inplace=True)

    ave_col = ['ch19l259', 'cs19l439', 'cp17i190', 'cw17j384', 'cv18j303', 'cd20m082', 'cw17j383',
             'nettoink_2020', 'cf17j397', 'cr18k120', 'ca20g075', 'nettohh_f_2020', 'brutohh_f_2019', 
             'cv19k302', 'cp18j193', 'cf20m397', 'cs20m415']
    
    for col in ave_col:
        df[col].fillna(df[col].mean(), inplace=True)

    for col in df.select_dtypes(include=['float64', 'int64', 'object']).columns:
        mode_series = df[col].mode(dropna=True)
        if not mode_series.empty:
            mode_value = mode_series[0]
        else:
            mode_value = np.nan

        if pd.isna(mode_value):
            non_na_values = df[col].dropna()
            if not non_na_values.empty:
                mode_value = non_na_values.mode()[0]
                
        df[col] = df[col].fillna(mode_value)

    id_column = df['nomem_encr']

    features = df.drop('nomem_encr', axis=1)

    scaled_df = scaler(features)

    df = pd.concat([id_column, scaled_df], axis=1)

    return df


def predict_outcomes(df, background_df=None, model_path="model.joblib"):
    """Generate predictions using the saved model and the input dataframe.

    The predict_outcomes function accepts a Pandas DataFrame as an argument
    and returns a new DataFrame with two columns: nomem_encr and
    prediction. The nomem_encr column in the new DataFrame replicates the
    corresponding column from the input DataFrame. The prediction
    column contains predictions for each corresponding nomem_encr. Each
    prediction is represented as a binary value: '0' indicates that the
    individual did not have a child during 2021-2023, while '1' implies that
    they did.

    Parameters:
    df (pd.DataFrame): The input dataframe for which predictions are to be made.
    background_df (pd.DataFrame): The background dataframe for which predictions are to be made.
    model_path (str): The path to the saved model file (which is the output of training.py).

    Returns:
    pd.DataFrame: A dataframe containing the identifiers and their corresponding predictions.
    """

    ## This script contains a bare minimum working example
    if "nomem_encr" not in df.columns:
        print("The identifier variable 'nomem_encr' should be in the dataset")

    # Load the model
    model = joblib.load(model_path)

    # Preprocess the fake / holdout data
    df = clean_df(df, background_df)

    # df = df[model.feature_names_in_]

    # Exclude the variable nomem_encr if this variable is NOT in your model
    # vars_in_model = model.feature_names_in_.shape[0]
    # vars_without_id = df.columns[df.columns != 'nomem_encr'][:vars_in_model]

    # vars_without_id = [col for col in df.columns if col != 'nomem_encr' and col in model.feature_names_in_]
    
    # vars_without_id = df.columns[df.columns != 'nomem_encr']
    vars_without_id = [col for col in df.columns if col != 'nomem_encr' and col in model.get_booster().feature_names]
    predictions = model.predict(df[vars_without_id])

    # Generate predictions from model, should be 0 (no child) or 1 (had child)
    # predictions = model.predict(df[vars_without_id])

    # Output file should be DataFrame with two columns, nomem_encr and predictions
    df_predict = pd.DataFrame(
        {"nomem_encr": df["nomem_encr"], "prediction": predictions}
    )

    # Return only dataset with predictions and identifier
    return df_predict
