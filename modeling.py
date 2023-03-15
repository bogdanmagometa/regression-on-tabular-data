"""
modeling.py

1. Train the regresion on train data from TRAIN_DATASET_PATH
2. Predict target variable from test data from TEST_DATASET_PATH
3. Write predicted values of target variable into csv file with name TARGET_OUTPUT_PATH
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

TRAIN_DATASET_PATH = 'internship_train.csv'
TEST_DATASET_PATH = 'internship_hidden_test.csv'
TARGET_OUTPUT_PATH = "predictions.csv"

def calc_rmse(x1, x2):
    return np.linalg.norm(x1 - x2) / x1.shape[0]**0.5

def main():
    train_df = pd.read_csv(TRAIN_DATASET_PATH)

    # Train on train dataset and print achieved RMSE
    y = train_df['target']
    train_df['53'] = train_df['6']**2
    x = sm.add_constant(train_df[['7', '53']])
    model = sm.OLS(y, x).fit()
    train_rmse = calc_rmse(x @ model.params, y)
    print(f"RMSE on train data: {train_rmse:.3g}")

    # Predicton on internship_hidden_test.csv
    test_df = pd.read_csv(TEST_DATASET_PATH)
    test_df['53'] = test_df['6']**2
    x_test = sm.add_constant(test_df[['7', '53']])
    predicted_target = x_test @ model.params
    predicted_target = pd.DataFrame({'target': predicted_target})
    predicted_target.to_csv(TARGET_OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()
