import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression


def print_evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')


def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square


def predict_price(model, X):
    y_preds = model.predict(X)
    df = pd.read_csv('output/results.csv', index_col=0)
    for index, y_pred in enumerate(list(y_preds)):
        inputs_list = X.loc[index].to_list()
        df.loc[len(df)] = inputs_list + [y_pred]
    df.to_csv('output/results.csv')
    return y_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputs", nargs='+', type=float,
            help="inputs housing data")
    parser.add_argument("-c", "--csv-file", type=str,
            help="csv file with housing data")
    parser.add_argument("-m", "--model", type=str, default="linear",
            help="csv file with housing datatype of the regression model")
    args = parser.parse_args()

    USAhousing = pd.read_csv('input/USA_Housing.csv')

    X = USAhousing[[
            'Avg. Area Income',
            'Avg. Area House Age',
            'Avg. Area Number of Rooms',
            'Avg. Area Number of Bedrooms',
            'Area Population'
    ]]
    y = USAhousing['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if args.model == 'linear':
        model = make_pipeline(StandardScaler(with_mean=False), LinearRegression())
    elif args.model == 'polynomial':
        model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print('Train set evaluation:\n_____________________________________')
    print_evaluate(y_train, train_pred)
    print('Test set evaluation:\n_____________________________________')
    print_evaluate(y_test, test_pred)

    metrics_df = pd.read_csv('output/metrics.csv', index_col=0)
    metrics_df.loc[len(metrics_df.index)] = [
        args.model,
        *evaluate(y_test, test_pred)
    ]
    metrics_df.to_csv('output/metrics.csv')

    if args.inputs:
        inputs_df = pd.DataFrame([args.inputs], columns=[
            'Avg. Area Income',
            'Avg. Area House Age',
            'Avg. Area Number of Rooms',
            'Avg. Area Number of Bedrooms',
            'Area Population'
        ])
        price = predict_price(model, inputs_df)

        print('_____________________________________\n')
        print(f"Predicted price: {price}")
    elif args.csv_file:
        inputs_df = pd.read_csv('input/inputs.csv')
        price = predict_price(model, inputs_df)

        print('_____________________________________\n')
        print('Done.')


if __name__ == '__main__':
    main()
