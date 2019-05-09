import numpy as np
import pandas as pd
from sklearn.svm import SVR
import pandas_datareader.data as web
import datetime as dt
from datetime import timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from pymongo import MongoClient
import csv
pd.plotting.register_matplotlib_converters(explicit=True)

def TrainTestModel(etf_fund, start_date, end_date):
    df_stocks = web.DataReader(etf_fund, 'yahoo', start_date, end_date)  # Data retrieval
    df_stocks.reset_index(inplace=True)  # Data cleansing from this point
    df_stock_price = df_stocks[['Date', 'Close']].copy()
    #size = len(df_stock_price.index)
    df_stock_price['DateFloat'] = df_stock_price['Date'].values.astype(float)

    X = df_stock_price['DateFloat'].values.reshape(-1, 1)  # Data normalization and splitting training and test data
    Y = df_stock_price['Close'].values.reshape(-1, 1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=100, test_size=0.25)

    model = LinearRegression()  # Instantiate Linear Regression model
    model.fit(X_train, Y_train)

    predicted_values = model.predict(X_test)  # Value prediction for test data based on model training
    r2 = r2_score(Y_test, predicted_values)
    mse = mean_squared_error(Y_test, predicted_values)
    intercept = model.intercept_  # Intercept and coefficient for future value
    coefficient = model.coef_

    X_test_date = [x[0] for x in X_test]
    pd.to_datetime(X_test_date)
    [time for time in pd.to_datetime(X_test_date)]  # Check this part where being used
    X_test_date_float = [time.date() for time in pd.to_datetime(X_test_date)]

    df_date_pred_actual_value = pd.DataFrame({"Date": X_test_date_float})
    df_date_pred_actual_value["PredictedPrice"] = predicted_values
    df_date_pred_actual_value["ActualPrice"] = Y_test
    df_date_pred_actual_value.sort_values(by="Date", inplace=True)

    # dict_stock_r2_PredActual_val contains a dictionary of stock, r2, intercept, coefficient
    # and data frame containing predicted and actual values
    dict_stock_r2_PredActual_val = {'Fund': etf_fund,
                                    'r2': r2,
                                    'MeanSqVal': mse,
                                    'Intercept': intercept,
                                    'Coefficient': coefficient,
                                    'Test_Pred_data': df_date_pred_actual_value}

    return dict_stock_r2_PredActual_val