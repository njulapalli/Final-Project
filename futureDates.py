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

def createFutureDates(days):  # Function to creat data-frame for future dates for price projection
    # This should be used when predicting data for future starting current day
    # date_today = dt.datetime.today()
    # This should be used to validate model against available data taking that as future value and today as a past date
    date_today = dt.datetime.today() - timedelta(days = days)
    #date_today = dt.datetime(2017, 5, 3)
    dates_future = [date_today]
    for day in range(1, days):
        date_future = date_today + timedelta(days=day)
        dates_future.append(date_future)
    df_future = pd.DataFrame({'Date': dates_future})
    df_future['DateFloat'] = df_future['Date'].values.astype(float)
    return df_future