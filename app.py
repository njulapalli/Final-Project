# import necessary libraries
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
import futureDates
import trainModel
import seaborn as sns
import time
from flask import Flask, render_template, request, redirect, url_for
import csv
import os

pd.plotting.register_matplotlib_converters(explicit=True)

# create instance of Flask app
app = Flask(__name__, static_url_path='/static')

def calculateModel(img_path, list_etf, future_proj_days):
    if future_proj_days == 0:
        future_proj_days = 750 # Number of days in future for which values to be predicted. In live code this is user provided
    print("calculating for future projection days: " + str(future_proj_days))
    dict_r2 = {} # Dictionary for storing r2 values
    dict_fund_model = {} # Dictionary to store ETF funds and their corresponding model
    start = dt.datetime(2005,1,1)
    end = dt.datetime.today()
    for etf in list_etf:
        model_etf = trainModel.TrainTestModel(etf, start, end)
        dict_r2[etf] = model_etf['r2']
        dict_fund_model[etf] = model_etf

    projection_images = []
    dict_future_projection = {}
    for etf_iterator in range(0,len(list_etf)):
        fund_lr_model = dict_fund_model[list_etf[etf_iterator]]
        r2_value = round(fund_lr_model['r2'],6)*100
        #coefficient = fund_lr_model['Coefficient'][0]
        #intercept = fund_lr_model['Intercept'][0]
        coefficient = fund_lr_model['Coefficient'][0]
        intercept = fund_lr_model['Intercept'][0]
        df_fund_model_dataset = pd.DataFrame(fund_lr_model['Test_Pred_data'])
        plt.plot(df_fund_model_dataset['Date'],df_fund_model_dataset['ActualPrice'])
        plt.plot(df_fund_model_dataset['Date'],df_fund_model_dataset['PredictedPrice'])
        plt.title(f'Performance Trend for {list_etf[etf_iterator]} with confidence level of {r2_value}% \n')
        #plt.show()
        image_name = 'Projection_' + list_etf[etf_iterator] + str(int(time.time()))
        plt.savefig(img_path + image_name + ".png")
        plt.close()
        projection_images.append(image_name)
        
        # Predicting future values
        df_future_projection = futureDates.createFutureDates(future_proj_days)
        df_future_projection['FutureValue'] = ((df_future_projection['DateFloat']).values.reshape(-1,1) * coefficient) + intercept
        dict_future_projection[list_etf[etf_iterator]] = df_future_projection
        #df_future_projection['FutureValue'] = ((df_future_projection['DateFloat']) * coefficient) + intercept
        file_name = 'FutureProjection_' + list_etf[etf_iterator] + str(int(time.time())) + '.csv'
        #image_future_projection = 'FutureProjections_' + list_etf[etf_iterator] + '.csv'
        df_future_projection.to_csv(img_path + file_name)
        # This part of code is to plot future value graph. This is irrelevant since with LR model growth/fall will always
        # be in a straight line. To test, you can un-comment these code-lines.
        #plt.plot(df_future_projection['Date'], df_future_projection['FutureValue'])
        #plt.title(f'Future growth pattern for {list_etf[etf_iterator]} with confidence level of {r2_value}% \n')
        #plt.show()    
    return projection_images

# create route that renders index.html template
#@app.route("/")
@app.route("/", methods=['GET', 'POST'])
def index():
    print(app.instance_path)
    selected = []
    flat_list = []
    curr_file = os.getcwd()+"/data/ETFSymbols.csv"
    img_path = "static/images/"

    if not os.path.exists(img_path):
        os.makedirs(img_path)        

    print(curr_file)
    with open(curr_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
    
        for sublist in list(readCSV):
            for item in sublist:
                flat_list.append(item)

    if request.method == 'POST':
        selected = request.form.getlist('option')
        projection_days = int(request.form.get('days'))
        print("Selected symbols: " + str(selected))
        print("Projection days:" + str(projection_days))
        image_list = calculateModel(img_path, selected, projection_days)
        return render_template("index.html", option_list=flat_list, item_list=image_list, sub_img_path="/" + img_path)
    else:
        #print(flat_list)
        return render_template("index.html", option_list=flat_list, item_list="")


if __name__ == "__main__":
    app.run(debug=True)
