# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:35:47 2019

@author: diana
"""

import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from eda import feature_engineer, live_demo

# Flask utils
from flask import Flask, request, render_template, current_app

# Define a flask app
app = Flask(__name__)
# Load model
print("Model is loading....")
app.clf = lgb.Booster(model_file=r"models\Sponsor_InvestorZip_offerview.txt")

print("master data is loaded...")
location = r"uploads\master.csv"
app.master_feature = feature_engineer(pd.read_csv(location, index_col=0))

print("app starting....")

@app.route('/')
@app.route('/input')
def render_input_html():
    return render_template("input.html")

@app.route('/output')
def render_output_html():
    #pull 'offering_id' from input field and store it
    offering_id = request.args.get('offering_id')

    results = live_demo(current_app.master_feature, current_app.clf, offering_id)
    results = results[results['Invest_Probability']>0.5]
    prediction_list = results.to_dict(orient='records')

    return render_template("output.html", prediction_list = prediction_list, offering_id = offering_id)


if __name__ == '__main__':
    app.run(port=5002, debug=True)

#name = '86e9d382c12b4362923fcf0a1b4c7115'
#name = '5d31dde054734746bc600c2d7c92a4e7'
#name = '0c68ec6b2b9c45b8bcd290b5afdfd4d1'
