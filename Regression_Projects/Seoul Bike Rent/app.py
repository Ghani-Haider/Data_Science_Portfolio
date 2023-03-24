# from joblib import load
from joblib import load
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# load the prediction model pipeline (Feature Scaler + Estimator)
pipeline = load('./model/bike_rent_count.mod')

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/predict', methods=['POST'])
def predict():
    # fetch form inputs containing the values
    data = [x for x in request.form.values()]

    # extract values, transform categorical features
    # and return correct input format ex: [[feature_1, feature_2, ...], ]
    x_input = feature_encode(data)
    # scale and predict
    predicted_val = pipeline.predict(x_input)[0]
    # display the predicted value
    return render_template('homepage.html', prediction_txt=f"The Predict Value is {predicted_val}")


### categorical feature encoding to dummmy variables ###
### following input features from the user will be received in given order ###
# --------------------------- # ---------------------------- #
# {"Hour", "Temperature(°C)", "Humidity(%)", "Wind speed (m/s)",
#  "Visibility (10m)", "Solar Radiation (MJ/m2)", "Rainfall(mm)",
#  "Snowfall (cm)", "Seasons", "Holiday", 
#  "Functioning Day", "Week Day"}
# --------------------------- # ---------------------------- #
def feature_encode(data):
    # temp_to_snowfall =  [data["Temperature(°C)"], data["Humidity(%)"], 
    #                          data["Wind speed (m/s)"], data["Visibility (10m)"],
    #                          data["Solar Radiation (MJ/m2)"], data["Rainfall(mm)"],
    #                          data["Snowfall (cm)"]]
    # ['23', '-5.2', '37', '2.2', '2000', '0.0', '0.0', '0.0', 'Winter', 'Holiday', 'Yes', 'Yes']
    # save numerical features as list
    temp_to_snowfall = list(map(float, data[1:8]))
    
    # encode hour feature to dummy var
    hour = [0] * 24
    # set the given hour 1
    hour[int(data[0])] = 1
    
    # encode seasons to dummy var
    season = [0] * 3
    if data[8] == 'Spring':
        season[0] = 1
    elif data[8] == 'Summer':
        season[1] = 1
    elif data[8] == 'Winter':
        season[2] = 1

    # encode holiday to dummy var
    holiday = [1] if data[9] == 'No Holiday' else [0]
    # encode functioning day to dummy var
    func_day = [1] if data[10] == 'Yes' else [0]
    # encode weekday to dummy var
    weekday = [1] if data[11] == 'Yes' else [0]

    # return the features with encoded values and as a test input instance ex: [[feature_1, feature_2, ...], ]
    return np.array(list(temp_to_snowfall + hour[1:] + season + holiday + func_day + weekday)).reshape(1,-1)

if __name__ == "__main__":
    app.run(debug=True)