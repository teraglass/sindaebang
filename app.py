from flask import Flask, render_template, request, redirect, Response
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import io
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

data = pd.read_csv('./dataset_sindaebang_2022.csv',date_parser = True)
data['Date'] = pd.to_datetime(data['Date'])

model_path = "./models"
model_name_front = "WL_sindaebang_case07_single_after_"
model_name_tail = "_drop-out_0_seq_length_24"

drop_out_rate = 0  # 0~1
output_dim = 1
seq_length = 24
data_dim = 15
seq_length = 24
lead_time = "0.5h"    # 0.5h, 1h, 2h, 3h, 4h, 5h, 6h , timeseries
Predict_time_index = {"0.5h":30, "1h":60, "2h":120, "3h":180, "4h":240, "5h":300, "6h":360}

# rate of training data size = training data size / total data size
train_size = int(len(data)*0.75)
test_start = int(len(data)*0.75)

# make data of training and test data
data_training = data[:train_size]
data_test = data[test_start-seq_length:]

scaler = MinMaxScaler()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=["GET","POST"])
def predict():
    global scaler
    if request.method == 'POST':
        lead_time = request.values['lead_time']
        file = request.files['file']
        if lead_time == "0.5h":
            column_lead_time = 31
        elif lead_time == "1h":
            column_lead_time = 32
        elif lead_time == "2h":
            column_lead_time = 33
        elif lead_time == "3h":
            column_lead_time = 34
        elif lead_time == "4h":
            column_lead_time = 35
        elif lead_time == "5h":
            column_lead_time = 36
        elif lead_time == "6h":
            column_lead_time = 37
        else:
            timeseries_model()

        # scaler transform
        data_training_drop = data_training.iloc[:,
                             [1, 2, 3, 4, 5, 6, 9, 13, 14, 17, 21, 22, 23, 24, 25, column_lead_time]]
        scaler.fit_transform(data_training_drop)

        model_name = os.path.join(model_path, model_name_front + lead_time + model_name_tail)
        print(model_name)
        model = tf.keras.models.load_model(model_name)

        if file != '':
            filename = os.path.splitext(secure_filename(file.filename))[0]
            file_bytes = file.read()
            data_test = pd.read_csv(io.BytesIO(file_bytes),
                             infer_datetime_format=True, header=None)
            y_test = scaler.transform(data_test)
            y_test = np.array(y_test).reshape(-1, seq_length, 16) # single:16-> 15

            ######
            y_pred = model.predict(y_test[:,:,:-1])
            dn = y_pred*scaler.data_range_[-1]+scaler.data_min_[-1]
            dn = dn.flatten().reshape(-1, 1)
            dn = pd.DataFrame(dn)

            # Make a prediction time for each leadtime
            Predict_time = pd.DataFrame()
            Predict_time_index = lead_time
            Predict_time_leadtime = Predict_time_index[lead_time]
            print(Predict_time_index)
            print(Predict_time_leadtime)
            Current_time_test = data_test.iloc[-240:, [0]]
            Current_time_test = Current_time_test.reset_index(drop=True)


            for i in range(0, 1):
                Predict_time[Predict_time_index] = pd.DatetimeIndex(Current_time_test['Date']) + timedelta(
                    minutes=Predict_time_leadtime[i])

            # make the result file
            result_all = pd.DataFrame()
            result_all = pd.concat([Current_time_test], axis=1)

            for i in range(0, output_dim):
                result_tmp = pd.DataFrame(
                    {"y_true_" + Predict_time_index[i]: y_test[:, i], "y_pred_" + Predict_time_index[i]: y_pred[:, i]})
                Predict_time_tmp = Predict_time.iloc[:, [i]]
                result_all = pd.concat([result_all, Predict_time_tmp, result_tmp], axis=1)

            output_stream = io.StringIO()
            result_all.to_csv(output_stream, index=False, header=False)
            response = Response(
                output_stream.getvalue(),
                mimetype='text/csv',
                content_type='application/octet-stream',
            )
            filename = os.path.join(filename, "_predict_result")
            response.headers["Content-Disposition"] = "attachment; filename=" + filename + ".csv"
            return response
    else:
        return redirect("/")

def timeseries_model():
    lead_time = "timeseries"  # 0.5h, 1h, 2h, 3h, 4h, 5h, 6h , timeseries
    Predict_time_index = ["0.5h", "1h", "2h", "3h", "4h", "5h", "6h"]
    Predict_time_leadtime = [30, 60, 120, 180, 240, 300, 360]

    model = tf.keras.models.load_model(os.path.join(model_path, "WL_sindaebang_case07_timeseries_drop-out_0_seq_length_24"))

    data_training_drop = data_training.iloc[:,
                         [1, 2, 3, 4, 5, 6, 9, 13, 14, 17, 21, 22, 23, 24, 25, 31, 32, 33, 34, 35, 36, 37]]
    data_test_drop = data_test.iloc[:,
                     [1, 2, 3, 4, 5, 6, 9, 13, 14, 17, 21, 22, 23, 24, 25, 31, 32, 33, 34, 35, 36, 37]]
    Current_time_test = data_test.iloc[:, [0]]
    Current_time_test = Current_time_test.reset_index(drop=True)

    Predict_time = pd.DataFrame()

    for i in range(0, len(Predict_time_leadtime)):
        Predict_time[Predict_time_index[i]] = pd.DatetimeIndex(Current_time_test['Date']) + timedelta(
            minutes=Predict_time_leadtime[i])

    # data normalize of training data
    scaler_TS = MinMaxScaler()
    data_training_scale = scaler_TS.fit_transform(data_training_drop)

if __name__ == '__main__':
    app.run()
