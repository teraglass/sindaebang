from flask import Flask, render_template, request, abort, send_file, redirect, jsonify, Response
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

model_path = "./models"

data = pd.read_csv('dataset_sindaebang_2022.csv', date_parser = True)

# Convert to datetime for Date column
data['Date'] = pd.to_datetime(data['Date'])


# training parameter
model_name = "WL_sindaebang_case07_timeseries"
lead_time = "timeseries"    # 0.5h, 1h, 2h, 3h, 4h, 5h, 6h , timeseries
Predict_time_index = ["0.5h","1h","2h","3h","4h","5h","6h"]
Predict_time_leadtime = [30,60,120,180,240,300,360]

drop_out_rate = 0  #  0~1

seq_length = 24
hidden_dim = 48
data_dim = 14
size_of_batch = 48   # 한번에 학습시킬 자료의 단위
output_dim = 7
learning_rate = 0.0001
iterations = 100


# rate of training data size = training data size / total data size
train_size = int(len(data)*0.75)
test_start = int(len(data)*0.75)

# make data of training and test data
data_training = data[:train_size]
data_test = data[test_start-seq_length:]


data_training_drop = data_training.iloc[:,[1,2,3,4,5,8,12,13,16,20,21,22,23,24,30,31,32,33,34,35,36]]
data_test_drop = data_test.iloc[:,[1,2,3,4,5,8,12,13,16,20,21,22,23,24,30,31,32,33,34,35,36]]
Current_time_test = data_test.iloc[:,[0]]
Current_time_test = Current_time_test.reset_index(drop=True)

# Make a prediction time for each leadtime
Predict_time = pd.DataFrame()

for i in range(0, len(Predict_time_leadtime)):
    Predict_time[Predict_time_index[i]] = pd.DatetimeIndex(Current_time_test['Date']) + timedelta(minutes=Predict_time_leadtime[i])

# data normalize of training data
scaler = MinMaxScaler() # feature scaling
scaler.fit_transform(data_training_drop)

@app.route('/')
def index():  # put application's code here
    return render_template('index.html')

@app.route("/predict",methods=["GET","POST"])
def predict():
    global scaler
    if request.method == 'POST':
        file = request.files['file']
        lead_time = request.values['lead_time']
        print(lead_time)
        model = tf.keras.models.load_model(os.path.join(model_path, lead_time))

        if file != '':
            filename = os.path.splitext(secure_filename(file.filename))[0]
            file_bytes = file.read()
            df = pd.read_csv(io.BytesIO(file_bytes),
                             infer_datetime_format=True, header=None)
            df = df.iloc[:, 1:]
            ##### normalization
            y_test = (df - scaler.data_min_[-1]) / (scaler.data_range_[-1] - scaler.data_min_[-1])
            # change dimension
            y_test = np.array(y_test).reshape(-1, seq_length, data_dim)
            # prediction
            y_pred = model.predict(y_test)
            ##### denormalization
            dn = y_pred * scaler.data_range_[-1] + scaler.data_min_[-1]
            dn = pd.DataFrame(dn)
            output_stream = io.StringIO()
            dn.to_csv(output_stream, index=False, header=False)
            response = Response(
                output_stream.getvalue(),
                mimetype='text/csv',
                content_type='application/octet-stream',
            )
            filename = os.path.join(filename, "result")
            response.headers["Content-Disposition"] = "attachment; filename=" + filename + ".csv"
            return response
    else:
        return redirect("/")
if __name__ == '__main__':
    app.run()

