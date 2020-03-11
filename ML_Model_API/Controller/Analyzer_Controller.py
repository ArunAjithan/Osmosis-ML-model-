from flask import Flask, json
from flask_cors import CORS
from ML_Model_API.Test_Models.Test_Model import TestMlModel
# from ML_Model_API.Train_Models.Train_Model import TrainMlModel
from ML_Model_API.Train_Models.Network_Anomaly_Detection import TrainMlModel
app = Flask(__name__)
co_rs = CORS(app)


@app.route('/Prediction')
def prediction_result():

    t_model = TestMlModel()

    result_set = t_model.custom_test_model()

    response = app.response_class(
        response=json.dumps(result_set),
        mimetype='application/json'
    )

    return response


@app.route('/Training_Accuracy')
def train_accuracy():

    t_model = TrainMlModel()

    result_set = t_model.data_set_analysis()

    response = json.dumps({"Accuracy": result_set})

    return response


if __name__ == '__main__':
    app.run(debug=True)
