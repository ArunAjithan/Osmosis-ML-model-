import pickle
import pandas as pd
import numpy as np
import glob
import os


class TestMlModel:

    @staticmethod
    def custom_test_model():

        t_module = TestMlModel()

        vector_name = '../../media/ML_Models/Vectorizer.pickle'
        model_name = '../../media/ML_Models/Finalized_Model.pickle'

        vectorizer = pickle.load(open(vector_name, 'rb'))
        loaded_model = pickle.load(open(model_name, 'rb'))

        df = pd.read_csv('../../media/Test_Data_Set/Testing_Set.csv')

        trail_test = df['question_text'].values.tolist()

        test_vector_trail = vectorizer.transform(trail_test)

        predicted = loaded_model.predict(test_vector_trail)

        predicted_array = np.array(predicted)

        df['target'] = predicted_array

        df.to_csv('../../media/Predicted_Data/Prediction.csv', encoding='utf-8', index=False)

        t_module.update_training_data_set(df)

        return predicted.tolist()

    @staticmethod
    def update_training_data_set(df_prediction):

        all_files = glob.glob('../../media/Train_Data_Set/*')

        latest_train_data_set = max(all_files, key=os.path.getctime)

        df_training = pd.read_csv(latest_train_data_set)

        df_prediction = df_prediction.append(df_training)

        df_prediction.to_csv('../../media/Feedback_Data_Set/Feedback.csv', encoding='utf-8', index=False)
