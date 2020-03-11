import pandas as pd                    # pandas is a data frame library
import numpy as np
import glob
import os
from sklearn.metrics import accuracy_score
import sklearn.model_selection as model_selection
from sklearn.naive_bayes import MultinomialNB


class TrainMlModel:

    all_files = glob.glob('../../media/Train_Data_Set/*')

    latest_train_data_set = max(all_files, key=os.path.getctime)

    df = pd.read_csv(latest_train_data_set)

    def data_set_analysis(self):

        train_ml_model = TrainMlModel()

        features = self.df.iloc[:, :-1]
        class_label = self.df.iloc[:, -1].values
        class_label = class_label.astype('int')

        features.service.value_counts().sort_values(ascending=False).head(20)
        features.flag.value_counts().sort_values(ascending=False).head(1)

        top_service = [x for x in features.service.value_counts().sort_values(ascending=False).head(10).index]
        top_flag = [x for x in features.flag.value_counts().sort_values(ascending=False).head(10).index]
        top_protocol_type = [x for x in
                             features.protocol_type.value_counts().sort_values(ascending=False).head(3).index]

        for label in top_service:
            features[label] = np.where(features['service'] == label, 1, 0)
        features[['service'] + top_service].head(20)

        for label in top_flag:
            features[label] = np.where(features['flag'] == label, 1, 0)
        features[['flag'] + top_flag].head(20)

        for label in top_protocol_type:
            features[label] = np.where(features['protocol_type'] == label, 1, 0)
        features[['protocol_type'] + top_protocol_type].head(1)

        features = features.drop(['protocol_type', 'service', 'flag'], axis=1)

        x_train, x_test, y_train, y_test = model_selection.train_test_split(features, class_label, test_size=0.20)

        return train_ml_model.train_machine_model(x_train, x_test, y_train, y_test)

    @staticmethod
    def train_machine_model(x_train, x_test, y_train, y_test):

        clf = MultinomialNB().fit(x_train, y_train.ravel())

        predicted = clf.predict(x_test)

        return accuracy_score(y_test, predicted)


if __name__ == "__main__":
    train_ml = TrainMlModel()
    train_ml.data_set_analysis()
