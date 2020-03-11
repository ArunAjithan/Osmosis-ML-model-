import pandas as pd
import sklearn.model_selection as model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
from sklearn.metrics import accuracy_score
import glob
import os


class TrainMlModel:
    all_files = glob.glob('../../media/Train_Data_Set/*')

    latest_train_data_set = max(all_files, key=os.path.getctime)

    df = pd.read_csv(latest_train_data_set)

    def data_set_analysis(self):
        tml_module = TrainMlModel()

        feature_col_names = ['question_text']
        predicted_class_names = ['target']

        x = self.df[feature_col_names].values
        y = self.df[predicted_class_names].values
        split_test_size = 0.20

        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=split_test_size)

        val = tml_module.train_machine_model(x_train, x_test, y_train, y_test)

        return int(val * 100)

    @staticmethod
    def train_machine_model(x_train, x_test, y_train, y_test):
        vectorizer = TfidfVectorizer()

        train_vectors = vectorizer.fit_transform(x_train.ravel())

        test_vectors = vectorizer.transform(x_test.ravel())

        clf = MultinomialNB().fit(train_vectors, y_train.ravel())

        predicted = clf.predict(test_vectors)

        vector_name = '../../media/ML_Models/Vectorizer.pickle'
        model_name = '../../media/ML_Models/Finalized_Model.pickle'

        pickle.dump(vectorizer, open(vector_name, "wb"))
        pickle.dump(clf, open(model_name, 'wb'))

        return accuracy_score(y_test, predicted)
