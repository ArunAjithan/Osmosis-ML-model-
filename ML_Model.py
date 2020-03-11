import pandas as pd                    # pandas is a dataframe library
import numpy as np

from sklearn.metrics import accuracy_score
import sklearn.model_selection as model_selection
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("C:\\Users\\m1055934\\Desktop\\ntwrk_trail.csv")

features = df.iloc[:, :-1]
class_label = df.iloc[:,-1].values
class_label = class_label.astype('int')

features.service.value_counts().sort_values(ascending=False).head(20)
features.flag.value_counts().sort_values(ascending=False).head(1)

top_service = [x for x in features.service.value_counts().sort_values(ascending=False).head(10).index]
top_flag = [x for x in features.flag.value_counts().sort_values(ascending=False).head(10).index]
top_protocol_type = [x for x in features.protocol_type.value_counts().sort_values(ascending=False).head(3).index]


for lable in top_service:
    features[lable] = np.where(features['service']==lable, 1, 0)
features[['service']+top_service].head(20)


for lable in top_flag:
    features[lable] = np.where(features['flag']==lable, 1, 0)
features[['flag']+top_flag].head(20)

for lable in top_protocol_type:
    features[lable] = np.where(features['protocol_type']==lable, 1, 0)
features[['protocol_type']+top_protocol_type].head(1)

features = features.drop(['protocol_type', 'service', 'flag' ], axis = 1)

X_train , X_test , y_train , y_test = model_selection.train_test_split(features, class_label, test_size = 0.20)

clf = MultinomialNB().fit(X_train, y_train.ravel())

predicted = clf.predict(X_test)

print(predicted)

print(accuracy_score(y_test,predicted))