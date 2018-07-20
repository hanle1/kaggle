import pandas as pd
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

y_train = train['label']
x_train = train.drop('label',1)
x_test = test

import tensorflow as tf
import skflow

# classifier = skflow.TensorFlowLinearRegressor(n_classes=10,batch_size=100,steps=10000,learning_rate=0.01)
# classifier.fit(x_train,y_train)
# linear_y_predict = classifier.predict(x_test)
# linear_submission = pd.DataFrame({'ImageId':range(1,28001),'Label': linear_y_predict})
# linear_submission.to_csv('./data/linear_submission.csv',index = False)

classifier = skflow.TensorFlowDNNClassifier(hidden_units=[200,50,10],n_classes=10,steps=5000,learning_rate=0.01,batch_size=50)
classifier.fit(x_train,y_train)
dnn_y_predict = classifier.predict(x_test)
dnn_submission = pd.DataFrame({'ImageId':range(1,28001),'Label': dnn_y_predict})
dnn_submission.to_csv('./data/dnn_submission.csv',index = False)
