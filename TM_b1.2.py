import pandas as pd
import numpy as np
import string
import warnings

from sklearn import svm

class Train_test_model:

    def prepare(self):

        X_train = train_data_tfidf
        X_test = test_data_tfidf
        y_train = df_train["label"].values[:31962]
        y_test = df["label"].values[31962:]

        return X_train, X_test, y_train, y_test

    def evaluate_on_test_data(self, X_test, X_train, y_train):

        self.X_test = X_test
        self.X_train = X_train
        self.y_train = y_train

        predictions = model.predict(X_test)
        corr_classifications = 0
        accuracy = model.score(X_train,y_train)
        return accuracy

    def model(self, X_train, y_train):

        self.X_train = X_train
        self.y_train = y_train
        
        model = svm.SVC(kernel = 'ploy')
        model.fit(X_train, Y_train)


train = Train_test_model()
X_train, X_test, y_train, y_test = train.prepare()

train.model(X_train, y_train)
acc = train.evaluate_on_test_data(model)
print('Accuracy with kernel: ',' is: ', acc*100)



    
    
