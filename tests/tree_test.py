from sklearn import datasets
from sklearn.model_selection import train_test_split
from anisotropic_decision_tree import TreeBase
import numpy as np
import time


X, Y = datasets.make_regression(n_samples=2500, n_features=10)
X_train, X_test, y_train, y_test = train_test_split(X, Y)

T = TreeBase(split_method='svmlike')
print('fitting')
s = time.clock()
T = T.fit(X_train, y_train)
s2 = time.clock()
print('fitting in {} s'.format(s2 - s))
print('predicting')
pred = T.predict(X_test)
print('predicting in {} s'.format(time.clock() - s2))
print(pred)