import numpy as np
from sklearn import linear_model

X = ...
y = ...

logr = linear_model.LogisticRegression()
logr.fit(X, y)

predicted = logr.predict(np.array([3.46]).reshape(-1,1))

