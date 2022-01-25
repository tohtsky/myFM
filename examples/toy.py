import numpy as np
from sklearn.feature_extraction import DictVectorizer

import myfm

train = [
    {"user": "1", "item": "5", "age": 19},
    {"user": "2", "item": "43", "age": 33},
    {"user": "3", "item": "20", "age": 55},
    {"user": "4", "item": "10", "age": 20},
]
v = DictVectorizer()
X = v.fit_transform(train)
y = np.asarray([0, 1, 1, 0])
fm = myfm.MyFMClassifier(rank=4)
fm.fit(X, y)
p = fm.predict_proba(v.transform({"user": "1", "item": "10", "age": 24}))
print(p)
