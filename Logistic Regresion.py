from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


iris = datasets.load_iris()

x = iris['data'][:, 3:]

y = (iris["target"]==2).astype(np.int64)

clf = LogisticRegression()

clf.fit(x,y)
predicted = clf.predict([[2.6]])

print(predicted)

x_new = np.linspace(0, 3, 100).reshape(-1, 1)
y_prob = clf.predict_proba(x_new)

plt.plot(x_new, y_prob[:, 1], "g-")
plt.title("Virginica")
plt.show()



# print(y)
# print(x)
#print(list(iris.keys()))
#print(iris['data']).shape
#print(iris['target'])
#print(iris['DESCR'])