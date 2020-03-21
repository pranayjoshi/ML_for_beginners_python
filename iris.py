import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
iris = load_iris()
print(iris.feature_names)
print(iris.target[0])
print(iris.data[0])
test_idx = [0, 50, 100]  

# Training Data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)
print(test_target)
print(clf.predict(test_data))
a = clf.predict(test_data)
b = test_target
if a.all() == b.all():
    print("true")