from sklearn import datasets
iris = datasets.load_iris()

# using Iris Data and Target
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

# Defining Classifier
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

# match the curvature of given data 
my_classifier.fit(X_train, y_train)

# using the code to predict
predict = my_classifier.predict(X_test)

# checking the score
from sklearn.metrics import accuracy_score
result = accuracy_score(y_test, predict)

# Printing result
print(result)


