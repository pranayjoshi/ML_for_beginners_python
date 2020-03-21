# using scipy
from scipy.spatial import distance
# Calculate distance using euclid method
def euc(a, b):
    return distance.euclidean(a, b)

# OWN CLASSIFIER
class pranay():
    # Creating our own fit to match the curvature
    def fit(self, X_test, y_test):
        self.X_train = X_train 
        self.y_train = y_train

    # using a simple algorithm to check the closest distance
    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0

        # Iterating to find the closest distance
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
            return self.y_train[best_index]   # Returning the closest or the best Index
            
    # Creating our own prediction method
    def predict(self, X_test):
        predictions = []  # Creating a 2d array called predict
        # taking random labels using a loop
        for row in X_test:
            label = self.closest(row)  # checks the closest point
            predictions.append(label)
        return predictions  # returning the 2D Array

from sklearn import datasets
iris = datasets.load_iris()

# using Iris Data and Target
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

# Defining Classifier
# from sklearn.neighbors import KNeighborsClassifier
my_classifier = pranay()

# fit = match the curvature of given data 
my_classifier.fit(X_train, y_train)

# using the code to predict
predictions = my_classifier.predict(X_test)

# checking the score
from sklearn.metrics import accuracy_score

# Printing result
print(accuracy_score(y_test, predictions))


