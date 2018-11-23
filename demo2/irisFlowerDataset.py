# How to test a model and determine accuracy

# Partition data into 2 sets, train and test

# import a dataset
from sklearn import datasets

iris = datasets.load_iris()

# Can think of classifier as a function f(x) = y
X = iris.data  # features
y = iris.target  # labels

# partition into training and testing sets

from sklearn.model_selection import train_test_split

# test_size=0.5 -> split in half
# X_train and y_train are the features and labels for the training set
# X_test and y_test are the features and labels for the testing set
# so we have 150 split 75 for test and 75 for train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# We will use two different classifier to compare their output
# Classifier
from sklearn import tree

# train the classifier using training data
my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(X_train, y_train)

# Predict
predictions = my_classifier.predict(X_test)
print(predictions)

# Test
# To calculate the accuracy we can compare the predicted labels to the true labels and tally the score
from sklearn.metrics import accuracy_score
print(accuracy_score(predictions, y_test))

# Repeat using KNN
# Classifier
from sklearn.neighbors import KNeighborsClassifier

my_classifier = KNeighborsClassifier()
my_classifier.fit(X_train, y_train)

# predict
predictions = my_classifier.predict(X_test)
print(predictions)

# test
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, predictions))