from sklearn import tree


# Training Data

# Store are list in the features variable consists of [height, weight, shoe_size]
features = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
          [190, 90, 47], [175, 64, 39],
          [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

# labels
labels = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
          'female', 'male', 'male']

# Classifier for supervised learning
# Classifier is a box of rules
# clf will store our DecisionTree classifier
clf = tree.DecisionTreeClassifier()


# Find patterns in the training data
# Training algorithm is included in the classifier object is called Fit
# Fit method trains the decision tree on our data set
clf = clf.fit(features, labels)


# Classifier notice males tends to weigh more and taller, so it'll crate a rule that the heavier
# person is, more likely it is to be male
prediction = clf.predict([[190, 70, 43]])

# print the output
print(prediction)
