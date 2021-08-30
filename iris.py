from numpy import not_equal
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import numpy as np

iris = load_iris()

features = iris.data
labels = iris.target

lim = input("Heden udaa oruulah we? ")

features_user = np.array([])

for i in range(int(lim)):
    print("Feature: {} \n" .format(i+1))
    sepal_length = float(input("Sepal length: "))
    sepal_width = float(input("Sepal width: "))
    petal_length = float(input("Petal length: "))
    petal_width = float(input("Petal width: "))
    features_user = np.vstack([features_user, [sepal_length, sepal_width, petal_length, petal_width]])

print(features_user)
#print(features_user)

# avgAccScore = 0

# for i in range(1):
#     features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size=.5)
#     classifier=tree.DecisionTreeClassifier()
#     #classifier=GaussianNB()
#     classifier.fit(features_train,labels_train)
#     predictions=classifier.predict(features_test)
#     print(predictions)
#     avgAccScore = avgAccScore + accuracy_score(labels_test,predictions)
#     print(labels_test)
    
# print("Average Acc Score: {} " .format(avgAccScore/1))
# print(classifier.feature_importances_)

#print(features)


# lim = input("Heden udaa oruulah we? ")

# features_user = np.empty([int(lim), 4])
# print(features_user)

# for i in range(int(lim)):
#     sepal_length = input("Sepal length: ")
#     sepal_width = input("Sepal width: ")
#     petal_length = input("Petal length: ")
#     petal_width = input("Petal width: ")
#     np.concatenate((features_user, [sepal_length, sepal_width, petal_length, petal_width]))

# print(features_user)