#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
# sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
# tempo de treinamento: 208.852 s
# tempo de test: 22.629 s
# 0.984072810011

# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

# tempo de treinamento: 0.124 s
# tempo de test: 1.32 s
# 0.884527872582

clf = SVC(kernel='rbf', C=10000)
# c=10 -> 0.616040955631
# c=100 -> 0.616040955631
# c=1000 -> 0.821387940842
# c=10000 -> 0.892491467577
t0 = time()
clf.fit(features_train,labels_train)
print "tempo de treinamento:", round(time()-t0, 3), "s"
t1 = time()
y_pred = clf.predict(features_test)
print "tempo de test:", round(time()-t1, 3), "s"
# result=accuracy_score(labels_test,y_pred)
# print(result)

# print(y_pred[10])
# print(y_pred[26])
# print(y_pred[50])

print(y_pred.tolist().count(0))
print(y_pred.tolist().count(1))

#########################################################
### your code goes here ###

#########################################################


