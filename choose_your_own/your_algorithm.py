#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show bbbyou the decision boundary

# # ###Decision Tree
# clf = DecisionTreeClassifier()
# clf.fit(features_train, labels_train)
# # predict
# predited = clf.predict(features_test)
# acc=accuracy_score(labels_test,predited)
# print(acc)
# # acc 0.912
# # time 2.7s

# # ### KNN
# clf = KNeighborsClassifier(3)
# clf.fit(features_train, labels_train)
# # predict
# predited = clf.predict(features_test)
# acc=accuracy_score(labels_test,predited)
# print(acc)
# ## acc 0.936
# ## time 3.4s

clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2))
clf.fit(features_train, labels_train)
# predict
predited = clf.predict(features_test)
acc=accuracy_score(labels_test,predited)
print(acc)
# acc 0.924
# time 2.8s

# clf = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=0)
# clf.fit(features_train, labels_train)
# # # predict
# predited = clf.predict(features_test)
# acc=accuracy_score(labels_test,predited)
# print(acc)
# # # 0.92
# # # time 2.8s


try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
