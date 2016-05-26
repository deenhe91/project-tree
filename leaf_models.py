import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.cross_validation import train_test_split, cross_val_score 
from sklearn import preprocessing, tree, svm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression


# I knew the colnames from the dataset

colnames = ['Species','SpecimenNumber', 'Eccentricity', 'AspectRatio', 'Elongation', 'Solidity', 'Stochastic','Convexity', 'IsoperimetricFactor',
            'MaxIndentationDepth','Lobedness','AverageIntensity','AverageContrast','Smoothness','ThirdMoment','Uniformity','Entropy']

# and had the dataset saved in '/leaf/':
data = pd.read_csv('leaf/leaf.csv', names=colnames)

#cleaned up data
data.replace(to_replace='nan', value = float('nan'), inplace = True)
data.groupby('Species')['Eccentricity'].std()
data['Ecc_squared'] = [e**2 for e in data['Eccentricity']]
data = data.drop('Entropy', axis=1)

#and pickled outcome:
pickle.dump(data, 'leaf_data.csv', 'rb')

data_reduced = data.drop(['SpecimenNumber'], axis=1).groupby('Species').agg(['mean'])



#just a little extra: loosely defined medicinal 'usefulness' - in case of scenario where
# identifying leaf species was required to find medicinally useful plants...
uses = [1,0,1,1,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,1,0,0,1]#Medicinal uses
data['Usefulness'] = [uses[i-1] for i in data.Species]
Counter(uses)



#Test/train splits - stratified because there are so few samples for each category:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


"""Feature Importance"""

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure(figsize=(15,10))
sns.set_style("white")
plt.title("FEATURE IMPORTANCE", size=20)
plt.bar(range(x_train.shape[1]), importances[indices],
       color="lightblue", align="center")
sns.despine()
plt.xticks(range(x_train.shape[1]), indices)
# plt.ylabel('feature importance')
# plt.xlabel('feature')
plt.xlim([-1, x_train.shape[1]])
plt.ylim(0.04,None)
plt.savefig('f_importance.png')
plt.show()


# RandomForest

clf = tree.DecisionTreeClassifier(max_depth=10)
clf = clf.fit(x_train, y_train)
clf.predict(x_test)
clf.score(x_test, y_test)

clf = tree.DecisionTreeClassifier(max_depth=10)
clf = clf.fit(x_train, y_train)
clf.predict(x_test)
tree.export_graphviz(clf,out_file='tree.dot') 
forest = RandomForestClassifier(n_estimators = 200, max_depth=30)
forest = forest.fit(x_train, y_train)
output = forest.predict(x_test)

print("RandomForest output", output, "\n")
print("score", forest.score(x_test,y_test))

#CONFUSION MATRIX

spec_number = set(data.Species)


cm = confusion_matrix(y_test, output)

def plot_confusion_matrix(cm, title='CONFUSION MATRIX', cmap=plt.cm.Blues):
    plt.figure(figsize=(15,15))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size=30)
    plt.colorbar()
    tick_marks = np.arange(len(spec_number))
    plt.xticks(tick_marks, spec_number, rotation=60, size = 20)
    plt.yticks(tick_marks, spec_number, size=20)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm2 = confusion_matrix(y_test, output)
plot_confusion_matrix(cm2)

#lOGISTIC REGRESSION

scores = []

logit_model = LogisticRegression()
logit_fit = logit_model.fit(x_train, y_train)
logit_predict = logit_model.predict(x_test)
logit_score = accuracy_score(y_test, logit_predict)
scores.append(logit_score)
print (logit_score)

#KNN

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

k_values = range(1,15)
k_value_scores = []

for k in k_values:
    neighbours = KNeighborsClassifier(n_neighbors=k)
    result = neighbours.fit(x_train, y_train)
    y_predict = result.predict(x_test)
    k_value_scores.append(accuracy_score(y_test,y_predict))

k_and_score = list(zip(k_values, k_value_scores))
print(k_and_score)

scores.append(max(k_value_scores))
max(k_and_score, key=lambda x:x[1]) 


#SVM
#scaled for SVM

x_scaled = preprocessing.scale(x_train)
x_t_train_scaled = preprocessing.scale(x_train)
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(x_t_train_scaled, y_train) 
dec = clf.decision_function(x_t_train_scaled)
clf.predict(x_t_scaled)
x_t_scaled = preprocessing.scale(x_test) #SCALED - but no stratified split -
clf.score(x_t_scaled, y_test)
scores.append(clf.score(x_t_scaled, y_test))
clf.score(x_test, y_test) #NOT SCALED







