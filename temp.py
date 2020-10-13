#import the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.cm import rainbow
import warnings
warnings.filterwarnings('ignore')

#import dataset
dataset = pd.read_csv('heart.csv')
dataset_original = pd.read_csv('heart.csv')

#visulaise data
rcParams['figure.figsize'] = 20, 14
plt.matshow(dataset.corr())
plt.yticks(np.arange(dataset.shape[1]), dataset.columns)
plt.xticks(np.arange(dataset.shape[1]), dataset.columns)
plt.colorbar()

dataset.hist()    #histogram
dataset_original.hist()

#categorical variables
dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

#feature scaling
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
scaling_these_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[scaling_these_columns] = standardScaler.fit_transform(dataset[scaling_these_columns])


# Splitting the dataset into the Training set and Test set
y = dataset['target']
X = dataset.drop(['target'], axis = 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

# SVM Classifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
svc_scores = []
svc_precision_scores = []
svc_recall_scores = []
svc_f1_scores = []
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for i in range(len(kernels)):
    classifier_svm = SVC(kernel = kernels[i])
    classifier_svm.fit(X_train, y_train)
    svm_y_pred = classifier_svm.predict(X_test)
    svc_scores.append(classifier_svm.score(X_test, y_test))
    svc_precision_scores.append(precision_score(y_test, svm_y_pred))
    svc_recall_scores.append(recall_score(y_test,svm_y_pred))
    svc_f1_scores.append(f1_score(y_test,svm_y_pred))
    

#Decision tree classifier   
#Entropy criterion    
from sklearn.tree import DecisionTreeClassifier
dt_scores_entropy = []
dt_entropy_precision_scores = []
dt_entropy_recall_scores  = []
dt_entropy_f1_scores = [] 

for i in range(1, len(X.columns) + 1):
    classifier_dt_entropy = DecisionTreeClassifier(criterion = 'entropy' , max_features = i, random_state = 0)
    classifier_dt_entropy.fit(X_train, y_train)
    dt_scores_entropy.append(classifier_dt_entropy.score(X_test, y_test))    

#maximum score is with max features  = 2 with 83 
classifier_dt_entropy = DecisionTreeClassifier(criterion = 'entropy' , max_features = 2, random_state = 0)    
classifier_dt_entropy.fit(X_train, y_train)
dt_entropy_y_pred = classifier_dt_entropy.predict(X_test)
dt_entropy_precision_scores.append(precision_score(y_test,dt_entropy_y_pred))
dt_entropy_recall_scores.append(recall_score(y_test,dt_entropy_y_pred))
dt_entropy_f1_scores.append(f1_score(y_test,dt_entropy_y_pred))
 
#Gini criterion
  
dt_scores_gini = []
for i in range(1, len(X.columns) + 1):
    classifier_dt_gini = DecisionTreeClassifier( max_features = i, random_state = 0)
    classifier_dt_gini.fit(X_train, y_train)
    dt_scores_gini.append(classifier_dt_gini.score(X_test, y_test))     

#maximum score is with max features  = 2,4,18 with 79
dt_gini_precion_scores = []
dt_gini_recall_scores = []
dt_gini_f1_scores = []    
no_features = (2,4,18)
for i in no_features:
    classifier_dt_gini = DecisionTreeClassifier( max_features = i, random_state = 0)
    classifier_dt_gini.fit(X_train, y_train)
    dt_gini_y_pred = classifier_dt_gini.predict(X_test)
    dt_gini_precion_scores.append(precision_score(y_test, dt_gini_y_pred))
    dt_gini_recall_scores.append(recall_score(y_test, dt_gini_y_pred))
    dt_gini_f1_scores.append(f1_score(y_test, dt_gini_y_pred))
    

#Random forrest classifier

#Entropy criterion
     
from sklearn.ensemble import RandomForestClassifier
rf_scores_entropy = []
estimators = [10, 100, 200, 500, 1000]
rf_entropy_precision_scores = []
rf_entropy_recall_scores = []
rf_entropy_f1_scores = [] 
for i in estimators:
    classifier_rf_entropy = RandomForestClassifier(criterion = 'entropy',n_estimators = i, random_state = 0)
    classifier_rf_entropy.fit(X_train, y_train)
    rf_entropy_y_pred = classifier_rf_entropy.predict(X_test)
    rf_scores_entropy.append(classifier_rf_entropy.score(X_test, y_test))
    rf_entropy_precision_scores.append(precision_score(y_test, rf_entropy_y_pred))
    rf_entropy_recall_scores.append(recall_score(y_test, rf_entropy_y_pred))
    rf_entropy_f1_scores.append(f1_score(y_test, rf_entropy_y_pred))

#Gini criterion
    
rf_scores_gini = []
estimators = [10, 100, 200, 500, 1000]
rf_gini_precision_scores = []
rf_gini_recall_scores = []
rf_gini_f1_scores = [] 
for i in estimators:
    classifier_rf_gini = RandomForestClassifier(n_estimators = i, random_state = 0)
    classifier_rf_gini.fit(X_train, y_train)
    rf_gini_y_pred = classifier_rf_gini.predict(X_test)
    rf_scores_gini.append(classifier_rf_gini.score(X_test, y_test))
    rf_gini_precision_scores.append(precision_score(y_test, rf_gini_y_pred))
    rf_gini_recall_scores.append(recall_score(y_test, rf_gini_y_pred))
    rf_gini_f1_scores.append(f1_score(y_test, rf_gini_y_pred))
    

#KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn_scores = []
knn_precision_scores = []
knn_recall_scores = []
knn_f1_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X_train, y_train)
    knn_scores.append(knn_classifier.score(X_test, y_test))    

#maximum score is with number of neighbors = 8 
knn_classifier = KNeighborsClassifier(n_neighbors = 8) 
knn_classifier.fit(X_train, y_train) 
knn_y_pred = knn_classifier.predict(X_test)
knn_precision_scores.append(precision_score(y_test,knn_y_pred))
knn_recall_scores.append(recall_score(y_test,knn_y_pred))
knn_f1_scores.append(f1_score(y_test,knn_y_pred))    

#Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
NB_scores = []
NB_precision_scores = []
NB_recall_scores = []
NB_f1_scores = []

classifier_NB = GaussianNB()
classifier_NB.fit(X_train,y_train)
NB_y_pred = classifier_NB.predict(X_test)

NB_scores.append(classifier_NB.score(X_test, y_test))    
NB_precision_scores.append(precision_score(y_test, NB_y_pred))
NB_recall_scores.append(recall_score(y_test, NB_y_pred))
NB_f1_scores.append(f1_score(y_test, NB_y_pred))


############################################################################################3
#SVC

labels = ['linear', 'poly', 'rbf', 'sigmoid']

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 1.5*width, svc_f1_scores, width, label='F1')
rects2 = ax.bar(x - 0.5*width, svc_precision_scores, width, label='Precision')
rects3 = ax.bar(x + 0.5*width, svc_recall_scores, width, label='Recall')
rects4 = ax.bar(x + 1.5*width, svc_scores, width, label='Score')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by group')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()

############################################################################################3
#DTree Gini

labels = ['2','4','18']

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 1.5*width, dt_gini_f1_scores, width, label='F1')
rects2 = ax.bar(x - 0.5*width, dt_gini_precion_scores, width, label='Precision')
rects3 = ax.bar(x + 0.5*width, dt_gini_recall_scores, width, label='Recall')
#rects4 = ax.bar(x + 1.5*width, dt_scores_gini, width, label='Score')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by group')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()

##############################################################################################
#DTree Entropy

labels = ['F1', 'Precision', 'Recall']
dt_entropy_sc = dt_entropy_f1_scores + dt_entropy_precision_scores + dt_entropy_recall_scores

plt.figure(figsize=(10,7))
plt.bar(labels,dt_entropy_sc)
plt.show()

##############################################################################################
#RF Entropy

labels = [10, 100, 200, 500, 1000]

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 1.5*width, rf_entropy_f1_scores, width, label='F1')
rects2 = ax.bar(x - 0.5*width, rf_entropy_precision_scores, width, label='Precision')
rects3 = ax.bar(x + 0.5*width, rf_entropy_recall_scores, width, label='Recall')
rects4 = ax.bar(x + 1.5*width, rf_scores_entropy, width, label='Score')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by group')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()

###############################################################################################
#RF Gini

labels = [10, 100, 200, 500, 1000]

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 1.5*width, rf_gini_f1_scores, width, label='F1')
rects2 = ax.bar(x - 0.5*width, rf_gini_precision_scores, width, label='Precision')
rects3 = ax.bar(x + 0.5*width, rf_gini_recall_scores, width, label='Recall')
rects4 = ax.bar(x + 1.5*width, rf_scores_gini, width, label='Score')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by group')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()

###############################################################################################
#KNN

labels = ['F1', 'Precision', 'Recall']
knn_sc = knn_f1_scores + knn_precision_scores + knn_recall_scores

plt.figure(figsize=(10,7))
plt.bar(labels,dt_entropy_sc)
plt.show()

############################################################################################3
#Naive Bayes

labels = ['F1', 'Precision', 'Recall','Scores']
NB_sc = NB_f1_scores + NB_precision_scores + NB_recall_scores + NB_scores

plt.figure(figsize=(10,7))
plt.bar(labels,NB_sc)
plt.show()
