# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:06:40 2021

@author: aracz
"""

import numpy as np;  # importing numerical computing package
from matplotlib import pyplot as plt;  # importing MATLAB-like plotting framework
from sklearn.linear_model import LogisticRegression; #  importing logistic regression classifier
from sklearn.model_selection import train_test_split, cross_val_score; # importing splitting
from sklearn.metrics import confusion_matrix, plot_confusion_matrix; #  importing performance metrics
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree;    # importing decision tree tools
from sklearn.feature_selection import SelectKBest; # importing feature selection 
from sklearn.decomposition import PCA;  # importing PCA
import seaborn as sns;  # importing the Seaborn library
import matplotlib.colors as col;  # importing coloring tools from MatPlotLib
from sklearn.cluster import KMeans; # Class for K-means clustering
from sklearn.metrics import davies_bouldin_score, roc_curve, auc;  # function for Davies-Bouldin goodness-of-fit 
from sklearn.metrics.cluster import contingency_matrix;
from sklearn.pipeline import make_pipeline;
from sklearn.preprocessing import StandardScaler;

#Getting the dataset
diamonds = pd.read_csv('https://raw.githubusercontent.com/raczandras/ML/main/diamonds.csv');
atts = diamonds.drop('expensive',axis=1);
atts2 = atts.to_numpy();
att_names = atts.columns;
target = diamonds['expensive'];
target2 = target.copy();
target = target.replace(to_replace = 0, value = "not expensive");
target = target.replace(to_replace = 1, value = "expensive");
target_names = ["not expensive", "expensive"];
diamonds2 = diamonds;
columns_titles= ["carat", "depth", "table", "expensive", "x", "y", "z"];

#The relationship of attributes
sns.pairplot(data=diamonds)
plt.show()

#PCA
pipe = make_pipeline(StandardScaler(), PCA());
diamonds_pc_scaled=pipe.fit_transform(atts)
var_ratio = pipe.named_steps['pca'].explained_variance_ratio_;
cum_var_ratio = np.cumsum(var_ratio);
plt.figure();
plt.title('Variance of attribute');
df = pd.DataFrame({'var':pipe.named_steps['pca'].explained_variance_ratio_,
             'PC':att_names})
sns.barplot(x='PC',y="var", 
           data=df, color="c");
plt.show();

plt.figure();
plt.title('CUM Variance');
df = pd.DataFrame({'var':cum_var_ratio,
             'PC':att_names})
sns.barplot(x='PC',y="var", 
           data=df, color="c");
plt.show();


#Scatter for attributes
colors = ['blue','red'];
plt.figure();
p = atts.shape[1]; # number of attributes
feature_selection = SelectKBest(k=2);
feature_selection.fit(atts,target);
scores = feature_selection.scores_; #attribute weight
features1 = feature_selection.transform(atts);
mask = feature_selection.get_support();
feature_indices = [];
for i in range(p):
    if mask[i] == True : feature_indices.append(i);
x_axis, y_axis = feature_indices;
print('Importance weight of input attributes')
for i in range(p):
    print(features1[i],': ',scores[i]);
plt.title('Scatterplot for 2 kbest attribs');
plt.xlabel(att_names[x_axis]);
plt.ylabel(att_names[y_axis]);
plt.scatter(atts2[:,0],atts2[:,1],s=50,c=target2, cmap=col.ListedColormap(colors));
plt.show();

#Divide data to test and train data
X_train, X_test, y_train, y_test = train_test_split(atts, target2, test_size=0.2, 
                                shuffle = True, random_state=2021);

# Fitting logistic regression
logreg_classifier = LogisticRegression(solver='liblinear')
logreg_classifier.fit(X_train,y_train)
score_train_logreg = logreg_classifier.score(X_train,y_train)
ypred_logreg = logreg_classifier.predict(X_train)
cm_logreg_train = confusion_matrix(y_train, ypred_logreg); # train confusion matrix
score_test_logreg = logreg_classifier.score(X_test,y_test)
ypred_logreg = logreg_classifier.predict(X_test)
cm_logreg_test = confusion_matrix(y_test, ypred_logreg); # test confusion matrix
yprobab_logreg = logreg_classifier.predict_proba(X_test)
scores = cross_val_score(logreg_classifier, atts, target, cv=5)
print("Logistic regression scores:")
print(scores)

#Results of logistic regression
plot_confusion_matrix(logreg_classifier, X_train, y_train, display_labels = target_names);
plt.title('Confusion matrix for TRAIN data (logreg)');
plot_confusion_matrix(logreg_classifier, X_test, y_test, display_labels = target_names);
plt.title('Confusion matrix for TEST data (logreg)');

# Plotting ROC curve
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, yprobab_logreg[:,1]);
roc_auc_logreg = auc(fpr_logreg, tpr_logreg);
plt.figure(7);
lw = 2;
plt.plot(fpr_logreg, tpr_logreg, color='red',
         lw=lw, label='Logistic regression (AUC = %0.2f)' % roc_auc_logreg);
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--');
plt.xlim([-0.01, 1.0]);
plt.ylim([-0.01, 1.05]);
plt.xlabel('False Positive Rate');
plt.ylabel('True Positive Rate');
plt.title('Receiver operating characteristic curve (logreg)');
plt.legend(loc="lower right");
plt.show();

#Fitting decision tree
depth = 3;
class_tree = DecisionTreeClassifier(criterion = "gini", max_depth=depth, min_samples_leaf=15)
class_tree.fit(X_train, y_train)
score_train = class_tree.score(X_train, y_train)
score_test = class_tree.score(X_test, y_test)
y_pred_gini = class_tree.predict(X_test)
yprobab_tree = class_tree.predict_proba(X_test)
fig = plt.figure(8,figsize = (12,6),dpi=100);
plot_tree(class_tree, feature_names = att_names, 
               class_names = target_names,
               filled = True, fontsize = 6)

plot_confusion_matrix(class_tree, X_train, y_train, display_labels = target_names);
plt.title('Confusion matrix for TRAIN data (decision tree)');
plot_confusion_matrix(class_tree, X_test, y_test, display_labels = target_names);
plt.title('Confusion matrix for TEST data (decision tree)');

# Plotting ROC curve
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, yprobab_tree[:,1]);
roc_auc_logreg = auc(fpr_logreg, tpr_logreg);
plt.figure(7);
lw = 2;
plt.plot(fpr_logreg, tpr_logreg, color='red',
         lw=lw, label='Logistic regression (AUC = %0.2f)' % roc_auc_logreg);
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--');
plt.xlim([-0.01, 1.0]);
plt.ylim([-0.01, 1.05]);
plt.xlabel('False Positive Rate');
plt.ylabel('True Positive Rate');
plt.title('Receiver operating characteristic curve (decision tree');
plt.legend(loc="lower right");
plt.show();


# Finding optimal cluster number
Max_K = 15;  # maximum cluster number
SSE = np.zeros((Max_K-2));
DB = np.zeros((Max_K-2));
for i in range(Max_K-2):
    n_c = i+2;
    kmeans = KMeans(n_clusters=n_c, random_state=2020);
    kmeans.fit(atts);
    labels = kmeans.labels_;
    SSE[i] = kmeans.inertia_;
    DB[i] = davies_bouldin_score(atts,labels);
    
# Visualization of SSE values    
plt.figure();
plt.title('Sum of squares of error curve');
plt.xlabel('Number of clusters');
plt.ylabel('SSE');
plt.plot(np.arange(2,Max_K),SSE, color='red')
plt.show();

# Visualization of DB scores
plt.figure();
plt.title('Davies-Bouldin score curve');
plt.xlabel('Number of clusters');
plt.ylabel('DB index');
plt.plot(np.arange(2,Max_K),DB, color='blue')
plt.show();

#KMEANS with optimal clusters
kmeans = KMeans(n_clusters=4, random_state=2020);
kmeans.fit(X_test);
data_labels = kmeans.labels_;
cm = contingency_matrix(y_test, kmeans.labels_);

