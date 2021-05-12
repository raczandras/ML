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
from sklearn.metrics import davies_bouldin_score;  # function for Davies-Bouldin goodness-of-fit 
from sklearn.metrics.cluster import contingency_matrix;
from sklearn.pipeline import make_pipeline;
from sklearn.preprocessing import StandardScaler;

#Getting the diamonds
diamonds = pd.read_csv('https://raw.githubusercontent.com/raczandras/ML/main/diamonds.csv');
