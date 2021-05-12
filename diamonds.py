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
diamonds = pd.read_csv('https://raw.githubusercontent.com/raczandras/ML/main/diamonds.csv', usecols=range(1,11));

#Replacing attributes to simple numbers
cols = ['price'];
cheap = diamonds['price'] <= 2000;
medium = diamonds['price'] <= 9000;
expensive = diamonds ['price'] > 9000;
diamonds['price'] = np.select([cheap, medium, expensive], [1,2,3], default=-1);
diamonds['cut'] = diamonds.cut.mask(diamonds.cut=="Fair", 1);
diamonds['cut'] = diamonds.cut.mask(diamonds.cut=="Good", 2);
diamonds['cut'] = diamonds.cut.mask(diamonds.cut=="Ideal", 3);
diamonds['cut'] = diamonds.cut.mask(diamonds.cut=="Premium", 4);
diamonds['cut'] = diamonds.cut.mask(diamonds.cut=="Very Good", 5);

diamonds['color'] = diamonds.color.mask(diamonds.color=="D", 1);
diamonds['color'] = diamonds.color.mask(diamonds.color=="E", 2);
diamonds['color'] = diamonds.color.mask(diamonds.color=="F", 3);
diamonds['color'] = diamonds.color.mask(diamonds.color=="G", 4);
diamonds['color'] = diamonds.color.mask(diamonds.color=="H", 5);
diamonds['color'] = diamonds.color.mask(diamonds.color=="I", 6);
diamonds['color'] = diamonds.color.mask(diamonds.color=="J", 7);

diamonds['clarity'] = diamonds.clarity.mask(diamonds.clarity=="I1", 1);
diamonds['clarity'] = diamonds.clarity.mask(diamonds.clarity=="IF", 2);
diamonds['clarity'] = diamonds.clarity.mask(diamonds.clarity=="SI1", 3);
diamonds['clarity'] = diamonds.clarity.mask(diamonds.clarity=="SI2", 4);
diamonds['clarity'] = diamonds.clarity.mask(diamonds.clarity=="VS1", 5);
diamonds['clarity'] = diamonds.clarity.mask(diamonds.clarity=="VS2", 6);
diamonds['clarity'] = diamonds.clarity.mask(diamonds.clarity=="VVS1", 7);
diamonds['clarity'] = diamonds.clarity.mask(diamonds.clarity=="VVS2", 8);
