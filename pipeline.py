# libraries

import pandas as pd
import numpy as np
import datetime
import re
import os
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
#from sklearn import 
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression, LassoCV, RidgeCV
from sklearn import linear_model
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
import random
import matplotlib.pyplot as plt
from scipy import optimize
import time
import seaborn as sns
import functools
from sklearn import svm
from sklearn import linear_model
from sklearn import metrics
from sklearn import tree
from sklearn import neighbors
get_ipython().magic('matplotlib inline')


#Constants
SKLEARN_MODELS = {
    'logistic_regression': linear_model.LogisticRegressionCV(), 
    'knn': neighbors.KNeighborsClassifier(),
    'decision_tree': tree.DecisionTreeClassifier(), 
    'random_forest': ensemble.RandomForestClassifier(), 
    'svm': svm.SVC(), 
    'boosting': ensemble.AdaBoostClassifier(),
    'bagging': ensemble.BaggingClassifier()
}


# make pandas dataframe from data
def get_data(path, filetype):
	'''
	function to get data
	inputs: 
		path - a string of the path where the data are stored
		filetype - a string of the type of file being read in
	outputs: 
		df - dataframe fo the data being pulled
	'''
	if filetype == "csv":
		df = pd.read_csv(path)
	elif filetype == 'json':
		df = pd.read_json(path) #add to this if needed in the future
	else:
		print('Error: Use a string of the path and a string of the filetype being read')

	return df

# get correlations
def get_correlations(dataframe):
	'''
	makes correlations of each of the columns of the dataframe
	inputs: dataframe
	outputs: correlations of the variables
	'''
	currcorr = dataframe.corr()
	return sns.heatmap(currcorr, cmap = "RdBu", vmin=-1, vmax=1)


# explore data

# scatterplot to see correlations between variables that aren't delinquency, personID, or zipcode.
def make_scatter(dataframe):
	'''
	function to make scatterplots showing how each variable changes based on whether they are
	delinquent or not
	input: dataframe
	output: scatterplots!
	'''
	for col in list(dataframe):
		for cols in list(dataframe):
			dataframe.plot.scatter(col, cols, figsize = (10,5))
			plt.title(col + ' vs ' + cols)
			plt.xlabel(col)
			plt.ylabel(cols)
			
			plt.show()

def make_boxplots(dataframe, by):
	'''
	function to make boxplots showing how each variable changes based on whether they are
	delinquent or not
	input: dataframe
	output: boxplots!
	'''
	for col in list(dataframe):
		dataframe.boxplot(col, by=by)
		plt.title('vs ' + col)
		plt.xlabel(by)
		plt.ylabel(col)
		plt.show()




def make_zscore_df(dataframe):
	'''
	calculates a dataframe of the z-score of all data points to identify which are outliers
	inputs: dataframe and the columns that won't be used to 
	'''    
	cols = list(dataframe)
	zscore_df = pd.DataFrame(columns=['num_outliers'])
	
	for col in cols:
		col_zscore = col + '_zscore'
		zscore_single_col = (dataframe[col] - dataframe[col].mean())/dataframe[col].std()
		zscore_df.loc[:,col_zscore] = zscore_single_col
	return zscore_df


# calculate number of outliers in the row
def calculate_outliers_per_row(zscore_df, score, threshold):
	'''
	function to calculate the number of outliers in the row and put that sum in a column
	inputs: zscore df made from make zscore df function and a predetermined z-score
	output:
	'''
	trans = zscore_df.transpose()
	sumsgreat = (trans > score).sum()
	sumsless = (trans < -score).sum()
	sums = sumsgreat + sumsless
	zscore_df['num_outliers'] = sumsgreat + sumsless
 
	# figure out which rows have any and many outliers.
	trans2 = zscore_df.transpose()
	rows_with_any_outliers = []
	rows_with_many_outliers = []
	for x in list(trans2):
		if trans2[x]['num_outliers'] > threshold: # if more than 33% of the entries in it are outliers, remove that row from the z-score table.
			rows_with_many_outliers.append(x)
			#print(trans2.head())
			trans2 = trans2.drop([x], axis=1)
		#if trans2[x]['num_outliers'] > 0:
		#	rows_with_any_outliers.append(x)
	return rows_with_many_outliers#, rows_with_any_outliers


# are there any nas?
def find_nas(dataframe):
	'''
	find if there are any columns with nas in the DataFrame
	input: dataframe
	output: each column with a boolean True or False if have any nas
	'''
	return dataframe.isna().any()

def cols_with_nas(series):
	return series.index[series].tolist()


# created function to fill nas based on specific inputs user want to use.
def fillnas_with_data(dataframe, stat, variables, object_cols):
    '''
    fill the nas with different values based on inputs.
    inputs:
        dataframe: either with outliers, without rows with many outliers, or without rows with at least one outlier
        stat: either median or mean
        variables: the variables to be filled.
    outputs: dataframe with nas filled
    '''

    for var in variables:
        if stat == 'median' and var not in object_cols:
            fill_type = 'median_' + var
            fill_name = dataframe[var].median()
            dataframe[var] = dataframe[var].fillna(fill_name)
        elif stat == 'mean' and var not in object_cols:
            fill_type = 'mean_' + var
            fill_name = dataframe[var].mean()
            dataframe[var] = dataframe[var].fillna(fill_name)
        elif stat == 'mode' and var not in object_cols:
            fill_type = 'mode_' + var
            fill_name = dataframe[var].mode()
            dataframe[var] = dataframe[var].fillna(fill_name)
        else:
            print('use only mean, median, or mode for stat')
    return dataframe


def change_to_one_zero(dataframe, value_for_one, value_for_zero):
	dataframe.replace((value_for_one, value_for_zero), (1, 0), inplace=True)
	return dataframe


def discretize_variables(dataframe, col_name, buckets, cut_type):
    '''
    function to discretize variables.
    inputs:
        dataframe: cred_df, cred_df_no_outliers, cred_df_less_outliers
        col_name: the name of the column from the dataframe user wants to discretize
        buckets: number of buckets want to discretize by
        cut_type: way to make the buckets - either equal-width bins (cut), or quantile bins (qcut)
    output:
        dataframe with a column that's changed from continuous to discrete.
    '''
    if cut_type == 'cut' or 'Cut':
        dataframe[col_name] = pd.cut(dataframe[col_name], buckets)
    elif cut_type == 'qcut' or 'Qcut':
        dataframe[col_name] = pd.qcut(dataframe[col_name], buckets)
    else:
        print('Error: Please only use cut or qcut')
    return dataframe


# create binary/dummy variables from categorical variable
def dummify_categories(dataframe, cols_with_nas, object_cols, ignore_with_predictor):
	'''
	function to create dummy variables from categorized data.
	inputs: 
		dataframe: either cred_df, cred_df_no_outliers, or cred_df_less_outliers
		col_name: column name of column to dummify
	outputs: pandas series of dummy data
	'''
	for col_name in object_cols:
		if (len(dataframe[col_name].unique()) <= 100) and (col_name in cols_with_nas) and (col_name not in ignore_with_predictor):
			dummies = pd.get_dummies(data=dataframe[col_name])
			dummies.columns = [str(col_name)+"_"+x for x in dummies.columns]
			dataframe = pd.concat([dataframe, dummies], axis=1)
	return dataframe

def sort_zip(t0, t1):
    '''
    CUSTOM SORT FUNCTION
    '''
    return t1[1] - t0[1]

def rf_features(df=None, var_excl=None, y_pred=None, n_jobs=10, random_state=0):
    '''
    PURPOSE: runs a random forest classifier for the sake of producing a list of features
        to include in a model 
        
    INPUTS: 
        df (pd dataframe): the dataframe to use
        var_excl (list of strings): a list of types of data to exclude
        y_pred (str): colname for the predicted variable
        n_jobs (int): sklearn default
        random_state (int): sklearn default
        
    RETURNS:
        sorted_coef_list (list): list of features to include in ranked order
    '''
    rf = ensemble.RandomForestClassifier(n_jobs=n_jobs, random_state=random_state)
    rf.fit(df.select_dtypes(exclude=var_excl), df[y_pred])
    coef_list = list(zip(df.select_dtypes(exclude=var_excl), rf.feature_importances_))
    
    return elim_zero_coef(coef_list = coef_list)


def elim_zero_coef(coef_list=None):
    '''
    PURPOSE: eliminate all coefficients in a list with 0 influence
    INPUT: coef_list (list of coeffs)
    RETURNS: sorted feature list with only important coeffs (important means != 0)
    '''
    rl = []
    for (x, y) in coef_list:
        if y != 0:
            rl.append((x,y))
            
    return sorted(rl, key=functools.cmp_to_key(sort_zip))

def lcv_features(df=None, y_pred=None, var_excl=None, features=None):
    '''
    PURPOSE: runs a lasso  regression for the sake of producing a list of features
        to include in a model 
        
    INPUTS: 
        df (pd dataframe): the dataframe to use
        y_pred (str): colname for the predicted variable
        var_excl (list): datatypes to exclude
        features (list, optional): list of features to include
        
    RETURNS:
        sorted_coef_list (list): list of features to include in ranked order
    '''
    df_use = df.select_dtypes(exclude=var_excl)
    if features is None:
        features = []
        for x in df_use.columns:
            if x != y_pred:
                features.append(x)
    lcv = LassoCV()
    lcv.fit(df_use[features], df_use[y_pred])
    coef_list = list(zip(features, abs(lcv.coef_)))
    return elim_zero_coef(coef_list = coef_list)


def rcv_features(df=None, y_pred=None, var_excl=None, features=None):
	'''
	PURPOSE: runs a ridge regression for the sake of producing a list of features
		to include in a model

	INPUTS: 
		df (pd dataframe): the dataframe to use
		y_pred (str): colname for the predicted variable
		var_excl (list): datatypes to exclude
		features (list, optional): list of features to include

	RETURNS:
		sorted_coef_list (list): list of features to include in ranked order
	'''

	df_use = df.select_dtypes(exclude=var_excl)
	if features is None:
		features = []
		for x in df_use.columns:
			if x != y_pred:
				features.append(x)
	rcv = LassoCV()
	rcv.fit(df_use[features], df_use[y_pred])
	coef_list = list(zip(features, abs(rcv.coef_)))
	return elim_zero_coef(coef_list = coef_list)


# for jupyter notebooks
#%matplotlib inline

# if you're running this in a jupyter notebook, print out the graphs
NOTEBOOK = 0

def define_clfs_params(grid_size):
    """Define defaults for different classifiers.
    Define three types of grids:
    Test: for testing your code
    Small: small grid
    Large: Larger grid that has a lot more parameter sweeps
    """

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3) 
            }

    large_grid = { 
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }
    
    small_grid = { 
    'RF':{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [10,100], 'learning_rate' : [0.001,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }
    
    test_grid = { 
    'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'LR': { 'penalty': ['l1'], 'C': [0.01]},
    'SGD': { 'loss': ['perceptron'], 'penalty': ['l2']},
    'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
    'NB' : {},
    'DT': {'criterion': ['gini'], 'max_depth': [1],'min_samples_split': [10]},
    'SVM' :{'C' :[0.01],'kernel':['linear']},
    'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']}
           }
    
    if (grid_size == 'large'):
        return clfs, large_grid
    elif (grid_size == 'small'):
        return clfs, small_grid
    elif (grid_size == 'test'):
        return clfs, test_grid
    else:
        return 0, 0

# a set of helper function to do machine learning evalaution

def joint_sort_descending(l1, l2):
    # l1 and l2 have to be numpy arrays
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]

def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary

def precision_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    #precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    #precision = precision[1]  # only interested in precision for label 1
    precision = precision_score(y_true, preds_at_k)
    return precision

def plot_precision_recall_n(y_true, y_prob, model_name):
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()
    

def clf_loop(models_to_run, clfs, grid, X, y):
    """Runs the loop using models_to_run, clfs, gridm and the data
    """
    results_df =  pd.DataFrame(columns=('model_type','clf', 'parameters', 'auc-roc','p_at_5', 'p_at_10', 'p_at_20'))
    tscv = TimeSeriesSplit(n_splits=3)
    for n in range(1, 2):
        # create training and valdation sets
        for train_index, test_index in tscv.split(X):
            print("TRAIN:", train_index, "TEST:", test_index)
            print(X)
            print(y)
            print((X[train_index]))
            X_train = (X[train_index])
            print((X[train_index]))
            X_test = (X[test_index])
            y_train = (y[train_index])
            y_test = (y[test_index])
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
            for index,clf in enumerate([clfs[x] for x in models_to_run]):
                print(models_to_run[index])
                parameter_values = grid[models_to_run[index]]
                for p in ParameterGrid(parameter_values):
                    try:
                        clf.set_params(**p)
                        y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                        # you can also store the model, feature importances, and prediction scores
                        # we're only storing the metrics for now
                        y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                        results_df.loc[len(results_df)] = [models_to_run[index],clf, p,
                                                       roc_auc_score(y_test, y_pred_probs),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0)]
                        if NOTEBOOK == 1:
                            plot_precision_recall_n(y_test,y_pred_probs,clf)
                    except IndexError as e:
                        print('Error:',e)
                        continue
    return results_df

def get_training_testing_datasets(df=None, time_col=None, period=None):
	'''
	PURPOSE: Get test/train splits by time
		    
	INPUTS: 
		df (pd dataframe): the dataframe to analyze
		time_col (str): the name of the time column
		period (str of the format '1M', '3M', '6M', etc.): how long the testing period should be

	RETURNS: 
		test_dfs (list of pd dataframes): list of testing dfs pairwise bound to the training dfs
		train_dfs (list of pd dataframes): list of training dfs pairwise bound to the testing dfs
		time_starts[:-1] (list): list of start times indexed to test and train lists
	
	'''
	time_starts=pd.date_range(start=df[time_col].min(), end=df[time_col].max(), freq=period)
	keep_times = []
	train_dfs = []
	test_dfs = []
	temp_df = df.copy()

	for i, time in enumerate(time_starts[:-1]):
		test_mask = (temp_df[time_col] > time) & (temp_df[time_col] <= time_starts[i+1])
		train_mask = (temp_df[time_col] <= time)
		test = temp_df.loc[test_mask]
		train = temp_df.loc[train_mask]
		if (test.shape[0] != 0) and (train.shape[0] != 0):
			test_dfs.append(test)
			train_dfs.append(train)
			keep_times.append(time)

	return(test_dfs, train_dfs, keep_times)

def create_validate_clf(train_data=None, test_data=None, threshold=None,
                        feature_list=None, y_column=None, clf_list=None):
    '''
    PURPOSE: To  create and validate all models in one fell swoop
    
    INPUTS:
        train_data (pandas df): the training data
        threshold (float): threshold for probabilities to be 1
        test_data (pandas df): the testing data
        feature_list (list of str): list of colnames to treat as features
        y_column (str): name of column to predict
        clf_list (list of str): list of classifiers to include, defaults to all
        
    RETURNS:
        models (dict): dictionary of type of clf to their models on the training set
        evaluations (dict): dictionary of type of clf to its evaluation metrics on the testing data
    '''
    clf_list = ['logistic_regression', 'knn', 'decision_tree', 'random_forest', 'svm', 
                'boosting', 'bagging'] if clf_list is None else clf_list
    if threshold is None:
        print ("Setting threshold to list")
        threshold = 0.5
        #threshold = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
    models = {}
    evaluations = {}
    #x_num = 0
    #for x in threshold:   
    #x_num += 1
    #print("x:", x, "x counter:", x_num, "out of:", len(threshold)) 	
    mod_num = 0
    for mod in clf_list:
        model = Pipeline([
            ('scaler',StandardScaler()),
            ('clf', SKLEARN_MODELS[mod])
        ])
        model = model.fit(train_data[feature_list], train_data[y_column])
        models[mod] = model
        if mod != 'svm':
            y_pred = model.predict_proba(test_data[feature_list])        
            y_metric = [1 if (i >= threshold) else 0 for i in y_pred[:,1]]
        else: 
            y_metric = model.predict(test_data[feature_list])
        accuracy = metrics.accuracy_score(test_data[y_column],y_metric)
        f1 = metrics.f1_score(test_data[y_column],y_metric)
        precision = metrics.precision_score(test_data[y_column],y_metric)
        recall = metrics.recall_score(test_data[y_column],y_metric)
        roc_auc = roc_auc_score(test_data[y_column], y_metric)
        evaluations[mod] = {'accuracy': accuracy, 'f1': f1, 'recall': recall, 
                        'precision': precision, 'roc': roc_auc}
        mod_num +=1
        print("mod", mod, "mod counter:", mod_num, "out of:", len(clf_list))
       
    return models, evaluations