# libraries

import pandas as pd
import numpy as np
import datetime
import re
import os
import matplotlib.pyplot
from sklearn import svm
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn import linear_model
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn import metrics
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble 
from sklearn import neighbors
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
import seaborn as sns
import functools
get_ipython().magic('matplotlib inline')


# Constants
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