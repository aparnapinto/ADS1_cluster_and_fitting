# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 16:43:11 2023

@author: aparn
"""

#Importing modules
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import warnings
import scipy.optimize as opt

#Ignore warnings
warnings.filterwarnings('ignore')

# Defining a function to read a datafile
def read(datafile):
    """
    This function is for reading datafile and its transpose and return the
    functions.
    
    """
    
    df = pd.read_csv(datafile)
    
    df.drop(['Series Name', 'Series Code', 'Country Code'], axis=1, inplace=True)
    df_transpose = df.set_index("Country Name").transpose()
    return df, df_transpose

#Reading the csv file to get a return of dataframes
df_gni, df_gni_transpose = read("GNI_per_capita.csv")
print(df_gni.describe())

def heat_corr(df_gni, size=10):
    """
    Function creates heatmap of correlation matrix for each pair of columns
    in the dataframe.
    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot (in inch)
    """
    corr = df_gni.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(corr, cmap='coolwarm')
    # setting ticks to column names
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    
    plt.show()
    return

def norm(array):
    """ Returns array normalised to [0,1]. Array can be a numpy array
    or a column of a dataframe"""
    min_val = np.min(array)
    max_val = np.max(array)
   
    scaled = (array-min_val) / (max_val-min_val)  
   
    return scaled

def norm_df(df, first=0, last=None):
    """
    Returns all columns of the dataframe normalised to [0,1] with the
    exception of the first (containing the names)
    Calls function norm to do the normalisation of one column, but
    doing all in one function is also fine.
    First, last: columns from first to last (including) are normalised.
    Defaulted to all. None is the empty entry. The default corresponds
    """
   
    # iterate over all numerical columns
    for col in df.columns[first:last]:
        df[col] = norm(df[col])
        return df

#Calling the function
heat_corr(df_gni)

pd.plotting.scatter_matrix(df_gni, figsize=(10.0, 10.0))
plt.tight_layout() 
plt.show()

# extracting columns for fitting
df_fit = df_gni[["2008 [YR2008]", "2018 [YR2018]"]].copy()
df_fit.fillna(df_fit.mean(), inplace=True)
# normalising dataframe and inspect result
df_fit = norm_df(df_fit)

for ic in range(2, 12):
    # setting up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_fit)    
    # extracting labels and calculate silhoutte score
    labels = kmeans.labels_
    print (ic, skmet.silhouette_score(df_fit, labels))
    
# Plot for two clusters
kmeans = cluster.KMeans(n_clusters=2)
kmeans.fit(df_fit)    
# extracting labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_
plt.figure(figsize=(6.0, 6.0))

plt.scatter(df_fit["2008 [YR2008]"], df_fit["2018 [YR2018]"], c=labels, cmap="Accent")

# show cluster centres
for ic in range(2):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=10)
   
plt.xlabel("2008", fontsize=16)
plt.ylabel("2018", fontsize=16)
plt.title("2 clusters", fontsize=22)
plt.show()

def exp_growth(t, scale, growth):
    """ 
    This function is for computing exponential function with scale and growth as free parameters.
    """
    f = scale * np.exp(growth * (t-1950))
    return f

#Reading the csv file to get a return of dataframes
df_cereal, df_cereal_transpose = read("cereal.csv")
print(df_cereal.describe())

years = np.array(df_cereal_transpose.index.values)
Argentina = np.array(df_cereal_transpose.Argentina.values)

#fitting exponential growth
cereal, covar = opt.curve_fit(exp_growth, years,Argentina)
print("Fit parameter", cereal)

# use *cereal to pass on the fit parameters
df_cereal_transpose["cereal_exp"] = exp_growth(Argentina, *cereal)
plt.figure()
plt.plot(years, Argentina, label="data")
plt.plot(years, df_cereal_transpose["cereal_exp"], label="fit")
plt.legend()
plt.title("First fit attempt")
plt.xlabel("Years")
plt.ylabel("Cereal yield of Argentina")
plt.show()
print()
 
cereal = [-0.5, 0]
df_cereal_transpose["cereal_exp"] = exp_growth(Argentina, *cereal)
plt.figure()
plt.plot(years, Argentina, label="data")
plt.plot(years, df_cereal_transpose["cereal_exp"], label="fit")
plt.legend()
plt.xlabel("Years")
plt.ylabel("Cereal yield of Argentina")
plt.title("Improved start value")
plt.show()

# fitting exponential growth
cereal, covar = opt.curve_fit(exp_growth, years,Argentina, p0=[-0.5, 0])
# much better
print("Fit parameter", cereal)
df_cereal_transpose["cereal_exp"] = exp_growth(Argentina, *cereal)
plt.figure()
plt.plot(years, Argentina, label="data")
plt.plot(years, df_cereal_transpose["cereal_exp"], label="fit")
plt.legend()
plt.xlabel("Years")
plt.ylabel("Cereal yield of Argentina")
plt.title("Final fit exponential growth")
plt.show()
print()