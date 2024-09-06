#To implement analysis using machine learning algorithms

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import os
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np

def perform_svr_analysis(data, target_column):
    
    #Perform support vector regression on the data.takes  data ,target_column as arguments and returns trained svr model
    
    try:
        data = data.select_dtypes(include=[np.number])

        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        #standardize features
        scalar=StandardScaler()
        x_scaled=scalar.fit_transform(X)
        y_scaled = scalar.fit_transform(y.values.reshape(-1, 1)).flatten()
        
        x_train,x_test,y_train,y_test=train_test_split(x_scaled,y_scaled,test_size=0.2,random_state=42)
        
        # Train the model
        model = SVR(kernel='rbf')
        model.fit(x_train, y_train)
        preds=model.predict(x_test)
        
        return model,preds
        
        
    except Exception as e:
        print(f"Error in svr: {e}")
        return None,None

def perform_kmeans_clustering(data, n_clusters):
     #Perform kmeans clustering on the data. takes  data ,N_clusters as arguments and returns trained kmeans model
  
    try:
        #select only numerical columns for clustering 
        data_numeric=data.select_dtypes(include=[float,int])
        if data_numeric.empty:
            raise ValueError("No numeric data available for clustering")
        
        model =  KMeans(n_clusters=3, n_init=3)

        model.fit(data_numeric)
        
        data['Cluster'] = model.labels_
        print(f"K-Means clustering completed with {n_clusters} clusters.")
        return model
    except Exception as e:
        print(f"Error in K-Means clustering: {e}")
        return None

def perform_random_forest_analysis(data, target_column):
    #Perform random forest regression on the data.takes  data ,target_column as arguments and returns trained  random forest regression model
    try:
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        
        preds=model.predict(X_test)
        mse=mean_squared_error(y_test,preds)
        
        print(f"Random forest model trained with Mean Squared Error: {mse}")
        return model,preds
    except Exception as e:
        print(f"Error in decision tree classification: {e}")
        return None,None
    
    
def analysis_engine(data, algorithm, target_column=None,n_clusters=3):
  #wrapped up function to apply the chosen algorithm to the data.
    if algorithm == 'svr' and target_column:
        return perform_svr_analysis(data, target_column)
    elif algorithm == 'kmeans':
        print(n_clusters)
        return perform_kmeans_clustering(data,n_clusters)
    elif algorithm == 'randomforest' and target_column:
        return perform_random_forest_analysis(data, target_column)
    else:
        print("Invalid algorithm or missing target_column for selected algorithm.")
        return None,None
    

        