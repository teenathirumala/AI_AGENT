# To generate reports with visualisation

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as Confusion_matrix

def plot_svr(data, target_column, model):
    #function to plot the support vector regression results.
    from sklearn.preprocessing import StandardScaler
    
    X = data.drop(columns=[target_column],axis=1)
    y = data[target_column]
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
    
    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, color='blue', label='Predicted vs Actual')
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red', label='Ideal Fit')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Support Vector Regression: {target_column}')
    plt.legend()
    plt.show()
    
    

def plot_kmeans_clusters(data, model):
    #function to plot the kmeans clustering results.
    data.loc[:, 'Cluster'] = model.labels_

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x=data.columns[0], y=data.columns[1], hue='Cluster', palette='viridis',s=100)
    plt.title('K-Means Clustering')
    plt.show()

def plot_random_forest_regression(data, target_column, model):
    #function to plot the random forest regression results.
    X = data.drop(columns=[target_column],axis=1)
    y = data[target_column]
    y_pred=model.predict(X);
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, color='blue', label='Predicted vs Actual')
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red', label='Ideal Fit')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Random forest regression:{target_colum}')
    plt.legend()
    plt.show()
def generate_summary(algorithm, model, data, target_column):
   #Generate a written summary based on the algorithm and model used.
    summary = f"Analysis Summary for {algorithm.capitalize()}:\n"
    if algorithm == 'svr':
        summary += f"SVR Kernel: {model.kernel}\n"
        summary += f"SVR C: {model.C}\n"
        summary += f"SVR Epsilon: {model.epsilon}\n"
        summary += f"R^2 Score: {model.score(data.drop(columns=target_column, axis=1), data[target_column]):.4f}\n"
    elif algorithm == 'kmeans':
        summary += f"Number of clusters: {model.n_clusters}\n"
        summary += f"Cluster Centers:\n{model.cluster_centers_}\n"
        
    elif algorithm == 'random_forest':
        summary += f"Number of Trees: {model.n_estimators}\n"
        summary += f"Feature Importances:\n{model.feature_importances_}\n"
        summary += f"R^2 Score: {model.score(data.drop(columns=target_column, axis=1), data[target_column]):.4f}\n"

    return summary

def generate_report(algorithm, model, data, target_column=None):
    #Generate a full report including visualizations and a written summary.
  try:
    if algorithm == 'svr' and target_column:
        plot_svr(data, target_column, model)
        summary = generate_summary('svr', model, data,target_column)
    elif algorithm == 'kmeans':
        plot_kmeans_clusters(data, model)
        summary = generate_summary('kmeans', model, data,target_column)
    elif algorithm == 'randomforest' and target_column:
        plot_random_forest_regression(data, target_column, model)
        summary = generate_summary('randomforest', model, data,target_column)
    else:
        print("invalid algorithm or missing target column for the report generation")
        return
    print(summary)
    # Generate summary
  except Exception as e:
        print(f"Error in generating report")
        return None
 
