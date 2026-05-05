import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
 
def preprocessing(df,features):
    X = df[features].copy()
    numerical = X.select_dtypes(include=['int64', 'float64']).columns
    categorical = X.select_dtypes(include=['object']).columns
 
    encoder = LabelEncoder()
    scaler = StandardScaler()
 
    X[numerical] = scaler.fit_transform(X[numerical])
    for col in categorical:
        X[col] = encoder.fit_transform(X[col])
    return X
 
def clustering(X,n_cluster,linkage):
    hierarchical = AgglomerativeClustering(n_clusters=n_cluster, linkage=linkage)
    labels = hierarchical.fit_predict(X)
    return hierarchical, labels