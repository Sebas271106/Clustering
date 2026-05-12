import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
def preprocessing(df, features, scaler_type="standard"):
    X = df[features].copy()
    numerical = X.select_dtypes(include=["int64", "float64"]).columns
    categorical = X.select_dtypes(include=["object"]).columns

    encoder = LabelEncoder()
    if len(numerical) > 0:
        scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()
        X[numerical] = scaler.fit_transform(X[numerical])
    for col in categorical:
        X[col] = encoder.fit_transform(X[col])
    return X


def clustering(X, algorithm, *, n_clusters=3, linkage="ward", eps=0.5, min_samples=5):
    X_arr = X.values if hasattr(X, "values") else X
    if algorithm == "K-Means":
        model = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        labels = model.fit_predict(X_arr)
    elif algorithm == "K-Medoids":
        try:
            from sklearn_extra.cluster import KMedoids  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError(
                "K-Medoids requiere scikit-learn-extra: pip install scikit-learn-extra"
            ) from exc
        model = KMedoids(n_clusters=n_clusters, random_state=0)
        labels = model.fit_predict(X_arr)
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_arr)
    elif algorithm == "Hierarchical Clustering":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = model.fit_predict(X_arr)
    else:
        raise ValueError(f"Algoritmo no soportado: {algorithm}")
    return model, labels
