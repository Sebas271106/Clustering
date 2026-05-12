import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors

from model import clustering, preprocessing


def plot_cluster_size_distribution(labels):
    fig, ax = plt.subplots(figsize=(8, 4))
    unique, counts = np.unique(labels, return_counts=True)
    names = [f"Ruido (-1)" if int(u) == -1 else f"Cluster {int(u)}" for u in unique]
    ax.bar(names, counts, color="steelblue", edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Número de observaciones")
    ax.set_title("Distribución de puntos por cluster")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    return fig


def sweep_partition_metrics(X_model, algorithm, linkage_method, k_min=2, k_max=10):
    ks = list(range(k_min, k_max + 1))
    silhouettes = []
    db_scores = []
    inertias = []
    for k in ks:
        try:
            _, lab = clustering(
                X_model,
                algorithm,
                n_clusters=k,
                linkage=linkage_method,
                eps=0.5,
                min_samples=5,
            )
            mask = lab >= 0
            if np.sum(mask) < 2 or len(np.unique(lab[mask])) < 2:
                silhouettes.append(np.nan)
                db_scores.append(np.nan)
            else:
                silhouettes.append(silhouette_score(X_model[mask], lab[mask]))
                db_scores.append(davies_bouldin_score(X_model[mask], lab[mask]))
            km = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X_model)
            inertias.append(km.inertia_)
        except Exception:
            silhouettes.append(np.nan)
            db_scores.append(np.nan)
            inertias.append(np.nan)
    return ks, silhouettes, db_scores, inertias


def plot_k_comparison(X_model, algorithm, linkage_method, show_inertia):
    ks, sils, dbs, inertias = sweep_partition_metrics(X_model, algorithm, linkage_method)
    n_rows = 3 if show_inertia else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 3 * n_rows))
    if n_rows == 2:
        axes = list(axes)
    else:
        axes = list(axes)

    axes[0].plot(ks, sils, "o-", color="tab:blue")
    axes[0].set_xlabel("k (número de clusters)")
    axes[0].set_ylabel("Silhouette (mayor es mejor)")
    axes[0].set_xticks(ks)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ks, dbs, "o-", color="tab:orange")
    axes[1].set_xlabel("k (número de clusters)")
    axes[1].set_ylabel("Davies–Bouldin (menor es mejor)")
    axes[1].set_xticks(ks)
    axes[1].grid(True, alpha=0.3)

    if show_inertia and len(axes) > 2:
        axes[2].plot(ks, inertias, "o-", color="tab:green")
        axes[2].set_xlabel("k (número de clusters)")
        axes[2].set_ylabel("Inercia (WCSS)")
        axes[2].set_xticks(ks)
        axes[2].grid(True, alpha=0.3)

    if show_inertia:
        fig.suptitle(
            "Comparación por k — Silhouette, Davies–Bouldin e inercia (método del codo, K-Means)",
            fontsize=11,
        )
    else:
        fig.suptitle(f"Comparación por k — Silhouette y Davies–Bouldin ({algorithm})", fontsize=11)
    fig.tight_layout()
    return fig


def plot_silhouette_diagram(X_eval, y_eval):
    cluster_ids = np.unique(y_eval)
    n_clusters = len(cluster_ids)
    sil_all = silhouette_samples(X_eval, y_eval)
    score_avg = float(np.mean(sil_all))
    fig, ax = plt.subplots(figsize=(9, 6))
    y_lower = 10
    cmap = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 3)))

    for idx, cid in enumerate(cluster_ids):
        vals = sil_all[y_eval == cid]
        vals.sort()
        size = vals.shape[0]
        y_upper = y_lower + size
        color = cmap[idx % len(cmap)]
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size, str(int(cid)))
        y_lower = y_upper + 10

    ax.axvline(score_avg, color="red", linestyle="--", linewidth=1.5, label=f"Media = {score_avg:.3f}")
    ax.set_title("Silhouette por muestra (modelo actual)")
    ax.set_xlabel("Coeficiente de silhouette")
    ax.set_ylabel("Índice de muestra (agrupado por cluster)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_dendrogram_safe(X_model, linkage_method, max_leaf):
    X64 = np.asarray(X_model, dtype=np.float64)
    n = X64.shape[0]
    if n > max_leaf:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_leaf, replace=False)
        X64 = X64[idx]
        subtitle = f"Muestra aleatoria de {max_leaf} filas (dataset tiene {n})."
    else:
        subtitle = "Todas las observaciones."
    Z = scipy_linkage(X64, method=linkage_method)
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, ax=ax, truncate_mode="level", p=8)
    ax.set_title(f"Dendrograma (enlace: {linkage_method})")
    ax.set_xlabel("Índices / subclusters fusionados")
    ax.set_ylabel("Distancia")
    fig.text(0.5, 0.01, subtitle, ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return fig


def plot_kdistance(X_model, min_samples):
    X_arr = np.asarray(X_model, dtype=np.float64)
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X_arr)
    dists, _ = nn.kneighbors(X_arr)
    kdist = np.sort(dists[:, -1])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(len(kdist)), kdist, color="purple")
    ax.set_xlabel("Puntos ordenados por distancia")
    ax.set_ylabel(f"Distancia al vecino {min_samples}-ésimo")
    ax.set_title("Gráfico k-distancia (orientativo para elegir eps en DBSCAN)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


st.title("Hands-on: Streamlit — modelos no supervisados")

file = st.file_uploader("Sube un archivo CSV", type=["csv"])
if file is None:
    st.info("Sube un conjunto de datos en formato CSV para comenzar.")
    st.stop()

df = pd.read_csv(file)

st.subheader("Vista previa del dataset")
st.dataframe(df.head())

st.subheader("Selección de características")
feature_mode = st.radio(
    "Modo de características",
    ["Selección manual", "Reducción con PCA"],
    horizontal=True,
)

available_cols = df.columns.tolist()
features = st.multiselect(
    "Columnas a usar"
    + (" para el clustering" if feature_mode == "Selección manual" else " antes de PCA"),
    available_cols,
)

if len(features) < 2:
    st.warning("Selecciona al menos 2 columnas.")
    st.stop()

n_components = 2
if feature_mode == "Reducción con PCA":
    max_components = len(features)
    n_components = int(
        st.number_input(
            "Número de componentes PCA (mínimo 2)",
            min_value=2,
            max_value=max_components,
            value=min(2, max_components),
            step=1,
        )
    )

st.subheader("Preprocesado")
scaler_choice = st.selectbox(
    "Escalado de variables numéricas",
    ["standard", "minmax"],
    format_func=lambda x: "StandardScaler (media 0, var 1)" if x == "standard" else "MinMaxScaler [0, 1]",
)
st.caption(
    "Las variables categóricas se codifican con LabelEncoder; las numéricas se escalan según tu elección."
)

X_encoded = preprocessing(df, features, scaler_type=scaler_choice)

if feature_mode == "Reducción con PCA":
    pca_reduce = PCA(n_components=n_components)
    X_model = pca_reduce.fit_transform(X_encoded.values)
else:
    X_model = X_encoded.values

st.subheader("Modelo de clustering")
algorithm = st.selectbox(
    "Algoritmo",
    ["K-Means", "DBSCAN", "Hierarchical Clustering"],
)

n_clusters = 3
linkage = "ward"
eps = 0.5
min_samples = 5

if algorithm in ("K-Means", "K-Medoids", "Hierarchical Clustering"):
    n_clusters = st.slider("Número de clusters", 2, 10, 3)
if algorithm == "Hierarchical Clustering":
    linkage = st.selectbox(
        "Tipo de enlace (linkage)",
        ["ward", "complete", "average", "single"],
    )
if algorithm == "DBSCAN":
    eps = st.slider("eps (vecindad)", 0.05, 5.0, 0.5, step=0.05)
    min_samples = st.slider("min_samples", 2, 50, 5)

try:
    _, labels = clustering(
        X_model,
        algorithm,
        n_clusters=n_clusters,
        linkage=linkage,
        eps=eps,
        min_samples=min_samples,
    )
except ImportError as e:
    st.error(str(e))
    st.stop()

df_out = df.copy()
df_out["Cluster"] = labels

st.subheader("Métricas (cuando aplica)")
mask_eval = labels >= 0
n_unique = len(np.unique(labels[mask_eval])) if np.any(mask_eval) else 0

if algorithm == "DBSCAN":
    n_noise = int(np.sum(labels == -1))
    st.write(f"Puntos clasificados como ruido (etiqueta -1): **{n_noise}**")

if n_unique >= 2 and np.sum(mask_eval) >= 2:
    X_eval = X_model[mask_eval]
    y_eval = labels[mask_eval]
    try:
        sil = silhouette_score(X_eval, y_eval)
        db = davies_bouldin_score(X_eval, y_eval)
        col1, col2 = st.columns(2)
        col1.metric("Silhouette score", f"{sil:.4f}")
        col2.metric("Davies–Bouldin (menor es mejor)", f"{db:.4f}")
    except Exception as exc:
        st.caption(f"No se pudieron calcular métricas: {exc}")
else:
    st.caption("Silhouette / Davies–Bouldin requieren al menos 2 clusters con muestras suficientes.")

st.subheader("Visualización de clusters (2D)")
if feature_mode == "Reducción con PCA" and X_model.shape[1] >= 2:
    X_plot = X_model[:, :2]
    xlab, ylab = "Componente 1", "Componente 2"
elif X_model.shape[1] == 2:
    X_plot = X_model
    xlab, ylab = "Dimensión 1", "Dimensión 2"
else:
    viz_pca = PCA(n_components=2)
    X_plot = viz_pca.fit_transform(X_model)
    xlab, ylab = "PCA 1 (solo visualización)", "PCA 2 (solo visualización)"

fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
scatter = ax_scatter.scatter(X_plot[:, 0], X_plot[:, 1], c=labels, cmap="tab10", alpha=0.75)
ax_scatter.set_xlabel(xlab)
ax_scatter.set_ylabel(ylab)
ax_scatter.set_title(f"{algorithm} — clusters")
plt.colorbar(scatter, ax=ax_scatter, label="Etiqueta")
st.pyplot(fig_scatter)
plt.close(fig_scatter)

st.subheader("Análisis del modelo")

tab_sizes, tab_k, tab_sil, tab_extra = st.tabs(
    ["Distribución por cluster", "Comparación por k", "Silhouette detallado", "Análisis específico"]
)

with tab_sizes:
    fig_sz = plot_cluster_size_distribution(labels)
    st.pyplot(fig_sz)
    plt.close(fig_sz)

with tab_k:
    if algorithm == "DBSCAN":
        st.info(
            "DBSCAN no usa un k fijo: la comparación por k aplica a K-Means, K-Medoids o clustering jerárquico."
        )
    else:
        show_inertia = algorithm == "K-Means"
        st.caption(
            "Las curvas Silhouette y Davies–Bouldin usan el algoritmo seleccionado. "
            "La inercia se muestra con **K-Means** como referencia común para el método del codo."
        )
        fig_k = plot_k_comparison(X_model, algorithm, linkage, show_inertia=show_inertia)
        st.pyplot(fig_k)
        plt.close(fig_k)

with tab_sil:
    if n_unique >= 2 and np.sum(mask_eval) >= 2:
        try:
            fig_sil = plot_silhouette_diagram(X_eval, y_eval)
            st.pyplot(fig_sil)
            plt.close(fig_sil)
        except Exception as exc:
            st.caption(f"No se pudo dibujar el diagrama de silhouette: {exc}")
    else:
        st.caption("Se necesitan al menos 2 clusters válidos (sin contar ruido aislado) para el diagrama.")

with tab_extra:
    if algorithm == "Hierarchical Clustering":
        max_leaf = st.slider("Máximo de filas en el dendrograma", 30, 250, 120, step=10)
        fig_den = plot_dendrogram_safe(X_model, linkage, max_leaf=max_leaf)
        st.pyplot(fig_den)
        plt.close(fig_den)
    elif algorithm == "DBSCAN":
        st.caption(
            "Busca el primer “codo” en la curva: valores de eps algo por debajo de ese tramo suelen separar mejor densidades."
        )
        fig_kd = plot_kdistance(X_model, min_samples)
        st.pyplot(fig_kd)
        plt.close(fig_kd)
    else:
        st.caption(
            "El dendrograma está disponible al elegir **Hierarchical Clustering**. "
            "El gráfico k-distancia aparece con **DBSCAN**."
        )

st.subheader("Datos con etiqueta de cluster")
st.dataframe(df_out)
