from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
# from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.cluster import KMeans

def plot_clusters(ax, embeddings, pred, method):
    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], c=pred, cmap='viridis', alpha=0.5)
    ax.set_title(f'Clustering Results using {method}')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.legend(*scatter.legend_elements(), title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')

def dim_red(mat, p, method):
    if method == 'ACP':
        pca = PCA(p)
        red_mat = pca.fit_transform(mat)
    elif method == 'TSNE':
        tsne = TSNE(n_components=2, random_state=42)
        red_mat = tsne.fit_transform(mat)
    elif method == 'UMAP':
        umap = UMAP(n_components=2, random_state=42)
        red_mat = umap.fit_transform(mat)
    else:
        raise Exception("Please select one of the three methods: APC, AFC, UMAP")

    return red_mat

def clust(mat, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    pred = kmeans.fit_predict(mat)
    return pred
  

def construct_and_save_embeddings() : 
    # import data
    ng20 = fetch_20newsgroups(subset='test')
    corpus = ng20.data[:2000]
    labels = ng20.target[:2000]
    k = len(set(labels))

    # embedding
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(corpus)

    # Save embeddings to a file
    save_embeddings(embeddings, 'embeddings.npy')


def evaluate() : 
    # import data
    ng20 = fetch_20newsgroups(subset='test')
    corpus = ng20.data[:2000]
    labels = ng20.target[:2000]
    k = len(set(labels))

    # embedding
    # Load embeddings from the file
    loaded_embeddings = load_embeddings('embeddings.npy')

    # Set up subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Perform dimensionality reduction and clustering for each method
    methods = ['ACP', 'TSNE', 'UMAP']
    for ax, method in zip(axs, methods):
        # Perform dimensionality reduction
        if method == "TSNE":
            red_emb = dim_red(loaded_embeddings, 3, method)
        else:
            red_emb = dim_red(loaded_embeddings, 20, method)

        # Perform clustering
        pred = clust(red_emb, k)

        # Plot clusters
        plot_clusters(ax, red_emb, pred, method)

        # Evaluate clustering results
        nmi_score = normalized_mutual_info_score(pred, labels)
        ari_score = adjusted_rand_score(pred, labels)

        # Print results
        print(f'Method: {method}\nNMI: {nmi_score:.2f} \nARI: {ari_score:.2f}\n')

    plt.tight_layout()
    plt.savefig("./results.png")
    plt.show()

def save_embeddings(embeddings, filename):
    np.save(filename, embeddings)

def load_embeddings(filename):
    return np.load(filename)


evaluate()