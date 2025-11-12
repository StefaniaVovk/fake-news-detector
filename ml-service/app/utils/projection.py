import numpy as np
from sklearn.manifold import TSNE
import umap

def compute_tsne(embeddings, n_components=2, perplexity=30):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    return tsne.fit_transform(embeddings)

def compute_umap(embeddings, n_components=2, n_neighbors=15, min_dist=0.1):
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    return reducer.fit_transform(embeddings)
