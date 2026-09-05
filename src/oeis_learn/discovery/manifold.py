"""Manifold dimensionality reduction and density clustering for latent discovery."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


def reduce_manifold_2d(
    embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = 42
) -> np.ndarray:
    """Reduces high-dimensional embeddings to 2D continuous manifold coords using UMAP/TSNE/PCA."""
    n_samples, d = embeddings.shape
    if n_samples < 5:
        return np.zeros((n_samples, 2), dtype=np.float32)

    try:
        from cuml.manifold import UMAP as cuUMAP
        reducer = cuUMAP(n_neighbors=min(n_neighbors, n_samples - 1), min_dist=min_dist, random_state=random_state)
        return np.array(reducer.fit_transform(embeddings), dtype=np.float32)
    except Exception:
        try:
            import umap
            reducer = umap.UMAP(n_neighbors=min(n_neighbors, n_samples - 1), min_dist=min_dist, random_state=random_state)
            return np.array(reducer.fit_transform(embeddings), dtype=np.float32)
        except Exception:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=random_state)
            return np.array(reducer.fit_transform(embeddings), dtype=np.float32)


def cluster_latent_manifold(
    embeddings: np.ndarray, min_cluster_size: int = 5, min_samples: int = 2
) -> np.ndarray:
    """Performs density clustering (HDBSCAN / DBSCAN) on latent embeddings.

    Returns array of integer cluster labels (-1 denotes noise/anomaly).
    """
    n_samples = embeddings.shape[0]
    if n_samples < 3:
        return np.zeros(n_samples, dtype=int)

    try:
        from cuml.cluster import HDBSCAN as cuHDBSCAN
        clusterer = cuHDBSCAN(min_cluster_size=min(min_cluster_size, n_samples), min_samples=min_samples)
        return np.array(clusterer.fit_predict(embeddings), dtype=int)
    except Exception:
        try:
            from sklearn.cluster import HDBSCAN
            clusterer = HDBSCAN(min_cluster_size=min(min_cluster_size, n_samples), min_samples=min_samples)
            return np.array(clusterer.fit_predict(embeddings), dtype=int)
        except Exception:
            from sklearn.cluster import DBSCAN
            clusterer = DBSCAN(eps=0.5, min_samples=min_samples)
            return np.array(clusterer.fit_predict(embeddings), dtype=int)
