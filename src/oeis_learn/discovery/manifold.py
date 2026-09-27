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

    # Try GPU-accelerated UMAP first (RAPIDS cuML)
    try:
        from cuml.manifold import UMAP as cuUMAP
        reducer = cuUMAP(n_neighbors=min(n_neighbors, n_samples - 1), min_dist=min_dist, random_state=random_state)
        return np.array(reducer.fit_transform(embeddings), dtype=np.float32)
    except Exception:
        pass
    
    # Try CPU UMAP
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=min(n_neighbors, n_samples - 1), min_dist=min_dist, random_state=random_state)
        return np.array(reducer.fit_transform(embeddings), dtype=np.float32)
    except Exception:
        pass
    
    # Fall back to PCA if sklearn is available
    try:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=random_state)
        return np.array(reducer.fit_transform(embeddings), dtype=np.float32)
    except Exception:
        pass
    
    # If no dimensionality reduction available, return random 2D projection
    logger.warning("No manifold reduction available (sklearn, umap, cuml), returning random projection")
    np.random.seed(random_state)
    return np.random.randn(n_samples, 2).astype(np.float32)


def cluster_latent_manifold(
    embeddings: np.ndarray, min_cluster_size: int = 5, min_samples: int = 2
) -> np.ndarray:
    """Performs density clustering (HDBSCAN / DBSCAN) on latent embeddings.

    Returns array of integer cluster labels (-1 denotes noise/anomaly).
    """
    n_samples = embeddings.shape[0]
    if n_samples < 3:
        return np.zeros(n_samples, dtype=int)

    # Try GPU-accelerated HDBSCAN first (RAPIDS cuML)
    try:
        from cuml.cluster import HDBSCAN as cuHDBSCAN
        clusterer = cuHDBSCAN(min_cluster_size=min(min_cluster_size, n_samples), min_samples=min_samples)
        return np.array(clusterer.fit_predict(embeddings), dtype=int)
    except Exception:
        pass
    
    # Try sklearn HDBSCAN
    try:
        from sklearn.cluster import HDBSCAN
        clusterer = HDBSCAN(min_cluster_size=min(min_cluster_size, n_samples), min_samples=min_samples)
        return np.array(clusterer.fit_predict(embeddings), dtype=int)
    except Exception:
        pass
    
    # Fall back to DBSCAN if sklearn available
    try:
        from sklearn.cluster import DBSCAN
        clusterer = DBSCAN(eps=0.5, min_samples=min_samples)
        return np.array(clusterer.fit_predict(embeddings), dtype=int)
    except Exception:
        pass
    
    # If no clustering available, return all same cluster
    logger.warning("No clustering available (sklearn, cuml), returning single cluster")
    return np.zeros(n_samples, dtype=int)
