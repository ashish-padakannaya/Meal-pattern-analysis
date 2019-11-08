from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def get_pca_vectors(vectors, k, **opts):
    """generates k PCA vectors 
    
    Arguments:
        vectors {np.array} -- 2D numpy array of features
        k {int} -- no of features
    
    Returns:
        np.array -- 2D numpy array of features
    """
    k = min(k, vectors.shape[1])
    std_scaler = StandardScaler()
    scaled_values = std_scaler.fit_transform(vectors)
    pca = PCA(n_components=k)
    pca_vectors = pca.fit_transform(scaled_values)
    print("Total variance accounted for: ", sum(pca.explained_variance_ratio_))

    return pca_vectors

