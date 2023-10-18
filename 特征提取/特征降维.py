import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis, FastICA, NMF, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import LocallyLinearEmbedding, Isomap, TSNE, SpectralEmbedding, MDS
from sklearn.preprocessing import StandardScaler
import joblib


class FeatureDimensionalityReduction:
    def __init__(self, method='PCA', normalize=False, n_component=3, **kwargs):
        self.method = method
        self.normalize = normalize
        self.kwargs = kwargs
        self.model = None
        self.n_component = n_component

    def fit(self, X, filePath="model.z", scalerPath="scaler.z"):
        if self.normalize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            joblib.dump(scaler, scalerPath)

        if self.method == 'PCA':
            self.model = PCA(self.n_component, **self.kwargs)
        elif self.method == 'LDA':
            self.model = LinearDiscriminantAnalysis(self.n_component, **self.kwargs)
        elif self.method == 'LLE':
            self.model = LocallyLinearEmbedding(n_components=self.n_component, **self.kwargs)
        elif self.method == 'Isomap':
            self.model = Isomap(n_components=self.n_component, **self.kwargs)
        elif self.method == 't-SNE':
            self.model = TSNE(n_components=self.n_component, **self.kwargs)
        elif self.method == 'Laplacian Eigenmaps':
            self.model = SpectralEmbedding(n_components=self.n_component, **self.kwargs)
        elif self.method == 'Modified Locally Linear Embedding':
            self.model = LocallyLinearEmbedding(n_components=self.n_component, **self.kwargs)
        elif self.method == 'Multidimensional Scaling':
            self.model = MDS(n_components=self.n_component, **self.kwargs)
        elif self.method == 'Spectral Embedding':
            self.model = SpectralEmbedding(n_components=self.n_component, **self.kwargs)
        elif self.method == 'Kernel PCA':
            self.model = KernelPCA(n_components=self.n_component, **self.kwargs)
        elif self.method == 'Factor Analysis':
            self.model = FactorAnalysis(n_components=self.n_component, **self.kwargs)
        elif self.method == 'Independent Component Analysis':
            self.model = FastICA(n_components=self.n_component, **self.kwargs)
        elif self.method == 'Non-negative Matrix Factorization':
            self.model = NMF(n_components=self.n_component, **self.kwargs)
        elif self.method == 'Truncated SVD':
            self.model = TruncatedSVD(n_components=self.n_component, **self.kwargs)
        else:
            raise ValueError('Invalid method specified.')

        self.model.fit(X)
        joblib.dump(self.model, filePath)

    def transform(self, X, scalerPath="scaler.z", filePath="model.z"):
        if self.normalize:
            scaler = joblib.load(scalerPath)
            X = scaler.fit_transform(X)
        model = joblib.load(filePath)
        return model.transform(X)
