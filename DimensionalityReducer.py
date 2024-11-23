from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

REDUCTION_METHODS = ['pca', 'tsne', 'quartiles']

class DimensionalityReducer():
    def __init__(self, reduction_method):
        self.reduction_methods = REDUCTION_METHODS
        assert reduction_method in self.reduction_methods, f"Invalid reduction method. Please choose from {self.reduction_methods}"
        self.reduction_method = reduction_method
    
    def reduce(self, data, components:int):
        match self.reduction_method:
            case 'quartiles':
                reduced_data = self.quartiles(data, components)
            case 'pca':
                reduced_data = self.pca(data, components)
            case 'tsne':
                reduced_data = self.tsne(data, components)
            case _:
                print(f'Invalid reduction method. Please choose from {self.reduction_methods}')
        return reduced_data

    def quartiles(self, data):
        """ Takes numpy array, returns numpy array of Q1, mean, Q3"""
        q75, q25 = np.percentile(data, [75 ,25])
        mean = np.mean(data)
        return np.array([abs(value) for value in [q25, mean, q75]])


    def pca(self, data, components:int):
        print('pca reducer')
        pca = PCA(n_components=components)
        reduced = pca.fit_transform(data)
        return reduced


    def tsne(self, data, components:int):
        tsne = TSNE(n_components=components)
        reduced = tsne.fit_transform(data)
        return reduced



