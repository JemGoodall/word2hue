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
                reduced_data = self.quartiles(data)
            case 'pca':
                reduced_data = self.pca(data, components)
            case 'tsne':
                reduced_data = self.tsne(data, components)
            case _:
                print(f'Invalid reduction method. Please choose from {self.reduction_methods}')
        return reduced_data

    def quartiles(self, data):
        """ Takes numpy array, returns numpy array of Q1, mean, Q3"""
        q75, q25 = np.percentile(data, [75 ,25], axis=1)
        mean = np.mean(data, axis=1)
        data_stacked = np.vstack((q25, mean, q75)).T
        return data_stacked


    def pca(self, data, components:int):
        print(data.shape)
        pca = PCA(n_components=components)
        reduced = pca.fit_transform(data)
        print(reduced.shape)
        return reduced


    def tsne(self, data, components:int):
        tsne = TSNE(n_components=components)
        reduced = tsne.fit_transform(data)
        return reduced



