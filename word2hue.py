import spacy # run in terminal to get model: python3 -m spacy download en_core_web_lg
import numpy as np
import sys
from tqdm import tqdm

# own imports:
from text_processing import get_tokens
from DimensionalityReducer import DimensionalityReducer

model = spacy.load('en_core_web_lg')


def normalize(x, new_range=(0, 1)): #x is an array. Default range is between zero and one
    xmin, xmax = np.min(x), np.max(x) #get max and min from input array
    norm = (x - xmin)/(xmax - xmin) # scale between zero and one
    if new_range != (0, 1):
        return norm * (new_range[1] - new_range[0]) + new_range[0] #scale to a different range.    
    return(norm)


def word2hue(text_input, n_tokens=-1, save_location='', keep_punctuation=True, reduction_method='pca'):
    tokens = get_tokens(text_input, keep_punctuation, n_tokens)
    word_embeddings = np.array([model(tokens).vector for tokens in tqdm(tokens,'Getting word vectors')])
    reduce_dimensions = DimensionalityReducer(reduction_method)
    reduced_embeddings = reduce_dimensions.reduce(word_embeddings, 3)
    print(f"Vector (size {word_embeddings.shape}) reduced to: {reduced_embeddings.shape}")
    word_rgb_values = normalize(reduced_embeddings, new_range=(0,255)).astype(int)

    return tokens, word_rgb_values

